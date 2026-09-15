import numpy as np
from .data_preprocessing import (training_mito_lengths, training_dna_numbers, training_mito_length_0dy, training_dna_number_0dy,
                                 pulse_data_summary_statistics, pulse_data_summary_statistics_variance, chase_data_summary_statistics,
                                 chase_data_summary_statistics_variance,
                                 pulse_data_summary_statistics_with_hetero, pulse_data_summary_statistics_with_hetero_variance,
                                 pulse_data_summary_statistics_with_edu_hetero, pulse_data_summary_statistics_with_edu_hetero_variance
)
import os
from numba import jit, prange
import scipy.stats as sc
from .stochastic_systems_models import (pulse_summary_statistics, logarithmic_three_population_pulse, logarithmic_two_population_pulse,
                                        logarithmic_one_population_pulse, logarithmic_three_population_chase, logarithmic_two_population_chase,
                                        chase_summary_statistics, ou_inhibition_three_population_pulse, logarithmic_dispersed_three_population_pulse,
                                        ou_inhibition_three_population_pulse, pulse_heteroscedastic_summary_statistics, pulse_edu_heteroscedastic_summary_statistics)
dirname = os.path.dirname(__file__)

#Gibbs sampler for beta0, beta1, sigma
error_correction = 1/(1-3.6/100)
X_pulse = np.concatenate([np.ones(len(training_mito_lengths))[:,np.newaxis], training_mito_lengths[:,np.newaxis]], axis=1)
Y_pulse = error_correction*training_dna_numbers
Lambda_pulse = np.diag(training_mito_lengths**2)

X_chase = np.concatenate([np.ones(len(training_mito_length_0dy))[:,np.newaxis], training_mito_length_0dy[:,np.newaxis]], axis=1)
Y_chase = error_correction*training_dna_number_0dy
Lambda_chase = np.diag(training_mito_length_0dy**2)

def sample_beta(sig_squared, X,Y, Lambda):
    XtX = np.dot(X.transpose(), X)
    XtX_inv = np.linalg.inv(XtX)
    mean = np.dot(XtX_inv, X.transpose())
    mean = np.dot(mean, Y)

    Omega = np.dot(X.transpose(), Lambda)
    Omega = np.dot(Omega, X)
    var = np.dot(XtX_inv, sig_squared*Omega)
    var = np.dot(var, XtX_inv)

    return np.random.multivariate_normal(mean, var)

def sample_sigma(beta, sig0_squared, v0, X, Y, Lambda):
    a = v0 + 2 + len(Y)
    E = np.divide(Y - np.dot(X, beta), np.diag(np.sqrt(Lambda)))
    delta = sig0_squared*v0 + np.linalg.norm(E)**2
    return 1/np.random.gamma(a/2, 2/delta)

def gibbs_sampler(n, sig0_squared, v0, experiment = "pulse"):

    if experiment == "pulse":
        X = X_pulse
        Y = Y_pulse
        Lambda = Lambda_pulse

    elif experiment == "chase":
        X = X_chase
        Y = Y_chase
        Lambda = Lambda_chase

    XtX = np.dot(X.transpose(), X)
    XtX_inv = np.linalg.inv(XtX)
    mean = np.dot(XtX_inv, X.transpose())

    beta = np.dot(mean, Y)
    sig_squared = sample_sigma(beta, sig0_squared, v0, X, Y, Lambda)

    betas = []
    sig_squareds = []
    for i in range(n):
        beta = sample_beta(sig_squared, X, Y, Lambda)
        sig_squared = sample_sigma(beta, sig0_squared,v0, X, Y, Lambda)
        betas.append(beta)
        sig_squareds.append(sig_squared)

    return np.array(betas), np.array(sig_squareds)

#Defining priors
def sample_three_population_pulse_prior(num_samples):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    betas, _ = gibbs_sampler(num_samples, 0.25**2, 1)
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    diffusion_prob_pulse = np.random.uniform(0,1,num_samples)
    mu_d_pulse = sc.loguniform.rvs(0.005, 1, size = num_samples)
    mu_b_pulse = sc.loguniform.rvs(10**(-5), 1, size = num_samples)
    true_birth = 1/np.random.uniform(0.1, 10, num_samples)
    c = sc.loguniform.rvs(10**-5, 10**-1, size = num_samples)

    return np.array([beta0,beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth, c])

def sample_preliminary_two_population_pulse_prior(num_samples):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    betas, _ = gibbs_sampler(num_samples, 0.5**2, 1)
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    #diffusion_prob_pulse = np.random.uniform(0,1,num_samples)
    diffusion_prob_pulse = np.ones(num_samples)
    mu_d_pulse = sc.loguniform.rvs(0.005, 0.1, size = num_samples)
    true_birth = 1/np.random.uniform(0.1, 10, num_samples)
    c = sc.loguniform.rvs(10**-5, 10**-2, size = num_samples)

    return np.array([beta0,beta1, diffusion_prob_pulse, mu_d_pulse, true_birth, c])

def sample_two_population_pulse_prior(num_samples):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    betas, _ = gibbs_sampler(num_samples, 0.25**2, 1)
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    diffusion_prob_pulse = np.random.uniform(0,1,num_samples)
    mu_d_pulse = sc.loguniform.rvs(0.005, 0.1, size = num_samples)
    true_birth = 1/np.random.uniform(0.1, 10, num_samples)
    c = sc.loguniform.rvs(10**-5, 10**-2, size = num_samples)

    return np.array([beta0,beta1, diffusion_prob_pulse, mu_d_pulse, true_birth, c])

def sample_one_population_pulse_prior(num_samples):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    betas, _ = gibbs_sampler(num_samples, 0.25**2, 1)
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    mu_d_pulse = sc.loguniform.rvs(0.005, 0.1, size = num_samples)
    c = sc.loguniform.rvs(10**-5, 10**-1, size = num_samples)

    return np.array([beta0,beta1, mu_d_pulse, c])

def sample_ou_prior(num_samples):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    betas, _ = gibbs_sampler(num_samples, 0.25**2, 1)
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    diffusion_prob_pulse = np.random.uniform(0,1,num_samples)
    mu_d_pulse = sc.loguniform.rvs(0.005, 1, size = num_samples)
    mu_b_pulse = sc.loguniform.rvs(10**(-5), 1, size = num_samples)
    true_birth = 1/np.random.uniform(0.1, 10, num_samples)
    c = mu_b_pulse*sc.loguniform.rvs(1, 30, size = num_samples)
    theta = sc.loguniform.rvs(10**-3, 10**0, size = num_samples)
    sd = np.random.uniform(0,1,num_samples)

    return np.array([beta0,beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth, c, theta, sd])

def sample_dispersed_prior(num_samples):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    betas, _ = gibbs_sampler(num_samples, 0.25**2, 1)
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    diffusion_prob_pulse = np.random.uniform(0,1,num_samples)
    mu_d_pulse = sc.loguniform.rvs(0.005, 1, size = num_samples)
    mu_b_pulse = sc.loguniform.rvs(10**(-5), 1, size = num_samples)
    true_birth = 1/np.random.uniform(0.1, 10, num_samples)
    c = sc.loguniform.rvs(10**-5, 10**-1, size = num_samples) 
    ksigs = np.random.uniform(0,1, num_samples)#logarithmic

    return np.array([beta0,beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth, c, ksigs])

#We use the posteriors of the pulse ABC fit to construct the priors of the chase ABC fit, so we need to specify pulse_samples
three_population_pulse_accepted_parameters = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))
two_population_pulse_accepted_parameters = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_accepted_params.npy"))

def sample_three_population_chase_prior(num_samples, pulse_samples = three_population_pulse_accepted_parameters):
        """
        Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
        sample of a specific parameter. If nums_samples=1, this is a column vector.
        """
        #Take 500000 samples
        pulse_parameter_index = np.random.choice(len(pulse_samples), num_samples)
        pulse_samples_with_rep = pulse_samples[pulse_parameter_index]

        #compute pertubation variance
        diffusion_prob_pulse_var = np.var(np.transpose(pulse_samples)[2])
        mu_d_pulse_log_var = np.var(np.log(np.transpose(pulse_samples)[3]))
        mu_b_pulse_log_var = np.var(np.log(np.transpose(pulse_samples)[4]))
        mu_r_pulse_recip_var = np.var(1/(np.transpose(pulse_samples)[5]))
        c_log_var = np.var(np.log(np.transpose(pulse_samples)[6]))

        #Extract parameters
        diffusion_prob_pulse = np.transpose(pulse_samples_with_rep)[2]
        mu_d_pulse = np.transpose(pulse_samples_with_rep)[3]
        mu_b_pulse = np.transpose(pulse_samples_with_rep)[4]
        true_birth = np.transpose(pulse_samples_with_rep)[5]
        c = np.transpose(pulse_samples_with_rep)[6]

        #Perturb parameters
        diffusion_prob_pulse_pert = np.minimum(np.abs(diffusion_prob_pulse + np.random.normal(0, np.sqrt(0.1*diffusion_prob_pulse_var), num_samples)),1)
        mu_d_pulse_pert = np.exp(np.log(mu_d_pulse) + np.random.normal(0, np.sqrt(0.1*mu_d_pulse_log_var), num_samples))
        mu_b_pulse_pert = np.exp(np.log(mu_b_pulse) + np.random.normal(0, np.sqrt(0.1*mu_b_pulse_log_var), num_samples))
        true_birth_pert = np.abs(1/(1/true_birth + np.random.normal(0,np.sqrt(0.1*mu_r_pulse_recip_var), num_samples)))
        c_pert = np.exp(np.log(c) + np.random.normal(0, np.sqrt(0.1*c_log_var), num_samples))
        
        mu_b_spike=np.random.binomial(1,0.5, num_samples)
        s=np.random.binomial(1,0.5, num_samples)
        mu_a_spike = (mu_b_spike + s > 0) #Given that mu_b=0, this is 0 with probability 1/2, otherwise, it is 1.

        mu_b_chase = mu_b_spike*sc.loguniform.rvs(10**(-6), 10**(-2), size = num_samples)
        mu_a_chase = mu_a_spike*sc.loguniform.rvs(10**(-6), 1, size = num_samples) + np.divide(mu_b_chase, diffusion_prob_pulse_pert)

        betas, sig = gibbs_sampler(num_samples, 0.5**2, 1, experiment = "chase")
        beta0 = np.transpose(betas)[0]
        beta1 = np.transpose(betas)[1]

        return np.array([beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_b_pulse_pert, true_birth_pert, mu_b_chase, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_a_chase, c_pert])

def sample_two_population_chase_prior(num_samples, pulse_samples = two_population_pulse_accepted_parameters):
        """
        Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
        sample of a specific parameter. If nums_samples=1, this is a column vector.
        """
        #Take 500000 samples
        pulse_parameter_index = np.random.choice(len(pulse_samples), num_samples)
        pulse_samples_with_rep = pulse_samples[pulse_parameter_index]

        #beta0,beta1, diffusion_prob_pulse, mu_d_pulse, true_birth, c
        #compute pertubation variance
        diffusion_prob_pulse_var = np.var(np.transpose(pulse_samples)[2])
        mu_d_pulse_log_var = np.var(np.log(np.transpose(pulse_samples)[3]))
        mu_r_pulse_recip_var = np.var(1/(np.transpose(pulse_samples)[4]))
        c_log_var = np.var(np.log(np.transpose(pulse_samples)[5]))

        #Extract parameters
        diffusion_prob_pulse = np.transpose(pulse_samples_with_rep)[2]
        mu_d_pulse = np.transpose(pulse_samples_with_rep)[3]
        true_birth = np.transpose(pulse_samples_with_rep)[4]
        c = np.transpose(pulse_samples_with_rep)[5]

        #Perturb parameters
        diffusion_prob_pulse_pert = np.minimum(np.abs(diffusion_prob_pulse + np.random.normal(0, np.sqrt(0.1*diffusion_prob_pulse_var), num_samples)),1)
        mu_d_pulse_pert = np.exp(np.log(mu_d_pulse) + np.random.normal(0, np.sqrt(0.1*mu_d_pulse_log_var), num_samples))
        true_birth_pert = np.abs(1/(1/true_birth + np.random.normal(0,np.sqrt(0.1*mu_r_pulse_recip_var), num_samples)))
        c_pert = np.exp(np.log(c) + np.random.normal(0, np.sqrt(0.1*c_log_var), num_samples))

        diffusion_prob_chase = diffusion_prob_pulse_pert

        mu_b_chase = np.random.binomial(1,0.5, num_samples)*sc.loguniform.rvs(10**(-6), 10**(-2), size = num_samples)
        mu_d_chase = sc.loguniform.rvs(10**(-6), 1, size = num_samples) + np.divide(mu_b_chase, diffusion_prob_chase)

        betas, sig = gibbs_sampler(num_samples, 0.5**2, 1, experiment = "chase")
        beta0 = np.transpose(betas)[0]
        beta1 = np.transpose(betas)[1]

        return np.array([beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, true_birth_pert, mu_b_chase, diffusion_prob_chase, mu_d_chase, c_pert])

def sample_variablep_two_population_chase_prior(num_samples, pulse_samples = two_population_pulse_accepted_parameters):
        """
        Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
        sample of a specific parameter. If nums_samples=1, this is a column vector.
        """
        #Take 500000 samples
        pulse_parameter_index = np.random.choice(len(pulse_samples), num_samples)
        pulse_samples_with_rep = pulse_samples[pulse_parameter_index]

        #beta0,beta1, diffusion_prob_pulse, mu_d_pulse, true_birth, c
        #compute pertubation variance
        diffusion_prob_pulse_var = np.var(np.transpose(pulse_samples)[2])
        mu_d_pulse_log_var = np.var(np.log(np.transpose(pulse_samples)[3]))
        mu_r_pulse_recip_var = np.var(1/(np.transpose(pulse_samples)[4]))
        c_log_var = np.var(np.log(np.transpose(pulse_samples)[5]))

        #Extract parameters
        diffusion_prob_pulse = np.transpose(pulse_samples_with_rep)[2]
        mu_d_pulse = np.transpose(pulse_samples_with_rep)[3]
        true_birth = np.transpose(pulse_samples_with_rep)[4]
        c = np.transpose(pulse_samples_with_rep)[5]

        #Perturb parameters
        diffusion_prob_pulse_pert = np.minimum(np.abs(diffusion_prob_pulse + np.random.normal(0, np.sqrt(0.1*diffusion_prob_pulse_var), num_samples)),1)
        mu_d_pulse_pert = np.exp(np.log(mu_d_pulse) + np.random.normal(0, np.sqrt(0.1*mu_d_pulse_log_var), num_samples))
        true_birth_pert = np.abs(1/(1/true_birth + np.random.normal(0,np.sqrt(0.1*mu_r_pulse_recip_var), num_samples)))
        c_pert = np.exp(np.log(c) + np.random.normal(0, np.sqrt(0.1*c_log_var), num_samples))

        diffusion_prob_chase = np.random.uniform(0,1,num_samples)


        mu_b_chase = np.random.binomial(1,0.5, num_samples)*sc.loguniform.rvs(10**(-6), 10**(-2), size = num_samples)
        mu_d_chase = sc.loguniform.rvs(10**(-6), 1, size = num_samples) + np.divide(mu_b_chase, diffusion_prob_chase)

        betas, sig = gibbs_sampler(num_samples, 0.5**2, 1, experiment = "chase")
        beta0 = np.transpose(betas)[0]
        beta1 = np.transpose(betas)[1]

        return np.array([beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, true_birth_pert, mu_b_chase, diffusion_prob_chase, mu_d_chase, c_pert])

# def sample_one_population_chase_prior(num_samples, pulse_samples):
#         """
#         Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
#         sample of a specific parameter. If nums_samples=1, this is a column vector.
#         """
#         #Take 500000 samples
#         pulse_parameter_index = np.random.choice(len(pulse_samples), num_samples)
#         pulse_samples_with_rep = pulse_samples[pulse_parameter_index]

#         #beta0,beta1, diffusion_prob_pulse, mu_d_pulse, true_birth, c
#         #compute pertubation variance
#         mu_d_pulse_log_var = np.var(np.log(np.transpose(pulse_samples)[2]))
#         c_log_var = np.var(np.log(np.transpose(pulse_samples)[3]))

#         #Extract parameters
#         mu_d_pulse = np.transpose(pulse_samples_with_rep)[2]
#         c = np.transpose(pulse_samples_with_rep)[3]

#         #Perturb parameters
#         mu_d_pulse_pert = np.exp(np.log(mu_d_pulse) + np.random.normal(0, np.sqrt(0.1*mu_d_pulse_log_var), num_samples))
#         c_pert = np.exp(np.log(c) + np.random.normal(0, np.sqrt(0.1*c_log_var), num_samples))


#         mu_b_chase = np.random.binomial(1,0.5, num_samples)*sc.loguniform.rvs(10**(-6), 10**(-2), size = num_samples)
#         mu_d_chase = sc.loguniform.rvs(10**(-6), 1, size = num_samples) + mu_b_chase

#         betas, sig = gibbs_sampler(num_samples, 0.5**2, 1, experiment = "chase")
#         beta0 = np.transpose(betas)[0]
#         beta1 = np.transpose(betas)[1]

#         return np.array([beta0, beta1, sig, mu_d_pulse_pert, mu_b_chase, mu_d_chase, c_pert])

@jit(parallel=True)
def run_parallel(simulated_summaries, simulated_params, stochastic_simulator, summary_statistic, mode = "training"):
    for i in prange(len(simulated_summaries)):
        z = stochastic_simulator(simulated_params[i], mode = mode)
        print("Simulation " + str(i) + " Finished")
        s = summary_statistic(z)
        simulated_summaries[i] = s

    print("Simulations finished, returning outputs")

    return  (simulated_summaries, simulated_params)

def run(number_sims, model = "three population", inference_portion = "pulse", hetero = False, edu_hetero = False):
    extra_dimensions = 0
    if hetero:
        extra_dimensions = 1
    if edu_hetero:
        extra_dimensions = 5
    
    if inference_portion == "pulse":
        summary_statistic_shape = 6 + extra_dimensions
        if model == "three population":
            summary_statistic = pulse_summary_statistics
            stochastic_simulator = logarithmic_three_population_pulse
            sample_prior = sample_three_population_pulse_prior
        elif model == "two population":
            summary_statistic = pulse_summary_statistics
            stochastic_simulator = logarithmic_two_population_pulse
            sample_prior = sample_two_population_pulse_prior
        elif model == "preliminary two population":
            summary_statistic = pulse_summary_statistics
            stochastic_simulator = logarithmic_two_population_pulse
            sample_prior = sample_preliminary_two_population_pulse_prior
        elif model == "one population":
            summary_statistic = pulse_summary_statistics
            stochastic_simulator = logarithmic_one_population_pulse
            sample_prior = sample_one_population_pulse_prior
        elif model == "stochastic three population":
            summary_statistic = pulse_heteroscedastic_summary_statistics
            stochastic_simulator = ou_inhibition_three_population_pulse
            sample_prior = sample_ou_prior
        elif model == "dispersed three population":
            summary_statistic = pulse_edu_heteroscedastic_summary_statistics
            stochastic_simulator = logarithmic_dispersed_three_population_pulse
            sample_prior = sample_dispersed_prior

    elif inference_portion == "chase":
        summary_statistic = chase_summary_statistics
        summary_statistic_shape = 15
        if model == "three population":
            stochastic_simulator = logarithmic_three_population_chase
            sample_prior = sample_three_population_chase_prior
        elif model == "two population":
            stochastic_simulator = logarithmic_two_population_chase
            sample_prior = sample_two_population_chase_prior
        elif model == "variable p two population":
            stochastic_simulator = logarithmic_two_population_chase
            sample_prior = sample_variablep_two_population_chase_prior


    simulated_summaries = np.zeros((number_sims, summary_statistic_shape)).astype(np.float64)
    simulated_params = np.transpose(sample_prior(number_sims)).astype(np.float64)

    return run_parallel(simulated_summaries, simulated_params, stochastic_simulator, summary_statistic)

def run_validation(number_sims = 500, inference_portion = "pulse", mode = "validation"):

    if inference_portion == "pulse":
        summary_statistic_shape = 6
        params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))[:number_sims]
        stochastic_simulator = logarithmic_three_population_pulse
        summary_statistic = pulse_summary_statistics
    elif inference_portion == "chase":
        summary_statistic_shape = 15
        params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))[:number_sims]
        stochastic_simulator = logarithmic_three_population_chase
        summary_statistic = chase_summary_statistics

    simulated_summaries = np.zeros((number_sims, summary_statistic_shape)).astype(np.float64)

    return run_parallel(simulated_summaries, params, stochastic_simulator, summary_statistic, mode)

def mahalanobis_distance(x, y, inverse_cov):
    v = (x-y.transpose()).transpose()
    a = np.diag(inverse_cov)[np.newaxis].transpose() * v
    #a = np.matmul(inverse_cov, v)

    return np.sum(v*a, axis=0)
    
def ABC_distances(simulated_summaries, inference_portion = "pulse", hetero = False, edu_hetero = False):
    if inference_portion == "pulse":

        if edu_hetero:
            true_data = pulse_data_summary_statistics_with_edu_hetero
            true_data_variance = pulse_data_summary_statistics_with_edu_hetero_variance
        elif hetero:
            true_data = pulse_data_summary_statistics_with_hetero
            true_data_variance = pulse_data_summary_statistics_with_hetero_variance
        else:
            true_data = pulse_data_summary_statistics
            true_data_variance = pulse_data_summary_statistics_variance

    elif inference_portion == "chase":

        true_data = chase_data_summary_statistics
        true_data_variance = chase_data_summary_statistics_variance


    precision_matrix = np.diag(1/true_data_variance)

    return mahalanobis_distance(simulated_summaries, true_data, precision_matrix)

def ABC_reject(simulated_summaries, simulated_params, proportion_accepted, inference_portion="pulse", hetero=False, edu_hetero=False):
    """
    Pick the smallest h such that the proportion of simulations accepted is 'proportion_accepted'
    """
    number_accepted = round(proportion_accepted*len(simulated_params))
    bandwidths = np.array(ABC_distances(simulated_summaries,  inference_portion = inference_portion, hetero = hetero, edu_hetero = edu_hetero))
    idx = np.argpartition(bandwidths, number_accepted)

    accepted_params = np.array(simulated_params)[idx[:number_accepted]]
    bandwidth = bandwidths[idx[number_accepted]]

    return accepted_params, bandwidth

#Functions to simulate posterior predictive trajectories
@jit(parallel = True)
def chase_posterior_predictive_simulator(model, stochastic_simulator, params, mode):
    if model == "three population":
        dim = 12
    elif model == "two population" or model == "variable p two population":
        dim = 9

    if mode == "full":
        num_cells = 265
    elif mode == "training":
        num_cells = 181
    elif mode == "validation":
        num_cells = 84

    trajectories = np.zeros((len(params), num_cells, 385, dim))
    variance_statistics = np.zeros(len(params))
    peak1_0day = np.zeros(len(params))
    peak1_4day = np.zeros(len(params))

    for i in prange(len(params)):
        param = params[i]
        (_, _, _, _, _, _,
                _, _, _, _, _, _,
                initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
                variance_statistic, trajectory_chase) = stochastic_simulator(param, verbose = False, full_trajectory=True, mode=mode)

        
        #appending the trajectory of all cells in this simulation
        trajectories[i] = trajectory_chase

        #the peak 1 proportions
        peak1_0day[i] = initial_average_peak1_proportion_chase
        peak1_4day[i] = final_average_peak1_proportion_chase

        #and the variance statistic
        variance_statistics[i] = variance_statistic

        print(str(i+1) +"/" + str(len(params)))

    return (trajectories, peak1_0day, peak1_4day, variance_statistics)

@jit(parallel=True)
def pulse_posterior_predictive_simulator(model, stochastic_simulator, params, mode):
    if model == "three population" or model == "stochastic three population" or model == "dispersed three population":
        dim = 9
    elif model == "two population" or model == "preliminary two population":
        dim = 6
    elif model == "one population":
        dim = 3
    if mode == "full":
        num_cells = 382
    elif mode == "training":
        num_cells = 247
    elif mode == "validation":
        num_cells = 135
    
    trajectories = np.zeros((len(params), num_cells, 97, dim))
    peak1 = np.zeros(len(params))
    variance_statistics = np.zeros(len(params))
    for i in prange(len(params)):
        param = params[i]
        (_, _, _, _, _, _, 
           _, _, _, _, _, _, 
           average_peak1_proportion, variance_statistic, trajectory, _,_,_,_,_) = stochastic_simulator(param, verbose = False, full_trajectory=True, mode=mode)

        #appending the trajectories of all cells in this simulation
        trajectories[i] = trajectory

        #the peak 1 proportions
        peak1[i] = average_peak1_proportion

        #and the variance statistics
        variance_statistics[i] = variance_statistic

        print(str(i+1) +"/" + str(len(params)))

    return (trajectories, peak1, variance_statistics)

def posterior_predictive(num_samples = 500, model = "three population", inference_portion = "pulse", mode = "full", my_params = False):
    #Pulse simulation
    if inference_portion == "pulse":
        if model == "three population":
            stochastic_simulator = logarithmic_three_population_pulse
            params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))
        elif model == "two population":
            stochastic_simulator = logarithmic_two_population_pulse
            params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_accepted_params.npy"))
        
        elif model == "preliminary two population":
            stochastic_simulator = logarithmic_two_population_pulse
            params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/preliminary/preliminary_two_population_pulse_accepted_params.npy"))
        
        elif model == "one population":
            stochastic_simulator = logarithmic_one_population_pulse
            params = np.load(os.path.join(dirname, "simulated_ABC_data/one_population_model/one_population_pulse_accepted_params.npy"))

        elif model == "stochastic three population":
            stochastic_simulator = ou_inhibition_three_population_pulse
            params = np.load(os.path.join(dirname, "simulated_ABC_data_birth_rate_selection/stochastic_inhibition/three_population_pulse_stocinh_accepted_params.npy"))

        elif model == "dispersed three population":
            stochastic_simulator = logarithmic_dispersed_three_population_pulse
            params = np.load(os.path.join(dirname, "simulated_ABC_data_birth_rate_selection/logarithmic/dispersed_three_population_pulse_accepted_params.npy"))

        if not isinstance(my_params, bool):
            params = my_params

        params = params[:num_samples]

        #Running the simulations in parallel
        (trajectories, peak1, variance_statistics)  = pulse_posterior_predictive_simulator(model, stochastic_simulator, params, mode)

        #post processing the trajectories into edu and nucleoid trajectories
        nucleoid_trajectories = np.zeros((len(params), 97))
        edu_trajectories = np.zeros((len(params), 97))
        for j in range(len(trajectories)):
            
            trajectory = trajectories[j]
            nucleoid_trajectory = np.sum(trajectory, axis = 2)
            if model == "three population" or model == "stochastic three population" or model == "dispersed three population":
                edu_trajectory = nucleoid_trajectory - trajectory[:,:,3] - trajectory[:,:,6]
            elif model == "two population" or model == "preliminary two population":
                edu_trajectory = nucleoid_trajectory - trajectory[:,:,3]
            elif model == "one population":
                edu_trajectory = nucleoid_trajectory - trajectory[:,:,0]
            
            #mean across all cells
            nucleoid_trajectories[j] = np.mean(nucleoid_trajectory, axis=0)
            edu_trajectories[j] = np.mean(edu_trajectory, axis=0)

        return (nucleoid_trajectories, edu_trajectories, peak1, variance_statistics)
    
    #Chase simulation
    elif inference_portion == "chase":
        if model == "three population":
            stochastic_simulator = logarithmic_three_population_chase
            params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))

        elif model == "two population":
            stochastic_simulator = logarithmic_two_population_chase
            params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_chase_accepted_params.npy"))

        elif model == "variable p two population":
            stochastic_simulator = logarithmic_two_population_chase
            params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/variable_p/two_population_chase_variablep_accepted_params.npy"))


        if not isinstance(my_params, bool):
            params = my_params
        params = params[:num_samples]
        
        #Running the simulations in parallel
        (trajectories, peak1_0day, peak1_4day, variance_statistics) = chase_posterior_predictive_simulator(model, stochastic_simulator, params, mode)

        #post processing the trajectories into edu and nucleoid trajectories
        nucleoid_trajectories = np.zeros((len(params), 385))
        edu_trajectories = np.zeros((len(params), 385))
        edu_proportions = np.zeros((len(params), 385))
        for j in range(len(trajectories)):
            
            trajectory = trajectories[j]
            nucleoid_trajectory = np.sum(trajectory, axis = 2)
            if model == "three population":
                edu_trajectory = nucleoid_trajectory - trajectory[:,:,0] - trajectory[:,:,3] - trajectory[:,:,9]
            elif model == "two population" or model == "variable p two population":
                edu_trajectory = nucleoid_trajectory - trajectory[:,:,3]
                edu_trajectory = nucleoid_trajectory - trajectory[:,:,0] - trajectory[:,:,3]

            #mean over all cells
            nucleoid_trajectories[j] = np.mean(nucleoid_trajectory, axis=0)
            edu_trajectories[j] = np.mean(edu_trajectory, axis=0)
            edu_proportions[j] = np.mean(np.divide(edu_trajectory,nucleoid_trajectory), axis = 0)

        return (nucleoid_trajectories, edu_trajectories, edu_proportions, peak1_0day, peak1_4day, variance_statistics)

# @jit(parallel=True)
# def single_cell_posterior_predictive(num_samples = 500, inference_portion = "pulse"):
    
#     if inference_portion == "pulse":
#         stochastic_simulator = logarithmic_three_population_pulse
#         params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))

#     else:
#         stochastic_simulator = logarithmic_three_population_chase    
#         params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    
#     k = min(num_samples, len(params))
#     nucleoid1 = np.zeros(k)
#     nucleoid2 = np.zeros(k)
#     nucleoid3 = np.zeros(k)
#     nucleoid4 = np.zeros(k)
#     edu1 = np.zeros(k)
#     edu2 = np.zeros(k)
#     edu3 = np.zeros(k)
#     edu4 = np.zeros(k)

#     for i in prange(k):
#         print(i)
#         param = params[i]
#         if inference_portion == "pulse":
#             n1,e1,_,n2,e2,_,n3,e3,_,n4,e4,_,_,_,_,_ = stochastic_simulator(params=param, verbose="False", full_trajectory="False", mode = "full", burn_in = 1000, increments=250)
#         else:
#             n1,e1,_,n2,e2,_,n3,e3,_,n4,e4,_,_,_,_,_,_ = stochastic_simulator(params=param, verbose="False", full_trajectory="False", mode = "full", burn_in=1000, increments=250)
#         nucleoid1[i] = n1
#         nucleoid2[i] = n2
#         nucleoid3[i] = n3
#         nucleoid4[i] = n4
#         edu1[i] = e1
#         edu2[i] = e2
#         edu3[i] = e3
#         edu4[i] = e4

#     return nucleoid1, nucleoid2, nucleoid3, nucleoid4, edu1, edu2, edu3, edu4


def single_cell_posterior_predictive(num_samples=500, inference_portion="pulse", dispersed=False):

    if not dispersed:
        if inference_portion == "pulse":
            stochastic_simulator = logarithmic_three_population_pulse
            params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))
        else:
            stochastic_simulator = logarithmic_three_population_chase    
            params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))

    else:
        stochastic_simulator = logarithmic_dispersed_three_population_pulse
        params = np.load(os.path.join(dirname, "simulated_ABC_data_birth_rate_selection/logarithmic/dispersed_three_population_pulse_accepted_params.npy"))

    k = min(num_samples, len(params))

    # Preallocate arrays
    if inference_portion == "pulse":
        results = _compute_posterior_pulse(k, params, stochastic_simulator)
    else:
        results = _compute_posterior_chase(k, params, stochastic_simulator)

    return results

@jit(parallel=True)
def _compute_posterior_pulse(k, params, stochastic_simulator):
    n1,e1,_,n2,e2,_,n3,e3,_,n4,e4,_,_,_,_,_,_,_,_,_ = stochastic_simulator(
    params = params[0], verbose = False, full_trajectory = False, mode = "full", burn_in = 10
        )
    nucleoid1 = np.zeros((k, len(n1)))
    nucleoid2 = np.zeros((k, len(n2)))
    nucleoid3 = np.zeros((k, len(n3)))
    nucleoid4 = np.zeros((k, len(n4)))
    edu1 = np.zeros((k, len(e1)))
    edu2 = np.zeros((k, len(e2)))
    edu3 = np.zeros((k, len(e3)))
    edu4 = np.zeros((k, len(e4)))

    for i in prange(k):
        print(i)
        param = params[i]
        n1,e1,_,n2,e2,_,n3,e3,_,n4,e4,_,_,_,_,_,_,_,_,_ = stochastic_simulator(
            params = param, verbose = False, full_trajectory = False, mode = "full", burn_in = 1000
        )
        nucleoid1[i] = n1
        nucleoid2[i] = n2
        nucleoid3[i] = n3
        nucleoid4[i] = n4
        edu1[i] = e1
        edu2[i] = e2
        edu3[i] = e3
        edu4[i] = e4

    return nucleoid1, nucleoid2, nucleoid3, nucleoid4, edu1, edu2, edu3, edu4

@jit(parallel=True)
def _compute_posterior_chase(k, params, stochastic_simulator):
    n1,e1,_,n2,e2,_,n3,e3,_,n4,e4,_,_,_,_,_ = stochastic_simulator(
    params = params[0], verbose = False, full_trajectory = False, mode = "full", burn_in = 10
        )
    nucleoid1 = np.zeros((k, len(n1)))
    nucleoid2 = np.zeros((k, len(n2)))
    nucleoid3 = np.zeros((k, len(n3)))
    nucleoid4 = np.zeros((k, len(n4)))
    edu1 = np.zeros((k, len(e1)))
    edu2 = np.zeros((k, len(e2)))
    edu3 = np.zeros((k, len(e3)))
    edu4 = np.zeros((k, len(e4)))

    for i in prange(k):
        print(i)
        param = params[i]
        n1,e1,_,n2,e2,_,n3,e3,_,n4,e4,_,_,_,_,_ = stochastic_simulator(
            params = param, verbose = False, full_trajectory = False, mode = "full", burn_in = 1000
        )
        nucleoid1[i] = n1
        nucleoid2[i] = n2
        nucleoid3[i] = n3
        nucleoid4[i] = n4
        edu1[i] = e1
        edu2[i] = e2
        edu3[i] = e3
        edu4[i] = e4

    return nucleoid1, nucleoid2, nucleoid3, nucleoid4, edu1, edu2, edu3, edu4