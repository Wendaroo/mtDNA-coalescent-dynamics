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
from .stochastic_systems_models import (pulse_summary_statistics, chase_summary_statistics, pulse_heteroscedastic_summary_statistics, pulse_edu_heteroscedastic_summary_statistics,
                                        extended_logarithmic_three_population_pulse, extended_logarithmic_three_population_chase)
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

def sample_three_population_pulse_prior(num_samples):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    betas, _ = gibbs_sampler(num_samples, 0.5**2, 1)
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    p = np.random.uniform(0,1,num_samples)
    #if mode
    mode = np.random.randint(3, size = num_samples)
    mode_r = (mode >= 1)
    mode_y = (mode >= 1)
    mode_o = (mode < 2)
    mu_d = sc.loguniform.rvs(0.005, 1, size = num_samples)
    mu_d_r = mode_r*mu_d
    mu_d_y = mode_y*mu_d
    mu_d_o = mode_o*mu_d
    mu_a = sc.loguniform.rvs(10**(-5), 1, size = num_samples)
    mu_r = 1/np.random.uniform(0.1, 1/np.maximum(mu_d_r, 0.1))
    mu_rej = np.maximum(np.random.binomial(1,0.5, num_samples), (mu_d_o == 0))*sc.loguniform.rvs(10**-6, 1, size = num_samples)
    c = sc.loguniform.rvs(10**-5, 10**-1, size = num_samples) #logarithmic
    #c = sc.loguniform.rvs(10**-8, 10**-3, size = num_samples) #differential

    return np.array([beta0,beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c])

#We use the posteriors of the pulse ABC fit to construct the priors of the chase ABC fit, so we need to specify pulse_samples
three_population_pulse_accepted_parameters = np.load(os.path.join(dirname, "simulated_ABC_data/pulse/accepted_extended_three_population_pulse_params.npy"))
three_population_chase_pulse_accepted_parameters = np.load(os.path.join(dirname, "simulated_ABC_data/chase/accepted_extended_three_population_chase_pulse_params.npy"))
three_population_chase_chase_accepted_parameters = np.load(os.path.join(dirname, "simulated_ABC_data/chase/accepted_extended_three_population_chase_chase_params.npy"))

def extended_recip(x):
    return 1/np.maximum(x, 10e-3)

def truncated_normal(mu,sig, a, b, num_samples):
    return np.maximum(np.minimum(np.random.normal(mu,sig, num_samples), b), a)

#beta0,beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c
def sample_three_population_chase_prior(num_samples,pulse_samples = three_population_pulse_accepted_parameters):
    """
    Outputs a matrix of parameters. If num_samples >1, each column is a sample of parameters, and each row is every 
    sample of a specific parameter. If nums_samples=1, this is a column vector.
    """
    #Take 500000 samples
    pulse_parameter_index = np.random.choice(len(pulse_samples), num_samples)
    pulse_samples_with_rep = pulse_samples[pulse_parameter_index]

    mode0 = (((pulse_samples[:,3] > 0).astype(int) + (pulse_samples[:,4] > 0).astype(int) + (pulse_samples[:,5] > 0).astype(int)) == 3).astype(bool)
    mode1 = (((pulse_samples[:,3] == 0).astype(int) + (pulse_samples[:,4] == 0).astype(int) + (pulse_samples[:,5] > 0).astype(int)) == 3).astype(bool)
    mode2 = (((pulse_samples[:,3] > 0).astype(int) + (pulse_samples[:,4] > 0).astype(int) + (pulse_samples[:,5] == 0).astype(int)) == 3).astype(bool)

    #compute pertubation variance
    diffusion_prob_pulse_var = np.var(np.transpose(pulse_samples)[2])
    mode0_pulse_log_var = np.var(np.log(pulse_samples[mode0, 3]))
    mode1_pulse_log_var = np.var(np.log(pulse_samples[mode1, 5]))
    mode2_pulse_log_var = np.var(np.log(pulse_samples[mode2, 3]))
    mu_a_pulse_log_var = np.var(np.log(np.transpose(pulse_samples)[6]))
    mu_r_pulse_recip_var = np.var(1/(np.transpose(pulse_samples)[7]))
    mu_rej_pulse_log_var = np.var(np.log(pulse_samples[pulse_samples[:,8] !=0,8]))
    c_log_var = np.var(np.log(np.transpose(pulse_samples)[9]))

    mode0_rep = (((pulse_samples_with_rep[:,3] > 0).astype(int) + (pulse_samples_with_rep[:,4] > 0).astype(int) + (pulse_samples_with_rep[:,5] > 0).astype(int)) == 3).astype(bool)
    mode1_rep = (((pulse_samples_with_rep[:,3] == 0).astype(int) + (pulse_samples_with_rep[:,4] == 0).astype(int) + (pulse_samples_with_rep[:,5] > 0).astype(int)) == 3).astype(bool)
    mode2_rep = (((pulse_samples_with_rep[:,3] > 0).astype(int) + (pulse_samples_with_rep[:,4] > 0).astype(int) + (pulse_samples_with_rep[:,5] == 0).astype(int)) == 3).astype(bool)

    #Extract parameters
    diffusion_prob_pulse = np.transpose(pulse_samples_with_rep)[2]
    mu_d_r_pulse = np.transpose(pulse_samples_with_rep)[3]
    mu_d_y_pulse = np.transpose(pulse_samples_with_rep)[4]
    mu_d_o_pulse = np.transpose(pulse_samples_with_rep)[5]
    mu_a_pulse = np.transpose(pulse_samples_with_rep)[6]
    true_birth = np.transpose(pulse_samples_with_rep)[7]
    mu_rej_pulse = np.transpose(pulse_samples_with_rep)[8]
    c = np.transpose(pulse_samples_with_rep)[9]

    #Perturb parameters
    mode0_normal_pert = np.random.normal(0, np.sqrt(0.1*mode0_pulse_log_var), np.sum(mode0_rep))
    mode1_normal_pert = np.random.normal(0, np.sqrt(0.1*mode1_pulse_log_var), np.sum(mode1_rep))
    mode2_normal_pert = np.random.normal(0, np.sqrt(0.1*mode2_pulse_log_var), np.sum(mode2_rep))
    diffusion_prob_pulse_pert = np.minimum(np.abs(diffusion_prob_pulse + np.random.normal(0, np.sqrt(0.1*diffusion_prob_pulse_var), num_samples)), 1)
    
    mu_d_r_pulse_pert = mu_d_r_pulse - 1 + 1
    mu_d_r_pulse_pert[mode0_rep] = np.exp(np.log(mu_d_r_pulse[mode0_rep]) + mode0_normal_pert)
    mu_d_r_pulse_pert[mode2_rep] = np.exp(np.log(mu_d_r_pulse[mode2_rep]) + mode2_normal_pert)

    mu_d_y_pulse_pert = mu_d_y_pulse - 1 + 1
    mu_d_y_pulse_pert[mode0_rep] = np.exp(np.log(mu_d_y_pulse[mode0_rep]) + mode0_normal_pert)
    mu_d_y_pulse_pert[mode2_rep] = np.exp(np.log(mu_d_y_pulse[mode2_rep]) + mode2_normal_pert)

    mu_d_o_pulse_pert = mu_d_o_pulse - 1 + 1
    mu_d_o_pulse_pert[mode0_rep] = np.exp(np.log(mu_d_o_pulse[mode0_rep]) + mode0_normal_pert)
    mu_d_o_pulse_pert[mode1_rep] = np.exp(np.log(mu_d_o_pulse[mode1_rep]) + mode1_normal_pert)

    mu_a_pulse_pert = np.exp(np.log(mu_a_pulse) + np.random.normal(0, np.sqrt(0.1*mu_a_pulse_log_var), num_samples))
    mu_r_pertubation = truncated_normal(0, np.sqrt(0.1*mu_a_pulse_log_var), -0.95*extended_recip(true_birth), extended_recip(1.05*mu_d_r_pulse_pert) - 1/true_birth, num_samples)
    #mu_r_pertubation = np.minimum(np.random.normal(0, np.sqrt(0.1*mu_r_pulse_recip_var), num_samples), 0.95*(extended_recip(mu_d_r_pulse_pert) - 1/true_birth))
    true_birth_pert = np.abs(1/(1/true_birth + mu_r_pertubation))
    mu_rej_pulse_pert = mu_rej_pulse + 1 - 1
    mu_rej_pulse_pert[mu_rej_pulse != 0] = np.exp(np.log(mu_rej_pulse[mu_rej_pulse != 0]) + np.random.normal(0, np.sqrt(0.1*mu_rej_pulse_log_var), np.sum(mu_rej_pulse != 0)))
    c_pert = np.exp(np.log(c) + np.random.normal(0, np.sqrt(0.1*c_log_var), num_samples))

    #We have a lot of evidence to suggest that death predominantly occurs in the old population during the chase portion of
    #the experiment. Since fitting to the chase is a means to an end to getting a good pulse fit, we assume these are
    #the chase dynamics
    zer = np.zeros(num_samples)
    mu_d_o_chase = sc.loguniform.rvs(0.001, 0.1, size = num_samples)

    betas, sig = gibbs_sampler(num_samples, 0.5**2, 1, experiment = "chase")
    beta0 = np.transpose(betas)[0]
    beta1 = np.transpose(betas)[1]
    #beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c, mu_b, sig
    return (np.array([beta0, beta1, diffusion_prob_pulse_pert, mu_d_r_pulse_pert, mu_d_y_pulse_pert, mu_d_o_pulse_pert, mu_a_pulse_pert, true_birth_pert, mu_rej_pulse_pert, c_pert]), 
            np.array([beta0, beta1, diffusion_prob_pulse_pert, zer, zer, mu_d_o_chase, zer, true_birth_pert, zer, zer, zer, sig]))


@jit(parallel=True)
def run_parallel(simulated_summaries, simulated_pulse_params, simulated_chase_params, stochastic_simulator, summary_statistic, inference_portion, mode = "training"):

    if inference_portion == "pulse":
        for i in prange(len(simulated_summaries)):
            z = stochastic_simulator(simulated_pulse_params[i], mode = mode)
            print("Simulation " + str(i) + " Finished")
            s = summary_statistic(z)
            simulated_summaries[i] = s
    
    elif inference_portion == "chase":
        for i in prange(len(simulated_summaries)):
            z = stochastic_simulator(simulated_pulse_params[i], simulated_chase_params[i], mode = mode)
            print("Simulation " + str(i) + " Finished")
            s = summary_statistic(z)
            simulated_summaries[i] = s        

    print("Simulations finished, returning outputs")

    return  simulated_summaries

def run(number_sims, inference_portion = "pulse", hetero = False, edu_hetero = False):
    extra_dimensions = 0
    if hetero:
        extra_dimensions = 1
    if edu_hetero:
        extra_dimensions = 5
    
    if inference_portion == "pulse":
        summary_statistic_shape = 6 + extra_dimensions
        summary_statistic = pulse_summary_statistics
        stochastic_simulator = extended_logarithmic_three_population_pulse
        sample_prior = sample_three_population_pulse_prior
        simulated_pulse_paramss = sample_prior(number_sims)
        simulated_chase_paramss = simulated_pulse_paramss

    elif inference_portion == "chase":
        summary_statistic = chase_summary_statistics
        summary_statistic_shape = 15
        stochastic_simulator = extended_logarithmic_three_population_chase
        sample_prior = sample_three_population_chase_prior
        simulated_pulse_paramss, simulated_chase_paramss = sample_prior(number_sims)

    simulated_summaries = np.zeros((number_sims, summary_statistic_shape)).astype(np.float64)
    simulated_pulse_params = np.transpose(simulated_pulse_paramss).astype(np.float64)
    simulated_chase_params = np.transpose(simulated_chase_paramss).astype(np.float64)

    return run_parallel(simulated_summaries, simulated_pulse_params, simulated_chase_params, stochastic_simulator, summary_statistic, inference_portion)

def run_validation(number_sims = 500, inference_portion = "pulse", mode = "validation"):

    if inference_portion == "pulse":
        summary_statistic_shape = 6
        pulse_params = three_population_pulse_accepted_parameters[:number_sims]
        chase_params = three_population_pulse_accepted_parameters[:number_sims]
        stochastic_simulator = extended_logarithmic_three_population_pulse
        summary_statistic = pulse_summary_statistics
    elif inference_portion == "chase":
        summary_statistic_shape = 15
        pulse_params = three_population_chase_pulse_accepted_parameters[:number_sims]
        chase_params = three_population_chase_chase_accepted_parameters[:number_sims]
        stochastic_simulator = extended_logarithmic_three_population_chase
        summary_statistic = chase_summary_statistics

    simulated_summaries = np.zeros((number_sims, summary_statistic_shape)).astype(np.float64)

    return run_parallel(simulated_summaries, pulse_params, chase_params, stochastic_simulator, summary_statistic, inference_portion, mode)

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

def chase_posterior_predictive_simulator(pulse_params, chase_params, mode, num_samples):

    dim = 12
    if mode == "full":
        num_cells = 265
    elif mode == "training":
        num_cells = 181
    elif mode == "validation":
        num_cells = 84

    pulse_params = pulse_params[:num_samples]
    chase_params = chase_params[:num_samples]
    trajectories = np.zeros((len(pulse_params), num_cells, 385, dim))
    variance_statistics = np.zeros(len(pulse_params))
    peak1_0day = np.zeros(len(pulse_params))
    peak1_4day = np.zeros(len(pulse_params))

    for i in prange(len(pulse_params)):
        pulse_param = pulse_params[i]
        chase_param = chase_params[i]
        (_, _, _, _, _, _,
                _, _, _, _, _, _,
                initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
                variance_statistic, trajectory_chase) = extended_logarithmic_three_population_chase(pulse_param, chase_param, verbose = False, full_trajectory=True, mode=mode)

        #appending the trajectory of all cells in this simulation
        trajectories[i] = trajectory_chase

        #the peak 1 proportions
        peak1_0day[i] = initial_average_peak1_proportion_chase
        peak1_4day[i] = final_average_peak1_proportion_chase

        #and the variance statistic
        variance_statistics[i] = variance_statistic

        print(str(i+1) +"/" + str(len(pulse_params)))

    return (trajectories, peak1_0day, peak1_4day, variance_statistics)


@jit(parallel=True)
def pulse_posterior_predictive_simulator(params, mode, num_samples = 100):
    dim = 9

    if mode == "full":
        num_cells = 382
    elif mode == "training":
        num_cells = 247
    elif mode == "validation":
        num_cells = 135
    
    params = params[:num_samples]
    trajectories = np.zeros((len(params), num_cells, 97, dim))
    peak1 = np.zeros(len(params))
    variance_statistics = np.zeros(len(params))
    for i in prange(len(params)):
        param = params[i]
        (_, _, _, _, _, _, 
           _, _, _, _, _, _, 
           average_peak1_proportion, variance_statistic, trajectory, _,_,_,_,_) = extended_logarithmic_three_population_pulse(param, verbose = False, full_trajectory=True, mode=mode)

        #appending the trajectories of all cells in this simulation
        trajectories[i] = trajectory

        #the peak 1 proportions
        peak1[i] = average_peak1_proportion

        #and the variance statistics
        variance_statistics[i] = variance_statistic

        print(str(i+1) +"/" + str(len(params)))

    return (trajectories, peak1, variance_statistics)

def posterior_predictive(num_samples = 500, inference_portion = "pulse", mode = "full"):
    #Pulse simulation
    if inference_portion == "pulse":
        params = three_population_pulse_accepted_parameters
        params = params[:num_samples]

        #Running the simulations in parallel
        (trajectories, peak1, variance_statistics)  = pulse_posterior_predictive_simulator(extended_logarithmic_three_population_pulse, params, mode)

        #post processing the trajectories into edu and nucleoid trajectories
        nucleoid_trajectories = np.zeros((len(params), 97))
        edu_trajectories = np.zeros((len(params), 97))
        for j in range(len(trajectories)):
            
            trajectory = trajectories[j]
            nucleoid_trajectory = np.sum(trajectory, axis = 2)
            edu_trajectory = nucleoid_trajectory - trajectory[:,:,3] - trajectory[:,:,6]
            #mean across all cells
            nucleoid_trajectories[j] = np.mean(nucleoid_trajectory, axis=0)
            edu_trajectories[j] = np.mean(edu_trajectory, axis=0)

        return (nucleoid_trajectories, edu_trajectories, peak1, variance_statistics)
    
    #Chase simulation
    elif inference_portion == "chase":

        pulse_params = three_population_chase_pulse_accepted_parameters
        chase_params = three_population_chase_chase_accepted_parameters
        pulse_params = pulse_params[:num_samples]
        chase_params = chase_params[:num_samples]
        
        #Running the simulations in parallel
        (trajectories, peak1_0day, peak1_4day, variance_statistics) = chase_posterior_predictive_simulator(extended_logarithmic_three_population_chase, pulse_params, chase_params, mode)

        #post processing the trajectories into edu and nucleoid trajectories
        nucleoid_trajectories = np.zeros((len(params), 385))
        edu_trajectories = np.zeros((len(params), 385))
        edu_proportions = np.zeros((len(params), 385))
        for j in range(len(trajectories)):
            
            trajectory = trajectories[j]
            nucleoid_trajectory = np.sum(trajectory, axis = 2)
            edu_trajectory = nucleoid_trajectory - trajectory[:,:,0] - trajectory[:,:,3] - trajectory[:,:,9]

            #mean over all cells
            nucleoid_trajectories[j] = np.mean(nucleoid_trajectory, axis=0)
            edu_trajectories[j] = np.mean(edu_trajectory, axis=0)
            edu_proportions[j] = np.mean(np.divide(edu_trajectory,nucleoid_trajectory), axis = 0)

        return (nucleoid_trajectories, edu_trajectories, edu_proportions, peak1_0day, peak1_4day, variance_statistics)