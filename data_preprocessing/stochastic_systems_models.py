import numpy as np
from numba import jit, prange
import numba
from .data_preprocessing import (training_mito_lengths, training_time_indicator, training_dna_numbers, training_mito_lengths_chase, training_chase_time_indicator,
                                 validation_mito_lengths, validation_time_indicator, validation_dna_numbers, validation_mito_lengths_chase, validation_chase_time_indicator,
                                 all_assays_mito_lengths, all_assays_time_indicator, all_assays_dna_numbers, all_assays_mito_lengths_chase, all_assays_chase_time_indicator,
                                 all_assays_edu_number_1hr, all_assays_edu_number_3hr, all_assays_edu_number_7hr, all_assays_edu_number_24hr,
                                 validation_edu_number_1hr, validation_edu_number_3hr, validation_edu_number_7hr, validation_edu_number_24hr,
                                 training_edu_number_1hr, training_edu_number_3hr, training_edu_number_7hr, training_edu_number_24hr)

#############################################################################################################################
#                                                                                                                           #
#                                                              HELPER                                                       #
#                                                            FUNCTIONS                                                      #
#                                                                                                                           #
#############################################################################################################################
@jit(nopython=True)
def normal_sample(num_samples):
    noise = np.empty(num_samples,dtype=np.float64)
    for i in range(num_samples):
        noise[i] = np.random.normal()
    return noise

sorted_mt_indices = np.argsort(training_mito_lengths)

@jit(nopython=True)
def gamma_sample(num_samples, shape, scale):
    noise = np.empty(num_samples,dtype=np.float64)
    for i in range(num_samples):
        noise[i] = np.random.gamma(shape, scale)
    return noise

@jit(nopython = True)
def moving_variance(dna, mt, b0, b1, sorted_indices,window_size=20):
    residuals = dna[sorted_indices] - b0 - b1*mt[sorted_indices]
    moving_variance = np.convolve(residuals**2, np.ones(window_size)/window_size, mode='valid')
    moving_variance_x_axis = np.convolve(mt[sorted_indices], np.ones(window_size)/window_size, mode='valid')

    return moving_variance_x_axis, moving_variance

#regression coefficients of dna v mtvol, edu_1hr v mtvol, edu_3hr v mtvol, edu_7hr v mtvol, edu_24hr v mtvol
#Training
#n: 214, 1.23
#e1: 53, 0.06
#e3: 42,0.13
#e7: 35, 0.27
#e24: 99, 0.51

#All assays
#n: 190, 1.24
#e1: 50, 0.06
#e3: 47, 0.12
#e7: 30, 0.24
#e24: 87, 0.51

#Validation
#n: 147, 1.26
#e1: 44, 0.08
#e3: 61, 0.11
#e7: 13, 0.21
#e24: 91, 0.45
training_regression_coefficients = np.array([[214,12.3],[53,0.06],[42,0.13],[35,0.27],[99,0.51]])
full_regression_coefficients = np.array([[190,1.24],[50,0.06],[47,0.12],[30,0.24],[87,0.51]])
validation_regression_coefficients = np.array([[147,1.26],[44,0.08],[61,0.11],[13,0.21],[91, 0.45]])

@jit(nopython=True)
def hetero_summary_statistic(dna, mt, window_size=20, mode = "training"):
    if mode == "full":
        data_dna = all_assays_dna_numbers
        b0, b1 = full_regression_coefficients[0]
    elif mode == "training":
        data_dna = training_dna_numbers
        b0, b1 = training_regression_coefficients[0]
    else:
        data_dna = validation_dna_numbers
        b0, b1 = validation_regression_coefficients[0]
    sorted_indices = np.argsort(mt)

    _, data_moving_variance = moving_variance(data_dna, mt, b0, b1, sorted_indices, window_size)
    _, log_moving_variance = moving_variance(dna, mt, b0, b1, sorted_indices, window_size)
    return np.linalg.norm(np.sqrt(data_moving_variance) - np.sqrt(log_moving_variance))

@jit(nopython=True)
def hetero_summary_statistic_e1(dna, mt, window_size=7, mode = "training"):
    if mode == "full":
        data_edu = all_assays_edu_number_1hr
        b0, b1 = full_regression_coefficients[1]
    elif mode == "training":
        data_edu = training_edu_number_1hr
        b0, b1 = training_regression_coefficients[1]
    else:
        data_edu = validation_edu_number_1hr
        b0, b1 = validation_regression_coefficients[1]

    sorted_indices = np.argsort(mt)
    _, data_moving_variance = moving_variance(data_edu, mt, b0, b1, sorted_indices, window_size)
    _, log_moving_variance = moving_variance(dna, mt, b0, b1, sorted_indices, window_size)

    return np.linalg.norm(np.sqrt(data_moving_variance) - np.sqrt(log_moving_variance))

@jit(nopython=True)
def hetero_summary_statistic_e3(dna, mt, window_size=7, mode = "training"):
    if mode == "full":
        data_edu = all_assays_edu_number_3hr
        b0, b1 = full_regression_coefficients[2]
    elif mode == "training":
        data_edu = training_edu_number_3hr
        b0, b1 = training_regression_coefficients[2]
    else:
        data_edu = validation_edu_number_3hr
        b0, b1 = validation_regression_coefficients[2]

    sorted_indices = np.argsort(mt)
    _, data_moving_variance = moving_variance(data_edu, mt, b0, b1, sorted_indices, window_size)
    _, log_moving_variance = moving_variance(dna, mt, b0, b1, sorted_indices, window_size)

    return np.linalg.norm(np.sqrt(data_moving_variance) - np.sqrt(log_moving_variance))

@jit(nopython=True)
def hetero_summary_statistic_e7(dna, mt, window_size=7, mode = "training"):
    if mode == "full":
        data_edu = all_assays_edu_number_7hr
        b0, b1 = full_regression_coefficients[2]
    elif mode == "training":
        data_edu = training_edu_number_7hr
        b0, b1 = training_regression_coefficients[2]
    else:
        data_edu = validation_edu_number_7hr
        b0, b1 = validation_regression_coefficients[2]

    sorted_indices = np.argsort(mt)
    _, data_moving_variance = moving_variance(data_edu, mt, b0, b1, sorted_indices, window_size)
    _, log_moving_variance = moving_variance(dna, mt, b0, b1, sorted_indices, window_size)

    return np.linalg.norm(np.sqrt(data_moving_variance) - np.sqrt(log_moving_variance))

@jit(nopython=True)
def hetero_summary_statistic_e24(dna, mt, window_size=7, mode = "training"):
    if mode == "full":
        data_edu = all_assays_edu_number_24hr
        b0, b1 = full_regression_coefficients[3]
    elif mode == "training":
        data_edu = training_edu_number_24hr
        b0, b1 = training_regression_coefficients[3]
    else:
        data_edu = validation_edu_number_24hr
        b0, b1 = validation_regression_coefficients[3]

    sorted_indices = np.argsort(mt)
    _, data_moving_variance = moving_variance(data_edu, mt, b0, b1, sorted_indices, window_size)
    _, log_moving_variance = moving_variance(dna, mt, b0, b1, sorted_indices, window_size)

    return np.linalg.norm(np.sqrt(data_moving_variance) - np.sqrt(log_moving_variance))

#if n < beta0, this birth rate breaks, so there is a max(n, beta0 + 0.0001) to ensure this doesn't happen
@jit(nopython=True)
def logarithmic_birth(e, n, mu, c, l, beta0 = 172, beta1 = 1.38, ou_addition=0.0):
    return e*np.maximum(0, ou_addition + mu + c*(np.log(beta1*l/beta0 + 1)/np.log(np.maximum(n,beta0+0.0001)/beta0)-1))

@jit(nopython=True)
def constant_death(e, n, mu, c, l, beta0 = 172, beta1 = 1.38):
    return mu*e

@jit(nopython=True)
def differential_birth(e, n, mu, c, l, beta0 = 172, beta1 = 1.38, ou_addition=0.0):
    return e*np.maximum(0, ou_addition + mu + c*(beta0+beta1*l-n))

@jit(nopython=True)
def ratiometric_birth(e, n, mu, c, l, beta0 = 172, beta1 = 1.38, ou_addition=0.0):
    return e*np.maximum(0, ou_addition + mu + c*((beta0+beta1*l)/n-1))

@jit(nopython=True)
def inhibition_birth(e, n, mu, c, l, beta0 = 172, beta1 = 1.38, ou_addition=0.0):
    l0 = np.log(c/mu)/beta1
    alpha = l0*beta0
    k = l/l0

    return e*np.maximum(0, ou_addition + c*(1+alpha/l)*(1-1/k)**(n-1))

@jit(nopython=True)
def OU_birth_func(time = 24, dt=0.01, mean = 0.012, theta = 0.1, sd = 0.01, initial_val = 0.012):
    
    #var = sig^2/2thet
    increment_num = round(time/dt)
    #print(increment_num)
    
    sig = np.sqrt(2*theta)*sd

    OU_births = [initial_val]
    #for i in range(time*increment_num):
    OU_times = np.linspace(0,time, increment_num)
    for i in OU_times[1:]:
        prev_length = OU_births[-1]
        new_length = prev_length + theta*(mean-prev_length)*dt + sig*np.sqrt(dt)*np.random.normal()
        OU_births.append(new_length)

    return (OU_times, np.array(OU_births))

@jit(nopython=True)
def _one_iter_burn_in_dispersed_three_population_model(diffusion_prob,c,true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths, initial_replicating, initial_young, initial_old, kappas):
    """
    Helper function. Simulates the burn-in period for 'burn-in-increments' hours, and records the the variance of the log-residuals.
    """
    #Defining the ageing rate to maintain equilibrium
    mu_a = mu_b/diffusion_prob

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    #
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    #
    # Order of populations (m): replicating population; young population; old population
    #
    # Order of events (n): replication without diffusion; replication with diffusion; Birth; ageing; death
    step_matrix = np.array([[-1,0,1,0,0],
                           [2,1,-1,-1,0],
                           [0,0,0,1,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()
    num_sims= len(mito_lengths)

    log_residuals = np.zeros(num_sims).astype(np.float64)
    replicating_output = np.zeros(num_sims).astype(np.int64)
    young_output = np.zeros(num_sims).astype(np.int64)
    old_output = np.zeros(num_sims).astype(np.int64)

    #Looping over each cell
    for i in prange(num_sims):
        mu_b2 = kappas[i]*mu_b
        mu_d2 = kappas[i]*mu_d
        true_birth_param2 = true_birth_param
        mu_a2 = kappas[i]*mu_a
    
        l=mito_lengths[i]

        nucleoid_state = np.array([round(initial_replicating[i]), round(initial_young[i]), round(initial_old[i])]).astype(np.int64)
        current_time =  0

        #Looping until the end of this iteration (usually 1 hour)
        while current_time <= burn_in_increments:
            current_replicating = nucleoid_state[0]
            current_young = nucleoid_state[1]
            current_old = nucleoid_state[2]

            n = int(np.sum(nucleoid_state))
            if n == 0:
                #print("breaking")
                break
            
            ##################----------------Generating the time that the next event takes place---------------------######################

            max_propensity = birth_rate(current_young, n, mu_b2, c, l, beta0, beta1) + death_rate(current_old, n, mu_d2, c, l, beta0, beta1) + current_replicating*true_birth_param2 + mu_a2*current_young
            next_event_time = np.random.exponential(1/max_propensity)
            
            #Updating the time
            current_time += next_event_time
            if current_time > burn_in_increments or (not np.any(nucleoid_state)):
                current_time = burn_in_increments
                break

            ##################-------------------------Generating what kind of event this is---------------------------#####################

            p_birth = birth_rate(current_young,n,mu_b2,c,l,beta0, beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d2,c,l,beta0, beta1)/max_propensity
            p_double_truebirth = diffusion_prob*current_replicating*true_birth_param2/max_propensity
            p_single_truebirth = (1-diffusion_prob)*current_replicating*true_birth_param2/max_propensity
            p_ageing = mu_a2*current_young/max_propensity

            probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_death])
            r2 = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r2)

            #Updating the nucleoid state based on which event occured
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()


        #Recording each subpopulation after the end of the iteration (usually an hour)
        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]
        replicating_output[i] = current_replicating
        young_output[i] = current_young
        old_output[i] = current_old

        #Recording the log residuals of the single cell
        log_residuals[i] = np.log(max(current_replicating + current_young + current_old,1)) - np.log(beta1*l + beta0)

    #Recording the variance of the log residuals of every cell
    variance_output = np.var(log_residuals)

    #Return each subpopulation for each cell to feed into the next iteration, as well as the log residual variance
    return (replicating_output.astype(np.float64), young_output.astype(np.float64), old_output.astype(np.float64), variance_output)

@jit(nopython=True)
def _burn_in_dispersed_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments = 1, beta0=172, beta1=1.38, 
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, ksig = 0.1, verbose = True, inference_portion = "pulse", mode = "training"):
    """
    Helper function. Simulates the burn-in period and records the control strength summary statistic S_cs.
    """
    if verbose:
        print("------------------------Beginning Burn In --------------------------")

    #Undercounting percentage error
    percent_error = 3.6

    #Initialising the nucleoid number
    if inference_portion == "pulse":
        #pulse data: we take the actual nucleoid number and error correct
        initial_nucleoid_number = dna_nums/(1-percent_error/100)
    else:
        #chase data: we draw from the empirical distribution of the 0 day data, and error correct
        initial_nucleoid_number = np.maximum((beta0 + beta1*mito_lengths + sig*mito_lengths*normal_sample(len(mito_lengths)))/(1-percent_error/100),1)

    # print("negative nucleoids:")
    # print(np.sum(initial_nucleoid_number < 0))
    #defining mu_a to maintain equilibrium
    mu_a = mu_b/diffusion_prob
    
    J = int(burn_in_time//burn_in_increments)

    #initialising each subpopulation via the equilibrium proportions (under a deterministic treatment)
    initialisation_denom = 1/true_birth_param + 1/mu_a + 1/mu_d    
    initial_replicating = initial_nucleoid_number/(true_birth_param*initialisation_denom)
    initial_young = initial_nucleoid_number/(mu_a*initialisation_denom)
    initial_old = initial_nucleoid_number/(mu_d*initialisation_denom)

    variances = np.zeros(J).astype(np.float64) 

    kappas = gamma_sample(len(mito_lengths), 1/ksig**2, ksig**2)
    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (replicating_output, young_output, old_output, variance_output) = _one_iter_burn_in_dispersed_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths = mito_lengths, initial_replicating = initial_replicating, initial_young = initial_young, initial_old = initial_old, kappas=kappas)

        variances[j] = variance_output
        initial_replicating = replicating_output.copy()
        initial_young = young_output.copy()
        initial_old = old_output.copy()
        if verbose:
            print("Iteration " +str(j) + " Finished")

    nucleoid_num = initial_replicating + initial_young + initial_old
    measurement_error = (1 - np.random.exponential(percent_error)/100)
    S_h = hetero_summary_statistic(measurement_error*nucleoid_num, mito_lengths, mode = mode) 
    #outputting each subpopulation to feed into the pulse portion of the simulation, as well as S_cs, and S_h
    return replicating_output, young_output, old_output, np.log(variances[-1]/variances[0]), S_h, kappas

@jit(nopython=True)
def _dispersed_three_population_pulse(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, replicating_output, young_output, old_output, kappas, beta0=172.431, beta1=1.3809, mito_lengths = training_mito_lengths, time_indicator = training_time_indicator, verbose = True, inference_portion = "pulse", full_trajectory = False, mode = "training"):
    """
    Helper function. Takes as input the output of the burn-in period and simulates the pulse portion of the experiment.
    """
    if verbose:
        print("------------------------Beginning Pulse Simulation --------------------------")

    percent_error = 3.6
    mu_a = mu_b/diffusion_prob
    
    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): replicating untagged, replicating single tagged, replicating double tagged,
    #                           young untagged, young single tagged, young double tagged,
    #                           old untagged, old single tagged, old double tagged (***)
    
    # Order of events (n): untagged birth, untagged replication with diffusion, untagged replication without diffusion,
    #                      single tagged birth, single tagged replication with diffusion,
    #                      single tagged replication without diffusion (single tagged daughter keeps replicating),
    #                      single tagged replication without diffusion (double tagged daughter keeps replicating),
    #                      double tagged birth, double tagged replication with diffusion, double tagged replication without diffusion,
    #                      untagged ageing, single tagged ageing, double tagged ageing
    #                      untagged death, single tagged death, double tagged death (*****)

    step_matrix = np.array([[1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,1,-1,0,-1,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,1,-1,0,0,0,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],
                            [0,2,1,-1,1,0,1,0,0,0,0,-1,0,0,0,0],
                            [0,0,0,0,1,1,0,-1,2,1,0,0,-1,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    #If full_trajectory (or its the chase portion), we simulate every cell for 24 hours, rather than some for 1, 3 , 7.
    if full_trajectory or inference_portion == "chase":
        time_indicator = 24*np.ones(len(mito_lengths))

    #If we are fitting to the pulse data, we record time points 1,3,7,24
    if inference_portion == "pulse":
        cell_number_1hr = np.sum(time_indicator == 1)
        cell_number_3hr = np.sum(time_indicator == 3)
        cell_number_7hr = np.sum(time_indicator == 7)
        cell_number_24hr = np.sum(time_indicator == 24)

        cell_number_chase = 0

    #If we are fitting to the chase data, we only record the final state to initialise the chase portion of the experiment
    elif inference_portion == "chase":
        cell_number_1hr = 0
        cell_number_3hr = 0
        cell_number_7hr = 0
        cell_number_24hr = 0

        cell_number_chase = np.sum(time_indicator > -1)

    #Initialising storage arrays for the 4 time points of interest
    nucleoid_num_1hr = [int(0)]*cell_number_1hr
    tagged_num_1hr = [int(0)]*cell_number_1hr
    mtvolume_1hr = [float(0)]*cell_number_1hr

    nucleoid_num_3hr = [int(0)]*cell_number_3hr
    tagged_num_3hr = [int(0)]*cell_number_3hr
    mtvolume_3hr = [float(0)]*cell_number_3hr

    nucleoid_num_7hr = [int(0)]*cell_number_7hr
    tagged_num_7hr = [int(0)]*cell_number_7hr
    mtvolume_7hr = [float(0)]*cell_number_7hr

    nucleoid_num_24hr = [int(0)]*cell_number_24hr
    tagged_num_24hr = [int(0)]*cell_number_24hr
    mtvolume_24hr = [float(0)]*cell_number_24hr

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 97,9)).astype(np.float64)

    #initialising a matrix to contain the sizes of each subpopulation for each cell after 24 hours, to use to initialise
    #the chase portion of the experiment
    chase_final_state = np.zeros((9, cell_number_chase)).astype(np.int64)

    #peak 1 proportion summary statistc
    peak1_proportion = [float(0)]*cell_number_24hr

    num_sims = len(mito_lengths)
    for i in prange(num_sims):
        #print(i)
        mu_b2 = kappas[i]*mu_b
        mu_d2 = kappas[i]*mu_d
        true_birth_param2 = true_birth_param
        mu_a2 = kappas[i]*mu_a

        l = mito_lengths[i]
        time_point = time_indicator[i]

        #Initialising the nucleoid state based on the output of the burn in.
        #Order of populations is (***)
        nucleoid_state = np.array([round(replicating_output[i]),0,0,round(young_output[i]),0,0, round(old_output[i]),0,0]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state

        #Looping for either 1,3,7, or 24 hours, depending on the cell
        while current_time <= time_point:
            current_replicating_DNA = nucleoid_state[0]
            current_replicating_single = nucleoid_state[1]
            current_replicating_double = nucleoid_state[2]
            current_young_DNA = nucleoid_state[3]
            current_young_single = nucleoid_state[4]
            current_young_double = nucleoid_state[5]
            current_old_DNA = nucleoid_state[6]
            current_old_single = nucleoid_state[7]
            current_old_double = nucleoid_state[8]

            current_replicating = current_replicating_DNA + current_replicating_single + current_replicating_double
            current_young = current_young_DNA + current_young_single + current_young_double
            current_old = current_old_DNA + current_old_single + current_old_double

            n = int(np.sum(nucleoid_state))

            #If the cell has ran out of nucleoids, end the loop
            if n == 0:
                break

            #Generating the time that the next event takes place
            max_propensity = birth_rate(current_young, n, mu_b2, c, l, beta0,beta1) + death_rate(current_old, n, mu_d2, c, l, beta0,beta1) + current_replicating*true_birth_param2 + current_young*mu_a2
            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<96:  
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            #If the event time is greater than the final time point, then this event doesn't happen, and we break before updating the nucleid state
            if current_time > time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################

            #Probability of a general event
            p_birth = birth_rate(current_young,n,mu_b2,c,l,beta0,beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d2,c,l,beta0,beta1)/max_propensity
            p_truebirth_dif = diffusion_prob*current_replicating*true_birth_param2/max_propensity
            p_truebirth_nodif = (1-diffusion_prob)*current_replicating*true_birth_param2/max_propensity
            p_ageing = mu_a2*current_young/max_propensity

            #probability that the event happens to an untagged, single, or double tagged molecule, given that the event affects ...
            
            #... the young population
            p_young_untagged = current_young_DNA/max(current_young,1)
            p_young_single = current_young_single/max(current_young,1)
            p_young_double = current_young_double/max(current_young,1)

            #... the old population
            p_old_untagged = current_old_DNA/max(current_old,1)
            p_old_single = current_old_single/max(current_old,1)
            p_old_double = current_old_double/max(current_old,1)

            #... the replicating population
            p_replicating_untagged = current_replicating_DNA/max(current_replicating,1)
            p_replicating_single = current_replicating_single/max(current_replicating,1)
            p_replicating_double = current_replicating_double/max(current_replicating,1)

            #Probabilities of each event (order given in (*****))
            probability_vector = np.array([p_birth*p_young_untagged, p_truebirth_dif*p_replicating_untagged, p_truebirth_nodif*p_replicating_untagged,
                                           p_birth*p_young_single, p_truebirth_dif*p_replicating_single, p_truebirth_nodif*p_replicating_single/2,
                                           p_truebirth_nodif*p_replicating_single/2, p_birth*p_young_double, p_truebirth_dif*p_replicating_double,
                                           p_truebirth_nodif*p_replicating_double, p_ageing*p_young_untagged, p_ageing*p_young_single, 
                                           p_ageing*p_young_double, p_death*p_old_untagged, p_death*p_old_single, p_death*p_old_double])
            
            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)

            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #Appending the data
        if inference_portion == "pulse":
            measurement_error = (1 - np.random.exponential(percent_error)/100)
            tagged = current_replicating_DNA + current_replicating_single + current_replicating_double + current_young_single + current_young_double + current_old_single + current_old_double
            tagged_output = round(tagged*measurement_error)
            untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
            if time_point == 1:
                nucleoid_num_1hr[i] = int(tagged_output + untagged_output)
                tagged_num_1hr[i] = int(tagged_output)
                mtvolume_1hr[i] = float(l)
            elif time_point == 3:
                j = i-cell_number_1hr
                nucleoid_num_3hr[j] = int(tagged_output + untagged_output)
                tagged_num_3hr[j] = int(tagged_output)
                mtvolume_3hr[j] = float(l)
            elif time_point == 7:
                j = i-cell_number_1hr - cell_number_3hr
                nucleoid_num_7hr[j] = int(tagged_output + untagged_output)
                tagged_num_7hr[j] = int(tagged_output)
                mtvolume_7hr[j] = float(l)
            elif time_point == 24:
                j = i-cell_number_1hr - cell_number_3hr - cell_number_7hr
                nucleoid_num_24hr[j] = int(tagged_output + untagged_output)
                tagged_num_24hr[j] = int(tagged_output)
                mtvolume_24hr[j] = float(l)

                peak1 = current_young_single + current_old_single
                peak1_proportion[j] = float(peak1/max(tagged,1))

        elif inference_portion == "chase":
            chase_final_state[0,i] = current_replicating_DNA
            chase_final_state[1,i] = current_replicating_single
            chase_final_state[2,i] = current_replicating_double
            chase_final_state[3,i] = current_young_DNA
            chase_final_state[4,i] = current_young_single
            chase_final_state[5,i] = current_young_double
            chase_final_state[6,i] = current_old_DNA
            chase_final_state[7,i] = current_old_single
            chase_final_state[8,i] = current_old_double

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_1hr = np.array(nucleoid_num_1hr)
    nucleoid_num_3hr = np.array(nucleoid_num_3hr)
    nucleoid_num_7hr = np.array(nucleoid_num_7hr)
    nucleoid_num_24hr = np.array(nucleoid_num_24hr)

    tagged_num_1hr = np.array(tagged_num_1hr)
    tagged_num_3hr = np.array(tagged_num_3hr)
    tagged_num_7hr = np.array(tagged_num_7hr)
    tagged_num_24hr = np.array(tagged_num_24hr)

    mtvolume_1hr = np.array(mtvolume_1hr)
    mtvolume_3hr = np.array(mtvolume_3hr)
    mtvolume_7hr = np.array(mtvolume_7hr)
    mtvolume_24hr = np.array(mtvolume_24hr)

    e1_h = 0.0
    e3_h = 0.0
    e7_h = 0.0
    e24_h = 0.0
    if inference_portion == "pulse":
        average_peak1_proportion = np.mean(np.array(peak1_proportion))
        if not full_trajectory:
            e1_h = hetero_summary_statistic_e1(measurement_error*tagged_num_1hr, mtvolume_1hr, mode = mode)
            e3_h = hetero_summary_statistic_e3(measurement_error*tagged_num_3hr, mtvolume_3hr, mode = mode)
            e7_h = hetero_summary_statistic_e7(measurement_error*tagged_num_7hr, mtvolume_7hr, mode = mode)
            e24_h = hetero_summary_statistic_e24(measurement_error*tagged_num_24hr, mtvolume_24hr, mode = mode)
    elif inference_portion == "chase":
        average_peak1_proportion = 0

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h)

@jit(nopython=True)
def _dispersed_three_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, birth_rate, death_rate, ksig, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_time_indicator, mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, verbose = True, full_trajectory = False, mode = "training"):
    """
    Helper function. Stitches together the burn-in phase and the pulse phase into one simulator.
    """

    (replicating_output, young_output, old_output, variance_statistic, S_h, kappas) = _burn_in_dispersed_three_population_model(diffusion_prob=diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b=mu_b_pulse, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = burn_in_time, verbose=verbose, ksig=ksig, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h) = _dispersed_three_population_pulse(diffusion_prob = diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b = mu_b_pulse, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, young_output=young_output, old_output=old_output, beta0=beta0, beta1=beta1, time_indicator = time_indicator, mito_lengths = mito_lengths, verbose=verbose, full_trajectory = full_trajectory, kappas=kappas, mode = mode)

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h)

@jit(nopython=True)
def _ou_one_iter_burn_in_three_population_model(diffusion_prob,c,true_birth_param, mu_d, mu_b, theta, sd, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths, initial_replicating, initial_young, initial_old, initial_ou_vals):
    
    #Defining the ageing rate to maintain equilibrium
    mu_a = mu_b/diffusion_prob

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    #
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    #
    # Order of populations (m): replicating population; young population; old population
    #
    # Order of events (n): replication without diffusion; replication with diffusion; Birth; ageing; death; fake event
    step_matrix = np.array([[-1,0,1,0,0,0],
                           [2,1,-1,-1,0,0],
                           [0,0,0,1,-1,0]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()
    num_sims= len(mito_lengths)

    log_residuals = np.zeros(num_sims).astype(np.float64)
    replicating_output = np.zeros(num_sims).astype(np.int64)
    young_output = np.zeros(num_sims).astype(np.int64)
    old_output = np.zeros(num_sims).astype(np.int64)
    final_ou_vals = np.zeros(num_sims).astype(np.float64)

    #Looping over each cell
    #start = time.time()
    for i in prange(num_sims):
        OU_times, OU_births = OU_birth_func(time = burn_in_increments, mean = 0, theta = theta, sd=sd*mu_b, initial_val = initial_ou_vals[i])
        final_ou_vals[i] = OU_births[-1]
    
        l=mito_lengths[i]

        nucleoid_state = np.array([round(initial_replicating[i]), round(initial_young[i]), round(initial_old[i])]).astype(np.int64)
        current_time =  0

        #Looping until the end of this iteration (usually 1 hour)
        while current_time <= burn_in_increments:
            current_replicating = nucleoid_state[0]
            current_young = nucleoid_state[1]
            current_old = nucleoid_state[2]

            n = int(np.sum(nucleoid_state))
            if n == 0:
                #print("breaking")
                break
            
            ##################----------------Generating the time that the next event takes place---------------------######################
            
            max_ou = np.max(OU_births[OU_times >= current_time])
            max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0, beta1, max_ou) + death_rate(current_old, n, mu_d, c, l, beta0, beta1) + current_replicating*true_birth_param + mu_a*current_young
            next_event_time = np.random.exponential(1/max_propensity)
            
            #Updating the time
            current_time += next_event_time
            if current_time > burn_in_increments or (not np.any(nucleoid_state)):
                current_time = burn_in_increments
                break

            ##################-------------------------Generating what kind of event this is---------------------------#####################
            birth_ou = np.interp(current_time, OU_times, OU_births)
            p_birth = birth_rate(current_young,n,mu_b,c,l,beta0, beta1, birth_ou)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0, beta1)/max_propensity
            p_double_truebirth = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_single_truebirth = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity
            p_ageing = mu_a*current_young/max_propensity
            p_fake = 1-p_birth-p_death-p_double_truebirth-p_single_truebirth-p_ageing

            probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_death,p_fake])
            r2 = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r2)

            #Updating the nucleoid state based on which event occured
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #Recording each subpopulation after the end of the iteration (usually an hour)
        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]
        replicating_output[i] = current_replicating
        young_output[i] = current_young
        old_output[i] = current_old

        #Recording the log residuals of the single cell
        log_residuals[i] = np.log(max(current_replicating + current_young + current_old,1)) - np.log(beta1*l + beta0)

    #Recording the variance of the log residuals of every cell
    variance_output = np.var(log_residuals)
    # end = time.time()
    #print(end - start)

    #Return each subpopulation for each cell to feed into the next iteration, as well as the log residual variance
    return (replicating_output.astype(np.float64), young_output.astype(np.float64), old_output.astype(np.float64), variance_output, final_ou_vals)

@jit(nopython=True)
def _ou_burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, theta, sd,birth_rate, death_rate, burn_in_increments = 1, beta0=172, beta1=1.38, 
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, verbose = True, inference_portion = "pulse", mode = "training"):

    if verbose:
        print("------------------------Beginning Burn In --------------------------")

    #Undercounting percentage error
    percent_error = 3.6

    #Initialising the nucleoid number
    if inference_portion == "pulse":
        #pulse data: we take the actual nucleoid number and error correct
        initial_nucleoid_number = dna_nums/(1-percent_error/100)
    else:
        #chase data: we draw from the empirical distribution of the 0 day data, and error correct
        initial_nucleoid_number = np.maximum((beta0 + beta1*mito_lengths + sig*mito_lengths*normal_sample(len(mito_lengths)))/(1-percent_error/100),1)

    # print("negative nucleoids:")
    # print(np.sum(initial_nucleoid_number < 0))
    #defining mu_a to maintain equilibrium
    mu_a = mu_b/diffusion_prob
    
    J = int(burn_in_time//burn_in_increments)

    #initialising each subpopulation via the equilibrium proportions (under a deterministic treatment)
    initialisation_denom = 1/true_birth_param + 1/mu_a + 1/mu_d    
    initial_replicating = initial_nucleoid_number/(true_birth_param*initialisation_denom)
    initial_young = initial_nucleoid_number/(mu_a*initialisation_denom)
    initial_old = initial_nucleoid_number/(mu_d*initialisation_denom)
    initial_ou_vals = np.repeat(0.0, len(initial_nucleoid_number)).astype(np.float64) 

    variances = np.zeros(J).astype(np.float64) 

    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (replicating_output, young_output, old_output, variance_output, ou_vals_output) = _ou_one_iter_burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, theta,sd,birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths = mito_lengths, initial_replicating = initial_replicating, initial_young = initial_young, initial_old = initial_old, initial_ou_vals = initial_ou_vals)
        
        initial_ou_vals = ou_vals_output.copy()
        variances[j] = variance_output
        initial_replicating = replicating_output.copy()
        initial_young = young_output.copy()
        initial_old = old_output.copy()
        if verbose:
            print("Iteration " +str(j) + " Finished")

        if np.max(initial_replicating+initial_young+initial_old) > 10000:
            break

    nucleoid_num = initial_replicating + initial_young + initial_old
    S_h = hetero_summary_statistic(nucleoid_num, mito_lengths, mode = mode) 
    #outputting each subpopulation to feed into the pulse portion of the simulation, as well as S_cs
    return replicating_output, young_output, old_output, np.log(variances[-1]/variances[0]), initial_ou_vals, S_h

@jit(nopython=True)
def _ou_testing_burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, theta,sd,birth_rate, death_rate, burn_in_increments = 1, beta0=172, beta1=1.38, 
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, verbose = True, inference_portion = "pulse"):

    if verbose:
        print("------------------------Beginning Burn In --------------------------")

    #Undercounting percentage error
    percent_error = 3.6

    #Initialising the nucleoid number
    if inference_portion == "pulse":
        #pulse data: we take the actual nucleoid number and error correct
        initial_nucleoid_number = dna_nums/(1-percent_error/100)
    else:
        #chase data: we draw from the empirical distribution of the 0 day data, and error correct
        initial_nucleoid_number = np.maximum((beta0 + beta1*mito_lengths + sig*mito_lengths*normal_sample(len(mito_lengths)))/(1-percent_error/100),1)

    # print("negative nucleoids:")
    # print(np.sum(initial_nucleoid_number < 0))
    #defining mu_a to maintain equilibrium
    mu_a = mu_b/diffusion_prob
    
    J = int(burn_in_time//burn_in_increments)

    #initialising each subpopulation via the equilibrium proportions (under a deterministic treatment)
    initialisation_denom = 1/true_birth_param + 1/mu_a + 1/mu_d    
    initial_replicating = initial_nucleoid_number/(true_birth_param*initialisation_denom)
    initial_young = initial_nucleoid_number/(mu_a*initialisation_denom)
    initial_old = initial_nucleoid_number/(mu_d*initialisation_denom)
    initial_ou_vals = np.repeat(0.0, len(initial_nucleoid_number)).astype(np.float64)

    variances = np.zeros(J).astype(np.float64) 

    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (replicating_output, young_output, old_output, variance_output, ou_vals_output) = _ou_one_iter_burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, theta, sd, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths = mito_lengths, initial_replicating = initial_replicating, initial_young = initial_young, initial_old = initial_old, initial_ou_vals=initial_ou_vals)
        
        initial_ou_vals = ou_vals_output.copy()
        variances[j] = variance_output
        initial_replicating = replicating_output.copy()
        initial_young = young_output.copy()
        initial_old = old_output.copy()
        if verbose:
            print("Iteration " +str(j) + " Finished")

    #outputting each subpopulation to feed into the pulse portion of the simulation, as well as S_cs
    return replicating_output, young_output, old_output, variances, initial_ou_vals


@jit(nopython=True)
def _ou_three_population_pulse(diffusion_prob,c, true_birth_param, mu_d, mu_b, theta, sd, initial_ou_vals, birth_rate, death_rate, replicating_output, young_output, old_output, beta0=172.431, beta1=1.3809, mito_lengths = training_mito_lengths, time_indicator = training_time_indicator, verbose = True, inference_portion = "pulse", full_trajectory = False, mode = "training"):

    if verbose:
        print("------------------------Beginning Pulse Simulation --------------------------")

    percent_error = 3.6
    mu_a = mu_b/diffusion_prob

    
    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): replicating untagged, replicating single tagged, replicating double tagged,
    #                           young untagged, young single tagged, young double tagged,
    #                           old untagged, old single tagged, old double tagged (***)
    
    # Order of events (n): untagged birth, untagged replication with diffusion, untagged replication without diffusion,
    #                      single tagged birth, single tagged replication with diffusion,
    #                      single tagged replication without diffusion (single tagged daughter keeps replicating),
    #                      single tagged replication without diffusion (double tagged daughter keeps replicating),
    #                      double tagged birth, double tagged replication with diffusion, double tagged replication without diffusion,
    #                      untagged ageing, single tagged ageing, double tagged ageing
    #                      untagged death, single tagged death, double tagged death, fake event (*****)

    step_matrix = np.array([[1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,1,-1,0,-1,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,1,-1,0,0,0,0,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0],
                            [0,2,1,-1,1,0,1,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,1,1,0,-1,2,1,0,0,-1,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()
    

    #If full_trajectory (or its the chase portion), we simulate every cell for 24 hours, rather than some for 1, 3 , 7.
    if full_trajectory or inference_portion == "chase":
        time_indicator = 24*np.ones(len(mito_lengths))

    #If we are fitting to the pulse data, we record time points 1,3,7,24
    if inference_portion == "pulse":
        cell_number_1hr = np.sum(time_indicator == 1)
        cell_number_3hr = np.sum(time_indicator == 3)
        cell_number_7hr = np.sum(time_indicator == 7)
        cell_number_24hr = np.sum(time_indicator == 24)

        cell_number_chase = 0

    #If we are fitting to the chase data, we only record the final state to initialise the chase portion of the experiment
    elif inference_portion == "chase":
        cell_number_1hr = 0
        cell_number_3hr = 0
        cell_number_7hr = 0
        cell_number_24hr = 0

        cell_number_chase = np.sum(time_indicator > -1)

    #Initialising storage arrays for the 4 time points of interest
    nucleoid_num_1hr = [int(0)]*cell_number_1hr
    tagged_num_1hr = [int(0)]*cell_number_1hr
    mtvolume_1hr = [float(0)]*cell_number_1hr

    nucleoid_num_3hr = [int(0)]*cell_number_3hr
    tagged_num_3hr = [int(0)]*cell_number_3hr
    mtvolume_3hr = [float(0)]*cell_number_3hr

    nucleoid_num_7hr = [int(0)]*cell_number_7hr
    tagged_num_7hr = [int(0)]*cell_number_7hr
    mtvolume_7hr = [float(0)]*cell_number_7hr

    nucleoid_num_24hr = [int(0)]*cell_number_24hr
    tagged_num_24hr = [int(0)]*cell_number_24hr
    mtvolume_24hr = [float(0)]*cell_number_24hr

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 97,9)).astype(np.float64)

    #initialising a matrix to contain the sizes of each subpopulation for each cell after 24 hours, to use to initialise
    #the chase portion of the experiment
    chase_final_state = np.zeros((9, cell_number_chase)).astype(np.int64)

    #peak 1 proportion summary statistc
    peak1_proportion = [float(0)]*cell_number_24hr

    num_sims = len(mito_lengths)
    final_ou_vals = np.zeros(num_sims).astype(np.float64)
    for i in prange(num_sims):
        #print(i)

        l = mito_lengths[i]
        time_point = time_indicator[i]
        #Generating the stochastic part of the birth rate function over the course of 24hrs
        OU_times, OU_births = OU_birth_func(time = time_point, mean = 0, theta = theta, sd=sd*mu_b, initial_val = initial_ou_vals[i])
        final_ou_vals[i] = OU_births[-1]
        #Initialising the nucleoid state based on the output of the burn in.
        #Order of populations is (***)
        nucleoid_state = np.array([round(replicating_output[i]),0,0,round(young_output[i]),0,0, round(old_output[i]),0,0]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state

        #Looping for either 1,3,7, or 24 hours, depending on the cell
        while current_time <= time_point:
            current_replicating_DNA = nucleoid_state[0]
            current_replicating_single = nucleoid_state[1]
            current_replicating_double = nucleoid_state[2]
            current_young_DNA = nucleoid_state[3]
            current_young_single = nucleoid_state[4]
            current_young_double = nucleoid_state[5]
            current_old_DNA = nucleoid_state[6]
            current_old_single = nucleoid_state[7]
            current_old_double = nucleoid_state[8]

            current_replicating = current_replicating_DNA + current_replicating_single + current_replicating_double
            current_young = current_young_DNA + current_young_single + current_young_double
            current_old = current_old_DNA + current_old_single + current_old_double

            n = int(np.sum(nucleoid_state))

            #If the cell has ran out of nucleoids, end the loop
            if n == 0:
                break
            
            max_ou = np.max(OU_births[OU_times >= current_time])
            #Generating the time that the next event takes place
            max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0,beta1,max_ou) + death_rate(current_old, n, mu_d, c, l, beta0,beta1) + current_replicating*true_birth_param + current_young*mu_a
            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<96:  
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            #If the event time is greater than the final time point, then this event doesn't happen, and we break before updating the nucleid state
            if current_time > time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################

            #Probability of a general event
            birth_ou = np.interp(current_time, OU_times, OU_births)
            p_birth = birth_rate(current_young,n,mu_b,c,l,beta0,beta1, birth_ou)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0,beta1)/max_propensity
            p_truebirth_dif = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_truebirth_nodif = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity
            p_ageing = mu_a*current_young/max_propensity
            p_fake = 1 - p_birth - p_death - p_truebirth_dif - p_truebirth_nodif - p_ageing

            #probability that the event happens to an untagged, single, or double tagged molecule, given that the event affects ...
            
            #... the young population
            p_young_untagged = current_young_DNA/max(current_young,1)
            p_young_single = current_young_single/max(current_young,1)
            p_young_double = current_young_double/max(current_young,1)

            #... the old population
            p_old_untagged = current_old_DNA/max(current_old,1)
            p_old_single = current_old_single/max(current_old,1)
            p_old_double = current_old_double/max(current_old,1)

            #... the replicating population
            p_replicating_untagged = current_replicating_DNA/max(current_replicating,1)
            p_replicating_single = current_replicating_single/max(current_replicating,1)
            p_replicating_double = current_replicating_double/max(current_replicating,1)

            #Probabilities of each event (order given in (*****))
            probability_vector = np.array([p_birth*p_young_untagged, p_truebirth_dif*p_replicating_untagged, p_truebirth_nodif*p_replicating_untagged,
                                           p_birth*p_young_single, p_truebirth_dif*p_replicating_single, p_truebirth_nodif*p_replicating_single/2,
                                           p_truebirth_nodif*p_replicating_single/2, p_birth*p_young_double, p_truebirth_dif*p_replicating_double,
                                           p_truebirth_nodif*p_replicating_double, p_ageing*p_young_untagged, p_ageing*p_young_single, 
                                           p_ageing*p_young_double, p_death*p_old_untagged, p_death*p_old_single, p_death*p_old_double, p_fake])
            
            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)

            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #Appending the data
        if inference_portion == "pulse":
            measurement_error = (1 - np.random.exponential(percent_error)/100)
            tagged = current_replicating_DNA + current_replicating_single + current_replicating_double + current_young_single + current_young_double + current_old_single + current_old_double
            tagged_output = round(tagged*measurement_error)
            untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
            if time_point == 1:
                nucleoid_num_1hr[i] = int(tagged_output + untagged_output)
                tagged_num_1hr[i] = int(tagged_output)
                mtvolume_1hr[i] = float(l)
            elif time_point == 3:
                j = i-cell_number_1hr
                nucleoid_num_3hr[j] = int(tagged_output + untagged_output)
                tagged_num_3hr[j] = int(tagged_output)
                mtvolume_3hr[j] = float(l)
            elif time_point == 7:
                j = i-cell_number_1hr - cell_number_3hr
                nucleoid_num_7hr[j] = int(tagged_output + untagged_output)
                tagged_num_7hr[j] = int(tagged_output)
                mtvolume_7hr[j] = float(l)
            elif time_point == 24:
                j = i-cell_number_1hr - cell_number_3hr - cell_number_7hr
                nucleoid_num_24hr[j] = int(tagged_output + untagged_output)
                tagged_num_24hr[j] = int(tagged_output)
                mtvolume_24hr[j] = float(l)

                peak1 = current_young_single + current_old_single
                peak1_proportion[j] = float(peak1/max(tagged,1))

        elif inference_portion == "chase":
            chase_final_state[0,i] = current_replicating_DNA
            chase_final_state[1,i] = current_replicating_single
            chase_final_state[2,i] = current_replicating_double
            chase_final_state[3,i] = current_young_DNA
            chase_final_state[4,i] = current_young_single
            chase_final_state[5,i] = current_young_double
            chase_final_state[6,i] = current_old_DNA
            chase_final_state[7,i] = current_old_single
            chase_final_state[8,i] = current_old_double

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_1hr = np.array(nucleoid_num_1hr)
    nucleoid_num_3hr = np.array(nucleoid_num_3hr)
    nucleoid_num_7hr = np.array(nucleoid_num_7hr)
    nucleoid_num_24hr = np.array(nucleoid_num_24hr)

    tagged_num_1hr = np.array(tagged_num_1hr)
    tagged_num_3hr = np.array(tagged_num_3hr)
    tagged_num_7hr = np.array(tagged_num_7hr)
    tagged_num_24hr = np.array(tagged_num_24hr)

    mtvolume_1hr = np.array(mtvolume_1hr)
    mtvolume_3hr = np.array(mtvolume_3hr)
    mtvolume_7hr = np.array(mtvolume_7hr)
    mtvolume_24hr = np.array(mtvolume_24hr)

    e1_h = 0.0
    e3_h = 0.0
    e7_h = 0.0
    e24_h = 0.0
    if inference_portion == "pulse":
        average_peak1_proportion = np.mean(np.array(peak1_proportion))
        if not full_trajectory:
            e1_h = hetero_summary_statistic_e1(measurement_error*tagged_num_1hr, mtvolume_1hr, mode = mode)
            e3_h = hetero_summary_statistic_e3(measurement_error*tagged_num_3hr, mtvolume_3hr, mode = mode)
            e7_h = hetero_summary_statistic_e7(measurement_error*tagged_num_7hr, mtvolume_7hr, mode = mode)
            e24_h = hetero_summary_statistic_e24(measurement_error*tagged_num_24hr, mtvolume_24hr, mode = mode)
    elif inference_portion == "chase":
        average_peak1_proportion = 0

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, final_ou_vals, e1_h, e3_h, e7_h, e24_h)


# @jit(nopython=True)
# def _ou_three_population_chase(mu_a, mu_b,c, true_birth_param, diffusion_prob, mu_d, theta, sd, initial_ou_vals, birth_rate, death_rate, beta0, beta1, initial_states,
#                                          mito_lengths = training_mito_lengths_chase, time_indicator = training_chase_time_indicator, verbose = True, full_trajectory = False):
    
#     percent_error = 3.6
#     if verbose:
#         print("------------------------Beginning Chase Simulation --------------------------")

#     #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
#     # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
#     # Order of populations (m): 0. replicating untagged, 1. replicating single tagged, 2. replicating double tagged,
#     #                           3. young untagged, 4. young single tagged, 5. young double tagged,
#     #                           6. remnant untagged, 7. remnant single tagged, 8. remnant double tagged,
#     #                           9. old untagged, 10. old single tagged, 11. old double tagged (***)
    
#     # Order of events (n): 0. untagged birth
#     #                      1. untagged replication with diffusion
#     #                      2. untagged replication without diffusion
#     #                      3. single tagged birth
#     #                      4. single tagged replication with diffusion
#     #                      5. single tagged replication without diffusion (single tagged daughter keeps replicating)
#     #                      6. single tagged replication without diffusion (untagged daughter keeps replicating)
#     #                      7. double tagged birth
#     #                      8. double tagged replication with diffusion 
#     #                      9. double tagged replication without diffusion
#     #                      
#     #                      10. untagged remnant replication with diffusion
#     #                      11. untagged remnant replication without diffusion (single tagged daughter keeps replicating),
#     #                      12. untagged remnant replication without diffusion (untagged daughter keeps replicating),
#     #                      13. single tagged remnant replication with diffusion (two single stranded daughters),
#     #                      14. single tagged remnant replication with diffusion (one untagged, one double stranded daughter),
#     #                      15. single tagged remnant replication without diffusion (two single stranded daughters),
#     #                      16. single tagged remnant replication without diffusion (one untagged, one double stranded daughter, untagged keeps replicating)
#     #                      17. single tagged remnant replication without diffusion (one untagged, one double stranded daughter, double strand keeps replicating)
#     #                      18. double tagged remnant replication with diffusion
#     #                      19. double tagged remnant replication without diffusion (double strand keeps replicating)
#     #                      20. double tagged remnant replication without diffusion (single strand keeps replicating)
#     #
#     #                      21.untagged ageing, 22. single tagged ageing, 23. double tagged ageing,
#     #                      24, untagged death, 25, single tagged death, 26, double tagged death

#     step_matrix = np.array([[1, -1, 0,  0,  0,  0,  1,  0,  0,  0,          0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,     0,0,0,  0,0,0,0],
#                             [0,  0, 0,  1, -1,  0, -1,  0,  0,  1,          0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  1,     0,0,0,  0,0,0,0],
#                             [0,  0, 0,  0,  0,  0,  0,  1, -1, -1,          0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,     0,0,0,  0,0,0,0],
#                             [-1, 2, 1,  0,  1,  1,  0,  0,  0,  0,          1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,     -1,0,0, 0,0,0,0],
#                             [0,  0, 0, -1,  1,  0,  1,  0,  2,  1,          1,  0,  1,  2,  0,  1,  0,  0,  1,  1,  0,     0,-1,0, 0,0,0,0],
#                             [0,  0, 0,  0,  0,  0,  0, -1,  0,  0,          0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,     0,0,-1, 0,0,0,0],
#                             [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,         -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,     0,0,0,  0,0,0,0],
#                             [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,     0,0,0,  0,0,0,0],
#                             [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1,     0,0,0,  0,0,0,0],
#                             [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     1,0,0, -1,0,0,0],
#                             [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     0,1,0,  0,-1,0,0],
#                             [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     0,0,1,  0,0,-1,0]]).astype(np.float64)
    
#     step_matrix = step_matrix.transpose()

#     #If full_trajectory, we simulate every cell for 4 days, rather than some for 0, 1, 2.
#     if full_trajectory:
#         time_indicator = 4*24*np.ones(len(mito_lengths))

#     num_sims = len(mito_lengths)

#     cell_number_0dy = np.sum(time_indicator == 0)
#     cell_number_1dy = np.sum(time_indicator == 24)
#     cell_number_2dy = np.sum(time_indicator == 48)
#     cell_number_4dy = np.sum(time_indicator == 96)

#     nucleoid_num_0dy = [int(0)]*cell_number_0dy
#     tagged_num_0dy = [int(0)]*cell_number_0dy
#     mtvolume_0dy = [float(0)]*cell_number_0dy

#     nucleoid_num_1dy = [int(0)]*cell_number_1dy
#     tagged_num_1dy = [int(0)]*cell_number_1dy
#     mtvolume_1dy = [float(0)]*cell_number_1dy

#     nucleoid_num_2dy = [int(0)]*cell_number_2dy
#     tagged_num_2dy = [int(0)]*cell_number_2dy
#     mtvolume_2dy = [float(0)]*cell_number_2dy

#     nucleoid_num_4dy = [int(0)]*cell_number_4dy
#     tagged_num_4dy = [int(0)]*cell_number_4dy
#     mtvolume_4dy = [float(0)]*cell_number_4dy
    
#     if full_trajectory:
#         initial_peak1_proportion = [float(0)]*cell_number_4dy
#     else:
#         initial_peak1_proportion = [float(0)]*cell_number_0dy
#     final_peak1_proportion = [float(0)]*cell_number_4dy

#     #the full trajectory, if full_trajectory == True
#     trajectory = np.zeros((len(mito_lengths), 24*4*4 + 1,12)).astype(np.float64)
    
#     #Looping over every cell
#     for i in prange(num_sims):
#         l = mito_lengths[i]
#         time_point = time_indicator[i]

#         pulse_replicating_DNA, pulse_replicating_single, pulse_replicating_double, pulse_young_DNA, pulse_young_single, pulse_young_double, pulse_old_DNA, pulse_old_single, pulse_old_double = np.transpose(initial_states)[i]

#         nucleoid_state = np.array([0,0,0,pulse_young_DNA,pulse_young_single,pulse_young_double, pulse_replicating_DNA, pulse_replicating_single, pulse_replicating_double, pulse_old_DNA, pulse_old_single, pulse_old_double]).astype(np.int64)
#         current_time =  0

#         #Generating the stochastic part of the birth rate function over the course of 24hrs
#         OU_times, OU_births = OU_birth_func(time = time_point, mean = 0, theta = theta, sd=sd*mu_b, initial_val = 0.0)

#         if full_trajectory:
#             trajectory[i][0] = (1 - percent_error/100)*nucleoid_state
#             initial_peak1_proportion[i] = float((pulse_young_single + pulse_old_single)/max(pulse_replicating_DNA + pulse_replicating_single + pulse_replicating_double + pulse_young_single + pulse_young_double + pulse_old_single + pulse_old_double,1))

#         #Looping for either 0, 1, 2, or 4 days, depending on the cell
#         while current_time <= time_point:

#             current_replicating_DNA = nucleoid_state[0]
#             current_replicating_single = nucleoid_state[1]
#             current_replicating_double = nucleoid_state[2]
#             current_young_DNA = nucleoid_state[3]
#             current_young_single = nucleoid_state[4]
#             current_young_double = nucleoid_state[5]
#             current_replicating_DNA_remnant = nucleoid_state[6]
#             current_replicating_single_remnant = nucleoid_state[7]
#             current_replicating_double_remnant = nucleoid_state[8]
#             current_old_DNA = nucleoid_state[9]
#             current_old_single = nucleoid_state[10]
#             current_old_double = nucleoid_state[11]

#             current_replicating = current_replicating_DNA + current_replicating_single + current_replicating_double
#             current_replicating_remnant = current_replicating_DNA_remnant + current_replicating_single_remnant + current_replicating_double_remnant
#             current_old = current_old_DNA + current_old_single + current_old_double
#             current_young = current_young_DNA + current_young_single + current_young_double

#             #If the cell is measured at 0 hours we need not simulate it further
#             if time_point == 0:
#                 break

#             n = np.sum(nucleoid_state)
            
#             if n == 0:
#                 break

#             max_ou = np.max(OU_births[OU_times >= current_time])
#             #Generating the time that the next event takes place
#             max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0, beta1, max_ou) + death_rate(current_old, n, mu_d, c, l, beta0, beta1) + (current_replicating + current_replicating_remnant)*true_birth_param + current_young*mu_a

#             #In the case where mu_b=mu_a=0, there is a chance all nucleoids get stuck in the young population, and no further events occur
#             if max_propensity == 0:
#                 break

#             next_event_time = np.random.exponential(1/max_propensity)

#             #every 15 minutes, we record the current nucleoid state
#             if full_trajectory:
#                 a = int(current_time//0.25)
#                 b = int((current_time+next_event_time)//0.25)

#                 for k in range(a,b):
#                     if k<384:  
#                         trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

#             current_time += next_event_time
#             if current_time >= time_point:
#                 break
            
#             ##################-------------------------Generating what kind of event this is---------------------------#####################
            
#             birth_ou = np.interp(current_time, OU_times, OU_births)
#             p_birth = birth_rate(current_young,n,mu_b,c,l,beta0, beta1, birth_ou)/max_propensity
#             p_death = death_rate(current_old,n,mu_d,c,l,beta0, beta1)/max_propensity
#             p_truebirth_dif = diffusion_prob*current_replicating*true_birth_param/max_propensity
#             p_truebirth_nodif = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity
#             p_remnant_truebirth_dif = diffusion_prob*current_replicating_remnant*true_birth_param/max_propensity
#             p_remnant_truebirth_nodif = (1-diffusion_prob)*current_replicating_remnant*true_birth_param/max_propensity
#             p_ageing = current_young*mu_a/max_propensity
#             p_fake = 1-p_birth-p_death-p_truebirth_dif-p_truebirth_nodif-p_remnant_truebirth_dif-p_remnant_truebirth_nodif-p_ageing

#             #probability that the event happens to an untagged, single, or double tagged molecule, given that the event affects ...
            
#             #... the young population
#             p_young_untagged = current_young_DNA/max(current_young,1)
#             p_young_single = current_young_single/max(current_young,1)
#             p_young_double = current_young_double/max(current_young,1)

#             #... the old population
#             p_old_untagged = current_old_DNA/max(current_old,1)
#             p_old_single = current_old_single/max(current_old,1)
#             p_old_double = current_old_double/max(current_old,1)

#             #... the replicating population
#             p_replicating_untagged = current_replicating_DNA/max(current_replicating,1)
#             p_replicating_single = current_replicating_single/max(current_replicating,1)
#             p_replicating_double = current_replicating_double/max(current_replicating,1)

#             #... the remnant replicating population
#             p_remnant_untagged = current_replicating_DNA_remnant/max(current_replicating_remnant,1)
#             p_remnant_single = current_replicating_single_remnant/max(current_replicating_remnant,1)
#             p_remnant_double = current_replicating_double_remnant/max(current_replicating_remnant,1)

#             probability_vector = np.array([p_birth*p_young_untagged, p_truebirth_dif*p_replicating_untagged, p_truebirth_nodif*p_replicating_untagged,
#                                            p_birth*p_young_single, p_truebirth_dif*p_replicating_single, p_truebirth_nodif*p_replicating_single/2, p_truebirth_nodif*p_replicating_single/2,
#                                            p_birth*p_young_double, p_truebirth_dif*p_replicating_double, p_truebirth_nodif*p_replicating_double,
#                                            p_remnant_truebirth_dif*p_remnant_untagged, p_remnant_truebirth_nodif*p_remnant_untagged/2, p_remnant_truebirth_nodif*p_remnant_untagged/2,
#                                            p_remnant_truebirth_dif*p_remnant_single/2, p_remnant_truebirth_dif*p_remnant_single/2,
#                                            p_remnant_truebirth_nodif*p_remnant_single/2, p_remnant_truebirth_nodif*p_remnant_single/4,p_remnant_truebirth_nodif*p_remnant_single/4,
#                                            p_remnant_truebirth_dif*p_remnant_double, p_remnant_truebirth_nodif*p_remnant_double/2, p_remnant_truebirth_nodif*p_remnant_double/2,
#                                            p_ageing*p_young_untagged, p_ageing*p_young_single, p_ageing*p_young_double,
#                                            p_death*p_old_untagged, p_death*p_old_single, p_death*p_old_double, p_fake])

#             r = np.random.uniform(0,1)
#             event_index = np.searchsorted(np.cumsum(probability_vector), r)
    
#             #Updating the nucleoid state based on what event happened
#             nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()


#         #outputting the final cell state
#         measurement_error = (1 - np.random.exponential(percent_error)/100)
#         tagged = current_replicating_single + current_replicating_double + current_replicating_DNA_remnant + current_replicating_single_remnant + current_replicating_double_remnant + current_young_single + current_young_double + current_old_single + current_old_double
#         tagged_output = round(tagged*measurement_error)
#         untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
#         peak1 = current_young_single + current_old_single + current_replicating_single

#         if time_point == 0:
#             nucleoid_num_0dy[i] = int(tagged_output + untagged_output)
#             tagged_num_0dy[i] = int(tagged_output)
#             mtvolume_0dy[i] = float(l)

#             initial_peak1_proportion[i] = float(peak1/max(tagged, 1)) 
#         elif time_point == 24:
#             j = i-cell_number_0dy
#             nucleoid_num_1dy[j] = int(tagged_output + untagged_output)
#             tagged_num_1dy[j] = int(tagged_output)
#             mtvolume_1dy[j] = float(l)
#         elif time_point == 48:
#             j = i-cell_number_0dy - cell_number_1dy
#             nucleoid_num_2dy[j] = int(tagged_output + untagged_output)
#             tagged_num_2dy[j] = int(tagged_output)
#             mtvolume_2dy[j] = float(l)
#         elif time_point == 96:
#             j = i-cell_number_0dy - cell_number_1dy - cell_number_2dy
#             nucleoid_num_4dy[j] = int(tagged_output + untagged_output)
#             tagged_num_4dy[j] = int(tagged_output)
#             mtvolume_4dy[j] = float(l)

#             final_peak1_proportion[j] = float(peak1/max(tagged, 1)) 

#         if verbose:
#             print("Cell " + str(i) + " Finished")

#     nucleoid_num_0dy = np.array(nucleoid_num_0dy)
#     nucleoid_num_1dy = np.array(nucleoid_num_1dy)
#     nucleoid_num_2dy = np.array(nucleoid_num_2dy)
#     nucleoid_num_4dy = np.array(nucleoid_num_4dy)

#     tagged_num_0dy = np.array(tagged_num_0dy)
#     tagged_num_1dy = np.array(tagged_num_1dy)
#     tagged_num_2dy = np.array(tagged_num_2dy)
#     tagged_num_4dy = np.array(tagged_num_4dy)

#     mtvolume_0dy = np.array(mtvolume_0dy)
#     mtvolume_1dy = np.array(mtvolume_1dy)
#     mtvolume_2dy = np.array(mtvolume_2dy)
#     mtvolume_4dy = np.array(mtvolume_4dy)

#     average_initial_peak1_proportion = np.mean(np.array(initial_peak1_proportion))
#     average_final_peak1_proportion = np.mean(np.array(final_peak1_proportion))

#     return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
#            nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
#             average_initial_peak1_proportion, average_final_peak1_proportion, trajectory)


@jit(nopython=True)
def _ou_three_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, theta, sd, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_time_indicator, mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, verbose = True, full_trajectory = False, mode = "training"):
    """
    Intermediary simulator function for the pulse data. Joins up the burn in function, and the pulse function.
    """
    (replicating_output, young_output, old_output, variance_statistic, ou_vals, S_h) = _ou_burn_in_three_population_model(diffusion_prob=diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b=mu_b_pulse, theta=theta, sd=sd, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = burn_in_time, verbose=verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, final_ou_vals, e1_h, e3_h, e7_h, e24_h) = _ou_three_population_pulse(diffusion_prob = diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b = mu_b_pulse, theta=theta, sd=sd, initial_ou_vals=ou_vals,birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, young_output=young_output, old_output=old_output, beta0=beta0, beta1=beta1, time_indicator = time_indicator, mito_lengths = mito_lengths, verbose=verbose, full_trajectory = full_trajectory, mode = mode)

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h)


# @jit(nopython=True)
# def _ou_three_population_burn_pulse_chase(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, theta, sd, mu_b_chase,diffusion_prob_chase, mu_d_chase, mu_a_chase, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
#                               time_indicator = training_chase_time_indicator, mito_lengths = training_mito_lengths_chase, sig=0.2, verbose = True, full_trajectory=False):

#     (replicating_output, young_output, old_output, variance_statistic, ou_vals, S_h) = _ou_burn_in_three_population_model(diffusion_prob=diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b = mu_b_pulse, theta=theta, sd=sd, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
#                          mito_lengths = mito_lengths, dna_nums = training_dna_numbers, burn_in_time = burn_in_time, sig=sig, inference_portion = "chase", verbose = verbose)

#     (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
#            nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
#             average_peak1_proportion, chase_final_state, trajectory_pulse, final_ou_vals) = _ou_three_population_pulse(diffusion_prob = diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b=mu_b_pulse, theta=theta, sd=sd, initial_ou_vals=ou_vals, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, young_output=young_output, old_output=old_output, beta0=beta0, beta1=beta1, time_indicator=time_indicator, mito_lengths = mito_lengths, inference_portion = "chase", verbose=verbose, full_trajectory = False)
    
#     (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
#            nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
#             initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, trajectory_chase) = _ou_three_population_chase(mu_b=mu_b_chase,c=0, true_birth_param=true_birth_param, diffusion_prob=diffusion_prob_chase, mu_d=mu_d_chase, theta=theta, sd=sd, initial_ou_vals=final_ou_vals, mu_a=mu_a_chase, birth_rate=birth_rate, death_rate=death_rate, beta0=beta0, beta1=beta1, 
#                                          mito_lengths = mito_lengths, time_indicator = time_indicator, initial_states = chase_final_state, full_trajectory=full_trajectory, verbose=verbose)
 
#     return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
#            nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
#            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
#            variance_statistic, trajectory_chase, S_h)

# @jit(nopython=True)
# def ou_logarithmic_three_population_pulse(params, verbose = False, full_trajectory = False, mode = "training"):
#     """
#     Simulator function for the purpose of ABC inference on the pulse data. For every cell in assays 1 and 2 of the pulse data,
#     this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

#     Args:
#         params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
#             - beta0 (float, beta0 >= 0)
#             - beta1 (float, beta1 >= 0)
#             - p (float, 0 <= p <= 1)
#             - mu_d (float, mu_d >= 0)
#             - mu_b (float, mu_b >= 0)
#             - mu_r (float, mu_r >= 0)
#             - c (float, c >= 0)
#         verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
#         mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
#               are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.

#     Returns:
#         Tuple (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
#            nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
#            average_peak1_proportion, variance_statistic), where:

#             - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
#             - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
#             - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
#             - average_peak1_proportion (float) is the mean over all 24 hour cells of the single tagged nucleoid proportion
#             - variance statistic (float) is as described in Wolf, Mjeku et al
#     """

#     beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c, theta, sd = params

#     if mode == "training":
#         time_indicator = training_time_indicator
#         mito_lengths = training_mito_lengths
#         dna_nums = training_dna_numbers
#     elif mode == "validation":
#         time_indicator = validation_time_indicator
#         mito_lengths = validation_mito_lengths
#         dna_nums = validation_dna_numbers
#     else:
#         time_indicator = all_assays_time_indicator
#         mito_lengths = all_assays_mito_lengths
#         dna_nums = all_assays_dna_numbers

#     return _ou_three_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, theta, sd, logarithmic_birth, constant_death, beta0=beta0, beta1=beta1, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths=mito_lengths, dna_nums=dna_nums, mode = mode)

# @jit(nopython=True)
# def ou_logarithmic_three_population_chase(params, verbose = False, full_trajectory = False, mode = "training"):
#     """
#     Simulator function for the purpose of ABC inference on the chase data. For every cell in assays 1 and 2 of the chase data,
#     this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

#     Args:
#         params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
#             - beta0 (float, beta0 >= 0)
#             - beta1 (float, beta1 >= 0)
#             - p (float, 0 <= p <= 1)
#             - mu_d (float, mu_d >= 0)
#             - mu_b (float, mu_b >= 0)
#             - mu_r (float, mu_r >= 0)
#             - mu_b^chase (float, mu_b^chase >= 0)
#             - p^chase (float, p^chase >= 0)
#             - mu_d^chase (float, mu_d^chase >= 0)
#             - mu_a^chase (float, mu_a^chase >= mu_b^chase/p^chase).
#             - c (float, c >= 0)
#         verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
#         mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
#               are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.

#     Returns:
#         Tuple (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
#            nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
#            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
#            variance_statistic), where:

#             - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
#             - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
#             - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
#             - initial_average_peak1_proportion (float) is the mean over all 0 day cells of the single tagged nucleoid proportion
#             - final_average_peak1_proportion (float) is the mean over all 4 day cells of the single tagged nucleoid proportion
#             - variance statistic (float) is as described in Wolf, Mjeku et al
#     """

#     if mode == "training":
#         time_indicator = training_chase_time_indicator
#         mito_lengths = training_mito_lengths_chase
#     elif mode == "validation":
#         time_indicator = validation_chase_time_indicator
#         mito_lengths = validation_mito_lengths_chase
#     elif mode == "full":
#         time_indicator = all_assays_chase_time_indicator
#         mito_lengths = all_assays_mito_lengths_chase

#     beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c, theta, sd = params

#     return _ou_three_population_burn_pulse_chase(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, theta, sd, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, _logarithmic_birth, _constant_death, beta0 = beta0, beta1 = beta1, sig=sig, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths = mito_lengths)

# @jit(nopython=True)
# def ou_inhibition_three_population_chase(params, verbose = False, full_trajectory = False, mode = "training"):
#     """
#     Simulator function for the purpose of ABC inference on the chase data. For every cell in assays 1 and 2 of the chase data,
#     this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

#     Args:
#         params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
#             - beta0 (float, beta0 >= 0)
#             - beta1 (float, beta1 >= 0)
#             - p (float, 0 <= p <= 1)
#             - mu_d (float, mu_d >= 0)
#             - mu_b (float, mu_b >= 0)
#             - mu_r (float, mu_r >= 0)
#             - mu_b^chase (float, mu_b^chase >= 0)
#             - p^chase (float, p^chase >= 0)
#             - mu_d^chase (float, mu_d^chase >= 0)
#             - mu_a^chase (float, mu_a^chase >= mu_b^chase/p^chase).
#             - c (float, c >= 0)
#         verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
#         mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
#               are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.

#     Returns:
#         Tuple (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
#            nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
#            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
#            variance_statistic), where:

#             - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
#             - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
#             - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
#             - initial_average_peak1_proportion (float) is the mean over all 0 day cells of the single tagged nucleoid proportion
#             - final_average_peak1_proportion (float) is the mean over all 4 day cells of the single tagged nucleoid proportion
#             - variance statistic (float) is as described in Wolf, Mjeku et al
#     """

#     if mode == "training":
#         time_indicator = training_chase_time_indicator
#         mito_lengths = training_mito_lengths_chase
#     elif mode == "validation":
#         time_indicator = validation_chase_time_indicator
#         mito_lengths = validation_mito_lengths_chase
#     elif mode == "full":
#         time_indicator = all_assays_chase_time_indicator
#         mito_lengths = all_assays_mito_lengths_chase

#     beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c, theta, sd = params

#     return _ou_three_population_burn_pulse_chase(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, theta, sd, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, _inhibition_birth, _constant_death, beta0 = beta0, beta1 = beta1, sig=sig, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths = mito_lengths)

@jit(nopython=True)
def ou_three_population_birth_rate_burn_in(params, birth_rate, time = 1000, verbose = False):
    """
    Simulator function for the purpose of selecting between different birth rates

    Args:
        params: List[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
        
        c: (float, c >= 0)

        birth_rate: birth rate function. Either _logarithmic_birth, _differential_birth, or _ratiomatric_birth

        time: (float, time > 0). Burn-in time

        verbose: Bool

    Returns:
        Tuple (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic), where:

            - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
            - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
            - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
            - average_peak1_proportion (float) is the mean over all 24 hour cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al
    """

    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c, theta, sd = params

    mito_lengths = all_assays_mito_lengths
    dna_nums = all_assays_dna_numbers

    replicating_output, young_output, old_output, variances, ou_vals = _ou_testing_burn_in_three_population_model(diffusion_prob_pulse,c, true_birth_param, mu_d_pulse, mu_b_pulse, theta, sd, birth_rate, constant_death, burn_in_increments = 1, beta0=beta0, beta1=beta1, 
                            mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = time, sig=0.2, verbose = verbose, inference_portion = "pulse")
    
    return replicating_output + young_output + old_output, variances

@jit(nopython=True)
def _one_iter_burn_in_three_population_model(diffusion_prob,c,true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths, initial_replicating, initial_young, initial_old):
    """
    Helper function. Simulates the burn-in period for 'burn-in-increments' hours, and records the the variance of the log-residuals.
    """
    #Defining the ageing rate to maintain equilibrium
    mu_a = mu_b/diffusion_prob

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    #
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    #
    # Order of populations (m): replicating population; young population; old population
    #
    # Order of events (n): replication without diffusion; replication with diffusion; Birth; ageing; death
    step_matrix = np.array([[-1,0,1,0,0],
                           [2,1,-1,-1,0],
                           [0,0,0,1,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()
    num_sims= len(mito_lengths)

    log_residuals = np.zeros(num_sims).astype(np.float64)
    replicating_output = np.zeros(num_sims).astype(np.int64)
    young_output = np.zeros(num_sims).astype(np.int64)
    old_output = np.zeros(num_sims).astype(np.int64)

    #Looping over each cell
    for i in prange(num_sims):
    
        l=mito_lengths[i]

        nucleoid_state = np.array([round(initial_replicating[i]), round(initial_young[i]), round(initial_old[i])]).astype(np.int64)
        current_time =  0

        #Looping until the end of this iteration (usually 1 hour)
        while current_time <= burn_in_increments:
            current_replicating = nucleoid_state[0]
            current_young = nucleoid_state[1]
            current_old = nucleoid_state[2]

            n = int(np.sum(nucleoid_state))
            if n == 0:
                #print("breaking")
                break
            
            ##################----------------Generating the time that the next event takes place---------------------######################

            max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0, beta1) + death_rate(current_old, n, mu_d, c, l, beta0, beta1) + current_replicating*true_birth_param + mu_a*current_young
            next_event_time = np.random.exponential(1/max_propensity)
            
            #Updating the time
            current_time += next_event_time
            if current_time > burn_in_increments or (not np.any(nucleoid_state)):
                current_time = burn_in_increments
                break

            ##################-------------------------Generating what kind of event this is---------------------------#####################

            p_birth = birth_rate(current_young,n,mu_b,c,l,beta0, beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0, beta1)/max_propensity
            p_double_truebirth = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_single_truebirth = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity
            p_ageing = mu_a*current_young/max_propensity

            probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_death])
            r2 = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r2)

            #Updating the nucleoid state based on which event occured
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()


        #Recording each subpopulation after the end of the iteration (usually an hour)
        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]
        replicating_output[i] = current_replicating
        young_output[i] = current_young
        old_output[i] = current_old

        #Recording the log residuals of the single cell
        log_residuals[i] = np.log(max(current_replicating + current_young + current_old,1)) - np.log(beta1*l + beta0)

    #Recording the variance of the log residuals of every cell
    variance_output = np.var(log_residuals)

    #Return each subpopulation for each cell to feed into the next iteration, as well as the log residual variance
    return (replicating_output.astype(np.float64), young_output.astype(np.float64), old_output.astype(np.float64), variance_output)

@jit(nopython=True)
def _burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments = 1, beta0=172, beta1=1.38, 
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, verbose = True, inference_portion = "pulse", mode = "training"):
    """
    Helper function. Simulates the burn-in period and records the control strength summary statistic S_cs.
    """
    if verbose:
        print("------------------------Beginning Burn In --------------------------")

    #Undercounting percentage error
    percent_error = 3.6

    #Initialising the nucleoid number
    if inference_portion == "pulse":
        #pulse data: we take the actual nucleoid number and error correct
        initial_nucleoid_number = dna_nums/(1-percent_error/100)
    else:
        #chase data: we draw from the empirical distribution of the 0 day data, and error correct
        initial_nucleoid_number = np.maximum((beta0 + beta1*mito_lengths + sig*mito_lengths*normal_sample(len(mito_lengths)))/(1-percent_error/100),1)

    # print("negative nucleoids:")
    # print(np.sum(initial_nucleoid_number < 0))
    #defining mu_a to maintain equilibrium
    mu_a = mu_b/diffusion_prob
    
    J = int(burn_in_time//burn_in_increments)

    #initialising each subpopulation via the equilibrium proportions (under a deterministic treatment)
    initialisation_denom = 1/true_birth_param + 1/mu_a + 1/mu_d    
    initial_replicating = initial_nucleoid_number/(true_birth_param*initialisation_denom)
    initial_young = initial_nucleoid_number/(mu_a*initialisation_denom)
    initial_old = initial_nucleoid_number/(mu_d*initialisation_denom)

    variances = np.zeros(J).astype(np.float64) 

    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (replicating_output, young_output, old_output, variance_output) = _one_iter_burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths = mito_lengths, initial_replicating = initial_replicating, initial_young = initial_young, initial_old = initial_old)

        variances[j] = variance_output
        initial_replicating = replicating_output.copy()
        initial_young = young_output.copy()
        initial_old = old_output.copy()
        if verbose:
            print("Iteration " +str(j) + " Finished")

    nucleoid_num = initial_replicating + initial_young + initial_old
    measurement_error = (1 - np.random.exponential(percent_error)/100)
    S_h = hetero_summary_statistic(measurement_error*nucleoid_num, mito_lengths, mode = mode) 
    #outputting each subpopulation to feed into the pulse portion of the simulation, as well as S_cs, and S_h
    return replicating_output, young_output, old_output, np.log(variances[-1]/variances[0]), S_h

@jit(nopython=True)
def _testing_burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments = 1, beta0=172, beta1=1.38, 
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, verbose = True, inference_portion = "pulse"):
    """
    Helper function for 'three_population_birth_rate_burn_in. Simulates the burn-in period and records the variance of the log-residuals over time.
    """
    if verbose:
        print("------------------------Beginning Burn In --------------------------")

    #Undercounting percentage error
    percent_error = 3.6

    #Initialising the nucleoid number
    if inference_portion == "pulse":
        #pulse data: we take the actual nucleoid number and error correct
        initial_nucleoid_number = dna_nums/(1-percent_error/100)
    else:
        #chase data: we draw from the empirical distribution of the 0 day data, and error correct
        initial_nucleoid_number = np.maximum((beta0 + beta1*mito_lengths + sig*mito_lengths*normal_sample(len(mito_lengths)))/(1-percent_error/100),1)

    # print("negative nucleoids:")
    # print(np.sum(initial_nucleoid_number < 0))
    #defining mu_a to maintain equilibrium
    mu_a = mu_b/diffusion_prob
    
    J = int(burn_in_time//burn_in_increments)

    #initialising each subpopulation via the equilibrium proportions (under a deterministic treatment)
    initialisation_denom = 1/true_birth_param + 1/mu_a + 1/mu_d    
    initial_replicating = initial_nucleoid_number/(true_birth_param*initialisation_denom)
    initial_young = initial_nucleoid_number/(mu_a*initialisation_denom)
    initial_old = initial_nucleoid_number/(mu_d*initialisation_denom)

    variances = np.zeros(J).astype(np.float64) 

    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (replicating_output, young_output, old_output, variance_output) = _one_iter_burn_in_three_population_model(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths = mito_lengths, initial_replicating = initial_replicating, initial_young = initial_young, initial_old = initial_old)

        variances[j] = variance_output
        initial_replicating = replicating_output.copy()
        initial_young = young_output.copy()
        initial_old = old_output.copy()
        if verbose:
            print("Iteration " +str(j) + " Finished")

    #outputting each subpopulation to feed into the pulse portion of the simulation, as well as S_cs
    return replicating_output, young_output, old_output, variances


@jit(nopython=True)
def _three_population_pulse(diffusion_prob,c, true_birth_param, mu_d, mu_b, birth_rate, death_rate, replicating_output, young_output, old_output, beta0=172.431, beta1=1.3809, mito_lengths = training_mito_lengths, time_indicator = training_time_indicator, verbose = True, inference_portion = "pulse", full_trajectory = False, mode = "training"):
    """
    Helper function. Takes as input the output of the burn-in period and simulates the pulse portion of the experiment.
    """
    if verbose:
        print("------------------------Beginning Pulse Simulation --------------------------")

    percent_error = 3.6
    mu_a = mu_b/diffusion_prob
    
    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): replicating untagged, replicating single tagged, replicating double tagged,
    #                           young untagged, young single tagged, young double tagged,
    #                           old untagged, old single tagged, old double tagged (***)
    
    # Order of events (n): untagged birth, untagged replication with diffusion, untagged replication without diffusion,
    #                      single tagged birth, single tagged replication with diffusion,
    #                      single tagged replication without diffusion (single tagged daughter keeps replicating),
    #                      single tagged replication without diffusion (double tagged daughter keeps replicating),
    #                      double tagged birth, double tagged replication with diffusion, double tagged replication without diffusion,
    #                      untagged ageing, single tagged ageing, double tagged ageing
    #                      untagged death, single tagged death, double tagged death (*****)

    step_matrix = np.array([[1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,1,-1,0,-1,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,1,-1,0,0,0,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0],
                            [0,2,1,-1,1,0,1,0,0,0,0,-1,0,0,0,0],
                            [0,0,0,0,1,1,0,-1,2,1,0,0,-1,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    #If full_trajectory (or its the chase portion), we simulate every cell for 24 hours, rather than some for 1, 3 , 7.
    if full_trajectory or inference_portion == "chase":
        time_indicator = 24*np.ones(len(mito_lengths))

    #If we are fitting to the pulse data, we record time points 1,3,7,24
    if inference_portion == "pulse":
        cell_number_1hr = np.sum(time_indicator == 1)
        cell_number_3hr = np.sum(time_indicator == 3)
        cell_number_7hr = np.sum(time_indicator == 7)
        cell_number_24hr = np.sum(time_indicator == 24)

        cell_number_chase = 0

    #If we are fitting to the chase data, we only record the final state to initialise the chase portion of the experiment
    elif inference_portion == "chase":
        cell_number_1hr = 0
        cell_number_3hr = 0
        cell_number_7hr = 0
        cell_number_24hr = 0

        cell_number_chase = np.sum(time_indicator > -1)

    #Initialising storage arrays for the 4 time points of interest
    nucleoid_num_1hr = [int(0)]*cell_number_1hr
    tagged_num_1hr = [int(0)]*cell_number_1hr
    mtvolume_1hr = [float(0)]*cell_number_1hr

    nucleoid_num_3hr = [int(0)]*cell_number_3hr
    tagged_num_3hr = [int(0)]*cell_number_3hr
    mtvolume_3hr = [float(0)]*cell_number_3hr

    nucleoid_num_7hr = [int(0)]*cell_number_7hr
    tagged_num_7hr = [int(0)]*cell_number_7hr
    mtvolume_7hr = [float(0)]*cell_number_7hr

    nucleoid_num_24hr = [int(0)]*cell_number_24hr
    tagged_num_24hr = [int(0)]*cell_number_24hr
    mtvolume_24hr = [float(0)]*cell_number_24hr

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 97,9)).astype(np.float64)

    #initialising a matrix to contain the sizes of each subpopulation for each cell after 24 hours, to use to initialise
    #the chase portion of the experiment
    chase_final_state = np.zeros((9, cell_number_chase)).astype(np.int64)

    #peak 1 proportion summary statistc
    peak1_proportion = [float(0)]*cell_number_24hr

    num_sims = len(mito_lengths)
    for i in prange(num_sims):
        #print(i)

        l = mito_lengths[i]
        time_point = time_indicator[i]

        #Initialising the nucleoid state based on the output of the burn in.
        #Order of populations is (***)
        nucleoid_state = np.array([round(replicating_output[i]),0,0,round(young_output[i]),0,0, round(old_output[i]),0,0]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state

        #Looping for either 1,3,7, or 24 hours, depending on the cell
        while current_time <= time_point:
            current_replicating_DNA = nucleoid_state[0]
            current_replicating_single = nucleoid_state[1]
            current_replicating_double = nucleoid_state[2]
            current_young_DNA = nucleoid_state[3]
            current_young_single = nucleoid_state[4]
            current_young_double = nucleoid_state[5]
            current_old_DNA = nucleoid_state[6]
            current_old_single = nucleoid_state[7]
            current_old_double = nucleoid_state[8]

            current_replicating = current_replicating_DNA + current_replicating_single + current_replicating_double
            current_young = current_young_DNA + current_young_single + current_young_double
            current_old = current_old_DNA + current_old_single + current_old_double

            n = int(np.sum(nucleoid_state))

            #If the cell has ran out of nucleoids, end the loop
            if n == 0:
                break

            #Generating the time that the next event takes place
            max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0,beta1) + death_rate(current_old, n, mu_d, c, l, beta0,beta1) + current_replicating*true_birth_param + current_young*mu_a
            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<96:  
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            #If the event time is greater than the final time point, then this event doesn't happen, and we break before updating the nucleid state
            if current_time > time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################

            #Probability of a general event
            p_birth = birth_rate(current_young,n,mu_b,c,l,beta0,beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0,beta1)/max_propensity
            p_truebirth_dif = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_truebirth_nodif = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity
            p_ageing = mu_a*current_young/max_propensity

            #probability that the event happens to an untagged, single, or double tagged molecule, given that the event affects ...
            
            #... the young population
            p_young_untagged = current_young_DNA/max(current_young,1)
            p_young_single = current_young_single/max(current_young,1)
            p_young_double = current_young_double/max(current_young,1)

            #... the old population
            p_old_untagged = current_old_DNA/max(current_old,1)
            p_old_single = current_old_single/max(current_old,1)
            p_old_double = current_old_double/max(current_old,1)

            #... the replicating population
            p_replicating_untagged = current_replicating_DNA/max(current_replicating,1)
            p_replicating_single = current_replicating_single/max(current_replicating,1)
            p_replicating_double = current_replicating_double/max(current_replicating,1)

            #Probabilities of each event (order given in (*****))
            probability_vector = np.array([p_birth*p_young_untagged, p_truebirth_dif*p_replicating_untagged, p_truebirth_nodif*p_replicating_untagged,
                                           p_birth*p_young_single, p_truebirth_dif*p_replicating_single, p_truebirth_nodif*p_replicating_single/2,
                                           p_truebirth_nodif*p_replicating_single/2, p_birth*p_young_double, p_truebirth_dif*p_replicating_double,
                                           p_truebirth_nodif*p_replicating_double, p_ageing*p_young_untagged, p_ageing*p_young_single, 
                                           p_ageing*p_young_double, p_death*p_old_untagged, p_death*p_old_single, p_death*p_old_double])
            
            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)

            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #Appending the data
        if inference_portion == "pulse":
            measurement_error = (1 - np.random.exponential(percent_error)/100)
            tagged = current_replicating_DNA + current_replicating_single + current_replicating_double + current_young_single + current_young_double + current_old_single + current_old_double
            tagged_output = round(tagged*measurement_error)
            untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
            if time_point == 1:
                nucleoid_num_1hr[i] = int(tagged_output + untagged_output)
                tagged_num_1hr[i] = int(tagged_output)
                mtvolume_1hr[i] = float(l)
            elif time_point == 3:
                j = i-cell_number_1hr
                nucleoid_num_3hr[j] = int(tagged_output + untagged_output)
                tagged_num_3hr[j] = int(tagged_output)
                mtvolume_3hr[j] = float(l)
            elif time_point == 7:
                j = i-cell_number_1hr - cell_number_3hr
                nucleoid_num_7hr[j] = int(tagged_output + untagged_output)
                tagged_num_7hr[j] = int(tagged_output)
                mtvolume_7hr[j] = float(l)
            elif time_point == 24:
                j = i-cell_number_1hr - cell_number_3hr - cell_number_7hr
                nucleoid_num_24hr[j] = int(tagged_output + untagged_output)
                tagged_num_24hr[j] = int(tagged_output)
                mtvolume_24hr[j] = float(l)

                peak1 = current_young_single + current_old_single
                peak1_proportion[j] = float(peak1/max(tagged,1))

        elif inference_portion == "chase":
            chase_final_state[0,i] = current_replicating_DNA
            chase_final_state[1,i] = current_replicating_single
            chase_final_state[2,i] = current_replicating_double
            chase_final_state[3,i] = current_young_DNA
            chase_final_state[4,i] = current_young_single
            chase_final_state[5,i] = current_young_double
            chase_final_state[6,i] = current_old_DNA
            chase_final_state[7,i] = current_old_single
            chase_final_state[8,i] = current_old_double

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_1hr = np.array(nucleoid_num_1hr)
    nucleoid_num_3hr = np.array(nucleoid_num_3hr)
    nucleoid_num_7hr = np.array(nucleoid_num_7hr)
    nucleoid_num_24hr = np.array(nucleoid_num_24hr)

    tagged_num_1hr = np.array(tagged_num_1hr)
    tagged_num_3hr = np.array(tagged_num_3hr)
    tagged_num_7hr = np.array(tagged_num_7hr)
    tagged_num_24hr = np.array(tagged_num_24hr)

    mtvolume_1hr = np.array(mtvolume_1hr)
    mtvolume_3hr = np.array(mtvolume_3hr)
    mtvolume_7hr = np.array(mtvolume_7hr)
    mtvolume_24hr = np.array(mtvolume_24hr)

    e1_h = 0.0
    e3_h = 0.0
    e7_h = 0.0
    e24_h = 0.0
    if inference_portion == "pulse":
        average_peak1_proportion = np.mean(np.array(peak1_proportion))
        if not full_trajectory:
            e1_h = hetero_summary_statistic_e1(measurement_error*tagged_num_1hr, mtvolume_1hr, mode = mode)
            e3_h = hetero_summary_statistic_e3(measurement_error*tagged_num_3hr, mtvolume_3hr, mode = mode)
            e7_h = hetero_summary_statistic_e7(measurement_error*tagged_num_7hr, mtvolume_7hr, mode = mode)
            e24_h = hetero_summary_statistic_e24(measurement_error*tagged_num_24hr, mtvolume_24hr, mode = mode)
    elif inference_portion == "chase":
        average_peak1_proportion = 0

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h)

@jit(nopython=True)
def _three_population_chase(mu_a, mu_b,c, true_birth_param, diffusion_prob, mu_d, birth_rate, death_rate, beta0, beta1, initial_states,
                                         mito_lengths = training_mito_lengths_chase, time_indicator = training_chase_time_indicator, verbose = True, full_trajectory = False):
    """
    Helper function. Takes as input the output of the pulse portion and simulates the chase portion of the experiment.
    """
    percent_error = 3.6
    if verbose:
        print("------------------------Beginning Chase Simulation --------------------------")

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): 0. replicating untagged, 1. replicating single tagged, 2. replicating double tagged,
    #                           3. young untagged, 4. young single tagged, 5. young double tagged,
    #                           6. remnant untagged, 7. remnant single tagged, 8. remnant double tagged,
    #                           9. old untagged, 10. old single tagged, 11. old double tagged (***)
    
    # Order of events (n): 0. untagged birth
    #                      1. untagged replication with diffusion
    #                      2. untagged replication without diffusion
    #                      3. single tagged birth
    #                      4. single tagged replication with diffusion
    #                      5. single tagged replication without diffusion (single tagged daughter keeps replicating)
    #                      6. single tagged replication without diffusion (untagged daughter keeps replicating)
    #                      7. double tagged birth
    #                      8. double tagged replication with diffusion 
    #                      9. double tagged replication without diffusion
    #                      
    #                      10. untagged remnant replication with diffusion
    #                      11. untagged remnant replication without diffusion (single tagged daughter keeps replicating),
    #                      12. untagged remnant replication without diffusion (untagged daughter keeps replicating),
    #                      13. single tagged remnant replication with diffusion (two single stranded daughters),
    #                      14. single tagged remnant replication with diffusion (one untagged, one double stranded daughter),
    #                      15. single tagged remnant replication without diffusion (two single stranded daughters),
    #                      16. single tagged remnant replication without diffusion (one untagged, one double stranded daughter, untagged keeps replicating)
    #                      17. single tagged remnant replication without diffusion (one untagged, one double stranded daughter, double strand keeps replicating)
    #                      18. double tagged remnant replication with diffusion
    #                      19. double tagged remnant replication without diffusion (double strand keeps replicating)
    #                      20. double tagged remnant replication without diffusion (single strand keeps replicating)
    #
    #                      21.untagged ageing, 22. single tagged ageing, 23. double tagged ageing,
    #                      24, untagged death, 25, single tagged death, 26, double tagged death

    step_matrix = np.array([[1, -1, 0,  0,  0,  0,  1,  0,  0,  0,          0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,     0,0,0,  0,0,0],
                            [0,  0, 0,  1, -1,  0, -1,  0,  0,  1,          0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  1,     0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  1, -1, -1,          0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,     0,0,0,  0,0,0],
                            [-1, 2, 1,  0,  1,  1,  0,  0,  0,  0,          1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,     -1,0,0, 0,0,0],
                            [0,  0, 0, -1,  1,  0,  1,  0,  2,  1,          1,  0,  1,  2,  0,  1,  0,  0,  1,  1,  0,     0,-1,0, 0,0,0],
                            [0,  0, 0,  0,  0,  0,  0, -1,  0,  0,          0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,     0,0,-1, 0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,         -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,     0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,     0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1,     0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     1,0,0, -1,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     0,1,0,  0,-1,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     0,0,1,  0,0,-1]]).astype(np.float64)
    
    step_matrix = step_matrix.transpose()

    #If full_trajectory, we simulate every cell for 4 days, rather than some for 0, 1, 2.
    if full_trajectory:
        time_indicator = 4*24*np.ones(len(mito_lengths))

    num_sims = len(mito_lengths)

    cell_number_0dy = np.sum(time_indicator == 0)
    cell_number_1dy = np.sum(time_indicator == 24)
    cell_number_2dy = np.sum(time_indicator == 48)
    cell_number_4dy = np.sum(time_indicator == 96)

    nucleoid_num_0dy = [int(0)]*cell_number_0dy
    tagged_num_0dy = [int(0)]*cell_number_0dy
    mtvolume_0dy = [float(0)]*cell_number_0dy

    nucleoid_num_1dy = [int(0)]*cell_number_1dy
    tagged_num_1dy = [int(0)]*cell_number_1dy
    mtvolume_1dy = [float(0)]*cell_number_1dy

    nucleoid_num_2dy = [int(0)]*cell_number_2dy
    tagged_num_2dy = [int(0)]*cell_number_2dy
    mtvolume_2dy = [float(0)]*cell_number_2dy

    nucleoid_num_4dy = [int(0)]*cell_number_4dy
    tagged_num_4dy = [int(0)]*cell_number_4dy
    mtvolume_4dy = [float(0)]*cell_number_4dy
    
    if full_trajectory:
        initial_peak1_proportion = [float(0)]*cell_number_4dy
    else:
        initial_peak1_proportion = [float(0)]*cell_number_0dy
    final_peak1_proportion = [float(0)]*cell_number_4dy

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 24*4*4 + 1,12)).astype(np.float64)
    
    #Looping over every cell
    for i in prange(num_sims):
        l = mito_lengths[i]
        time_point = time_indicator[i]

        pulse_replicating_DNA, pulse_replicating_single, pulse_replicating_double, pulse_young_DNA, pulse_young_single, pulse_young_double, pulse_old_DNA, pulse_old_single, pulse_old_double = np.transpose(initial_states)[i]

        nucleoid_state = np.array([0,0,0,pulse_young_DNA,pulse_young_single,pulse_young_double, pulse_replicating_DNA, pulse_replicating_single, pulse_replicating_double, pulse_old_DNA, pulse_old_single, pulse_old_double]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state
            initial_peak1_proportion[i] = float((pulse_young_single + pulse_old_single)/max(pulse_replicating_DNA + pulse_replicating_single + pulse_replicating_double + pulse_young_single + pulse_young_double + pulse_old_single + pulse_old_double,1))

        #Looping for either 0, 1, 2, or 4 days, depending on the cell
        while current_time <= time_point:

            current_replicating_DNA = nucleoid_state[0]
            current_replicating_single = nucleoid_state[1]
            current_replicating_double = nucleoid_state[2]
            current_young_DNA = nucleoid_state[3]
            current_young_single = nucleoid_state[4]
            current_young_double = nucleoid_state[5]
            current_replicating_DNA_remnant = nucleoid_state[6]
            current_replicating_single_remnant = nucleoid_state[7]
            current_replicating_double_remnant = nucleoid_state[8]
            current_old_DNA = nucleoid_state[9]
            current_old_single = nucleoid_state[10]
            current_old_double = nucleoid_state[11]

            current_replicating = current_replicating_DNA + current_replicating_single + current_replicating_double
            current_replicating_remnant = current_replicating_DNA_remnant + current_replicating_single_remnant + current_replicating_double_remnant
            current_old = current_old_DNA + current_old_single + current_old_double
            current_young = current_young_DNA + current_young_single + current_young_double

            #If the cell is measured at 0 hours we need not simulate it further
            if time_point == 0:
                break

            n = np.sum(nucleoid_state)
            
            if n == 0:
                break

            #Generating the time that the next event takes place
            max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0, beta1) + death_rate(current_old, n, mu_d, c, l, beta0, beta1) + (current_replicating + current_replicating_remnant)*true_birth_param + current_young*mu_a

            #In the case where mu_b=mu_a=0, there is a chance all nucleoids get stuck in the young population, and no further events occur
            if max_propensity == 0:
                break

            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<384:  
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            if current_time >= time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################
            
            p_birth = birth_rate(current_young,n,mu_b,c,l,beta0, beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0, beta1)/max_propensity
            p_truebirth_dif = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_truebirth_nodif = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity
            p_remnant_truebirth_dif = diffusion_prob*current_replicating_remnant*true_birth_param/max_propensity
            p_remnant_truebirth_nodif = (1-diffusion_prob)*current_replicating_remnant*true_birth_param/max_propensity
            p_ageing = current_young*mu_a/max_propensity

            #probability that the event happens to an untagged, single, or double tagged molecule, given that the event affects ...
            
            #... the young population
            p_young_untagged = current_young_DNA/max(current_young,1)
            p_young_single = current_young_single/max(current_young,1)
            p_young_double = current_young_double/max(current_young,1)

            #... the old population
            p_old_untagged = current_old_DNA/max(current_old,1)
            p_old_single = current_old_single/max(current_old,1)
            p_old_double = current_old_double/max(current_old,1)

            #... the replicating population
            p_replicating_untagged = current_replicating_DNA/max(current_replicating,1)
            p_replicating_single = current_replicating_single/max(current_replicating,1)
            p_replicating_double = current_replicating_double/max(current_replicating,1)

            #... the remnant replicating population
            p_remnant_untagged = current_replicating_DNA_remnant/max(current_replicating_remnant,1)
            p_remnant_single = current_replicating_single_remnant/max(current_replicating_remnant,1)
            p_remnant_double = current_replicating_double_remnant/max(current_replicating_remnant,1)

            probability_vector = np.array([p_birth*p_young_untagged, p_truebirth_dif*p_replicating_untagged, p_truebirth_nodif*p_replicating_untagged,
                                           p_birth*p_young_single, p_truebirth_dif*p_replicating_single, p_truebirth_nodif*p_replicating_single/2, p_truebirth_nodif*p_replicating_single/2,
                                           p_birth*p_young_double, p_truebirth_dif*p_replicating_double, p_truebirth_nodif*p_replicating_double,
                                           p_remnant_truebirth_dif*p_remnant_untagged, p_remnant_truebirth_nodif*p_remnant_untagged/2, p_remnant_truebirth_nodif*p_remnant_untagged/2,
                                           p_remnant_truebirth_dif*p_remnant_single/2, p_remnant_truebirth_dif*p_remnant_single/2,
                                           p_remnant_truebirth_nodif*p_remnant_single/2, p_remnant_truebirth_nodif*p_remnant_single/4,p_remnant_truebirth_nodif*p_remnant_single/4,
                                           p_remnant_truebirth_dif*p_remnant_double, p_remnant_truebirth_nodif*p_remnant_double/2, p_remnant_truebirth_nodif*p_remnant_double/2,
                                           p_ageing*p_young_untagged, p_ageing*p_young_single, p_ageing*p_young_double,
                                           p_death*p_old_untagged, p_death*p_old_single, p_death*p_old_double])

            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)
    
            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()


        #outputting the final cell state
        measurement_error = (1 - np.random.exponential(percent_error)/100)
        tagged = current_replicating_single + current_replicating_double + current_replicating_DNA_remnant + current_replicating_single_remnant + current_replicating_double_remnant + current_young_single + current_young_double + current_old_single + current_old_double
        tagged_output = round(tagged*measurement_error)
        untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
        peak1 = current_young_single + current_old_single + current_replicating_single

        if time_point == 0:
            nucleoid_num_0dy[i] = int(tagged_output + untagged_output)
            tagged_num_0dy[i] = int(tagged_output)
            mtvolume_0dy[i] = float(l)

            initial_peak1_proportion[i] = float(peak1/max(tagged, 1)) 
        elif time_point == 24:
            j = i-cell_number_0dy
            nucleoid_num_1dy[j] = int(tagged_output + untagged_output)
            tagged_num_1dy[j] = int(tagged_output)
            mtvolume_1dy[j] = float(l)
        elif time_point == 48:
            j = i-cell_number_0dy - cell_number_1dy
            nucleoid_num_2dy[j] = int(tagged_output + untagged_output)
            tagged_num_2dy[j] = int(tagged_output)
            mtvolume_2dy[j] = float(l)
        elif time_point == 96:
            j = i-cell_number_0dy - cell_number_1dy - cell_number_2dy
            nucleoid_num_4dy[j] = int(tagged_output + untagged_output)
            tagged_num_4dy[j] = int(tagged_output)
            mtvolume_4dy[j] = float(l)

            final_peak1_proportion[j] = float(peak1/max(tagged, 1)) 

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_0dy = np.array(nucleoid_num_0dy)
    nucleoid_num_1dy = np.array(nucleoid_num_1dy)
    nucleoid_num_2dy = np.array(nucleoid_num_2dy)
    nucleoid_num_4dy = np.array(nucleoid_num_4dy)

    tagged_num_0dy = np.array(tagged_num_0dy)
    tagged_num_1dy = np.array(tagged_num_1dy)
    tagged_num_2dy = np.array(tagged_num_2dy)
    tagged_num_4dy = np.array(tagged_num_4dy)

    mtvolume_0dy = np.array(mtvolume_0dy)
    mtvolume_1dy = np.array(mtvolume_1dy)
    mtvolume_2dy = np.array(mtvolume_2dy)
    mtvolume_4dy = np.array(mtvolume_4dy)

    average_initial_peak1_proportion = np.mean(np.array(initial_peak1_proportion))
    average_final_peak1_proportion = np.mean(np.array(final_peak1_proportion))

    return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
            average_initial_peak1_proportion, average_final_peak1_proportion, trajectory)

@jit(nopython=True)
def _three_population_burn_pulse_chase(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, mu_b_chase,diffusion_prob_chase, mu_d_chase, mu_a_chase, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_chase_time_indicator, mito_lengths = training_mito_lengths_chase, sig=0.2, verbose = True, full_trajectory=False, mode = "training"):

    """
    Helper function. Stitches together the burn-in phase, the pulse phase, and the chase phase into one simulator.
    
    """
    (replicating_output, young_output, old_output, variance_statistic, S_h) = _burn_in_three_population_model(diffusion_prob=diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b = mu_b_pulse, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = training_dna_numbers, burn_in_time = burn_in_time, sig=sig, inference_portion = "chase", verbose = verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory_pulse, e1_h, e3_h, e7_h, e24_h) = _three_population_pulse(diffusion_prob = diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b=mu_b_pulse, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, young_output=young_output, old_output=old_output, beta0=beta0, beta1=beta1, time_indicator=time_indicator, mito_lengths = mito_lengths, inference_portion = "chase", verbose=verbose, full_trajectory = False, mode = mode)
    
    (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, trajectory_chase) = _three_population_chase(mu_b=mu_b_chase,c=0, true_birth_param=true_birth_param, diffusion_prob=diffusion_prob_chase, mu_d=mu_d_chase, mu_a=mu_a_chase, birth_rate=birth_rate, death_rate=death_rate, beta0=beta0, beta1=beta1, 
                                         mito_lengths = mito_lengths, time_indicator = time_indicator, initial_states = chase_final_state, full_trajectory=full_trajectory, verbose=verbose)
 
    return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
           initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
           variance_statistic, trajectory_chase)

@jit(nopython=True)
def _three_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_time_indicator, mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, verbose = True, full_trajectory = False, mode = "training"):
    """
    Helper function. Stitches together the burn-in phase and the pulse phase into one simulator.
    """

    (replicating_output, young_output, old_output, variance_statistic, S_h) = _burn_in_three_population_model(diffusion_prob=diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b=mu_b_pulse, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = burn_in_time, verbose=verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h) = _three_population_pulse(diffusion_prob = diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, mu_b = mu_b_pulse, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, young_output=young_output, old_output=old_output, beta0=beta0, beta1=beta1, time_indicator = time_indicator, mito_lengths = mito_lengths, verbose=verbose, full_trajectory = full_trajectory, mode = mode)

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h)

@jit(nopython=True)
def _one_iter_burn_in_two_population_model(diffusion_prob,c,true_birth_param, mu_d, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths, initial_replicating, initial_old):
    
    #Defining the ageing rate to maintain equilibrium
    mu_b = mu_d*diffusion_prob

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    #
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    #
    # Order of populations (m): replicating population; old population
    #
    # Order of events (n): replication with diffusion; replication without diffusion; Birth; death
    step_matrix = np.array([[-1,0,1,0],
                           [2,1,-1,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()
    num_sims= len(mito_lengths)

    log_residuals = np.zeros(num_sims).astype(np.float64)
    replicating_output = np.zeros(num_sims).astype(np.int64)
    old_output = np.zeros(num_sims).astype(np.int64)
    
    #Looping over each cell
    for i in prange(num_sims):
    
        l=mito_lengths[i]

        nucleoid_state = np.array([round(initial_replicating[i]), round(initial_old[i])]).astype(np.int64)
        current_time =  0

        #Looping until the end of this iteration (usually 1 hour)
        while current_time <= burn_in_increments:
            current_replicating = nucleoid_state[0]
            current_old = nucleoid_state[1]

            n = int(np.sum(nucleoid_state))
            if n == 0:
                break
            
            ##################----------------Generating the time that the next event takes place---------------------######################

            max_propensity = birth_rate(current_old, n, mu_b, c, l, beta0, beta1) + death_rate(current_old, n, mu_d, c, l, beta0, beta1) + current_replicating*true_birth_param
            next_event_time = np.random.exponential(1/max_propensity)
            
            #Updating the time
            current_time += next_event_time
            if current_time > burn_in_increments or (not np.any(nucleoid_state)):
                current_time = burn_in_increments
                break

            ##################-------------------------Generating what kind of event this is---------------------------#####################

            p_birth = birth_rate(current_old,n,mu_b,c,l,beta0, beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0, beta1)/max_propensity
            p_diffusion_truebirth = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_nodiffusion_truebirth = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity

            probability_vector = np.array([p_diffusion_truebirth, p_nodiffusion_truebirth, p_birth, p_death])
            r2 = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r2)

            #Updating the nucleoid state based on which event occured
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()


        #Recording each subpopulation after the end of the iteration (usually an hour)
        current_replicating = nucleoid_state[0]
        current_old = nucleoid_state[1]
        replicating_output[i] = current_replicating
        old_output[i] = current_old

        #Recording the log residuals of the single cell
        log_residuals[i] = np.log(current_replicating + current_old) - np.log(beta1*l + beta0)

    #Recording the variance of the log residuals of every cell
    variance_output = np.var(log_residuals)

    #Return each subpopulation for each cell to feed into the next iteration, as well as the log residual variance
    return (replicating_output.astype(np.float64), old_output.astype(np.float64), variance_output)

@jit(nopython=True)
def _burn_in_two_population_model(diffusion_prob,c, true_birth_param, mu_d, birth_rate, death_rate, burn_in_increments = 1, beta0=172, beta1=1.38, 
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, verbose = True, inference_portion = "pulse", mode = "training"):

    if verbose:
        print("------------------------Beginning Burn In --------------------------")

    #Undercounting percentage error
    percent_error = 3.6

    #Initialising the nucleoid number
    if inference_portion == "pulse":
        #pulse data: we take the actual nucleoid number and error correct
        initial_nucleoid_number = dna_nums/(1-percent_error/100)
    else:
        #chase data: we draw from the empirical distribution of the 0 day data, and error correct
        initial_nucleoid_number = np.maximum((beta0 + beta1*mito_lengths + sig*mito_lengths*normal_sample(len(mito_lengths)))/(1-percent_error/100),1)
    
    J = int(burn_in_time//burn_in_increments)

    #initialising each subpopulation via the equilibrium proportions (under a deterministic treatment)
    initialisation_denom = true_birth_param + mu_d    
    initial_replicating = initial_nucleoid_number*mu_d/initialisation_denom
    initial_old = initial_nucleoid_number*true_birth_param/initialisation_denom

    variances = np.zeros(J).astype(np.float64) 

    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (replicating_output, old_output, variance_output) = _one_iter_burn_in_two_population_model(diffusion_prob,c, true_birth_param, mu_d, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths = mito_lengths, initial_replicating = initial_replicating, initial_old = initial_old)

        variances[j] = variance_output
        initial_replicating = replicating_output.copy()
        initial_old = old_output.copy()
        if verbose:
            print("Iteration " +str(j) + " Finished")

    nucleoid_num = initial_replicating + initial_old
    measurement_error = (1 - np.random.exponential(percent_error)/100)
    S_h = hetero_summary_statistic(measurement_error*nucleoid_num, mito_lengths, mode = mode) 

    #outputting each subpopulation to feed into the pulse portion of the simulation, as well as S_cs
    return replicating_output, old_output, np.log(variances[-1]/variances[0]), S_h

@jit(nopython=True)
def _two_population_pulse(diffusion_prob,c, true_birth_param, mu_d, birth_rate, death_rate, replicating_output, old_output, beta0=172.431, beta1=1.3809, mito_lengths = training_mito_lengths, time_indicator = training_time_indicator, verbose = True, inference_portion = "pulse", full_trajectory = False, mode = "training"):

    if verbose:
        print("------------------------Beginning Pulse Simulation --------------------------")

    percent_error = 3.6
    mu_b = mu_d*diffusion_prob
    
    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): replicating untagged, replicating single tagged, replicating double tagged,
    #                           old untagged, old single tagged, old double tagged (***)
    
    # Order of events (n): untagged birth, untagged replication with diffusion, untagged replication without diffusion,
    #                      single tagged birth, single tagged replication with diffusion,
    #                      single tagged replication without diffusion (single tagged daughter keeps replicating),
    #                      single tagged replication without diffusion (double tagged daughter keeps replicating),
    #                      double tagged birth, double tagged replication with diffusion, double tagged replication without diffusion,
    #                      untagged death, single tagged death, double tagged death (*****)

    step_matrix = np.array([[1,-1,-1,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,1,-1,0,-1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,1,-1,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,0,-1,0,0],
                            [0,2,1,-1,1,0,1,0,0,0,0,-1,0],
                            [0,0,0,0,1,1,0,-1,2,1,0,0,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    #If full_trajectory (or its the chase portion), we simulate every cell for 24 hours, rather than some for 1, 3 , 7.
    if full_trajectory or inference_portion == "chase":
        time_indicator = 24*np.ones(len(mito_lengths))

    #If we are fitting to the pulse data, we record time points 1,3,7,24
    if inference_portion == "pulse":
        cell_number_1hr = np.sum(time_indicator == 1)
        cell_number_3hr = np.sum(time_indicator == 3)
        cell_number_7hr = np.sum(time_indicator == 7)
        cell_number_24hr = np.sum(time_indicator == 24)

        cell_number_chase = 0

    #If we are fitting to the chase data, we only record the final state to initialise the chase portion of the experiment
    elif inference_portion == "chase":
        cell_number_1hr = 0
        cell_number_3hr = 0
        cell_number_7hr = 0
        cell_number_24hr = 0

        cell_number_chase = np.sum(time_indicator > -1)

    #Initialising storage arrays for the 4 time points of interest
    nucleoid_num_1hr = [int(0)]*cell_number_1hr
    tagged_num_1hr = [int(0)]*cell_number_1hr
    mtvolume_1hr = [float(0)]*cell_number_1hr

    nucleoid_num_3hr = [int(0)]*cell_number_3hr
    tagged_num_3hr = [int(0)]*cell_number_3hr
    mtvolume_3hr = [float(0)]*cell_number_3hr

    nucleoid_num_7hr = [int(0)]*cell_number_7hr
    tagged_num_7hr = [int(0)]*cell_number_7hr
    mtvolume_7hr = [float(0)]*cell_number_7hr

    nucleoid_num_24hr = [int(0)]*cell_number_24hr
    tagged_num_24hr = [int(0)]*cell_number_24hr
    mtvolume_24hr = [float(0)]*cell_number_24hr

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 97,6)).astype(np.float64)

    #initialising a matrix to contain the sizes of each subpopulation for each cell after 24 hours, to use to initialise
    #the chase portion of the experiment
    chase_final_state = np.zeros((6, cell_number_chase)).astype(np.int64)

    #peak 1 proportion summary statistc
    peak1_proportion = [float(0)]*cell_number_24hr

    num_sims = len(mito_lengths)
    for i in prange(num_sims):
        #print(i)

        l = mito_lengths[i]
        time_point = time_indicator[i]

        #Initialising the nucleoid state based on the output of the burn in.
        #Order of populations is (***)
        nucleoid_state = np.array([round(replicating_output[i]),0,0, round(old_output[i]),0,0]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state

        #Looping for either 1,3,7, or 24 hours, depending on the cell
        while current_time <= time_point:
            current_replicating_DNA = nucleoid_state[0]
            current_replicating_single = nucleoid_state[1]
            current_replicating_double = nucleoid_state[2]
            current_old_DNA = nucleoid_state[3]
            current_old_single = nucleoid_state[4]
            current_old_double = nucleoid_state[5]
            current_replicating = current_replicating_DNA + current_replicating_single + current_replicating_double
            current_old = current_old_DNA + current_old_single + current_old_double

            n = int(np.sum(nucleoid_state))

            #If the cell has ran out of nucleoids, end the loop
            if n == 0:
                break

            #Generating the time that the next event takes place
            max_propensity = birth_rate(current_old, n, mu_b, c, l, beta0,beta1) + death_rate(current_old, n, mu_d, c, l, beta0,beta1) + current_replicating*true_birth_param
            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<96:  
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            #If the event time is greater than the final time point, then this event doesn't happen, and we break before updating the nucleid state
            if current_time > time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################

            #Probability of a general event
            p_birth = birth_rate(current_old,n,mu_b,c,l,beta0,beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0,beta1)/max_propensity
            p_truebirth_dif = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_truebirth_nodif = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity

            #probability that the event happens to an untagged, single, or double tagged molecule, given that the event affects ...

            #... the old population
            p_old_untagged = current_old_DNA/max(current_old,1)
            p_old_single = current_old_single/max(current_old,1)
            p_old_double = current_old_double/max(current_old,1)

            #... the replicating population
            p_replicating_untagged = current_replicating_DNA/max(current_replicating,1)
            p_replicating_single = current_replicating_single/max(current_replicating,1)
            p_replicating_double = current_replicating_double/max(current_replicating,1)

            #Probabilities of each event (order given in (*****))
            probability_vector = np.array([p_birth*p_old_untagged, p_truebirth_dif*p_replicating_untagged, p_truebirth_nodif*p_replicating_untagged,
                                           p_birth*p_old_single, p_truebirth_dif*p_replicating_single, p_truebirth_nodif*p_replicating_single/2,
                                           p_truebirth_nodif*p_replicating_single/2, p_birth*p_old_double, p_truebirth_dif*p_replicating_double,
                                           p_truebirth_nodif*p_replicating_double, p_death*p_old_untagged, p_death*p_old_single, p_death*p_old_double])
            
            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)

            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #Appending the data
        if inference_portion == "pulse":
            measurement_error = (1 - np.random.exponential(percent_error)/100)
            tagged = current_replicating_DNA + current_replicating_single + current_replicating_double + current_old_single + current_old_double
            tagged_output = round(tagged*measurement_error)
            untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
            if time_point == 1:
                nucleoid_num_1hr[i] = int(tagged_output + untagged_output)
                tagged_num_1hr[i] = int(tagged_output)
                mtvolume_1hr[i] = float(l)
            elif time_point == 3:
                j = i-cell_number_1hr
                nucleoid_num_3hr[j] = int(tagged_output + untagged_output)
                tagged_num_3hr[j] = int(tagged_output)
                mtvolume_3hr[j] = float(l)
            elif time_point == 7:
                j = i-cell_number_1hr - cell_number_3hr
                nucleoid_num_7hr[j] = int(tagged_output + untagged_output)
                tagged_num_7hr[j] = int(tagged_output)
                mtvolume_7hr[j] = float(l)
            elif time_point == 24:
                j = i-cell_number_1hr - cell_number_3hr - cell_number_7hr
                nucleoid_num_24hr[j] = int(tagged_output + untagged_output)
                tagged_num_24hr[j] = int(tagged_output)
                mtvolume_24hr[j] = float(l)

                peak1 = current_old_single
                peak1_proportion[j] = float(peak1/max(tagged,1))

        elif inference_portion == "chase":
            chase_final_state[0,i] = current_replicating_DNA
            chase_final_state[1,i] = current_replicating_single
            chase_final_state[2,i] = current_replicating_double
            chase_final_state[3,i] = current_old_DNA
            chase_final_state[4,i] = current_old_single
            chase_final_state[5,i] = current_old_double

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_1hr = np.array(nucleoid_num_1hr)
    nucleoid_num_3hr = np.array(nucleoid_num_3hr)
    nucleoid_num_7hr = np.array(nucleoid_num_7hr)
    nucleoid_num_24hr = np.array(nucleoid_num_24hr)

    tagged_num_1hr = np.array(tagged_num_1hr)
    tagged_num_3hr = np.array(tagged_num_3hr)
    tagged_num_7hr = np.array(tagged_num_7hr)
    tagged_num_24hr = np.array(tagged_num_24hr)

    mtvolume_1hr = np.array(mtvolume_1hr)
    mtvolume_3hr = np.array(mtvolume_3hr)
    mtvolume_7hr = np.array(mtvolume_7hr)
    mtvolume_24hr = np.array(mtvolume_24hr)

    e1_h = 0.0
    e3_h = 0.0
    e7_h = 0.0
    e24_h = 0.0
    if inference_portion == "pulse":
        average_peak1_proportion = np.mean(np.array(peak1_proportion))
        if not full_trajectory:
            e1_h = hetero_summary_statistic_e1(measurement_error*tagged_num_1hr, mtvolume_1hr, mode = mode)
            e3_h = hetero_summary_statistic_e3(measurement_error*tagged_num_3hr, mtvolume_3hr, mode = mode)
            e7_h = hetero_summary_statistic_e7(measurement_error*tagged_num_7hr, mtvolume_7hr, mode = mode)
            e24_h = hetero_summary_statistic_e24(measurement_error*tagged_num_24hr, mtvolume_24hr, mode = mode)
    elif inference_portion == "chase":
        average_peak1_proportion = 0

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h)

@jit(nopython=True)
def _two_population_chase(mu_b,c, true_birth_param, diffusion_prob, mu_d, birth_rate, death_rate, beta0, beta1, initial_states,
                                         mito_lengths = training_mito_lengths_chase, time_indicator = training_chase_time_indicator, verbose = True, full_trajectory = False):
    
    percent_error = 3.6
    if verbose:
        print("------------------------Beginning Chase Simulation --------------------------")

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): 0. replicating untagged, 1. replicating single tagged, 2. replicating double tagged,
    #                           3. old untagged, 4. old single tagged, 5. old double tagged,
    #                           6. remnant untagged, 7. remnant single tagged, 8. remnant double tagged(***)
    
    # Order of events (n): 0. untagged birth
    #                      1. untagged replication with diffusion
    #                      2. untagged replication without diffusion
    #                      3. single tagged birth
    #                      4. single tagged replication with diffusion
    #                      5. single tagged replication without diffusion (single tagged daughter keeps replicating)
    #                      6. single tagged replication without diffusion (untagged daughter keeps replicating)
    #                      7. double tagged birth
    #                      8. double tagged replication with diffusion 
    #                      9. double tagged replication without diffusion
    #                      
    #                      10. untagged remnant replication with diffusion
    #                      11. untagged remnant replication without diffusion (single tagged daughter keeps replicating),
    #                      12. untagged remnant replication without diffusion (untagged daughter keeps replicating),
    #                      13. single tagged remnant replication with diffusion (two single stranded daughters),
    #                      14. single tagged remnant replication with diffusion (one untagged, one double stranded daughter),
    #                      15. single tagged remnant replication without diffusion (two single stranded daughters),
    #                      16. single tagged remnant replication without diffusion (one untagged, one double stranded daughter, untagged keeps replicating)
    #                      17. single tagged remnant replication without diffusion (one untagged, one double stranded daughter, double strand keeps replicating)
    #                      18. double tagged remnant replication with diffusion
    #                      19. double tagged remnant replication without diffusion (double strand keeps replicating)
    #                      20. double tagged remnant replication without diffusion (single strand keeps replicating)
    #
    #                      21, untagged death, 21, single tagged death, 23, double tagged death

    step_matrix = np.array([[1, -1, 0,  0,  0,  0,  1,  0,  0,  0,          0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,     0,0,0],
                            [0,  0, 0,  1, -1,  0, -1,  0,  0,  1,          0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  1,     0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  1, -1, -1,          0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,     0,0,0],
                            [-1, 2, 1,  0,  1,  1,  0,  0,  0,  0,          1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,     -1,0,0],
                            [0,  0, 0, -1,  1,  0,  1,  0,  2,  1,          1,  0,  1,  2,  0,  1,  0,  0,  1,  1,  0,     0,-1,0],
                            [0,  0, 0,  0,  0,  0,  0, -1,  0,  0,          0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,     0,0,-1],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,         -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,     0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,     0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1,     0,0,0]]).astype(np.float64)
    
    step_matrix = step_matrix.transpose()

    #If full_trajectory, we simulate every cell for 4 days, rather than some for 0, 1, 2.
    if full_trajectory:
        time_indicator = 4*24*np.ones(len(mito_lengths))

    num_sims = len(mito_lengths)

    cell_number_0dy = np.sum(time_indicator == 0)
    cell_number_1dy = np.sum(time_indicator == 24)
    cell_number_2dy = np.sum(time_indicator == 48)
    cell_number_4dy = np.sum(time_indicator == 96)

    nucleoid_num_0dy = [int(0)]*cell_number_0dy
    tagged_num_0dy = [int(0)]*cell_number_0dy
    mtvolume_0dy = [float(0)]*cell_number_0dy

    nucleoid_num_1dy = [int(0)]*cell_number_1dy
    tagged_num_1dy = [int(0)]*cell_number_1dy
    mtvolume_1dy = [float(0)]*cell_number_1dy

    nucleoid_num_2dy = [int(0)]*cell_number_2dy
    tagged_num_2dy = [int(0)]*cell_number_2dy
    mtvolume_2dy = [float(0)]*cell_number_2dy

    nucleoid_num_4dy = [int(0)]*cell_number_4dy
    tagged_num_4dy = [int(0)]*cell_number_4dy
    mtvolume_4dy = [float(0)]*cell_number_4dy

    if full_trajectory:
        initial_peak1_proportion = [float(0)]*cell_number_4dy
    else:
        initial_peak1_proportion = [float(0)]*cell_number_0dy
    final_peak1_proportion = [float(0)]*cell_number_4dy

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 24*4*4 + 1,9)).astype(np.float64)
    
    #Looping over every cell
    for i in prange(num_sims):
        l = mito_lengths[i]
        time_point = time_indicator[i]

        pulse_replicating_DNA, pulse_replicating_single, pulse_replicating_double, pulse_old_DNA, pulse_old_single, pulse_old_double = np.transpose(initial_states)[i]

        nucleoid_state = np.array([0,0,0,pulse_old_DNA,pulse_old_single,pulse_old_double, pulse_replicating_DNA, pulse_replicating_single, pulse_replicating_double]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state
            initial_peak1_proportion[i] = float(pulse_old_single/max(pulse_replicating_DNA + pulse_replicating_single + pulse_replicating_double + pulse_old_single + pulse_old_double, 1))

        #Looping for either 0, 1, 2, or 4 days, depending on the cell
        while current_time <= time_point:

            current_replicating_DNA = nucleoid_state[0]
            current_replicating_single = nucleoid_state[1]
            current_replicating_double = nucleoid_state[2]
            current_old_DNA = nucleoid_state[3]
            current_old_single = nucleoid_state[4]
            current_old_double = nucleoid_state[5]
            current_replicating_DNA_remnant = nucleoid_state[6]
            current_replicating_single_remnant = nucleoid_state[7]
            current_replicating_double_remnant = nucleoid_state[8]

            current_replicating = current_replicating_DNA + current_replicating_single + current_replicating_double
            current_replicating_remnant = current_replicating_DNA_remnant + current_replicating_single_remnant + current_replicating_double_remnant
            current_old = current_old_DNA + current_old_single + current_old_double

            #If the cell is measured at 0 hours we need not simulate it further
            if time_point == 0:
                break

            n = np.sum(nucleoid_state)
            
            if n == 0:
                break

            #Generating the time that the next event takes place
            max_propensity = birth_rate(current_old, n, mu_b, c, l, beta0, beta1) + death_rate(current_old, n, mu_d, c, l, beta0, beta1) + (current_replicating + current_replicating_remnant)*true_birth_param
            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<384:                       
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            if current_time >= time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################
            
            p_birth = birth_rate(current_old,n,mu_b,c,l,beta0, beta1)/max_propensity
            p_death = death_rate(current_old,n,mu_d,c,l,beta0, beta1)/max_propensity
            p_truebirth_dif = diffusion_prob*current_replicating*true_birth_param/max_propensity
            p_truebirth_nodif = (1-diffusion_prob)*current_replicating*true_birth_param/max_propensity
            p_remnant_truebirth_dif = diffusion_prob*current_replicating_remnant*true_birth_param/max_propensity
            p_remnant_truebirth_nodif = (1-diffusion_prob)*current_replicating_remnant*true_birth_param/max_propensity

            #probability that the event happens to an untagged, single, or double tagged molecule, given that the event affects ...

            #... the old population
            p_old_untagged = current_old_DNA/max(current_old,1)
            p_old_single = current_old_single/max(current_old,1)
            p_old_double = current_old_double/max(current_old,1)

            #... the replicating population
            p_replicating_untagged = current_replicating_DNA/max(current_replicating,1)
            p_replicating_single = current_replicating_single/max(current_replicating,1)
            p_replicating_double = current_replicating_double/max(current_replicating,1)

            #... the remnant replicating population
            p_remnant_untagged = current_replicating_DNA_remnant/max(current_replicating_remnant,1)
            p_remnant_single = current_replicating_single_remnant/max(current_replicating_remnant,1)
            p_remnant_double = current_replicating_double_remnant/max(current_replicating_remnant,1)

            probability_vector = np.array([p_birth*p_old_untagged, p_truebirth_dif*p_replicating_untagged, p_truebirth_nodif*p_replicating_untagged,
                                           p_birth*p_old_single, p_truebirth_dif*p_replicating_single, p_truebirth_nodif*p_replicating_single/2, p_truebirth_nodif*p_replicating_single/2,
                                           p_birth*p_old_double, p_truebirth_dif*p_replicating_double, p_truebirth_nodif*p_replicating_double,
                                           p_remnant_truebirth_dif*p_remnant_untagged, p_remnant_truebirth_nodif*p_remnant_untagged/2, p_remnant_truebirth_nodif*p_remnant_untagged/2,
                                           p_remnant_truebirth_dif*p_remnant_single/2, p_remnant_truebirth_dif*p_remnant_single/2,
                                           p_remnant_truebirth_nodif*p_remnant_single/2, p_remnant_truebirth_nodif*p_remnant_single/4,p_remnant_truebirth_nodif*p_remnant_single/4,
                                           p_remnant_truebirth_dif*p_remnant_double, p_remnant_truebirth_nodif*p_remnant_double/2, p_remnant_truebirth_nodif*p_remnant_double/2,
                                           p_death*p_old_untagged, p_death*p_old_single, p_death*p_old_double])

            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)
    
            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()


        #outputting the final cell state
        measurement_error = (1 - np.random.exponential(percent_error)/100)
        tagged = current_replicating_single + current_replicating_double + current_replicating_DNA_remnant + current_replicating_single_remnant + current_replicating_double_remnant + current_old_single + current_old_double
        tagged_output = round(tagged*measurement_error)
        untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
        peak1 = current_old_single + current_replicating_single

        if time_point == 0:
            nucleoid_num_0dy[i] = int(tagged_output + untagged_output)
            tagged_num_0dy[i] = int(tagged_output)
            mtvolume_0dy[i] = float(l)

            initial_peak1_proportion[i] = float(peak1/max(tagged, 1)) 
        elif time_point == 24:
            j = i-cell_number_0dy
            nucleoid_num_1dy[j] = int(tagged_output + untagged_output)
            tagged_num_1dy[j] = int(tagged_output)
            mtvolume_1dy[j] = float(l)
        elif time_point == 48:
            j = i-cell_number_0dy - cell_number_1dy
            nucleoid_num_2dy[j] = int(tagged_output + untagged_output)
            tagged_num_2dy[j] = int(tagged_output)
            mtvolume_2dy[j] = float(l)
        elif time_point == 96:
            j = i-cell_number_0dy - cell_number_1dy - cell_number_2dy
            nucleoid_num_4dy[j] = int(tagged_output + untagged_output)
            tagged_num_4dy[j] = int(tagged_output)
            mtvolume_4dy[j] = float(l)

            final_peak1_proportion[j] = float(peak1/max(tagged, 1)) 

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_0dy = np.array(nucleoid_num_0dy)
    nucleoid_num_1dy = np.array(nucleoid_num_1dy)
    nucleoid_num_2dy = np.array(nucleoid_num_2dy)
    nucleoid_num_4dy = np.array(nucleoid_num_4dy)

    tagged_num_0dy = np.array(tagged_num_0dy)
    tagged_num_1dy = np.array(tagged_num_1dy)
    tagged_num_2dy = np.array(tagged_num_2dy)
    tagged_num_4dy = np.array(tagged_num_4dy)

    mtvolume_0dy = np.array(mtvolume_0dy)
    mtvolume_1dy = np.array(mtvolume_1dy)
    mtvolume_2dy = np.array(mtvolume_2dy)
    mtvolume_4dy = np.array(mtvolume_4dy)

    average_initial_peak1_proportion = np.mean(np.array(initial_peak1_proportion))
    average_final_peak1_proportion = np.mean(np.array(final_peak1_proportion))

    return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
            average_initial_peak1_proportion, average_final_peak1_proportion, trajectory)

@jit(nopython=True)
def _two_population_burn_pulse_chase(diffusion_prob_pulse,mu_d_pulse, c, true_birth_param, mu_b_chase,diffusion_prob_chase, mu_d_chase, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_chase_time_indicator, mito_lengths = training_mito_lengths_chase, sig=0.2, verbose = True, full_trajectory=False, mode = "training"):

    (replicating_output, old_output, variance_statistic, S_h) = _burn_in_two_population_model(diffusion_prob=diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = training_dna_numbers, burn_in_time = burn_in_time, sig=sig, inference_portion = "chase", verbose = verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory_pulse, e1_h, e3_h, e7_h, e24_h) = _two_population_pulse(diffusion_prob = diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, old_output=old_output, beta0=beta0, beta1=beta1, time_indicator=time_indicator, mito_lengths = mito_lengths, inference_portion = "chase", verbose=verbose, full_trajectory = False, mode = mode)
    
    (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, trajectory_chase) = _two_population_chase(mu_b=mu_b_chase,c=0, true_birth_param=true_birth_param, diffusion_prob=diffusion_prob_chase, mu_d=mu_d_chase, birth_rate=birth_rate, death_rate=death_rate, beta0=beta0, beta1=beta1, 
                                         mito_lengths = mito_lengths, time_indicator = time_indicator, initial_states = chase_final_state, full_trajectory=full_trajectory, verbose=verbose)
 
    return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
           initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
           variance_statistic, trajectory_chase)

@jit(nopython=True)
def _two_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, c, true_birth_param, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_time_indicator, mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, verbose = True, full_trajectory = False, mode = "training"):
    """
    Intermediary simulator function for the pulse data. Joins up the burn in function, and the pulse function.
    """

    (replicating_output, old_output, variance_statistic,S_h) = _burn_in_two_population_model(diffusion_prob=diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = burn_in_time, verbose=verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h) = _two_population_pulse(diffusion_prob = diffusion_prob_pulse,c=c,true_birth_param=true_birth_param, mu_d=mu_d_pulse, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, old_output=old_output, beta0=beta0, beta1=beta1, time_indicator = time_indicator, mito_lengths = mito_lengths, verbose=verbose, full_trajectory = full_trajectory, mode = mode)

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h)

@jit(nopython=True)
def _one_iter_burn_in_one_population_model(c,mu, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths, initial_nucleoid):
    

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    #
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    #
    # Order of populations (m): nucleoids
    #
    # Order of events (n): Birth; death
    step_matrix = np.array([[1,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()
    num_sims= len(mito_lengths)

    log_residuals = np.zeros(num_sims).astype(np.float64)
    nucleoid_output = np.zeros(num_sims).astype(np.int64)
    
    #Looping over each cell
    for i in prange(num_sims):
    
        l=mito_lengths[i]

        nucleoid_state = np.array([round(initial_nucleoid[i])]).astype(np.int64)
        current_time =  0

        #Looping until the end of this iteration (usually 1 hour)
        while current_time <= burn_in_increments:
            n = nucleoid_state[0]

            if n == 0:
                break
            
            ##################----------------Generating the time that the next event takes place---------------------######################

            max_propensity = birth_rate(n, n, mu, c, l, beta0, beta1) + death_rate(n, n, mu, c, l, beta0, beta1)
            next_event_time = np.random.exponential(1/max_propensity)
            
            #Updating the time
            current_time += next_event_time
            if current_time > burn_in_increments or (not np.any(nucleoid_state)):
                current_time = burn_in_increments
                break

            ##################-------------------------Generating what kind of event this is---------------------------#####################

            p_birth = birth_rate(n,n,mu,c,l,beta0, beta1)/max_propensity
            p_death = death_rate(n,n,mu,c,l,beta0, beta1)/max_propensity

            probability_vector = np.array([p_birth, p_death])
            r2 = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r2)

            #Updating the nucleoid state based on which event occured
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()


        #Recording each subpopulation after the end of the iteration (usually an hour)
        n = nucleoid_state[0]
        nucleoid_output[i] = n

        #Recording the log residuals of the single cell
        log_residuals[i] = np.log(n) - np.log(beta1*l + beta0)

    #Recording the variance of the log residuals of every cell
    variance_output = np.var(log_residuals)

    #Return each subpopulation for each cell to feed into the next iteration, as well as the log residual variance
    return (nucleoid_output.astype(np.float64), variance_output)

@jit(nopython=True)
def _burn_in_one_population_model(c, mu, birth_rate, death_rate, burn_in_increments = 1, beta0=172, beta1=1.38, 
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, verbose = True, inference_portion = "pulse", mode = "training"):

    if verbose:
        print("------------------------Beginning Burn In --------------------------")

    #Undercounting percentage error
    percent_error = 3.6

    #Initialising the nucleoid number
    if inference_portion == "pulse":
        #pulse data: we take the actual nucleoid number and error correct
        initial_nucleoid_number = dna_nums/(1-percent_error/100)
    else:
        #chase data: we draw from the empirical distribution of the 0 day data, and error correct
        initial_nucleoid_number = np.maximum((beta0 + beta1*mito_lengths + sig*mito_lengths*normal_sample(len(mito_lengths)))/(1-percent_error/100),1)
    
    J = int(burn_in_time//burn_in_increments)

    variances = np.zeros(J).astype(np.float64) 

    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (nucleoid_output, variance_output) = _one_iter_burn_in_one_population_model(c, mu, birth_rate, death_rate, burn_in_increments, beta0, beta1,
                                          mito_lengths = mito_lengths, initial_nucleoid = initial_nucleoid_number)

        variances[j] = variance_output
        initial_nucleoid_number = nucleoid_output.copy()
        if verbose:
            print("Iteration " +str(j) + " Finished")

    measurement_error = (1 - np.random.exponential(percent_error)/100)
    S_h = hetero_summary_statistic(measurement_error*initial_nucleoid_number, mito_lengths, mode = mode) 

    #outputting each subpopulation to feed into the pulse portion of the simulation, as well as S_cs
    return nucleoid_output, np.log(variances[-1]/variances[0]), S_h

@jit(nopython=True)
def _one_population_pulse(c, mu, birth_rate, death_rate, nucleoid_output, beta0=172.431, beta1=1.3809, mito_lengths = training_mito_lengths, time_indicator = training_time_indicator, verbose = True, inference_portion = "pulse", full_trajectory = False, mode = "training"):

    if verbose:
        print("------------------------Beginning Pulse Simulation --------------------------")

    percent_error = 3.6
    
    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): untagged, single tagged, double tagged (***)
    
    # Order of events (n): untagged birth,single tagged birth,double tagged birth,
    #                      untagged death, single tagged death, double tagged death (*****)

    step_matrix = np.array([[-1,0,0,-1,0,0],
                            [2,0,0,0,-1,0],
                            [0,1,1,0,0,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    #If full_trajectory (or its the chase portion), we simulate every cell for 24 hours, rather than some for 1, 3 , 7.
    if full_trajectory or inference_portion == "chase":
        time_indicator = 24*np.ones(len(mito_lengths))

    #If we are fitting to the pulse data, we record time points 1,3,7,24
    if inference_portion == "pulse":
        cell_number_1hr = np.sum(time_indicator == 1)
        cell_number_3hr = np.sum(time_indicator == 3)
        cell_number_7hr = np.sum(time_indicator == 7)
        cell_number_24hr = np.sum(time_indicator == 24)

        cell_number_chase = 0

    #If we are fitting to the chase data, we only record the final state to initialise the chase portion of the experiment
    elif inference_portion == "chase":
        cell_number_1hr = 0
        cell_number_3hr = 0
        cell_number_7hr = 0
        cell_number_24hr = 0

        cell_number_chase = np.sum(time_indicator > -1)

    #Initialising storage arrays for the 4 time points of interest
    nucleoid_num_1hr = [int(0)]*cell_number_1hr
    tagged_num_1hr = [int(0)]*cell_number_1hr
    mtvolume_1hr = [float(0)]*cell_number_1hr

    nucleoid_num_3hr = [int(0)]*cell_number_3hr
    tagged_num_3hr = [int(0)]*cell_number_3hr
    mtvolume_3hr = [float(0)]*cell_number_3hr

    nucleoid_num_7hr = [int(0)]*cell_number_7hr
    tagged_num_7hr = [int(0)]*cell_number_7hr
    mtvolume_7hr = [float(0)]*cell_number_7hr

    nucleoid_num_24hr = [int(0)]*cell_number_24hr
    tagged_num_24hr = [int(0)]*cell_number_24hr
    mtvolume_24hr = [float(0)]*cell_number_24hr

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 97,3)).astype(np.float64)

    #initialising a matrix to contain the sizes of each subpopulation for each cell after 24 hours, to use to initialise
    #the chase portion of the experiment
    chase_final_state = np.zeros((3, cell_number_chase)).astype(np.int64)

    #peak 1 proportion summary statistc
    peak1_proportion = [float(0)]*cell_number_24hr

    num_sims = len(mito_lengths)
    for i in prange(num_sims):
        #print(i)

        l = mito_lengths[i]
        time_point = time_indicator[i]

        #Initialising the nucleoid state based on the output of the burn in.
        #Order of populations is (***)
        nucleoid_state = np.array([round(nucleoid_output[i]),0,0]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state

        #Looping for either 1,3,7, or 24 hours, depending on the cell
        while current_time <= time_point:
            current_DNA = nucleoid_state[0]
            current_single = nucleoid_state[1]
            current_double = nucleoid_state[2]
            n = int(np.sum(nucleoid_state))

            #If the cell has ran out of nucleoids, end the loop
            if n == 0:
                break

            #Generating the time that the next event takes place
            max_propensity = birth_rate(n, n, mu, c, l, beta0,beta1) + death_rate(n, n, mu, c, l, beta0,beta1)
            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<96:  
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            #If the event time is greater than the final time point, then this event doesn't happen, and we break before updating the nucleid state
            if current_time > time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################

            #Probability of a general event
            p_birth = birth_rate(n,n,mu,c,l,beta0,beta1)/max_propensity
            p_death = death_rate(n,n,mu,c,l,beta0,beta1)/max_propensity

            #probability that the event happens to an untagged, single, or double tagged molecule
            p_untagged = current_DNA/max(n,1)
            p_single = current_single/max(n,1)
            p_double = current_double/max(n,1)

            #Probabilities of each event (order given in (*****))
            probability_vector = np.array([p_birth*p_untagged, p_birth*p_single, p_birth*p_double,
                                           p_death*p_untagged, p_death*p_single, p_death*p_double])
            
            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)

            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #Appending the data
        if inference_portion == "pulse":
            measurement_error = (1 - np.random.exponential(percent_error)/100)
            tagged = current_single + current_double
            tagged_output = round(tagged*measurement_error)
            untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
            if time_point == 1:
                nucleoid_num_1hr[i] = int(tagged_output + untagged_output)
                tagged_num_1hr[i] = int(tagged_output)
                mtvolume_1hr[i] = float(l)
            elif time_point == 3:
                j = i-cell_number_1hr
                nucleoid_num_3hr[j] = int(tagged_output + untagged_output)
                tagged_num_3hr[j] = int(tagged_output)
                mtvolume_3hr[j] = float(l)
            elif time_point == 7:
                j = i-cell_number_1hr - cell_number_3hr
                nucleoid_num_7hr[j] = int(tagged_output + untagged_output)
                tagged_num_7hr[j] = int(tagged_output)
                mtvolume_7hr[j] = float(l)
            elif time_point == 24:
                j = i-cell_number_1hr - cell_number_3hr - cell_number_7hr
                nucleoid_num_24hr[j] = int(tagged_output + untagged_output)
                tagged_num_24hr[j] = int(tagged_output)
                mtvolume_24hr[j] = float(l)

                peak1 = current_single
                peak1_proportion[j] = float(peak1/max(tagged,1))

        elif inference_portion == "chase":
            chase_final_state[0,i] = current_DNA
            chase_final_state[1,i] = current_single
            chase_final_state[2,i] = current_double

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_1hr = np.array(nucleoid_num_1hr)
    nucleoid_num_3hr = np.array(nucleoid_num_3hr)
    nucleoid_num_7hr = np.array(nucleoid_num_7hr)
    nucleoid_num_24hr = np.array(nucleoid_num_24hr)

    tagged_num_1hr = np.array(tagged_num_1hr)
    tagged_num_3hr = np.array(tagged_num_3hr)
    tagged_num_7hr = np.array(tagged_num_7hr)
    tagged_num_24hr = np.array(tagged_num_24hr)

    mtvolume_1hr = np.array(mtvolume_1hr)
    mtvolume_3hr = np.array(mtvolume_3hr)
    mtvolume_7hr = np.array(mtvolume_7hr)
    mtvolume_24hr = np.array(mtvolume_24hr)

    e1_h = 0.0
    e3_h = 0.0
    e7_h = 0.0
    e24_h = 0.0
    if inference_portion == "pulse":
        average_peak1_proportion = np.mean(np.array(peak1_proportion))
        if not full_trajectory:
            e1_h = hetero_summary_statistic_e1(measurement_error*tagged_num_1hr, mtvolume_1hr, mode = mode)
            e3_h = hetero_summary_statistic_e3(measurement_error*tagged_num_3hr, mtvolume_3hr, mode = mode)
            e7_h = hetero_summary_statistic_e7(measurement_error*tagged_num_7hr, mtvolume_7hr, mode = mode)
            e24_h = hetero_summary_statistic_e24(measurement_error*tagged_num_24hr, mtvolume_24hr, mode = mode)
    elif inference_portion == "chase":
        average_peak1_proportion = 0
        e1_h = 0.0
        e3_h = 0.0
        e7_h = 0.0
        e24_h = 0.0

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h)

@jit(nopython=True)
def _one_population_chase(mu_b,c, mu_d, birth_rate, death_rate, beta0, beta1, initial_states,
                                         mito_lengths = training_mito_lengths_chase, time_indicator = training_chase_time_indicator, verbose = True, full_trajectory = False):
    
    percent_error = 3.6
    if verbose:
        print("------------------------Beginning Chase Simulation --------------------------")

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    
    # Order of populations (m): untagged, single tagged, double tagged,(***)
    
    # Order of events (n): untagged birth, single tagged birth, double tagged birth,
    #                      untagged death, single tagged death, double tagged death

    step_matrix = np.array([[1,1, 0,-1, 0, 0],
                            [0,0, 2, 0,-1, 0],
                            [0,0,-1, 0, 0,-1]]).astype(np.float64)
    
    step_matrix = step_matrix.transpose()

    #If full_trajectory, we simulate every cell for 4 days, rather than some for 0, 1, 2.
    if full_trajectory:
        time_indicator = 4*24*np.ones(len(mito_lengths))

    num_sims = len(mito_lengths)

    cell_number_0dy = np.sum(time_indicator == 0)
    cell_number_1dy = np.sum(time_indicator == 24)
    cell_number_2dy = np.sum(time_indicator == 48)
    cell_number_4dy = np.sum(time_indicator == 96)

    nucleoid_num_0dy = [int(0)]*cell_number_0dy
    tagged_num_0dy = [int(0)]*cell_number_0dy
    mtvolume_0dy = [float(0)]*cell_number_0dy

    nucleoid_num_1dy = [int(0)]*cell_number_1dy
    tagged_num_1dy = [int(0)]*cell_number_1dy
    mtvolume_1dy = [float(0)]*cell_number_1dy

    nucleoid_num_2dy = [int(0)]*cell_number_2dy
    tagged_num_2dy = [int(0)]*cell_number_2dy
    mtvolume_2dy = [float(0)]*cell_number_2dy

    nucleoid_num_4dy = [int(0)]*cell_number_4dy
    tagged_num_4dy = [int(0)]*cell_number_4dy
    mtvolume_4dy = [float(0)]*cell_number_4dy
    
    if full_trajectory:
        initial_peak1_proportion = [float(0)]*cell_number_4dy
    else:
        initial_peak1_proportion = [float(0)]*cell_number_0dy
    final_peak1_proportion = [float(0)]*cell_number_4dy

    #the full trajectory, if full_trajectory == True
    trajectory = np.zeros((len(mito_lengths), 24*4*4 + 1,3)).astype(np.float64)
    
    #Looping over every cell
    for i in prange(num_sims):
        l = mito_lengths[i]
        time_point = time_indicator[i]

        pulse_DNA, pulse_single, pulse_double = np.transpose(initial_states)[i]

        nucleoid_state = np.array([pulse_DNA, pulse_single, pulse_double]).astype(np.int64)
        current_time =  0

        if full_trajectory:
            trajectory[i][0] = (1 - percent_error/100)*nucleoid_state
            initial_peak1_proportion[i] = pulse_single/max(pulse_single+pulse_double)

        #Looping for either 0, 1, 2, or 4 days, depending on the cell
        while current_time <= time_point:

            current_DNA = nucleoid_state[0]
            current_single = nucleoid_state[1]
            current_double = nucleoid_state[2]

            #If the cell is measured at 0 hours we need not simulate it further
            if time_point == 0:
                break

            n = np.sum(nucleoid_state)
            
            if n == 0:
                break

            #Generating the time that the next event takes place
            max_propensity = birth_rate(n, n, mu_b, c, l, beta0, beta1) + death_rate(n, n, mu_d, c, l, beta0, beta1)
            next_event_time = np.random.exponential(1/max_propensity)

            #every 15 minutes, we record the current nucleoid state
            if full_trajectory:
                a = int(current_time//0.25)
                b = int((current_time+next_event_time)//0.25)

                for k in range(a,b):
                    if k<384:  
                        trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state

            current_time += next_event_time
            if current_time >= time_point:
                break
            
            ##################-------------------------Generating what kind of event this is---------------------------#####################
            
            p_birth = birth_rate(n,n,mu_b,c,l,beta0, beta1)/max_propensity
            p_death = death_rate(n,n,mu_d,c,l,beta0, beta1)/max_propensity

            #probability that the event happens to an untagged, single, or double tagged molecule
            p_untagged = current_DNA/max(n,1)
            p_single = current_single/max(n,1)
            p_double = current_double/max(n,1)

            probability_vector = np.array([p_birth*p_untagged, p_birth*p_single, p_birth*p_double,
                                           p_death*p_untagged, p_death*p_single, p_death*p_double])

            r = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r)
    
            #Updating the nucleoid state based on what event happened
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #outputting the final cell state
        measurement_error = (1 - np.random.exponential(percent_error)/100)
        tagged = current_single + current_double
        tagged_output = round(tagged*measurement_error)
        untagged_output = round((np.sum(nucleoid_state) - tagged)*measurement_error)
        peak1 = current_single

        if time_point == 0:
            nucleoid_num_0dy[i] = int(tagged_output + untagged_output)
            tagged_num_0dy[i] = int(tagged_output)
            mtvolume_0dy[i] = float(l)

            initial_peak1_proportion[i] = float(peak1/max(tagged, 1)) 
        elif time_point == 24:
            j = i-cell_number_0dy
            nucleoid_num_1dy[j] = int(tagged_output + untagged_output)
            tagged_num_1dy[j] = int(tagged_output)
            mtvolume_1dy[j] = float(l)
        elif time_point == 48:
            j = i-cell_number_0dy - cell_number_1dy
            nucleoid_num_2dy[j] = int(tagged_output + untagged_output)
            tagged_num_2dy[j] = int(tagged_output)
            mtvolume_2dy[j] = float(l)
        elif time_point == 96:
            j = i-cell_number_0dy - cell_number_1dy - cell_number_2dy
            nucleoid_num_4dy[j] = int(tagged_output + untagged_output)
            tagged_num_4dy[j] = int(tagged_output)
            mtvolume_4dy[j] = float(l)

            final_peak1_proportion[j] = float(peak1/max(tagged, 1)) 

        if verbose:
            print("Cell " + str(i) + " Finished")

    nucleoid_num_0dy = np.array(nucleoid_num_0dy)
    nucleoid_num_1dy = np.array(nucleoid_num_1dy)
    nucleoid_num_2dy = np.array(nucleoid_num_2dy)
    nucleoid_num_4dy = np.array(nucleoid_num_4dy)

    tagged_num_0dy = np.array(tagged_num_0dy)
    tagged_num_1dy = np.array(tagged_num_1dy)
    tagged_num_2dy = np.array(tagged_num_2dy)
    tagged_num_4dy = np.array(tagged_num_4dy)

    mtvolume_0dy = np.array(mtvolume_0dy)
    mtvolume_1dy = np.array(mtvolume_1dy)
    mtvolume_2dy = np.array(mtvolume_2dy)
    mtvolume_4dy = np.array(mtvolume_4dy)

    average_initial_peak1_proportion = np.mean(np.array(initial_peak1_proportion))
    average_final_peak1_proportion = np.mean(np.array(final_peak1_proportion))

    return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
            average_initial_peak1_proportion, average_final_peak1_proportion, trajectory)

@jit(nopython=True)
def _one_population_burn_pulse_chase(mu_pulse, c, mu_b_chase,mu_d_chase, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_chase_time_indicator, mito_lengths = training_mito_lengths_chase, sig=0.2, verbose = True, full_trajectory=False):

    (nucleoid_output, variance_statistic, S_h) = _burn_in_one_population_model(c=c,mu=mu_pulse, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = training_dna_numbers, burn_in_time = burn_in_time, sig=sig, inference_portion = "chase", verbose = verbose)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory_pulse, e1_h, e3_h, e7_h, e24_h) = _one_population_pulse(c=c,mu=mu_pulse, birth_rate=birth_rate, death_rate=death_rate, nucleoid_output=nucleoid_output, beta0=beta0, beta1=beta1, time_indicator=time_indicator, mito_lengths = mito_lengths, inference_portion = "chase", verbose=verbose, full_trajectory = False)
    
    (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, trajectory_chase) = _one_population_chase(mu_b=mu_b_chase,c=0, mu_d=mu_d_chase, birth_rate=birth_rate, death_rate=death_rate, beta0=beta0, beta1=beta1, 
                                         mito_lengths = mito_lengths, time_indicator = time_indicator, initial_states = chase_final_state, full_trajectory=full_trajectory,verbose=verbose)
 
    return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
           initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
           variance_statistic, trajectory_chase)

@jit(nopython=True)
def _one_population_burn_pulse(mu_pulse, c, birth_rate, death_rate, burn_in_time = 250, beta0=172.4, beta1=1.38, burn_in_increments=1,
                              time_indicator = training_time_indicator, mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, verbose = True, full_trajectory = False, mode = "training"):
    """
    Intermediary simulator function for the pulse data. Joins up the burn in function, and the pulse function.
    """

    (nucleoid_output, variance_statistic, S_h) = _burn_in_one_population_model(c=c, mu=mu_pulse, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, beta0=beta0, beta1=beta1, 
                         mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = burn_in_time, verbose=verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h) = _one_population_pulse(c=c,mu=mu_pulse, birth_rate=birth_rate, death_rate=death_rate, nucleoid_output=nucleoid_output, beta0=beta0, beta1=beta1, time_indicator = time_indicator, mito_lengths = mito_lengths, verbose=verbose, full_trajectory = full_trajectory, mode = mode)

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h)

#############################################################################################################################
#                                                                                                                           #
#                                                       STOCHASTIC SYSTEMS                                                  #
#                                                            MODEL                                                          #
#                                                                                                                           #
#############################################################################################################################

#Dispersed logarithmic control
@jit(nopython=True)
def logarithmic_dispersed_three_population_pulse(params, verbose = False, full_trajectory = False, mode = "training", burn_in=250):
    """
    Simulator function for the purpose of ABC inference on the pulse data, for dispersed logarithmic control. For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)
            - ksig (float, ksig > 0) [standard deviation of kappa, which encodes cell to cell variation (Supplementary Information Section 6.5)]
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        full_trajectory: Bool specifying whether to output the full trajectory every 15 simulation minutes, for posterior predictive plots. Should 
                        be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h

    Returns:
        Tuple (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, full_trajectory, S_h, e1_h, e3_h, e7_h, e24_h), where:

            - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
            - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
            - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
            - average_peak1_proportion (float) is the mean over all 24 hour cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al (Supplementary Information Section 6.2)
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _dispersed_three_population_pulse
            - S_h (float) is the heteroscedasticity summary statistic (Supplementary Information Section 6.5)
            - ei_h (float) is the heteroscedasticity summary statistic for the i'th hour EdU values (Supplementary Information Section 6.5)

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """

    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c, ksig = params

    if mode == "training":
        time_indicator = training_time_indicator
        mito_lengths = training_mito_lengths
        dna_nums = training_dna_numbers
    elif mode == "validation":
        time_indicator = validation_time_indicator
        mito_lengths = validation_mito_lengths
        dna_nums = validation_dna_numbers
    else:
        time_indicator = all_assays_time_indicator
        mito_lengths = all_assays_mito_lengths
        dna_nums = all_assays_dna_numbers

    return _dispersed_three_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, logarithmic_birth, constant_death, burn_in_time = burn_in, beta0=beta0, beta1=beta1, burn_in_increments = 1, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths=mito_lengths, dna_nums=dna_nums, ksig=ksig, mode = mode)

#Stochastic inhibition model
@jit(nopython=True)
def ou_inhibition_three_population_pulse(params, verbose = False, full_trajectory = False, mode = "training", burn_in=250):
    """
    Simulator function for the purpose of ABC inference on the pulse data, for the stochastic inhibition model. For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)
            - theta (float, theta>0) [parameter modulating speed of OU process oscillations (Supplementary Information Section 6.4)]
            - sigma (float, theta>0) [standard deviation of the OU process (Supplementary Information Section 6.4)]
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        full_trajectory: Bool specifying whether to output the full trajectory every 15 simulation minutes, for posterior predictive plots. Should 
                be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h

    Returns:
        Tuple (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h), where:

            - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
            - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
            - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
            - average_peak1_proportion (float) is the mean over all 24 hour cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al (Supplementary Information Section 6.2)
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _ou_three_population_pulse            
            - S_h (float) is the heteroscedasticity summary statistic (Supplementary Information Section 6.5)
            - ei_h (float) is the heteroscedasticity summary statistic for the i'th hour EdU values (Supplementary Information Section 6.5)

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """

    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c, theta, sd = params

    if mode == "training":
        time_indicator = training_time_indicator
        mito_lengths = training_mito_lengths
        dna_nums = training_dna_numbers
    elif mode == "validation":
        time_indicator = validation_time_indicator
        mito_lengths = validation_mito_lengths
        dna_nums = validation_dna_numbers
    else:
        time_indicator = all_assays_time_indicator
        mito_lengths = all_assays_mito_lengths
        dna_nums = all_assays_dna_numbers

    return _ou_three_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, theta, sd, inhibition_birth, constant_death, beta0=beta0, beta1=beta1, burn_in_time = burn_in, burn_in_increments=1, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths=mito_lengths, dna_nums=dna_nums, mode = mode)

#Pulse-chase three population model
@jit(nopython=True)
def logarithmic_three_population_chase(params, verbose = False, full_trajectory = False, mode = "training", burn_in=250, birth_rate = logarithmic_birth):
    """
    Simulator function for the purpose of ABC inference on the pulse-chase data, for the three population model (default logarithmic control). For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - mu_b^chase (float, mu_b^chase <= p^chase * mu_d^chase)
            - p^chase (float, p^chase >= 0)
            - mu_d^chase (float, mu_d^chase >= 0)
            - c (float, c >= 0)
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        full_trajectory: Bool specifying whether to output the full trajectory every 15 simulation minutes, for posterior predictive plots. Should 
                        be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h
        birth_rate: func - default is logarithmic_birth, but can be differential_birth, ratiometric_birth, inhibition_birth. If inhibition_birth, we need c > mu_b

    Returns:
        Tuple (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy, 
           average_initial_peak1_proportion, average_final_peak1_proportion, variance_statistic), where:

            - nucleoid_num_idy (array[int]) is the final array of nucleoid numbers for the i days (following the pulse) cells.
            - tagged_num_idy (array[int]) is the final array of tagged numbers for the i days (following the pulse) cells.
            - mtvolume_idy (array[int]) is the array of mitochondrial volumes associated to the i days (following the pulse) cells.
            - average_initial_peak1_proportion (float) is the mean over all 0dy (following the pulse) cells of the single tagged nucleoid proportion
            - average_final_peak1_proportion (float) is the mean over all 4dy (following the pulse) cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al (Supplementary Information Section 6.2)
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _ou_three_population_pulse 

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """

    if mode == "training":
        time_indicator = training_chase_time_indicator
        mito_lengths = training_mito_lengths_chase
    elif mode == "validation":
        time_indicator = validation_chase_time_indicator
        mito_lengths = validation_mito_lengths_chase
    elif mode == "full":
        time_indicator = all_assays_chase_time_indicator
        mito_lengths = all_assays_mito_lengths_chase

    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = params

    return _three_population_burn_pulse_chase(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, birth_rate, constant_death, burn_in_time = burn_in, beta0 = beta0, beta1 = beta1, burn_in_increments = 1, sig=sig, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths = mito_lengths, mode = mode)

#Pulse three population model
@jit(nopython=True)
def logarithmic_three_population_pulse(params, verbose = False, full_trajectory = False, mode = "training", burn_in=250, birth_rate = logarithmic_birth):
    """
    Simulator function for the purpose of ABC inference on the pulse data, for the three population model (default logarithmic control). For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        full_trajectory: Bool specifying whether to output the full trajectory every 15 simulation minutes, for posterior predictive plots. Should 
        be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h
        birth_rate: func - default is logarithmic_birth, but can be differential_birth, ratiometric_birth, inhibition_birth. If inhibition_birth, we need c > mu_b

    Returns:
        Tuple (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, S_h, e1_h, e3_h, e7_h, e24_h), where:

            - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
            - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
            - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
            - average_peak1_proportion (float) is the mean over all 24 hour cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al (Supplementary Information Section 6.2)
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _ou_three_population_pulse            
            - S_h (float) is the heteroscedasticity summary statistic (Supplementary Information Section 6.5)
            - ei_h (float) is the heteroscedasticity summary statistic for the i'th hour EdU values (Supplementary Information Section 6.5)

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """
    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c = params

    if mode == "training":
        time_indicator = training_time_indicator
        mito_lengths = training_mito_lengths
        dna_nums = training_dna_numbers
    elif mode == "validation":
        time_indicator = validation_time_indicator
        mito_lengths = validation_mito_lengths
        dna_nums = validation_dna_numbers
    else:
        time_indicator = all_assays_time_indicator
        mito_lengths = all_assays_mito_lengths
        dna_nums = all_assays_dna_numbers

    return _three_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, mu_b_pulse, c, true_birth_param, birth_rate, constant_death, burn_in_time = burn_in, beta0=beta0, beta1=beta1, burn_in_increments = 1, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths=mito_lengths, dna_nums=dna_nums, mode = mode)

@jit(nopython=True)
def three_population_birth_rate_burn_in(params, birth_rate, time = 1000, verbose = False):
    """
    Simulator function for the purpose of probing the burn in period specifically. Simulates every cell in the pulse dataset over the
    burn in period for the three population model.

    Args:
        params: List[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
        
        c: (float, c >= 0)
        birth_rate: (func) birth rate function. Either logarithmic_birth, differential_birth, ratiometric_birth, or inhibition_birth.
            If inhibition birth, we must have mu_b > c
        time: (float, time > 0). Burn-in time
        verbose: Bool specifying whether to print progress updates.

    Returns:
        Tuple (replicating_output, young_output, old_output, variance), where:
            - replicating_output (array[int]): number of replicating nucleoids in each simulated cell following the burn in
            - young_output (array[int]): number of young nucleoids in each simulated cell following the burn in
            - old_output (array[int]): number of old nucleoids in each simulated cell following the burn in
            - variance (array[float]): the variance of the log-residuals every one hour.
    """

    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c = params

    mito_lengths = all_assays_mito_lengths
    dna_nums = all_assays_dna_numbers

    replicating_output, young_output, old_output, variances = _testing_burn_in_three_population_model(diffusion_prob_pulse,c, true_birth_param, mu_d_pulse, mu_b_pulse, birth_rate, constant_death, burn_in_increments = 1, beta0=beta0, beta1=beta1, 
                            mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = time, sig=0.2, verbose = verbose, inference_portion = "pulse")
    
    return replicating_output + young_output + old_output, variances

@jit(nopython=True)
def logarithmic_two_population_chase(params, verbose = False, full_trajectory = False, burn_in=250, mode = "training", birth_rate = logarithmic_birth):
    """
    Simulator function for the purpose of ABC inference on the pulse-chase data, for the two population model (default logarithmic control). For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_r (float, mu_r >= 0)
            - mu_b^chase (float, mu_b^chase <= p^chase * mu_d^chase)
            - p^chase (float, p^chase >= 0)
            - mu_d^chase (float, mu_d^chase >= 0)
            - c (float, c >= 0)
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        full_trajectory: Bool specifying whether to output the full trajectory every 15 simulation minutes, for posterior predictive plots. Should 
                        be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h
        birth_rate: func - default is logarithmic_birth, but can be differential_birth, ratiometric_birth, inhibition_birth. If inhibition_birth, we need c > mu_b

    Returns:
        Tuple (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy, 
           average_initial_peak1_proportion, average_final_peak1_proportion, variance_statistic), where:

            - nucleoid_num_idy (array[int]) is the final array of nucleoid numbers for the i days (following the pulse) cells.
            - tagged_num_idy (array[int]) is the final array of tagged numbers for the i days (following the pulse) cells.
            - mtvolume_idy (array[int]) is the array of mitochondrial volumes associated to the i days (following the pulse) cells.
            - average_initial_peak1_proportion (float) is the mean over all 0dy (following the pulse) cells of the single tagged nucleoid proportion
            - average_final_peak1_proportion (float) is the mean over all 4dy (following the pulse) cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al (Supplementary Information Section 6.2)
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _two_population_chase 

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """

    if mode == "training":
        time_indicator = training_chase_time_indicator
        mito_lengths = training_mito_lengths_chase
    elif mode == "validation":
        time_indicator = validation_chase_time_indicator
        mito_lengths = validation_mito_lengths_chase
    elif mode == "full":
        time_indicator = all_assays_chase_time_indicator
        mito_lengths = all_assays_mito_lengths_chase

    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, c = params

    return _two_population_burn_pulse_chase(diffusion_prob_pulse,mu_d_pulse, c, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, birth_rate, constant_death, beta0 = beta0, beta1 = beta1, sig=sig, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths = mito_lengths, mode = mode, burn_in_time = burn_in)

@jit(nopython=True)
def logarithmic_two_population_pulse(params, verbose = False, full_trajectory = False, burn_in = 250, mode = "training", birth_rate = logarithmic_birth):
    """
    Simulator function for the purpose of ABC inference on the pulse data, for the two population model (default logarithmic control). For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        full_trajectory: Bool specifying whether to output the full trajectory every 15 simulation minutes, for posterior predictive plots. Should 
                        be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h
        birth_rate: func - default is logarithmic_birth, but can be differential_birth, ratiometric_birth, inhibition_birth. If inhibition_birth, we need c > mu_b

    Returns:
        Tuple (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, S_h, e1_h, e3_h, e7_h, e24_h), where:

            - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
            - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
            - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
            - average_peak1_proportion (float) is the mean over all 24 hour cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _two_population_pulse
            - S_h (float) is the heteroscedasticity summary statistic (Supplementary Information Section 6.5)
            - ei_h (float) is the heteroscedasticity summary statistic for the i'th hour EdU values (Supplementary Information Section 6.5)

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """

    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, true_birth_param, c = params

    if mode == "training":
        time_indicator = training_time_indicator
        mito_lengths = training_mito_lengths
        dna_nums = training_dna_numbers
    elif mode == "validation":
        time_indicator = validation_time_indicator
        mito_lengths = validation_mito_lengths
        dna_nums = validation_dna_numbers
    else:
        time_indicator = all_assays_time_indicator
        mito_lengths = all_assays_mito_lengths
        dna_nums = all_assays_dna_numbers

    return _two_population_burn_pulse(diffusion_prob_pulse,mu_d_pulse, c, true_birth_param, birth_rate, constant_death, beta0=beta0, beta1=beta1, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths=mito_lengths, dna_nums=dna_nums, mode = mode, burn_in_time = burn_in)

@jit(nopython=True)
def logarithmic_one_population_chase(params, verbose = False, full_trajectory = False, burn_in = 250, mode = "training", birth_rate = logarithmic_birth):
    """
    Simulator function for the purpose of ABC inference on the pulse-chase data, for the one population model (default logarithmic control). For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - mu (float, mu_d >= 0)
            - mu_b^chase (float, mu_b^chase <= mu_d^chase)
            - mu_d^chase (float, mu_d^chase >= 0)
            - c (float, c >= 0)
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h
        birth_rate: func - default is logarithmic_birth, but can be differential_birth, ratiometric_birth, inhibition_birth. If inhibition_birth, we need c > mu_b

    Returns:
        Tuple (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
           initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
           variance_statistic), where:

            - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
            - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
            - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
            - initial_average_peak1_proportion (float) is the mean over all 0 day cells of the single tagged nucleoid proportion
            - final_average_peak1_proportion (float) is the mean over all 4 day cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al (Supplementary Information Section 6.2)
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _one_population_chase

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """

    if mode == "training":
        time_indicator = training_chase_time_indicator
        mito_lengths = training_mito_lengths_chase
    elif mode == "validation":
        time_indicator = validation_chase_time_indicator
        mito_lengths = validation_mito_lengths_chase
    elif mode == "full":
        time_indicator = all_assays_chase_time_indicator
        mito_lengths = all_assays_mito_lengths_chase

    beta0, beta1, sig, mu_pulse, mu_b_chase, mu_d_chase, c = params

    return _one_population_burn_pulse_chase(mu_pulse, c, mu_b_chase, mu_d_chase, birth_rate, constant_death, beta0 = beta0, beta1 = beta1, sig=sig, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths = mito_lengths, mode = mode, burn_in_time = burn_in)

@jit(nopython=True)
def logarithmic_one_population_pulse(params, verbose = False, full_trajectory = False, burn_in = 250, mode = "training", birth_rate = logarithmic_birth):
    """
    Simulator function for the purpose of ABC inference on the pulse data, for the one population model (default logarithmic control). For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - mu (float, mu_d >= 0)
            - c (float, c >= 0)
        verbose: Bool specifying whether to print progress updates. Should be set to False if doing ABC.
        mode: str which is either "training", "validation or "full". If "training", the analogous cells of assays 1 and 2
              are simulated. If "validation", only assay 3 is simulated. If "full", all assays are simulated.
        burn_in: (int>0) specifying the burn in time t_b. Default is 250h
        birth_rate: func - default is logarithmic_birth, but can be differential_birth, ratiometric_birth, inhibition_birth. If inhibition_birth, we need c > mu_b

    Returns:
        Tuple (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, S_h, e1_h, e3_h, e7_h, e24_h), where:

            - nucleoid_num_ihr (array[int]) is the final array of nucleoid numbers for the i hour cells.
            - tagged_num_ihr (array[int]) is the final array of tagged numbers for the i hour cells.
            - mtvolume_ihr (array[int]) is the array of mitochondrial volumes associated to the i hour cells.
            - average_peak1_proportion (float) is the mean over all 24 hour cells of the single tagged nucleoid proportion
            - variance statistic (float) is as described in Wolf, Mjeku et al
            - trajectory (3d array[int]), where full_trajectory[i,j,k] is the number of molecules in the k'th subpopulation, at 15*j hours, for the i'th cell.
              For subpopulation orderings, check the comments in _one_population_pulse
            - S_h (float) is the heteroscedasticity summary statistic (Supplementary Information Section 6.5)
            - ei_h (float) is the heteroscedasticity summary statistic for the i'th hour EdU values (Supplementary Information Section 6.5)

        Note: if full_trajectory = True, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be 0 arrays, while trajectory will be populated.
        If full_trajectory = False, nucleoid_num_ihr, tagged_num_ihr, mtvolume_ihr will be populated, while trajectory will be a 0 array
    """

    beta0, beta1, mu_pulse, c = params

    if mode == "training":
        time_indicator = training_time_indicator
        mito_lengths = training_mito_lengths
        dna_nums = training_dna_numbers
    elif mode == "validation":
        time_indicator = validation_time_indicator
        mito_lengths = validation_mito_lengths
        dna_nums = validation_dna_numbers
    else:
        time_indicator = all_assays_time_indicator
        mito_lengths = all_assays_mito_lengths
        dna_nums = all_assays_dna_numbers

    return _one_population_burn_pulse(mu_pulse, c, birth_rate, constant_death, beta0=beta0, beta1=beta1, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths=mito_lengths, dna_nums=dna_nums, mode = mode, burn_in_time = burn_in)

#############################################################################################################################
#                                                                                                                           #
#                                                              SUMMARY                                                      #
#                                                             STATISTICS                                                    #
#                                                                                                                           #
#############################################################################################################################
@jit(nopython=True)
def pulse_summary_statistics(data):

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion,variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h) = data
    
    return  [np.mean(tagged_num_1hr),np.mean(tagged_num_3hr),np.mean(tagged_num_7hr),np.mean(tagged_num_24hr),
            average_peak1_proportion, variance_statistic]

@jit(nopython=True)
def pulse_heteroscedastic_summary_statistics(data):

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion,variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h) = data
    
    return  [np.mean(tagged_num_1hr),np.mean(tagged_num_3hr),np.mean(tagged_num_7hr),np.mean(tagged_num_24hr),
            average_peak1_proportion, variance_statistic, S_h]

@jit(nopython=True)
def pulse_edu_heteroscedastic_summary_statistics(data):

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion,variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h) = data
    
    return  [np.mean(tagged_num_1hr),np.mean(tagged_num_3hr),np.mean(tagged_num_7hr),np.mean(tagged_num_24hr),
            average_peak1_proportion, variance_statistic, S_h, e1_h, e3_h, e7_h, e24_h]

@jit(nopython=True)
def chase_summary_statistics(data):

    (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
           initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
           variance_statistic, trajectory) = data

    return  [np.mean(tagged_num_0dy),np.mean(tagged_num_1dy),np.mean(tagged_num_2dy),np.mean(tagged_num_4dy),
            np.mean(nucleoid_num_0dy),np.mean(nucleoid_num_1dy),np.mean(nucleoid_num_2dy),np.mean(nucleoid_num_4dy),
            np.mean(np.divide(tagged_num_0dy, np.maximum(nucleoid_num_0dy,1))),
            np.mean(np.divide(tagged_num_1dy, np.maximum(nucleoid_num_1dy,1))), 
            np.mean(np.divide(tagged_num_2dy, np.maximum(nucleoid_num_2dy,1))), 
            np.mean(np.divide(tagged_num_4dy, np.maximum(nucleoid_num_4dy,1))), 
            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, variance_statistic]
