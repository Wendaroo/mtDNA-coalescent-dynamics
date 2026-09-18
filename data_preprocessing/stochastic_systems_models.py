import numpy as np
from numba import jit, prange
import numba
from .data_preprocessing import (training_mito_lengths, training_time_indicator, training_dna_numbers, training_mito_lengths_chase, training_chase_time_indicator,
                                 validation_mito_lengths, validation_time_indicator, validation_dna_numbers, validation_mito_lengths_chase, validation_chase_time_indicator,
                                 all_assays_mito_lengths, all_assays_time_indicator, all_assays_dna_numbers, all_assays_mito_lengths_chase, all_assays_chase_time_indicator,
                                 all_assays_edu_number_1hr, all_assays_edu_number_3hr, all_assays_edu_number_7hr, all_assays_edu_number_24hr,
                                 validation_edu_number_1hr, validation_edu_number_3hr, validation_edu_number_7hr, validation_edu_number_24hr,
                                 training_edu_number_1hr, training_edu_number_3hr, training_edu_number_7hr, training_edu_number_24hr)
import ete3
from math import comb

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
def logarithmic_birth(n_y,n_r,  mu,c,l, f_y, f_r, beta0=172, beta1=1.38, n = 0):
    return n_y*np.maximum(0, mu + c*(np.log(beta1*l/beta0 + 1)/np.log(np.maximum(n,beta0+0.0001)/beta0)-1))

@jit(nopython=True)
def constant_death(e, n, mu, c, l, beta0 = 172, beta1 = 1.38):
    return mu*e

# @jit(nopython = True)
# def logarithmic_birth(e, n, mu, c, l, beta0 = 172, beta1 = 1.38, ou_addition=0.0):
#     #print(beta1*l/beta0 + 1)
#     return e*np.maximum(0, ou_addition + mu + c*(np.log(beta1*l/beta0 + 1)/np.log(np.maximum(n,beta0+0.0001)/beta0)-1))


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
def _extended_one_iter_burn_in_three_population_model(params, birth_rate, death_rate, burn_in_increments, mito_lengths, initial_replicating, initial_young, initial_old):
    """
    Helper function. Simulates the burn-in period for 'burn-in-increments' hours, and records the the variance of the log-residuals.
    """

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params
    #Defining the birth rate to maintain equilibrium
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    
    f_r = mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    f_y = 1/initialisation_denom

    #STEP MATRIX: dimensions (m,n), where m = number of populations, n = number of possible events
    #
    # If event i is chosen to occur, the ith column of the step matrix will be added to the nucleoid state vector
    #
    # Order of populations (m): replicating population; young population; old population
    #
    # Order of events (n): replication without diffusion; replication with diffusion; Birth; ageing; rep death; young death; old death; rejuvination
    step_matrix = np.array([[-1,0,1,0,-1,0,0,0],
                           [2,1,-1,-1,0,-1,0,1],
                           [0,0,0,1,0,0,-1,-1]]).astype(np.float64)
    
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
            #n_y,n_r,  mu,c,l, f_y, f_yr, beta0=172, beta1=1.38, n = 0
            max_propensity = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n) + death_rate(current_old, n, mu_d_o, c, l, beta0, beta1) + \
                death_rate(current_young, n, mu_d_y, c, l, beta0, beta1) + death_rate(current_replicating, n, mu_d_r, c, l, beta0, beta1) + \
                    current_replicating*mu_r + mu_a*current_young + mu_rej*current_old
            next_event_time = np.random.exponential(1/max_propensity)
            
            #Updating the time
            current_time += next_event_time
            if current_time > burn_in_increments or (not np.any(nucleoid_state)):
                current_time = burn_in_increments
                break

            ##################-------------------------Generating what kind of event this is---------------------------#####################

            p_birth = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n)/max_propensity
            p_rep_death = death_rate(current_replicating,n,mu_d_r,c,l,beta0, beta1)/max_propensity
            p_young_death = death_rate(current_young,n,mu_d_y,c,l,beta0, beta1)/max_propensity
            p_old_death = death_rate(current_old,n,mu_d_o,c,l,beta0, beta1)/max_propensity
            p_double_truebirth = p*current_replicating*mu_r/max_propensity
            p_single_truebirth = (1-p)*current_replicating*mu_r/max_propensity
            p_ageing = mu_a*current_young/max_propensity
            p_rej = mu_rej*current_old/max_propensity

            probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej])
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
def _extended_burn_in_three_population_model(params, birth_rate, death_rate, burn_in_increments = 1,
                         mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, burn_in_time = 250, sig=0.2, verbose = True, inference_portion = "pulse", mode = "training"):
    """
    Helper function. Simulates the burn-in period and records the control strength summary statistic S_cs.
    """

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params
    #Defining the birth rate to maintain equilibrium
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)


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
    
    J = int(burn_in_time//burn_in_increments)

    #initialising each subpopulation via the equilibrium proportions (under a deterministic treatment)
    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)

    initial_replicating = initial_nucleoid_number * mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    initial_young = initial_nucleoid_number/initialisation_denom
    initial_old = initial_nucleoid_number * mu_a/((mu_d_o + mu_rej)*initialisation_denom)

    variances = np.zeros(J).astype(np.float64) 

    #burning in for 250 hours, and recording the variance of the log residuals every hour to construct summary statistic S_cs
    for j in range(J):
        #print(j)
        (replicating_output, young_output, old_output, variance_output) = _extended_one_iter_burn_in_three_population_model(params, birth_rate, death_rate, burn_in_increments,
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
def _extended_three_population_pulse(params, birth_rate, death_rate, replicating_output, young_output, old_output, mito_lengths = training_mito_lengths, time_indicator = training_time_indicator, verbose = True, inference_portion = "pulse", full_trajectory = False, mode = "training"):
    """
    Helper function. Takes as input the output of the burn-in period and simulates the pulse portion of the experiment.
    """
    if verbose:
        print("------------------------Beginning Pulse Simulation --------------------------")

    percent_error = 3.6
    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params

    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    
    f_r = mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    f_y = 1/initialisation_denom
    
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
    #                      untagged rep death, single tagged rep death, double tagged rep death,
    #                      untagged young death, single tagged young death, double tagged young death,
    #                      untagged old death, single tagged old death, double tagged old death,
    #                      untagged rejuv, single tagged rejuv, double tagged rejuv (*****)

    step_matrix = np.array([[1,-1,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,1,-1,0,-1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,1,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],
                            [-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,1,0,0],
                            [0,2,1,-1,1,0,1,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,1,0],
                            [0,0,0,0,1,1,0,-1,2,1,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,1],
                            [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-1,0,0,-1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-1,0,0,-1,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,-1,0,0,-1]]).astype(np.float64)
    
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
            max_propensity = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n) + death_rate(current_old, n, mu_d_o, c, l, beta0, beta1) + \
                death_rate(current_young, n, mu_d_y, c, l, beta0, beta1) + death_rate(current_replicating, n, mu_d_r, c, l, beta0, beta1) + \
                    current_replicating*mu_r + mu_a*current_young + mu_rej*current_old
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
            p_birth = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n)/max_propensity
            p_rep_death = death_rate(current_replicating,n,mu_d_r,c,l,beta0, beta1)/max_propensity
            p_young_death = death_rate(current_young,n,mu_d_y,c,l,beta0, beta1)/max_propensity
            p_old_death = death_rate(current_old,n,mu_d_o,c,l,beta0, beta1)/max_propensity
            p_double_truebirth = p*current_replicating*mu_r/max_propensity
            p_single_truebirth = (1-p)*current_replicating*mu_r/max_propensity
            p_ageing = mu_a*current_young/max_propensity
            p_rej = mu_rej*current_old/max_propensity

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
            probability_vector = np.array([p_birth*p_young_untagged, p_double_truebirth*p_replicating_untagged, p_single_truebirth*p_replicating_untagged,
                                           p_birth*p_young_single, p_double_truebirth*p_replicating_single, p_single_truebirth*p_replicating_single/2,
                                           p_single_truebirth*p_replicating_single/2, p_birth*p_young_double, p_double_truebirth*p_replicating_double,
                                           p_single_truebirth*p_replicating_double, p_ageing*p_young_untagged, p_ageing*p_young_single, 
                                           p_ageing*p_young_double, p_rep_death*p_replicating_untagged, p_rep_death*p_replicating_single, p_rep_death*p_replicating_double,
                                           p_young_death*p_young_untagged, p_young_death*p_young_single, p_young_death*p_young_double,
                                           p_old_death*p_old_untagged, p_old_death*p_old_single, p_old_death*p_old_double,
                                           p_rej*p_old_untagged, p_rej*p_old_single, p_rej*p_old_double])
            
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
def _extended_three_population_chase(params, birth_rate, death_rate, initial_states,
                                         mito_lengths = training_mito_lengths_chase, time_indicator = training_chase_time_indicator, verbose = True, full_trajectory = False):
    """
    Helper function. Takes as input the output of the pulse portion and simulates the chase portion of the experiment.
    """

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c, mu_b, sig = params
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
    #                      24, untagged rep death, 25, single tagged rep death, 26, double tagged rep death,
    #                      27, untagged young death, 28, single tagged young death, 29, double tagged young death,
    #                      30, untagged remnant death, 31, single tagged remnant death, 32, double tagged remnant death,
    #                      33, untagged old death, 34, single tagged old death, 35, double tagged old death,
    #                      36. untagged rej, 37. single tagged rej, 38. double tagged rej

    step_matrix = np.array([[1, -1, 0,  0,  0,  0,  1,  0,  0,  0,          0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,     0,0,0,  -1,0,0,0,0,0,0,0,0,0,0,0,  0,0,0],
                            [0,  0, 0,  1, -1,  0, -1,  0,  0,  1,          0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  1,     0,0,0,  0,-1,0,0,0,0,0,0,0,0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  1, -1, -1,          0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,     0,0,0,  0,0,-1,0,0,0,0,0,0,0,0,0,  0,0,0],
                            [-1, 2, 1,  0,  1,  1,  0,  0,  0,  0,          1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,     -1,0,0, 0,0,0,-1,0,0,0,0,0,0,0,0,  1,0,0],
                            [0,  0, 0, -1,  1,  0,  1,  0,  2,  1,          1,  0,  1,  2,  0,  1,  0,  0,  1,  1,  0,     0,-1,0, 0,0,0,0,-1,0,0,0,0,0,0,0,  0,1,0],
                            [0,  0, 0,  0,  0,  0,  0, -1,  0,  0,          0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,     0,0,-1, 0,0,0,0,0,-1,0,0,0,0,0,0,  0,0,1],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,         -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,     0,0,0,  0,0,0,0,0,0,-1,0,0,0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,     0,0,0,  0,0,0,0,0,0,0,-1,0,0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1,     0,0,0,  0,0,0,0,0,0,0,0,-1,0,0,0,  0,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     1,0,0,  0,0,0,0,0,0,0,0,0,-1,0,0,  -1,0,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     0,1,0,  0,0,0,0,0,0,0,0,0,0,-1,0,  0,-1,0],
                            [0,  0, 0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,     0,0,1,  0,0,0,0,0,0,0,0,0,0,0,-1,  0,0,-1]]).astype(np.float64)
    
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
            max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0, beta1) + death_rate(current_old, n, mu_d_o, c, l, beta0, beta1) + \
                death_rate(current_young, n, mu_d_y, c, l, beta0, beta1) + death_rate(current_replicating, n, mu_d_r, c, l, beta0, beta1) + \
                    death_rate(current_replicating_remnant, n, mu_d_r, c, l, beta0, beta1) + (current_replicating + current_replicating_remnant)*mu_r + mu_a*current_young + mu_rej*current_old
            
            #In the case where mu_b=mu_a=0, there is a chance all nucleoids get stuck in the young population, and no further events occur
            if max_propensity == 0:
                if full_trajectory:
                    a = int(current_time//0.25)
                    b = 384

                    for k in range(a,b):
                        if k<384:  
                            trajectory[i][k+1] = (1 - percent_error/100)*nucleoid_state
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
            p_rep_death = death_rate(current_replicating,n,mu_d_r,c,l,beta0, beta1)/max_propensity
            p_young_death = death_rate(current_young,n,mu_d_y,c,l,beta0, beta1)/max_propensity
            p_remnant_death = death_rate(current_replicating_remnant,n,mu_d_r,c,l,beta0, beta1)/max_propensity
            p_old_death = death_rate(current_old,n,mu_d_o,c,l,beta0, beta1)/max_propensity
            p_double_truebirth = p*current_replicating*mu_r/max_propensity
            p_single_truebirth = (1-p)*current_replicating*mu_r/max_propensity
            p_remnant_double_truebirth = p*current_replicating_remnant*mu_r/max_propensity
            p_remnant_single_truebirth = (1-p)*current_replicating_remnant*mu_r/max_propensity
            p_ageing = mu_a*current_young/max_propensity
            p_rej = mu_rej*current_old/max_propensity

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

            probability_vector = np.array([p_birth*p_young_untagged, p_double_truebirth*p_replicating_untagged, p_single_truebirth*p_replicating_untagged,
                                           p_birth*p_young_single, p_double_truebirth*p_replicating_single, p_single_truebirth*p_replicating_single/2, p_single_truebirth*p_replicating_single/2,
                                           p_birth*p_young_double, p_double_truebirth*p_replicating_double, p_single_truebirth*p_replicating_double,
                                           p_remnant_double_truebirth*p_remnant_untagged, p_remnant_single_truebirth*p_remnant_untagged/2, p_remnant_single_truebirth*p_remnant_untagged/2,
                                           p_remnant_double_truebirth*p_remnant_single/2, p_remnant_double_truebirth*p_remnant_single/2,
                                           p_remnant_single_truebirth*p_remnant_single/2, p_remnant_single_truebirth*p_remnant_single/4,p_remnant_single_truebirth*p_remnant_single/4,
                                           p_remnant_double_truebirth*p_remnant_double, p_remnant_single_truebirth*p_remnant_double/2, p_remnant_single_truebirth*p_remnant_double/2,
                                           p_ageing*p_young_untagged, p_ageing*p_young_single, p_ageing*p_young_double,
                                           p_rep_death*p_replicating_untagged, p_rep_death*p_replicating_single, p_rep_death*p_replicating_double,
                                           p_young_death*p_young_untagged, p_young_death*p_young_single, p_young_death*p_young_double,
                                           p_remnant_death*p_remnant_untagged, p_remnant_death*p_remnant_single, p_remnant_death*p_remnant_double,
                                           p_old_death*p_old_untagged, p_old_death*p_old_single, p_old_death*p_old_double,                                                                              
                                           p_rej*p_old_untagged, p_rej*p_old_single, p_rej*p_old_double])

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
def _extended_three_population_burn_pulse_chase(pulse_params, chase_params, birth_rate, death_rate, burn_in_time = 250, burn_in_increments=1,
                              time_indicator = training_chase_time_indicator, mito_lengths = training_mito_lengths_chase, sig=0.2, verbose = True, full_trajectory=False, mode = "training"):

    """
    Helper function. Stitches together the burn-in phase, the pulse phase, and the chase phase into one simulator.
    
    """
    (replicating_output, young_output, old_output, variance_statistic, S_h) = _extended_burn_in_three_population_model(params=pulse_params, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, 
                         mito_lengths = mito_lengths, dna_nums = training_dna_numbers, burn_in_time = burn_in_time, sig=sig, inference_portion = "chase", verbose = verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory_pulse, e1_h, e3_h, e7_h, e24_h) = _extended_three_population_pulse(params = pulse_params, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, young_output=young_output, old_output=old_output, time_indicator=time_indicator, mito_lengths = mito_lengths, inference_portion = "chase", verbose=verbose, full_trajectory = False, mode = mode)
    
    (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy, 
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
            initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, trajectory_chase) = _extended_three_population_chase(params = chase_params, birth_rate=birth_rate, death_rate=death_rate, 
                                         mito_lengths = mito_lengths, time_indicator = time_indicator, initial_states = chase_final_state, full_trajectory=full_trajectory, verbose=verbose)
 
    return (nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy,
           initial_average_peak1_proportion_chase, final_average_peak1_proportion_chase, 
           variance_statistic, trajectory_chase)

@jit(nopython=True)
def _extended_three_population_burn_pulse(params, birth_rate, death_rate, burn_in_time = 250, burn_in_increments=1,
                              time_indicator = training_time_indicator, mito_lengths = training_mito_lengths, dna_nums = training_dna_numbers, verbose = True, full_trajectory = False, mode = "training"):
    """
    Helper function. Stitches together the burn-in phase and the pulse phase into one simulator.
    """

    (replicating_output, young_output, old_output, variance_statistic, S_h) = _extended_burn_in_three_population_model(params = params, birth_rate=birth_rate, death_rate=death_rate, burn_in_increments = burn_in_increments, 
                         mito_lengths = mito_lengths, dna_nums = dna_nums, burn_in_time = burn_in_time, verbose=verbose, mode = mode)

    (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr,
            average_peak1_proportion, chase_final_state, trajectory, e1_h, e3_h, e7_h, e24_h) = _extended_three_population_pulse(params = params, birth_rate=birth_rate, death_rate=death_rate, replicating_output=replicating_output, young_output=young_output, old_output=old_output, time_indicator = time_indicator, mito_lengths = mito_lengths, verbose=verbose, full_trajectory = full_trajectory, mode = mode)

    return (nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr, 
           average_peak1_proportion, variance_statistic, trajectory, S_h, e1_h, e3_h, e7_h, e24_h)


#Pulse-chase three population model
@jit(nopython=True)
def extended_logarithmic_three_population_chase(pulse_params, chase_params, verbose = False, full_trajectory = False, mode = "training", burn_in=250, birth_rate = logarithmic_birth):
    """
    Simulator function for the purpose of ABC inference on the pulse-chase data, for the three population model (default logarithmic control). For every cell in selected assays,
    this function simulates an analagous cell exposed to EdU for the same amount of time the measured cell was.

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - sigma (float, sigma > 0 (as defined in Supplementary Information Section 3.1.2, used in 3.3.1))
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - mu_rej (float, mu_rej >=0)
            - mu_b^chase (float, mu_b^chase <= p^chase * mu_a^chase)
            - p^chase (float, p^chase >= 0)
            - mu_d^chase (float, mu_d^chase >= 0)
            - mu_a^chase (float, mu_a^chase >= 0)
            - mu_rej^chase (float, mu_rej^chase >= 0)
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
    _, _, _, _, _, _, _, _, _, _, _, sig = chase_params

    if mode == "training":
        time_indicator = training_chase_time_indicator
        mito_lengths = training_mito_lengths_chase
    elif mode == "validation":
        time_indicator = validation_chase_time_indicator
        mito_lengths = validation_mito_lengths_chase
    elif mode == "full":
        time_indicator = all_assays_chase_time_indicator
        mito_lengths = all_assays_mito_lengths_chase

    return _extended_three_population_burn_pulse_chase(pulse_params, chase_params, birth_rate, constant_death, burn_in_time = burn_in, burn_in_increments = 1, sig=sig, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths = mito_lengths, mode = mode)

#Pulse three population model
@jit(nopython=True)
def extended_logarithmic_three_population_pulse(params, verbose = False, full_trajectory = False, mode = "training", burn_in=250, birth_rate = logarithmic_birth):
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
            - mu_rej (float, mu_r >= 0)
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

    return _extended_three_population_burn_pulse(params, birth_rate, constant_death, burn_in_time = burn_in, burn_in_increments = 1, verbose = verbose, full_trajectory=full_trajectory, time_indicator=time_indicator, mito_lengths=mito_lengths, dna_nums=dna_nums, mode = mode)#############################################################################################################################


#############################################################################################################################
#                                                                                                                           #
#                                                              SSD                                                          #
#                                                             FUNCTIONS                                                     #
#                                                                                                                           #
#############################################################################################################################

@jit(nopython = True)
def preferential_death(e, n, mu, c, l, beta0 = 172, beta1 = 1.38, pref_deg_fac = 1.02):
    return mu*(e-n) + pref_deg_fac*mu*n

@jit(nopython=True)
def SSD_birth(n_y,m_y,n_r, m_r, mu,c,l, f_yr, delta,beta0=172, beta1=1.38, n = 0, m = 0):
    n_opt = (beta0 + beta1*l)*f_yr
    return (n_y+m_y)*np.maximum(0, mu + c*(n_opt - (n_y+n_r) - delta*(m_y + m_r)))

#@jit(nopython = True)
def extended_three_population_SSD_fixation_prob(params, num_iter = 100, l = 500, N=0, delta = 1, birth_rate = SSD_birth, death_rate = preferential_death, pref_deg_factor = 1):
    """

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)):
    """

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params
    #Defining the birth rate to maintain equilibrium
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    if N !=0:
        n_init = N
        l = (N-beta0)/beta1

    else:
        n_init = int(beta0 + beta1*l)
    
    step_matrix = np.array([[-1,0,1,0,-1,0,0,0,   0,0,0,0,0,0,0,0],
                           [2,1,-1,-1,0,-1,0,1,   0,0,0,0,0,0,0,0],
                           [0,0,0,1,0,0,-1,-1,    0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,   -1,0,1,0,-1,0,0,0],
                           [0,0,0,0,0,0,0,0,   2,1,-1,-1,0,-1,0,1],
                           [0,0,0,0,0,0,0,0,   0,0,0,1,0,0,-1,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    initial_replicating = n_init * mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    initial_young = n_init/initialisation_denom
    initial_old = n_init * mu_a/((mu_d_o + mu_rej)*initialisation_denom)
    n_init = initial_replicating + initial_young + initial_old
    f_yr = 1-mu_a/((mu_d_o + mu_rej)*initialisation_denom)

    fixed_arr = np.zeros(num_iter)
    final_states = np.zeros((num_iter, 3))
    for i in range(num_iter):
        print(i)
        #initialise with one mutated molecule in the young population
        nucleoid_state = np.array([round(initial_replicating), round(initial_young)-1, round(initial_old), 0, 1, 0]).astype(np.int64)

        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]
        current_mutated_replicating = nucleoid_state[3]
        current_mutated_young = nucleoid_state[4]
        current_mutated_old = nucleoid_state[5]
        current_unmutated = current_replicating + current_young + current_old
        current_mutated= current_mutated_replicating + current_mutated_young + current_mutated_old


        #print(current_mutated_replicating)
        #print(current_mutated_young)
        #print(current_mutated_old)


        current_time =  0
        fixed = False
        eliminated = False
        while (not fixed) and (not eliminated):

            n = int(np.sum(nucleoid_state))
            #print(n)
            #print(current_young+current_mutated_young)
            if n == 0:
                print("here")
                break
            
            ##################----------------Generating the time that the next event takes place---------------------######################

            max_propensity = birth_rate(current_young, current_mutated_young, current_replicating, current_mutated_replicating, mu_b, c, l, f_yr, delta, beta0, beta1, n = current_unmutated,m = current_mutated) + death_rate(current_old+current_mutated_old, current_mutated_old, mu_d_o, c, l, beta0, beta1, pref_deg_factor) + \
                death_rate(current_young + current_mutated_young, current_mutated_young, mu_d_y, c, l, beta0, beta1, pref_deg_factor) + death_rate(current_replicating + current_mutated_replicating, current_mutated_replicating, mu_d_r, c, l, beta0, beta1, pref_deg_factor) + \
                    (current_replicating + current_mutated_replicating)*mu_r + mu_a*(current_young + current_mutated_young) + mu_rej*(current_old + current_mutated_old)
            next_event_time = np.random.exponential(1/max_propensity)

            # if current_time //(0.25*24*365) < (current_time + next_event_time)//(0.25*24*365):
            #     print(current_mutated/(current_mutated+current_unmutated))
                
            current_time += next_event_time

            ##################-------------------------Generating what kind of event this is---------------------------#####################

            p_birth = birth_rate(current_young, current_mutated_young, current_replicating, current_mutated_replicating, mu_b, c, l, f_yr, delta, beta0, beta1, n = current_unmutated,m = current_mutated)/max_propensity
            p_rep_death = death_rate(current_replicating+current_mutated_replicating,current_mutated_replicating,mu_d_r,c,l,beta0, beta1, pref_deg_factor)/max_propensity
            p_young_death = death_rate(current_young+current_mutated_young,current_mutated_young,mu_d_y,c,l,beta0, beta1, pref_deg_factor)/max_propensity
            p_old_death = death_rate(current_old+current_mutated_old,current_mutated_old,mu_d_o,c,l,beta0, beta1, pref_deg_factor)/max_propensity
            p_double_truebirth = p*(current_replicating+current_mutated_replicating)*mu_r/max_propensity
            p_single_truebirth = (1-p)*(current_replicating+current_mutated_replicating)*mu_r/max_propensity
            p_ageing = mu_a*(current_young + current_mutated_young)/max_propensity
            p_rej = mu_rej*(current_old + current_mutated_old)/max_propensity

            rep_mut_frac = current_mutated_replicating/max((current_replicating+current_mutated_replicating),1)
            young_mut_frac = current_mutated_young/max((current_young+current_mutated_young),1)
            old_mut_frac = current_mutated_old/max((current_old+current_mutated_old),1)

            pref_rep_mut_frac = pref_deg_factor*current_mutated_replicating/max((current_replicating+pref_deg_factor*current_mutated_replicating),1)
            pref_young_mut_frac = pref_deg_factor*current_mutated_young/max((current_young+pref_deg_factor*current_mutated_young),1)
            pref_old_mut_frac = pref_deg_factor*current_mutated_old/max((current_old+pref_deg_factor*current_mutated_old),1)

            proportion_mutated_vector = np.array([1-rep_mut_frac, 1-rep_mut_frac, 1-young_mut_frac, 1-young_mut_frac, 1-rep_mut_frac, 1-young_mut_frac, 1-old_mut_frac, 1-old_mut_frac,
                                                rep_mut_frac, rep_mut_frac, young_mut_frac, young_mut_frac, pref_rep_mut_frac, pref_young_mut_frac, pref_old_mut_frac, old_mut_frac])

            probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej,
                                        p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej])*proportion_mutated_vector
            #print(probability_vector)
            r2 = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r2)
            #print(event_index)

            #Updating the nucleoid state based on which event occured
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

            current_replicating = nucleoid_state[0]
            current_young = nucleoid_state[1]
            current_old = nucleoid_state[2]
            current_mutated_replicating = nucleoid_state[3]
            current_mutated_young = nucleoid_state[4]
            current_mutated_old = nucleoid_state[5]
            # print(current_young)
            # print(current_mutated_young)

            current_unmutated = current_replicating + current_young + current_old
            current_mutated= current_mutated_replicating + current_mutated_young + current_mutated_old

            if current_unmutated == 0:
                fixed = True
                fixed_arr[i] = 1
                final_states[i] = np.array([current_mutated_replicating, current_mutated_young, current_mutated_old])

            elif current_mutated == 0:
                eliminated = True
                final_states[i] = np.array([current_replicating, current_young, current_old])

    return fixed_arr, final_states

def extended_three_population_SSD_heteroplasmy_dynamics(params, num_iter = 10, l = 500, N=0, delta = 1, birth_rate = SSD_birth, death_rate = preferential_death, pref_deg_factor = 1):
    """
    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)):

            w+dm=N -> a+da=N ->a = N/(1+d)
    """

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params
    #Defining the birth rate to maintain equilibrium
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    if N !=0:
        n_init = N
        l = (N-beta0)/beta1

    else:
        n_init = int(beta0 + beta1*l)
    
    step_matrix = np.array([[-1,0,1,0,-1,0,0,0,   0,0,0,0,0,0,0,0],
                           [2,1,-1,-1,0,-1,0,1,   0,0,0,0,0,0,0,0],
                           [0,0,0,1,0,0,-1,-1,    0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,   -1,0,1,0,-1,0,0,0],
                           [0,0,0,0,0,0,0,0,   2,1,-1,-1,0,-1,0,1],
                           [0,0,0,0,0,0,0,0,   0,0,0,1,0,0,-1,-1]]).astype(np.float64)
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    initial_replicating = n_init * mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    initial_young = n_init/initialisation_denom
    initial_old = n_init * mu_a/((mu_d_o + mu_rej)*initialisation_denom)
    n_init = initial_replicating + initial_young + initial_old
    f_yr = 1-mu_a/((mu_d_o + mu_rej)*initialisation_denom)

    het_arr = np.zeros((num_iter, 50))
    for i in range(num_iter):
        print(i)
        #initialise with one mutated molecule in the young population
        nucleoid_state = np.array([round(initial_replicating*n_init/(1+delta)), round(initial_young*n_init/(1+delta)), round(initial_old*n_init/(1+delta)), 
                                   round(initial_replicating*n_init/(1+delta)), round(initial_young*n_init/(1+delta)), round(initial_old*n_init/(1+delta))]).astype(np.int64)

        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]
        current_mutated_replicating = nucleoid_state[3]
        current_mutated_young = nucleoid_state[4]
        current_mutated_old = nucleoid_state[5]
        current_unmutated = current_replicating + current_young + current_old
        current_mutated= current_mutated_replicating + current_mutated_young + current_mutated_old

        #print(current_mutated_replicating)
        #print(current_mutated_young)
        #print(current_mutated_old)


        current_time =  0
        fixed = False
        eliminated = False
        while (current_time < 24*180) and (not fixed) and (not eliminated):

            n = int(np.sum(nucleoid_state))
            #print(n)
            #print(current_young+current_mutated_young)
            if n == 0:
                break
            
            ##################----------------Generating the time that the next event takes place---------------------######################

            max_propensity = birth_rate(current_young, current_mutated_young, current_replicating, current_mutated_replicating, mu_b, c, l, f_yr, delta, beta0, beta1, n = current_unmutated,m = current_mutated) + death_rate(current_old+current_mutated_old, current_mutated_old, mu_d_o, c, l, beta0, beta1, pref_deg_factor) + \
                death_rate(current_young + current_mutated_young, current_mutated_young, mu_d_y, c, l, beta0, beta1, pref_deg_factor) + death_rate(current_replicating + current_mutated_replicating, current_mutated_replicating, mu_d_r, c, l, beta0, beta1, pref_deg_factor) + \
                    (current_replicating + current_mutated_replicating)*mu_r + mu_a*(current_young + current_mutated_young) + mu_rej*(current_old + current_mutated_old)
            next_event_time = np.random.exponential(1/max_propensity)

            if current_time //(24*180/50) < (current_time + next_event_time)//(24*180/50):
                j = int(current_time//(24*180/50))
                het_arr[i,j] = current_mutated/(current_unmutated+current_mutated)
                
            current_time += next_event_time

            ##################-------------------------Generating what kind of event this is---------------------------#####################

            p_birth = birth_rate(current_young, current_mutated_young, current_replicating, current_mutated_replicating, mu_b, c, l, f_yr, delta, beta0, beta1, n = current_unmutated,m = current_mutated)/max_propensity
            p_rep_death = death_rate(current_replicating+current_mutated_replicating,current_mutated_replicating,mu_d_r,c,l,beta0, beta1, pref_deg_factor)/max_propensity
            p_young_death = death_rate(current_young+current_mutated_young,current_mutated_young,mu_d_y,c,l,beta0, beta1, pref_deg_factor)/max_propensity
            p_old_death = death_rate(current_old+current_mutated_old,current_mutated_old,mu_d_o,c,l,beta0, beta1, pref_deg_factor)/max_propensity
            p_double_truebirth = p*(current_replicating+current_mutated_replicating)*mu_r/max_propensity
            p_single_truebirth = (1-p)*(current_replicating+current_mutated_replicating)*mu_r/max_propensity
            p_ageing = mu_a*(current_young + current_mutated_young)/max_propensity
            p_rej = mu_rej*(current_old + current_mutated_old)/max_propensity

            rep_mut_frac = current_mutated_replicating/max((current_replicating+current_mutated_replicating),1)
            young_mut_frac = current_mutated_young/max((current_young+current_mutated_young),1)
            old_mut_frac = current_mutated_old/max((current_old+current_mutated_old),1)

            pref_rep_mut_frac = pref_deg_factor*current_mutated_replicating/max((current_replicating+pref_deg_factor*current_mutated_replicating),1)
            pref_young_mut_frac = pref_deg_factor*current_mutated_young/max((current_young+pref_deg_factor*current_mutated_young),1)
            pref_old_mut_frac = pref_deg_factor*current_mutated_old/max((current_old+pref_deg_factor*current_mutated_old),1)

            proportion_mutated_vector = np.array([1-rep_mut_frac, 1-rep_mut_frac, 1-young_mut_frac, 1-young_mut_frac, 1-rep_mut_frac, 1-young_mut_frac, 1-old_mut_frac, 1-old_mut_frac,
                                                rep_mut_frac, rep_mut_frac, young_mut_frac, young_mut_frac, pref_rep_mut_frac, pref_young_mut_frac, pref_old_mut_frac, old_mut_frac])

            probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej,
                                        p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej])*proportion_mutated_vector
            #print(probability_vector)
            r2 = np.random.uniform(0,1)
            event_index = np.searchsorted(np.cumsum(probability_vector), r2)
            #print(event_index)

            #Updating the nucleoid state based on which event occured
            nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

            current_replicating = nucleoid_state[0]
            current_young = nucleoid_state[1]
            current_old = nucleoid_state[2]
            current_mutated_replicating = nucleoid_state[3]
            current_mutated_young = nucleoid_state[4]
            current_mutated_old = nucleoid_state[5]
            # print(current_young)
            # print(current_mutated_young)

            current_unmutated = current_replicating + current_young + current_old
            current_mutated= current_mutated_replicating + current_mutated_young + current_mutated_old

            if current_unmutated == 0:
                fixed = True
                j = int(current_time//(24*180/50))
                het_arr[i,j:] = 1

            elif current_mutated == 0:
                eliminated = True

    return het_arr

#############################################################################################################################
#                                                                                                                           #
#                                                              PULSING                                                      #
#                                                             FUNCTIONS                                                     #
#                                                                                                                           #
#############################################################################################################################

#@jit(nopython = True)
def extended_three_population_pulsing_params(params1, params2, pulsing_freq = 1, pulsing_length=0.1, pulsing_stagger = 0, track_every = 0.1, truncation_time = 4, mutation_probability = 0.01, l = 500, N=0, birth_rate = logarithmic_birth, death_rate = constant_death):
    """

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)):
    """

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params1
    print(mu_a)
    #Defining the birth rate to maintain equilibrium
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    if N !=0:
        n_init = N
        l = (N-beta0)/beta1

    else:
        n_init = int(beta0 + beta1*l)
    
    step_matrix = np.array([[-1,0,1,0,-1,0,0,0],
                           [2,1,-1,-1,0,-1,0,1],
                           [0,0,0,1,0,0,-1,-1]]).astype(np.float64)
    
    #which population does the event act on
    event_to_population = np.array([0,0,1,1,0,1,2,2])
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    initial_replicating = n_init * mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    initial_young = n_init/initialisation_denom
    initial_old = n_init * mu_a/((mu_d_o + mu_rej)*initialisation_denom)
    n_init = initial_replicating + initial_young + initial_old

    nucleoid_state = np.array([round(initial_replicating), round(initial_young), round(initial_old)]).astype(np.int64)

    current_time =  0
    replicating_mutants = [[] for _ in range(round(initial_replicating))]
    young_mutants = [[] for _ in range(round(initial_young))]
    old_mutants = [[] for _ in range(round(initial_old))]

    f_ys = np.zeros(int(truncation_time//track_every)+1)
    Ns = np.zeros(int(truncation_time//track_every)+1)
    mu_turns = np.zeros(int(truncation_time//track_every)+1)
    
    f_ys[0] = initial_young/n_init
    Ns[0] = n_init
    mu_turns[0] = mu_r*initial_replicating/n_init
    

    mutant_number = 0
    while current_time < truncation_time*24*365:
        #print(mu_a)

        if (current_time-pulsing_stagger*24*365)%(pulsing_freq*24*365) < pulsing_length*24*365:
            beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params2
        else:
            beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params1

        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]

        n = int(np.sum(nucleoid_state))
        if n == 0:
            break
        
        ##################----------------Generating the time that the next event takes place---------------------######################

        max_propensity = birth_rate(current_young, n, mu_b, c, l, beta0, beta1) + death_rate(current_old, n, mu_d_o, c, l, beta0, beta1) + \
            death_rate(current_young, n, mu_d_y, c, l, beta0, beta1) + death_rate(current_replicating, n, mu_d_r, c, l, beta0, beta1) + \
                current_replicating*mu_r + mu_a*current_young + mu_rej*current_old
        next_event_time = np.random.exponential(1/max_propensity)

        if current_time //(0.25*24*365) < (current_time + next_event_time)//(0.25*24*365):
            print(current_time/(24*365))

        a = int((current_time/(24*365))//track_every)
        b = int(((current_time+next_event_time)/(24*365))//track_every)

        for k in range(a,min(b, int(truncation_time//track_every))): 
            f_ys[k+1] = current_young/n
            Ns[k+1] = n
            mu_turns[k+1] = mu_r*current_replicating/n
            
        current_time += next_event_time

        ##################-------------------------Generating what kind of event this is---------------------------#####################

        p_birth = birth_rate(current_young,n,mu_b,c,l,beta0, beta1)/max_propensity
        p_rep_death = death_rate(current_replicating,n,mu_d_r,c,l,beta0, beta1)/max_propensity
        p_young_death = death_rate(current_young,n,mu_d_y,c,l,beta0, beta1)/max_propensity
        p_old_death = death_rate(current_old,n,mu_d_o,c,l,beta0, beta1)/max_propensity
        p_double_truebirth = p*current_replicating*mu_r/max_propensity
        p_single_truebirth = (1-p)*current_replicating*mu_r/max_propensity
        p_ageing = mu_a*current_young/max_propensity
        p_rej = mu_rej*current_old/max_propensity

        probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej])
        r2 = np.random.uniform(0,1)
        event_index = np.searchsorted(np.cumsum(probability_vector), r2)

        #Picking a population which this event acts on
        molecule_index = int(np.random.uniform(0,1)*nucleoid_state[event_to_population[event_index]])

        #Updating the nucleoid state based on which event occured
        nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #tracking the SFS
        #non preferential rep
        if event_index == 0: 
            mut = np.random.binomial(1,mutation_probability)

            young_mutants.append(replicating_mutants[molecule_index]*1)
            young_mutants.append(replicating_mutants[molecule_index]*1)
            replicating_mutants.pop(molecule_index)
            if mut:
                young_mutants[-1].append(mutant_number)
                mutant_number += 1
        
        #preferential rep
        elif event_index == 1:
            mut = np.random.binomial(1,mutation_probability)
            young_mutants.append(replicating_mutants[molecule_index]*1)
            if mut:
                k = np.random.binomial(1, 0.5)
                if k:
                    young_mutants[-1].append(mutant_number)
                else:
                    replicating_mutants[molecule_index].append(mutant_number)
                mutant_number += 1
        
        #birth
        elif event_index == 2:
            replicating_mutants.append(young_mutants[molecule_index])
            young_mutants.pop(molecule_index)

        #ageing
        elif event_index == 3:
            old_mutants.append(young_mutants[molecule_index])
            young_mutants.pop(molecule_index)

        #replicating death
        elif event_index == 4:
            replicating_mutants.pop(molecule_index)

        #young death
        elif event_index == 5:
            young_mutants.pop(molecule_index)

        #old death
        elif event_index == 6:
            old_mutants.pop(molecule_index)

        #rejuvination
        else:
            young_mutants.append(old_mutants[molecule_index])
            old_mutants.pop(molecule_index)

    
    #make this output mutation heteroplasmy time evolution. This will give me a sense of the dynamics
    #and whether they are correct. If it is correct, the number should behave as a simple random walk.
    #(ignoring the times each step time takes)
    cSFS = np.zeros(mutant_number)
    n = int(np.sum(nucleoid_state))
    n = nucleoid_state[0] + nucleoid_state[1]
    for i in range(mutant_number):
        mutant_counter = 0
        for replicating_mutant in replicating_mutants:
            mutant_counter += (i in replicating_mutant)
        for young_mutant in young_mutants:
            mutant_counter += (i in young_mutant)
        # for old_mutant in old_mutants:
        #     mutant_counter += (i in old_mutant)

        cSFS[i] = mutant_counter/n
        
    return cSFS[cSFS != 0], replicating_mutants, young_mutants, old_mutants, f_ys, Ns, mu_turns



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

#############################################################################################################################
#                                                                                                                           #
#                                                          COALESCENT TREE                                                  #
#                                                             FUNCTIONS                                                     #
#                                                                                                                           #
#############################################################################################################################

@jit(nopython = True)
def extended_three_population_forward_coalescent(params, verbose = False, l = 500, N= 1000, birth_rate = logarithmic_birth, death_rate = constant_death):

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params
    #Defining the birth rate to maintain equilibrium
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    
    f_r = mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    f_y = 1/initialisation_denom


    if N !=0:
        n_init = N
        l = (N-beta0)/beta1

    else:
        n_init = int(beta0 + beta1*l)
    
    step_matrix = np.array([[-1,0,1,0,-1,0,0,0],
                           [2,1,-1,-1,0,-1,0,1],
                           [0,0,0,1,0,0,-1,-1]]).astype(np.float64)
    
    #which population does the event act on
    event_to_population = np.array([0,0,1,1,0,1,2,2])
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    initial_replicating = n_init * mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    initial_young = n_init/initialisation_denom
    initial_old = n_init * mu_a/((mu_d_o + mu_rej)*initialisation_denom)
    n_init = initial_replicating + initial_young + initial_old

    nucleoid_state = np.array([round(initial_replicating), round(initial_young), round(initial_old)]).astype(np.int64)

    current_time =  0

    #Looping until the end of this iteration (usually 1 hour)
    event_times = np.zeros(2000000)
    event_indexes = np.zeros(2000000)
    molecule_indexes = np.zeros(2000000)

    for counter in range(2000000):
        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]

        n = int(np.sum(nucleoid_state))
        if n == 0:
            break
        
        ##################----------------Generating the time that the next event takes place---------------------######################

        max_propensity = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n)  + death_rate(current_old, n, mu_d_o, c, l, beta0, beta1) + \
            death_rate(current_young, n, mu_d_y, c, l, beta0, beta1) + death_rate(current_replicating, n, mu_d_r, c, l, beta0, beta1) + \
                current_replicating*mu_r + mu_a*current_young + mu_rej*current_old
        next_event_time = np.random.exponential(1/max_propensity)
        
        #Updating the time
        current_time += next_event_time

        ##################-------------------------Generating what kind of event this is---------------------------#####################

        p_birth = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n) /max_propensity
        p_rep_death = death_rate(current_replicating,n,mu_d_r,c,l,beta0, beta1)/max_propensity
        p_young_death = death_rate(current_young,n,mu_d_y,c,l,beta0, beta1)/max_propensity
        p_old_death = death_rate(current_old,n,mu_d_o,c,l,beta0, beta1)/max_propensity
        p_double_truebirth = p*current_replicating*mu_r/max_propensity
        p_single_truebirth = (1-p)*current_replicating*mu_r/max_propensity
        p_ageing = mu_a*current_young/max_propensity
        p_rej = mu_rej*current_old/max_propensity
        # if counter % 100 == 0:
        #     print(nucleoid_state)
        #     print(n)
        #     print(p_birth)

        probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej])
        r2 = np.random.uniform(0,1)
        event_index = np.searchsorted(np.cumsum(probability_vector), r2)

        #Picking a population which this event acts on
        molecule_index = int(np.random.uniform(0,1)*nucleoid_state[event_to_population[event_index]])

        #Updating the nucleoid state based on which event occured
        nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        event_times[counter] = current_time
        event_indexes[counter] = event_index
        molecule_indexes[counter] = molecule_index

    return (event_times, event_indexes.astype(np.int32),  molecule_indexes.astype(np.int32), nucleoid_state)

#Inputs the information of the forward simlation, and backpropagates to extract only those events which affected the nucleoids which 
#survived till the end. These are the events which are necessary to build the coalescent tree
def extract_coalescent_info(event_times, event_indexes, molecule_indexes, final_nucleoid_state):#winning mutant index
    #coalescent_times = [0]
    coalescent_times = []
    coalescent_event_indexes = []
    coalescent_molecule_indexes = []

    #Tracks the number of lineages left in each subpopulation
    rep_coalescent = [int(final_nucleoid_state[0])]
    young_coalescent = [int(final_nucleoid_state[1])]
    old_coalescent = [int(final_nucleoid_state[2])]

    #ones encode positions of molecules we are tracking. zeros encoding molecules we are not tracking
    rep_fullsample = np.ones(int(final_nucleoid_state[0]))
    young_fullsample = np.ones(int(final_nucleoid_state[1]))
    old_fullsample = np.ones(int(final_nucleoid_state[2]))

    for i in range(len(event_times)):
        #print(i)
        #stop when we've reached the common ancestor
        if np.sum(rep_fullsample) + np.sum(young_fullsample) + np.sum(old_fullsample) == 1:
            #print(i)
            break
        j = -i-1
        if event_indexes[j] == 0:

            #If the replication event produces one or two of the molecules we are tracking, update the coalescent tree
            indicator = (young_fullsample[-1] == 1) or (young_fullsample[-2] == 1)
            indicator2 = (young_fullsample[-1] == 1) and (young_fullsample[-2] == 1)
            if indicator:
                coalescent_event_indexes.append(event_indexes[j])

                if indicator2:
                    young_coalescent.append(young_coalescent[-1] - 2)
                else:
                    young_coalescent.append(young_coalescent[-1] - 1)
                rep_coalescent.append(rep_coalescent[-1] + 1)
                old_coalescent.append(old_coalescent[-1])

                coalescent_times.append(event_times[-1] - event_times[j])
            
            #Update the state vector:

            #pop out the molecule indexes
            young_fullsample = np.delete(young_fullsample, [-1, -2])

            #insert in the replicating population a 0 or 1 depending if the molecule is one we were tracking or not
            rep_fullsample = np.insert(rep_fullsample, molecule_indexes[j], indicator)

            if indicator:
                coalescent_molecule_indexes.append(int(np.sum(rep_fullsample[:molecule_indexes[j]])))

        elif event_indexes[j] == 1:
            #If the replication event produces a young molecule we are tracking, update the coalescent tree
            indicator = (young_fullsample[-1] == 1)
            if indicator:
                coalescent_event_indexes.append(event_indexes[j])

                if rep_fullsample[molecule_indexes[j]] == 1:
                    rep_coalescent.append(rep_coalescent[-1])
                else:
                    rep_coalescent.append(rep_coalescent[-1] + 1)
                young_coalescent.append(young_coalescent[-1] - 1)
                old_coalescent.append(old_coalescent[-1])

                coalescent_times.append(event_times[-1] - event_times[j])

            #Update the state vector:

            #pop out the molecule indexes
            young_fullsample = np.delete(young_fullsample, -1)

            #change the replicating entry based on whether we are now tracking the molecule or not
            rep_fullsample[molecule_indexes[j]] = max(indicator, rep_fullsample[molecule_indexes[j]])

            if indicator:
                coalescent_molecule_indexes.append(int(np.sum(rep_fullsample[:molecule_indexes[j]])))

        elif event_indexes[j] == 2:
            #If the birth event results in a replicating molecule we are tracking, update the coalescent tree
            indicator = rep_fullsample[-1] == 1
            if indicator:
                coalescent_event_indexes.append(event_indexes[j])


                rep_coalescent.append(rep_coalescent[-1] - 1)
                young_coalescent.append(young_coalescent[-1] + 1)
                old_coalescent.append(old_coalescent[-1])

                coalescent_times.append(event_times[-1] - event_times[j])
            
            #Update the state vector:

            #pop out the molecule indexes
            rep_fullsample = np.delete(rep_fullsample, -1)

            #change the young entry based on whether we are now tracking the molecule or not
            young_fullsample = np.insert(young_fullsample, molecule_indexes[j], indicator)

            if indicator:
                coalescent_molecule_indexes.append(int(np.sum(young_fullsample[:molecule_indexes[j]])))

        elif event_indexes[j] == 3:
            #If the ageing event results in an old molecule we are tracking, update the coalescent tree
            indicator = old_fullsample[-1] == 1
            if indicator:
                coalescent_event_indexes.append(event_indexes[j])

                rep_coalescent.append(rep_coalescent[-1])
                young_coalescent.append(young_coalescent[-1] + 1)
                old_coalescent.append(old_coalescent[-1] - 1)

                coalescent_times.append(event_times[-1] - event_times[j])
            
            #Update the state vector:

            #pop out the molecule indexes
            old_fullsample = np.delete(old_fullsample, -1)

            #change the young entry based on whether we are now tracking the molecule or not
            young_fullsample = np.insert(young_fullsample, molecule_indexes[j], indicator)

            if indicator:
                coalescent_molecule_indexes.append(int(np.sum(young_fullsample[:molecule_indexes[j]])))

        elif event_indexes[j] == 4:
            #A rep death event will always produce (backwards in time) a molecule we are not tracking
            
            #Update the state vector:
            rep_fullsample = np.insert(rep_fullsample, molecule_indexes[j], 0)

        elif event_indexes[j] == 5:
            #A young death event will always produce (backwards in time) a molecule we are not tracking
            
            #Update the state vector:
            young_fullsample = np.insert(young_fullsample, molecule_indexes[j], 0)

        elif event_indexes[j] == 6:
            #An old death event will always produce (backwards in time) a molecule we are not tracking
            
            #Update the state vector:
            old_fullsample = np.insert(old_fullsample, molecule_indexes[j], 0)

        elif event_indexes[j] == 7:
            #If the rejuvenation event results in an old molecule we are tracking, update the coalescent tree
            indicator = young_fullsample[-1] == 1
            if indicator:
                coalescent_event_indexes.append(event_indexes[j])

                rep_coalescent.append(rep_coalescent[-1])
                young_coalescent.append(young_coalescent[-1] - 1)
                old_coalescent.append(old_coalescent[-1] + 1)

                coalescent_times.append(event_times[-1] - event_times[j])
            
            #Update the state vector:

            #pop out the molecule indexes
            young_fullsample = np.delete(young_fullsample, -1)

            #change the old entry based on whether we are now tracking the molecule or not
            old_fullsample = np.insert(old_fullsample, molecule_indexes[j], indicator)

            if indicator:
                coalescent_molecule_indexes.append(int(np.sum(old_fullsample[:molecule_indexes[j]])))

    return rep_coalescent, young_coalescent, old_coalescent, coalescent_times, coalescent_event_indexes, coalescent_molecule_indexes

#Helper functions for build tree object
def add_parents_event0(young_coalescent, rep_tree, young_tree, old_tree, i, subsample_molecule_indexes):

    #produces two molecules we are tracking (coalescence)
    if young_coalescent[i] - young_coalescent[i+1] == 2:

        rep_tree[i+1][subsample_molecule_indexes[i]].add_child(young_tree[i][-1])
        rep_tree[i+1][subsample_molecule_indexes[i]].add_child(young_tree[i][-2])

        #adds parents to the rest of the dna
        j0=0
        for j in range(len(rep_tree[i])):                    
            if j != subsample_molecule_indexes[i]:
                rep_tree[i+1][j0].add_child(rep_tree[i][j])
                
            elif j == subsample_molecule_indexes[i]:
                j0+=1
                rep_tree[i+1][j0].add_child(rep_tree[i][j])

            j0+=1

        #young tree
        for j in range(len(young_tree[i])-2):
            young_tree[i+1][j].add_child(young_tree[i][j])
        
        #old tree
        for j in range(len(old_tree[i])):
            old_tree[i+1][j].add_child(old_tree[i][j])

    #produces one molecule we are tracking
    else:
        rep_tree[i+1][subsample_molecule_indexes[i]].add_child(young_tree[i][-1])

        #adds parents to the rest of the dna

        #replicating tree
        j0 = 0
        if len(rep_tree[i]) != len(rep_tree[i+1]):
            for j in range(len(rep_tree[i])):                    
                if j != subsample_molecule_indexes[i]:
                    rep_tree[i+1][j0].add_child(rep_tree[i][j])
                    
                elif j == subsample_molecule_indexes[i]:
                    j0+=1
                    rep_tree[i+1][j0].add_child(rep_tree[i][j])

                j0+=1

        else:
            for j in range(len(rep_tree[i])):                    
                    rep_tree[i+1][j].add_child(rep_tree[i][j])


        #young tree
        for j in range(len(young_tree[i])-1):
            young_tree[i+1][j].add_child(young_tree[i][j])
        
        #old tree
        for j in range(len(old_tree[i])):
            old_tree[i+1][j].add_child(old_tree[i][j])

def add_parents_event1(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes):
    rep_tree[i+1][subsample_molecule_indexes[i]].add_child(young_tree[i][-1])

    #replicating tree
    j0 = 0
    if len(rep_tree[i]) != len(rep_tree[i+1]):
        for j in range(len(rep_tree[i])):                    
            if j != subsample_molecule_indexes[i]:
                rep_tree[i+1][j0].add_child(rep_tree[i][j])
                
            elif j == subsample_molecule_indexes[i]:
                j0+=1
                rep_tree[i+1][j0].add_child(rep_tree[i][j])

            j0+=1

    else:
        for j in range(len(rep_tree[i])):                    
            rep_tree[i+1][j].add_child(rep_tree[i][j])

    #young tree
    for j in range(len(young_tree[i])-1):
        young_tree[i+1][j].add_child(young_tree[i][j])
    
    #old tree
    for j in range(len(old_tree[i])):
        old_tree[i+1][j].add_child(old_tree[i][j])

def add_parents_event2(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes):
    young_tree[i+1][subsample_molecule_indexes[i]].add_child(rep_tree[i][-1])

    for j in range(len(rep_tree[i]) - 1):                    
        rep_tree[i+1][j].add_child(rep_tree[i][j])

    j0=0
    for j in range(len(young_tree[i])):                    
        if j != subsample_molecule_indexes[i]:
            young_tree[i+1][j0].add_child(young_tree[i][j])
            
        elif j == subsample_molecule_indexes[i]:
            j0+=1
            young_tree[i+1][j0].add_child(young_tree[i][j])

        j0+=1
    
    #old tree
    for j in range(len(old_tree[i])):
        old_tree[i+1][j].add_child(old_tree[i][j])

def add_parents_event3(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes):
    young_tree[i+1][subsample_molecule_indexes[i]].add_child(old_tree[i][-1])

    for j in range(len(rep_tree[i])):                    
        rep_tree[i+1][j].add_child(rep_tree[i][j])

    j0=0
    for j in range(len(young_tree[i])):                    
        if j != subsample_molecule_indexes[i]:
            young_tree[i+1][j0].add_child(young_tree[i][j])
            
        elif j == subsample_molecule_indexes[i]:
            j0+=1
            young_tree[i+1][j0].add_child(young_tree[i][j])

        j0+=1
    
    #old tree
    for j in range(len(old_tree[i]) - 1):
        old_tree[i+1][j].add_child(old_tree[i][j])

def add_parents_event7(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes):
    old_tree[i+1][subsample_molecule_indexes[i]].add_child(young_tree[i][-1])

    for j in range(len(rep_tree[i])):                    
        rep_tree[i+1][j].add_child(rep_tree[i][j])

    j0=0
    for j in range(len(old_tree[i])):                    
        if j != subsample_molecule_indexes[i]:
            old_tree[i+1][j0].add_child(old_tree[i][j])
            
        elif j == subsample_molecule_indexes[i]:
            j0+=1
            old_tree[i+1][j0].add_child(old_tree[i][j])

        j0+=1
    
    #young tree
    for j in range(len(young_tree[i]) - 1):
        young_tree[i+1][j].add_child(young_tree[i][j])

#Inputs the information of extract_coalescent info and builds a coalescent tree object 
def build_tree_object(rep_coalescent, young_coalescent, old_coalescent, events, subsample_molecule_indexes):
    
    rep_tree = []
    young_tree = []
    old_tree = []

    trees = [rep_tree, young_tree, old_tree]

    #Build the tree nodes with no edges
    for i in range(len(rep_coalescent)):
        #print(i)
        rep_leaves = []
        young_leaves = []
        old_leaves = []
        for j in range(rep_coalescent[i]):
            rep_leaves.append(ete3.Tree(name = "rep"))

        for j in range(young_coalescent[i]):
            young_leaves.append(ete3.Tree(name = "young"))
        
        for j in range(old_coalescent[i]):
            old_leaves.append(ete3.Tree(name = "old"))
        
        rep_tree.append(rep_leaves)
        young_tree.append(young_leaves)
        old_tree.append(old_leaves)

    #Construct the edges between nodes
    for i in range(len(events)):
        #Diffusive replication
        if events[i] == 0:
            add_parents_event0(young_coalescent, rep_tree, young_tree, old_tree, i, subsample_molecule_indexes)
            
        #Non-diffusive replication
        elif events[i] == 1:
            add_parents_event1(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes)

        #Birth event
        elif events[i] == 2:
            add_parents_event2(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes)
        #Ageing event
        elif events[i] == 3:
            add_parents_event3(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes)
        #Rejuvenation event
        elif events[i] == 7:
            add_parents_event7(rep_tree, young_tree, old_tree, i, subsample_molecule_indexes)

    return trees

#Inputs the tree object and subsamples a tree from it
subpop_colors = {"rep": "#0C4202", "young": "#00B0F0", "old": "#A6A6A6"}
def subsample_tree(trees, rep_num, young_num, old_num, coalescent_times, seed = 0, random = False):

    #np.random.seed(seed)

    rep_tree, young_tree, old_tree = trees

    #select indices of leaves that we will construct the subtree from
    if random:
        indices = np.random.choice(len(rep_tree[0]) + len(young_tree[0]) + len(old_tree[0]), rep_num + young_num + old_num, replace = False)
        
        rep_indices = []
        young_indices = []
        old_indices = []
        for index in indices:
            if index < len(rep_tree[0]):
                rep_indices.append(index)
            elif index - len(rep_tree[0]) < len(young_tree[0]):
                young_indices.append(index - len(rep_tree[0]))
            else:
                old_indices.append(index - len(rep_tree[0]) - len(young_tree[0]))

        rep_num = len(rep_indices)
        young_num = len(young_indices)
        old_num = len(old_indices)
    
    else:
        rep_indices = np.random.choice(len(rep_tree[0]), rep_num, replace = False)
        young_indices = np.random.choice(len(young_tree[0]), young_num, replace = False)
        old_indices = np.random.choice(len(old_tree[0]), old_num, replace = False)

    rep_subtree = []
    young_subtree = []
    old_subtree = []

    #extracting the nodes associated to the above selected indices
    for index in rep_indices:
        rep_subtree.append(rep_tree[0][index])

    for index in young_indices:
        young_subtree.append(young_tree[0][index])

    for index in old_indices:
        old_subtree.append(old_tree[0][index])

    repyoungold = []
    repyoungold.extend(rep_subtree)
    repyoungold.extend(young_subtree)
    repyoungold.extend(old_subtree)

    #current new tree is a list of nodes of the current 'time slice' of the NEW tree we are creating (which is a subtree of the tree we're inputting). 
    # We will iterate backwards in time updating this based on the state of the new time slice

    print(rep_num)
    print(young_num)
    print(old_num)
    current_new_tree = []
    for i in range(rep_num):
        current_new_tree.append(ete3.Tree(name = "rep"))

    for i in range(young_num):
        current_new_tree.append(ete3.Tree(name = "young"))

    for i in range(old_num):
        current_new_tree.append(ete3.Tree(name = "old"))

    #list of times associated with current_new_tree
    current_node_times = list(np.zeros(len(current_new_tree)))

    for node in current_new_tree:
        style = ete3.NodeStyle()
        style["hz_line_color"] = subpop_colors[node.name]
        style["fgcolor"] = subpop_colors[node.name]
        style["size"] = 0
        style["hz_line_width"] = 10
        node.set_style(style)
                    
    #Iteratively constructing the subtree by moving up from the leaves. Current nodes is a list of nodes of the current 'time slice' of the tree we are inputting
    current_nodes = repyoungold
    k=0
    while True:
        #print(k)

        #break if common ancestor has been reached
        if len(current_nodes) == 1:
            break

        #make a list of parents of the input tree
        next_nodes = []
        for i in range(len(current_nodes)):
            current_node = current_nodes[i]
            next_nodes.append(current_node.up)

        #break if we traversed the whole simulation without finding a common ancestor
        if current_node.up is None:
            break

        
        #Check if there are two of the same parent. If so, a coalescent event has occured - delete the duplicate and coalesce the two children
        break_next_loop = False
        coalesced_node_indexes = []
        for i in range(len(next_nodes)-1):
            for j in range(len(next_nodes[i+1:])):
                if next_nodes[i] == next_nodes[i+1+j]:

                    #recreate the parent node and coalesce both children to it
                    node0 = ete3.Tree(name = next_nodes[i].name)
                    node0.add_child(current_new_tree[i])
                    node0.add_child(current_new_tree[i+1+j])

                    #set the length of the branches to be the recorded times
                    current_new_tree[i]._set_dist(coalescent_times[k] - current_node_times[i])
                    current_new_tree[i+1+j]._set_dist(coalescent_times[k] - current_node_times[i+1+j])

                    #Set the colour of the branch spanning from the new parent as green (as it is replicating)
                    style = ete3.NodeStyle()
                    style["hz_line_color"] = subpop_colors[node0.name]
                    style["vt_line_color"] = subpop_colors[node0.name]
                    style["fgcolor"] = subpop_colors[node0.name]
                    style["size"] = 0
                    style["vt_line_width"] = 10
                    style["hz_line_width"] = 10
                    node0.set_style(style)        

                    #Update the time slice to remove the two children and add the parent
                    current_new_tree[i] = node0
                    current_new_tree.pop(i+1+j)
                    #Update the parents
                    next_nodes.pop(i+1+j)

                    #Update the time of this new node and remove the old
                    current_node_times[i] = coalescent_times[k]
                    current_node_times.pop(i+1+j)

                    #Keep track of the molecules which coalesced
                    coalesced_node_indexes.append(i)
                    coalesced_node_indexes.append(i+1+j)
                    
                    break_next_loop = True
                    break
            if break_next_loop:
                break
        
        #Check if theres been a migration event
        for i in range(len(current_nodes)):
            current_node = current_nodes[i]
            if (current_node.up.name != current_node.name) and (i not in coalesced_node_indexes):
                node0 = ete3.Tree(name = current_node.up.name)
                node0.add_child(current_new_tree[i])

                current_new_tree[i]._set_dist(coalescent_times[k] - current_node_times[i])

                style = ete3.NodeStyle()
                style["hz_line_color"] = subpop_colors[node0.name]
                style["fgcolor"] = subpop_colors[node0.name]
                style["size"] = 0
                style["hz_line_width"] = 10
                node0.set_style(style)

                current_node_times[i] = coalescent_times[k]
                current_new_tree[i] = node0
            
        current_nodes = next_nodes
        k +=1

    return current_new_tree

def subsample_uncoloured_tree(trees, rep_indices, young_indices, old_indices, coalescent_times, seed = 0, random = False):

    np.random.seed(seed)

    rep_tree, young_tree, old_tree = trees

    rep_num = len(rep_indices)
    young_num = len(young_indices)
    old_num = len(old_indices)

    rep_subtree = []
    young_subtree = []
    old_subtree = []

    #extracting the nodes associated to the above selected indices
    for index in rep_indices:
        rep_subtree.append(rep_tree[0][index])

    for index in young_indices:
        young_subtree.append(young_tree[0][index])

    for index in old_indices:
        old_subtree.append(old_tree[0][index])

    repyoungold = []
    repyoungold.extend(rep_subtree)
    repyoungold.extend(young_subtree)
    repyoungold.extend(old_subtree)

    #current new tree is a list of nodes of the current 'time slice' of the NEW tree we are creating (which is a subtree of the tree we're inputting). 
    # We will iterate backwards in time updating this based on the state of the new time slice
    current_new_tree = []
    for i in range(rep_num):
        current_new_tree.append(ete3.Tree(name = "rep"))

    for i in range(young_num):
        current_new_tree.append(ete3.Tree(name = "young"))

    for i in range(old_num):
        current_new_tree.append(ete3.Tree(name = "old"))

    #list of times associated with current_new_tree
    current_node_times = list(np.zeros(len(current_new_tree)))

    for node in current_new_tree:
        style = ete3.NodeStyle()
        style["hz_line_color"] = subpop_colors[node.name]
        style["fgcolor"] = subpop_colors[node.name]
        style["size"] = 0
        style["hz_line_width"] = 10
        node.set_style(style)
                    
    #Iteratively constructing the subtree by moving up from the leaves. Current nodes is a list of nodes of the current 'time slice' of the tree we are inputting
    current_nodes = repyoungold
    k=0
    while True:

        #break if common ancestor has been reached
        if len(current_nodes) == 1:
            break

        #make a list of parents of the input tree
        next_nodes = []
        for i in range(len(current_nodes)):
            current_node = current_nodes[i]
            next_nodes.append(current_node.up)
        
        #Check if there are two of the same parent. If so, a coalescent event has occured - delete the duplicate and coalesce the two children
        break_next_loop = False
        coalesced_node_indexes = []
        for i in range(len(next_nodes)-1):
            for j in range(len(next_nodes[i+1:])):
                if next_nodes[i] == next_nodes[i+1+j]:

                    #recreate the parent node and coalesce both children to it
                    node0 = ete3.Tree(name = next_nodes[i].name)
                    node0.add_child(current_new_tree[i])
                    node0.add_child(current_new_tree[i+1+j])

                    #set the length of the branches to be the recorded times
                    current_new_tree[i]._set_dist(coalescent_times[k] - current_node_times[i])
                    current_new_tree[i+1+j]._set_dist(coalescent_times[k] - current_node_times[i+1+j])

                    #Set the colour of the branch spanning from the new parent as green (as it is replicating)
                    style = ete3.NodeStyle()
                    style["hz_line_color"] = subpop_colors[node0.name]
                    style["vt_line_color"] = subpop_colors[node0.name]
                    style["fgcolor"] = subpop_colors[node0.name]
                    style["size"] = 0
                    style["vt_line_width"] = 10
                    style["hz_line_width"] = 10
                    node0.set_style(style)        

                    #Update the time slice to remove the two children and add the parent
                    current_new_tree[i] = node0
                    current_new_tree.pop(i+1+j)
                    #Update the parents
                    next_nodes.pop(i+1+j)

                    #Update the time of this new node and remove the old
                    current_node_times[i] = coalescent_times[k]
                    current_node_times.pop(i+1+j)

                    #Keep track of the molecules which coalesced
                    coalesced_node_indexes.append(i)
                    coalesced_node_indexes.append(i+1+j)
                    
                    break_next_loop = True
                    break
            if break_next_loop:
                break
            
        current_nodes = next_nodes
        k +=1

    return current_new_tree[0]

#############################################################################################################################
#                                                                                                                           #
#                                                          SFS SIMULATORS                                                   #
#                                                          AND ANALYTICS                                                    #
#                                                                                                                           #
#############################################################################################################################
def extended_three_population_mutational_forward_coalescent(params, truncation_time = 4, mutation_probability = 0.01, l = 500, N=0, birth_rate = logarithmic_birth, death_rate = constant_death, n_coal = 20):
    """

    Args:
        params: List[float] or array[float] containing the following elements in the following order (as defined in Wolf, Mjeku et al):
            - beta0 (float, beta0 >= 0)
            - beta1 (float, beta1 >= 0)
            - p (float, 0 <= p <= 1)
            - mu_d (float, mu_d >= 0)
            - mu_b (float, mu_b >= 0)
            - mu_r (float, mu_r >= 0)
            - c (float, c >= 0)):
    """

    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params
    #Defining the birth rate to maintain equilibrium
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)

    f_r = mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    f_y = 1/initialisation_denom

    if N !=0:
        n_init = N
        l = (N-beta0)/beta1

    else:
        n_init = int(beta0 + beta1*l)
    
    step_matrix = np.array([[-1,0,1,0,-1,0,0,0],
                           [2,1,-1,-1,0,-1,0,1],
                           [0,0,0,1,0,0,-1,-1]]).astype(np.float64)
    
    #which population does the event act on
    event_to_population = np.array([0,0,1,1,0,1,2,2])
    
    #We only transpose this matrix so that we can easily access the columns later
    step_matrix = step_matrix.transpose()

    initialisation_denom = mu_b/(mu_d_r + p*mu_r) + 1 + mu_a/(mu_d_o + mu_rej)
    initial_replicating = n_init * mu_b/((mu_d_r+ p*mu_r)*initialisation_denom)
    initial_young = n_init/initialisation_denom
    initial_old = n_init * mu_a/((mu_d_o + mu_rej)*initialisation_denom)
    n_init = initial_replicating + initial_young + initial_old

    nucleoid_state = np.array([round(initial_replicating), round(initial_young), round(initial_old)]).astype(np.int64)

    current_time =  0
    replicating_mutants = [[] for _ in range(round(initial_replicating))]
    young_mutants = [[] for _ in range(round(initial_young))]
    old_mutants = [[] for _ in range(round(initial_old))]

    mutant_number = 0
    while current_time < truncation_time*24*365:

        current_replicating = nucleoid_state[0]
        current_young = nucleoid_state[1]
        current_old = nucleoid_state[2]

        n = int(np.sum(nucleoid_state))
        if n == 0:
            break
        
        ##################----------------Generating the time that the next event takes place---------------------######################

        max_propensity = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n) + death_rate(current_old, n, mu_d_o, c, l, beta0, beta1) + \
            death_rate(current_young, n, mu_d_y, c, l, beta0, beta1) + death_rate(current_replicating, n, mu_d_r, c, l, beta0, beta1) + \
                current_replicating*mu_r + mu_a*current_young + mu_rej*current_old
        next_event_time = np.random.exponential(1/max_propensity)

        # if current_time //(0.25*24*365) < (current_time + next_event_time)//(0.25*24*365):
        #     print(current_time/(24*365))
            
        current_time += next_event_time

        ##################-------------------------Generating what kind of event this is---------------------------#####################

        p_birth = birth_rate(current_young, current_replicating, mu_b, c, l, f_y, f_r, beta0, beta1, n)/max_propensity
        p_rep_death = death_rate(current_replicating,n,mu_d_r,c,l,beta0, beta1)/max_propensity
        p_young_death = death_rate(current_young,n,mu_d_y,c,l,beta0, beta1)/max_propensity
        p_old_death = death_rate(current_old,n,mu_d_o,c,l,beta0, beta1)/max_propensity
        p_double_truebirth = p*current_replicating*mu_r/max_propensity
        p_single_truebirth = (1-p)*current_replicating*mu_r/max_propensity
        p_ageing = mu_a*current_young/max_propensity
        p_rej = mu_rej*current_old/max_propensity

        probability_vector = np.array([p_double_truebirth, p_single_truebirth, p_birth, p_ageing, p_rep_death, p_young_death, p_old_death, p_rej])
        r2 = np.random.uniform(0,1)
        event_index = np.searchsorted(np.cumsum(probability_vector), r2)

        #Picking a population which this event acts on
        molecule_index = int(np.random.uniform(0,1)*nucleoid_state[event_to_population[event_index]])

        #Updating the nucleoid state based on which event occured
        nucleoid_state += step_matrix[event_index].astype(np.int64).flatten()

        #tracking the SFS
        #non preferential rep
        if event_index == 0: 
            mut = np.random.binomial(1,mutation_probability)

            young_mutants.append(replicating_mutants[molecule_index]*1)
            young_mutants.append(replicating_mutants[molecule_index]*1)
            replicating_mutants.pop(molecule_index)
            if mut:
                young_mutants[-1].append(mutant_number)
                mutant_number += 1
        
        #preferential rep
        elif event_index == 1:
            mut = np.random.binomial(1,mutation_probability)
            young_mutants.append(replicating_mutants[molecule_index]*1)
            if mut:
                k = np.random.binomial(1, 0.5)
                if k:
                    young_mutants[-1].append(mutant_number)
                else:
                    replicating_mutants[molecule_index].append(mutant_number)
                mutant_number += 1
        
        #birth
        elif event_index == 2:
            replicating_mutants.append(young_mutants[molecule_index])
            young_mutants.pop(molecule_index)

        #ageing
        elif event_index == 3:
            old_mutants.append(young_mutants[molecule_index])
            young_mutants.pop(molecule_index)

        #replicating death
        elif event_index == 4:
            replicating_mutants.pop(molecule_index)

        #young death
        elif event_index == 5:
            young_mutants.pop(molecule_index)

        #old death
        elif event_index == 6:
            old_mutants.pop(molecule_index)

        #rejuvination
        else:
            young_mutants.append(old_mutants[molecule_index])
            old_mutants.pop(molecule_index)

    
    #make this output mutation heteroplasmy time evolution. This will give me a sense of the dynamics
    #and whether they are correct. If it is correct, the number should behave as a simple random walk.
    #(ignoring the times each step time takes)
    current_replicating = nucleoid_state[0]
    current_young = nucleoid_state[1]
    current_old = nucleoid_state[2]
    cSFS = np.zeros(mutant_number)
    n = int(np.sum(nucleoid_state))

    indices = np.random.choice(current_replicating + current_young + current_old, n_coal, replace = False)
    #indices = np.random.choice(current_replicating + current_young, n_coal, replace = False)
        
    rep_indices = []
    young_indices = []
    old_indices = []
    for index in indices:
        if index < current_replicating:
            rep_indices.append(index)
        elif index - current_replicating < current_young:
            young_indices.append(index - current_replicating)
        else:
            old_indices.append(index - current_replicating - current_young)

    replicating_mutants_subindexed = [replicating_mutants[i] for i in rep_indices]
    young_mutants_subindexed = [young_mutants[i] for i in young_indices]
    old_mutants_subindexed = [old_mutants[i] for i in old_indices]
    for i in range(mutant_number):
        mutant_counter = 0
        for replicating_mutant in replicating_mutants_subindexed:
            mutant_counter += (i in replicating_mutant)
        for young_mutant in young_mutants_subindexed:
            mutant_counter += (i in young_mutant)
        for old_mutant in old_mutants_subindexed:
            mutant_counter += (i in old_mutant)

        cSFS[i] = mutant_counter/n_coal

    #print(mutant_number)
        
    return cSFS[cSFS != 0], replicating_mutants, young_mutants, old_mutants


def Z(k,j,n):
    prod = 1
    for i in range(k, n+1):
        if i != j:
            prod = prod*comb(i,2)/(comb(i,2) - comb(j,2))

    return prod

def expected_mutant_num(params, truncation_time, b, n = 200, l =500, N = 1000, mu_c=0):
    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params
    if mu_d_o != 0:
        mu_b = (mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r))

    else:
        mu_b = (mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r)

    f_y = 1/(1 + mu_b/(mu_d_r + p*mu_r) + mu_a/(mu_d_o + mu_rej))
    f_r = f_y*mu_b/(mu_d_r + p*mu_r)
    
    if N == 0:
        N_y = int((beta0 + beta1*l)*f_y)
        N_r = int((beta0 + beta1*l)*f_r)
    
    else:
        N_y = int(f_y*N)
        N_r = int(f_r*N)

    pi_y = 1/(1 + (1+p)*mu_b*mu_r/(mu_d_r+p*mu_r)**2 + mu_a*mu_rej/(mu_d_o + mu_rej)**2)

    if mu_c==0:
        coal = (pi_y/N_y)**2 * (mu_r + mu_d_r*p)/(mu_r*p + mu_d_r) * 2 * mu_r*N_r
    else:
        coal=mu_c

    # W = truncation_time*24*365/N**2 * 2 * mu_r*N_r
    # print(mu_r*N_r)
    # print(W)
    W = truncation_time*24*365 * coal

    if b < n:
        doub_sum = 0
        for k in range(2,n+1):
            sum = 0
            for j in range(k, n+1):
                sum += Z(k,j,n)*np.exp(-comb(j,2)*W)

            doub_sum += sum*comb(n-k, b-1)/comb(n-1,b)
        
        return (2/b)*(1-doub_sum)
    
    if b == n:
        sum = 0
        for j in range(2,n+1):
            sum += Z(2,j,n)*np.exp(-comb(j,2)*W)/comb(j,2)

        return W - 2*(1-1/n) + sum
    

def expected_SFS(params, truncation_time=2, n = 200, l = 500, N = 1000, mu_c = 0):
    SFS = np.zeros(n)
    for i in range(1, n+1):
        #print(i)
        SFS[i-1] = expected_mutant_num(params, truncation_time, i, n = n, l = l, N=N, mu_c=mu_c)
    
    return SFS/np.sum(SFS)


