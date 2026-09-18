import numpy as np
from .data_preprocessing import (all_assays_edu_number_chase_means, all_assays_edu_number_chase_err, all_assays_dna_number_chase_means,
                                 all_assays_dna_number_chase_err, all_assays_edu_proportion_chase_means, all_assays_edu_proportion_chase_err,
                                 validation_edu_number_chase_means, validation_edu_number_chase_err, validation_dna_number_chase_means,
                                 validation_dna_number_chase_err, validation_edu_proportion_chase_means, validation_edu_proportion_chase_err,
                                 training_edu_number_chase_means, training_edu_number_chase_err, training_dna_number_chase_means,
                                 training_dna_number_chase_err, training_edu_proportion_chase_means, training_edu_proportion_chase_err,
                                 all_assays_edu_number_pulse_means, all_assays_edu_number_pulse_err, all_assays_dna_number_pulse_means,
                                 all_assays_dna_number_pulse_err,
                                 validation_edu_number_pulse_means, validation_edu_number_pulse_err, validation_dna_number_pulse_means,
                                 validation_dna_number_pulse_err,
                                 training_edu_number_pulse_means, training_edu_number_pulse_err, training_dna_number_pulse_means,
                                 training_dna_number_pulse_err,
                                 all_assays_dna_number_1hr, all_assays_dna_number_3hr, all_assays_dna_number_7hr, all_assays_dna_number_24hr, 
                                 all_assays_dna_number_0dy, all_assays_dna_number_1dy, all_assays_dna_number_2dy, all_assays_dna_number_4dy,
                                 all_assays_mito_length_1hr, all_assays_mito_length_3hr,all_assays_mito_length_7hr,all_assays_mito_length_24hr,
                                 all_assays_mito_length_0dy, all_assays_mito_length_1dy, all_assays_mito_length_2dy, all_assays_mito_length_4dy,
                                 all_assays_edu_number_1hr, all_assays_edu_number_3hr, all_assays_edu_number_7hr, all_assays_edu_number_24hr, 
                                 all_assays_edu_number_0dy, all_assays_edu_number_1dy, all_assays_edu_number_2dy, all_assays_edu_number_4dy,
                                 all_assays_mito_lengths, all_assays_dna_numbers)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go

dirname = os.path.dirname(__file__)

def extended_log10(x):
    return np.log10(np.maximum(x, 10e-5))

def extended_log10_2(x):
    return np.log10(np.maximum(x, 10e-9))

#death modes: 0=combined, 1=equal death, 2=old death, 3=replicative death
#beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c, mu_b, sig
def extended_three_population_posteriors(all_params, accepted_params, all_chase_pulse_params = 0, accepted_chase_pulse_params = 0, all_chase_chase_params = 0, accepted_chase_chase_params = 0, mode = "pulse", dimensions = 1, MAP = False, death_mode = 0):
    if death_mode == 0:
        indices = np.ones(len(all_params)).astype(bool)
        accepted_indices = np.ones(len(accepted_params)).astype(bool)
    elif death_mode == 1:
        indices = (((all_params[:,3] > 0).astype(int) + (all_params[:,4] > 0).astype(int) + (all_params[:,5] > 0).astype(int)) == 3).astype(bool)
        accepted_indices = (((accepted_params[:,3] > 0).astype(int) + (accepted_params[:,4] > 0).astype(int) + (accepted_params[:,5] > 0).astype(int)) == 3).astype(bool)
    elif death_mode == 2:
        indices = (((all_params[:,3] == 0).astype(int) + (all_params[:,4] == 0).astype(int) + (all_params[:,5] > 0).astype(int)) == 3).astype(bool)
        accepted_indices = (((accepted_params[:,3] == 0).astype(int) + (accepted_params[:,4] == 0).astype(int) + (accepted_params[:,5] > 0).astype(int)) == 3).astype(bool)
    elif death_mode == 3:
        indices = (((all_params[:,3] > 0).astype(int) + (all_params[:,4] > 0).astype(int) + (all_params[:,5] == 0).astype(int)) == 3).astype(bool)
        accepted_indices = (((accepted_params[:,3] > 0).astype(int) + (accepted_params[:,4] > 0).astype(int) + (accepted_params[:,5] == 0).astype(int)) == 3).astype(bool)

    if dimensions == 1:
        if mode == "pulse":
            fig, ax = plt.subplots(2,5, figsize=(18,5))
            ax[0,0].hist(np.transpose(all_params[indices])[2], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params[accepted_indices])[2], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")
            ax[1,0].set_ylabel("Density")

            ax[0,1].hist(extended_log10(np.transpose(all_params[indices])[3]), density = True, label = "prior", bins=50)
            ax[0,1].hist(extended_log10(np.transpose(accepted_params[accepted_indices])[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_{d,r})$")

            ax[0,2].hist(extended_log10(np.transpose(all_params[indices])[4]), density = True, label = "prior", bins=50)
            ax[0,2].hist(extended_log10(np.transpose(accepted_params[accepted_indices])[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_{d,a})$")

            ax[0,3].hist(extended_log10(np.transpose(all_params[indices])[5]), density = True, label = "prior", bins=50)
            ax[0,3].hist(extended_log10(np.transpose(accepted_params[accepted_indices])[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,3].set_title("$log_{10}(\mu_{d,o})$")

            ax[0,4].hist(np.log10(np.transpose(all_params[indices])[6]), density = True, label = "prior", bins=50)
            ax[0,4].hist(np.log10(np.transpose(accepted_params[accepted_indices])[6]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,4].set_title("$log_{10}(\mu_a)$")

            # ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
            # ax[0,3].hist(np.log10(np.transpose(accepted_params)[4]/np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
            # ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            ax[1,0].hist(1/(np.transpose(all_params[indices])[7]), density = True, label = "prior", bins=50)
            ax[1,0].hist(1/(np.transpose(accepted_params[accepted_indices])[7]), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title("$1/\mu_r$")

            ax[1,1].hist(extended_log10_2(np.transpose(all_params[indices])[8]), density = True, label = "Prior", bins=50)
            ax[1,1].hist(extended_log10_2(np.transpose(accepted_params[accepted_indices])[8]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,1].set_title("$log_{10}(\mu_a')$")

            ax[1,2].hist(np.log10(np.transpose(all_params[indices])[9]), density = True, label = "Prior", bins=50)
            ax[1,2].hist(np.log10(np.transpose(accepted_params[accepted_indices])[9]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,2].set_title("$log_{10}(c)$")

            ax[1,3].hist(np.transpose(all_params[indices])[0], density = True, label = "prior", bins=50)
            ax[1,3].hist(np.transpose(accepted_params[accepted_indices])[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,3].set_title('$\\beta_0$')

            ax[1,4].hist(np.transpose(all_params[indices])[1], density = True, label = "Prior", bins=50)
            ax[1,4].hist(np.transpose(accepted_params[accepted_indices])[1], density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,4].set_title('$\\beta_1$')
            ax[1,4].legend()

            _ = ax[0,1].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[0,2].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[0,3].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[1,1].set_xticks([-8,-6,-4,-2, 0], ["$-\infty$",-6,-4,-2, 0])

            for j in range(2):
                for i in range(5):
                    ax[j,i].set_yticks([],[])

            

            
            plt.tight_layout()

            # - beta0 (float, beta0 >= 0)
            # - beta1 (float, beta1 >= 0)
            # - sigma (float, sigma > 0 (as defined in Supplementary Information Section 3.1.2, used in 3.3.1))
            # - p (float, 0 <= p <= 1)
            # - mu_d (float, mu_d >= 0)
            # - mu_b (float, mu_b >= 0)
            # - mu_r (float, mu_r >= 0)
            # - mu_rej (float, mu_rej >=0)
            # - mu_b^chase (float, mu_b^chase <= p^chase * mu_a^chase)
            # - p^chase (float, p^chase >= 0)
            # - mu_d^chase (float, mu_d^chase >= 0)
            # - mu_a^chase (float, mu_a^chase >= 0)
            # - mu_rej^chase (float, mu_rej^chase >= 0)
            # - c (float, c >= 0)
        if mode == "chase":
            if death_mode == 0:
                chase_indices = np.ones(len(all_chase_pulse_params)).astype(bool)
                chase_accepted_indices = np.ones(len(accepted_chase_pulse_params)).astype(bool)
            elif death_mode == 1:
                chase_indices = (((all_chase_pulse_params[:,3] > 0).astype(int) + (all_chase_pulse_params[:,4] > 0).astype(int) + (all_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
                chase_accepted_indices = (((accepted_chase_pulse_params[:,3] > 0).astype(int) + (accepted_chase_pulse_params[:,4] > 0).astype(int) + (accepted_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
            elif death_mode == 2:
                chase_indices = (((all_chase_pulse_params[:,3] == 0).astype(int) + (all_chase_pulse_params[:,4] == 0).astype(int) + (all_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
                chase_accepted_indices = (((accepted_chase_pulse_params[:,3] == 0).astype(int) + (accepted_chase_pulse_params[:,4] == 0).astype(int) + (accepted_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
            elif death_mode == 3:
                chase_indices = (((all_chase_pulse_params[:,3] > 0).astype(int) + (all_chase_pulse_params[:,4] > 0).astype(int) + (all_chase_pulse_params[:,5] == 0).astype(int)) == 3).astype(bool)
                chase_accepted_indices = (((accepted_chase_pulse_params[:,3] > 0).astype(int) + (accepted_chase_pulse_params[:,4] > 0).astype(int) + (accepted_chase_pulse_params[:,5] == 0).astype(int)) == 3).astype(bool)
            # - beta0 (float, beta0 >= 0)
            # - beta1 (float, beta1 >= 0)
            # - p (float, 0 <= p <= 1)
            # - mu_d (float, mu_d >= 0)
            # - mu_b (float, mu_b >= 0)
            # - mu_r (float, mu_r >= 0)
            # - c (float, c >= 0)
            fig, ax = plt.subplots(2,6, figsize=(18,5))
            ax[0,0].hist(np.transpose(all_params[indices])[2], density = True, label = "original pulse data prior", color = "darkviolet")
            ax[0,0].hist(np.transpose(all_chase_pulse_params[chase_indices])[2], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[2], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")
            ax[1,0].set_ylabel("Density")

            ax[0,1].hist(extended_log10(np.transpose(all_params[indices])[3]), density = True, label = "original pulse data prior", bins=50, color = "darkviolet")
            ax[0,1].hist(extended_log10(np.transpose(all_chase_pulse_params[chase_indices])[3]), density = True, label = "prior", bins=50)
            ax[0,1].hist(extended_log10(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_{d,r})$")

            ax[0,2].hist(extended_log10(np.transpose(all_params[indices])[4]), density = True, label = "original pulse data prior", bins=50, color = "darkviolet")
            ax[0,2].hist(extended_log10(np.transpose(all_chase_pulse_params[chase_indices])[4]), density = True, label = "prior", bins=50)
            ax[0,2].hist(extended_log10(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_{d,a})$")

            ax[0,3].hist(extended_log10(np.transpose(all_params[indices])[5]), density = True, label = "prior", bins=50, color = "darkviolet")
            ax[0,3].hist(extended_log10(np.transpose(all_chase_pulse_params[chase_indices])[5]), density = True, label = "prior", bins=50)
            ax[0,3].hist(extended_log10(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,3].set_title("$log_{10}(\mu_{d,i})$")

            ax[0,4].hist(np.log10(np.transpose(all_params[indices])[6]), density = True, label = "prior", bins=50, color = "darkviolet")
            ax[0,4].hist(np.log10(np.transpose(all_chase_pulse_params[chase_indices])[6]), density = True, label = "prior", bins=50)
            ax[0,4].hist(np.log10(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[6]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,4].set_title("$log_{10}(\mu_a)$")

            ax[0,5].hist(np.array([]), label = "Original pulse \n data prior", color = "darkviolet")
            ax[0,5].hist(np.array([]), label = "Prior")
            ax[0,5].hist(np.array([]), label = "Posterior", color = "orange")
            ax[0,5].legend()
            ax[0,5].spines['top'].set_visible(False)
            ax[0,5].spines['right'].set_visible(False)
            ax[0,5].spines['bottom'].set_visible(False)
            ax[0,5].spines['left'].set_visible(False)
            ax[0,5].set_xticks([],[])
            ax[0,5].set_yticks([],[])

            # ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
            # ax[0,3].hist(np.log10(np.transpose(accepted_params)[4]/np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
            # ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            ax[1,0].hist(1/(np.transpose(all_params[indices])[7]), density = True, label = "prior", bins=50, color = "darkviolet")
            ax[1,0].hist(1/(np.transpose(all_chase_pulse_params[chase_indices])[7]), density = True, label = "prior", bins=50)
            ax[1,0].hist(1/(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[7]), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title("$1/\mu_r$")

            ax[1,1].hist(extended_log10_2(np.transpose(all_params[indices])[8]), density = True, label = "Prior", bins=50, color = "darkviolet")
            ax[1,1].hist(extended_log10_2(np.transpose(all_chase_pulse_params[chase_indices])[8]), density = True, label = "Prior", bins=50)
            ax[1,1].hist(extended_log10_2(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[8]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,1].set_title("$log_{10}(\mu_a')$")

            ax[1,2].hist(np.log10(np.transpose(all_params[indices])[9]), density = True, label = "Prior", bins=50, color = "darkviolet")
            ax[1,2].hist(np.log10(np.transpose(all_chase_pulse_params[chase_indices])[9]), density = True, label = "Prior", bins=50)
            ax[1,2].hist(np.log10(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[9]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,2].set_title("$log_{10}(c)$")

            ax[1,3].hist(np.transpose(all_chase_pulse_params[chase_indices])[0], density = True, label = "prior", bins=50)
            ax[1,3].hist(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,3].set_title('$\\beta_0$')

            ax[1,4].hist(np.transpose(all_chase_pulse_params[chase_indices])[1], density = True, label = "Prior", bins=50)
            ax[1,4].hist(np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[1], density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,4].set_title('$\\beta_1$')
            #ax[1,4].legend()

            ax[1,5].hist(np.log10(np.transpose(all_chase_chase_params[chase_indices])[5]), density = True, label = "Prior", bins=50)
            ax[1,5].hist(np.log10(np.transpose(accepted_chase_chase_params[chase_accepted_indices])[5]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,5].set_title("$log_{10}(\mu_{d,i}^{chase})$")
            #ax[1,5].legend()

            #_ = ax[1,2].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])

            for j in range(2):
                for i in range(5):
                    ax[j,i].set_yticks([],[])

            _ = ax[0,1].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[0,2].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[0,3].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[1,1].set_xticks([-8,-6,-4,-2, 0], ["$-\infty$",-6,-4,-2, 0])
            
            plt.tight_layout()
    if dimensions == 2:
        if mode == "pulse":
            print("pulse plot")
            fig, ax = plt.subplots(8,8, figsize=(18,18))
            #beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c, mu_b, sig
            funcs = [np.log10, np.log10, lambda x: x, extended_log10, extended_log10, extended_log10, np.log10, lambda x: 1/x, extended_log10, np.log10]
            for i in range(8):
                for j in range(8):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        a = ax[j][i].scatter(funcs[i+2](np.transpose(all_params[indices])[i+2][:1000]), funcs[j+2](np.transpose(all_params[indices])[j+2][:1000]), label = "Prior")
                        b = ax[j][i].scatter(funcs[i+2](np.transpose(accepted_params[accepted_indices])[i+2]), funcs[j+2](np.transpose(accepted_params[accepted_indices])[j+2]), color = "orange", label = "Posterior")
                        if type(MAP) != bool:
                            ax[j,i].scatter(funcs[i+2](MAP[i+2]), funcs[j+2](MAP[j+2]), color = "black", marker="x")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=7 and i !=7) or (j!=6 and i ==7):
                        ax[j,i].set_xticks([],[])

        
            ax[2,2].scatter(np.array([]), np.array([]),label = "Prior")
            ax[2,2].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[2,2].legend(loc="center")

            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_{d,r})$')
            ax[0,1].set_title('$log_{10}(\mu_{d,r})$')
            ax[2,0].set_ylabel('$log_{10}(\mu_{d,a})$')
            ax[0,2].set_title('$log_{10}(\mu_{d,a})$')
            ax[3,0].set_ylabel('$log_{10}(\mu_{d,i})$')
            ax[0,3].set_title('$log_{10}(\mu_{d,i})$')
            ax[4,0].set_ylabel("$log_{10}(\mu_a)$")
            ax[0,4].set_title("$log_{10}(\mu_a)$")
            ax[5,0].set_ylabel('$1/\mu_r$')
            ax[0,5].set_title('$1/\mu_r$')
            ax[6,0].set_ylabel('$log_{10}(\mu_a\')$')
            ax[0,6].set_title('$log_{10}(\mu_a\')$')
            ax[7,0].set_ylabel('$log_{10}(c)$')
            ax[0,7].set_title('$log_{10}(c)$')

            _ = ax[1,0].set_yticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[7,1].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[2,0].set_yticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[7,2].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[3,0].set_yticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[7,3].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[-2,0].set_yticks([-8,-6,-4,-2, 0], ["$-\infty$",-6,-4,-2, 0])
            _ = ax[7,-2].set_xticks([-8, -6, -4 ,-2, 0], ["$-\infty$",-6, -4, -2, 0])


        #beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_b_pulse_pert, true_birth_pert, mu_rej, mu_b_chase, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_a_chase, mu_rej_chase, c_pert
        if mode == "chase":
            if death_mode == 0:
                chase_indices = np.ones(len(all_chase_pulse_params)).astype(bool)
                chase_accepted_indices = np.ones(len(accepted_chase_pulse_params)).astype(bool)
            elif death_mode == 1:
                chase_indices = (((all_chase_pulse_params[:,3] > 0).astype(int) + (all_chase_pulse_params[:,4] > 0).astype(int) + (all_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
                chase_accepted_indices = (((accepted_chase_pulse_params[:,3] > 0).astype(int) + (accepted_chase_pulse_params[:,4] > 0).astype(int) + (accepted_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
            elif death_mode == 2:
                chase_indices = (((all_chase_pulse_params[:,3] == 0).astype(int) + (all_chase_pulse_params[:,4] == 0).astype(int) + (all_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
                chase_accepted_indices = (((accepted_chase_pulse_params[:,3] == 0).astype(int) + (accepted_chase_pulse_params[:,4] == 0).astype(int) + (accepted_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
            elif death_mode == 3:
                chase_indices = (((all_chase_pulse_params[:,3] > 0).astype(int) + (all_chase_pulse_params[:,4] > 0).astype(int) + (all_chase_pulse_params[:,5] == 0).astype(int)) == 3).astype(bool)
                chase_accepted_indices = (((accepted_chase_pulse_params[:,3] > 0).astype(int) + (accepted_chase_pulse_params[:,4] > 0).astype(int) + (accepted_chase_pulse_params[:,5] == 0).astype(int)) == 3).astype(bool)
            print("chase")

            fig, ax = plt.subplots(9,9, figsize=(18,18))
            #beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c, mu_b, sig
            funcs = [np.log10, np.log10, lambda x: x, extended_log10, extended_log10, extended_log10, np.log10, lambda x: 1/x, extended_log10_2, np.log10, np.log10]
            for i in range(8):
                for j in range(8):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        a = ax[j][i].scatter(funcs[i+2](np.transpose(all_chase_pulse_params[chase_indices])[i+2][:1000]), funcs[j+2](np.transpose(all_chase_pulse_params[chase_indices])[j+2][:1000]), label = "Prior")
                        b = ax[j][i].scatter(funcs[i+2](np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[i+2]), funcs[j+2](np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[j+2]), color = "orange", label = "Posterior")
                        if type(MAP) != bool:
                            ax[j,i].scatter(funcs[i+2](MAP[i+2]), funcs[j+2](MAP[j+2]), color = "black", marker="x")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=8 and i !=8) or (j!=7 and i ==8):
                        ax[j,i].set_xticks([],[])

            for i in range(8):
                ax[-1,i].scatter(funcs[i+2](np.transpose(all_chase_pulse_params[chase_indices])[i+2][:1000]), funcs[-1](np.transpose(all_chase_chase_params[chase_indices])[5][:1000]), label = "Prior")
                ax[-1,i].scatter(funcs[i+2](np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[i+2]), funcs[-1](np.transpose(accepted_chase_chase_params[chase_accepted_indices])[5]), color = "orange", label = "Posterior")
                if i != 0:
                    ax[-1,i].set_yticks([],[])

                ax[i,-1].scatter(funcs[-1](np.transpose(all_chase_chase_params[chase_indices])[5][:1000]), funcs[i+2](np.transpose(all_chase_pulse_params[chase_indices])[i+2][:1000]), label = "Prior")
                ax[i,-1].scatter(funcs[-1](np.transpose(accepted_chase_chase_params[chase_accepted_indices])[5]), funcs[i+2](np.transpose(accepted_chase_pulse_params[chase_accepted_indices])[i+2]), color = "orange", label = "Posterior")
                if i != 7:
                    ax[i,-1].set_xticks([],[])
                ax[i,-1].set_yticks([],[])
        
            ax[2,2].scatter(np.array([]), np.array([]),label = "Prior")
            ax[2,2].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[2,2].legend(loc="center")

            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_{d,r})$')
            ax[0,1].set_title('$log_{10}(\mu_{d,r})$')
            ax[2,0].set_ylabel('$log_{10}(\mu_{d,a})$')
            ax[0,2].set_title('$log_{10}(\mu_{d,a})$')
            ax[3,0].set_ylabel('$log_{10}(\mu_{d,i})$')
            ax[0,3].set_title('$log_{10}(\mu_{d,i})$')
            ax[4,0].set_ylabel("$log_{10}(\mu_a)$")
            ax[0,4].set_title("$log_{10}(\mu_a)$")
            ax[5,0].set_ylabel('$1/\mu_r$')
            ax[0,5].set_title('$1/\mu_r$')
            ax[6,0].set_ylabel('$log_{10}(\mu_a\')$')
            ax[0,6].set_title('$log_{10}(\mu_a\')$')
            ax[7,0].set_ylabel('$log_{10}(c)$')
            ax[0,7].set_title('$log_{10}(c)$')
            ax[8,0].set_ylabel('$log_{10}(\mu_{d,i}^{chase})$')
            ax[0,8].set_title('$log_{10}(\mu_{d,i}^{chase})$')

            ax[8,8].spines['top'].set_visible(False)
            ax[8,8].spines['right'].set_visible(False)
            ax[8,8].spines['bottom'].set_visible(False)
            ax[8,8].spines['left'].set_visible(False)
            ax[8,8].set_xticks([],[])
            ax[8,8].set_yticks([],[])
            _ = ax[1,0].set_yticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[8,1].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[2,0].set_yticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[8,2].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[3,0].set_yticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[8,3].set_xticks([-4,-2, -1,0], ["$-\infty$",-2, -1,0])
            _ = ax[-3,0].set_yticks([-8,-6,-4,-2, 0], ["$-\infty$",-6,-4,-2, 0])
            _ = ax[8,-3].set_xticks([-8, -6, -4 ,-2, 0], ["$-\infty$",-6, -4, -2, 0])

def mu_b_posterior(params):
    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params.transpose()
    mu_b = np.zeros(len(beta0))
    mu_b[mu_d_o != 0] = ((mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r)))[mu_d_o != 0]
    mu_b[mu_d_o == 0] = ((mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r))[mu_d_o == 0]

    fig, ax = plt.subplots(2,5, figsize = (15,5))
    ax[0,0].scatter(params.transpose()[2], np.log10(mu_b), c = "orange")
    ax[0,1].scatter(extended_log10(params.transpose()[3]), np.log10(mu_b), c = "orange")
    ax[0,2].scatter(extended_log10(params.transpose()[4]), np.log10(mu_b), c = "orange")
    ax[0,3].scatter(extended_log10(params.transpose()[5]), np.log10(mu_b), c = "orange")
    ax[0,4].scatter(extended_log10(params.transpose()[6]), np.log10(mu_b), c = "orange")
    ax[1,0].scatter(1/(params.transpose()[7]), np.log10(mu_b), c = "orange")
    ax[1,1].scatter(extended_log10_2(params.transpose()[8]), np.log10(mu_b), c = "orange")
    ax[1,2].scatter(extended_log10(params.transpose()[9]), np.log10(mu_b), c = "orange")

    ax[1,4].spines['top'].set_visible(False)
    ax[1,4].spines['right'].set_visible(False)
    ax[1,4].spines['bottom'].set_visible(False)
    ax[1,4].spines['left'].set_visible(False)
    ax[1,4].set_xticks([],[])
    ax[1,4].set_yticks([],[])
    ax[1,3].spines['top'].set_visible(False)
    ax[1,3].spines['right'].set_visible(False)
    ax[1,3].spines['bottom'].set_visible(False)
    ax[1,3].spines['left'].set_visible(False)
    ax[1,3].set_xticks([],[])
    ax[1,3].set_yticks([],[])

    ax[0,0].set_ylabel("$\\log_{10}(\\mu_b)$")
    ax[1,0].set_ylabel("$\\log_{10}(\\mu_b)$")

    ax[0,0].set_xlabel("$p$")
    ax[0,1].set_xlabel("$\\log_{10}(\\mu_{d,r})$")
    ax[0,2].set_xlabel("$\\log_{10}(\\mu_{d,a})$")
    ax[0,3].set_xlabel("$\\log_{10}(\\mu_{d,i})$")
    ax[0,4].set_xlabel("$\\log_{10}(\\mu_a)$")
    ax[1,0].set_xlabel("$1/\\mu_r$")
    ax[1,1].set_xlabel("$\\log_{10}(\\mu_a')$")
    ax[1,2].set_xlabel("$\\log_{10}(c)$")

    _ = ax[0,1].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,2].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,3].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[1,1].set_xticks([-8,-6,-4,-2, 0], ["$-\\infty$",-6,-4,-2, 0])

    plt.tight_layout()

def death_mode_posterior(all_chase_pulse_params, accepted_chase_pulse_params):

    chase_indices_1 = (((all_chase_pulse_params[:,3] > 0).astype(int) + (all_chase_pulse_params[:,4] > 0).astype(int) + (all_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
    chase_accepted_indices_1 = (((accepted_chase_pulse_params[:,3] > 0).astype(int) + (accepted_chase_pulse_params[:,4] > 0).astype(int) + (accepted_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)

    chase_indices_2 = (((all_chase_pulse_params[:,3] == 0).astype(int) + (all_chase_pulse_params[:,4] == 0).astype(int) + (all_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)
    chase_accepted_indices_2 = (((accepted_chase_pulse_params[:,3] == 0).astype(int) + (accepted_chase_pulse_params[:,4] == 0).astype(int) + (accepted_chase_pulse_params[:,5] > 0).astype(int)) == 3).astype(bool)

    chase_indices_3 = (((all_chase_pulse_params[:,3] > 0).astype(int) + (all_chase_pulse_params[:,4] > 0).astype(int) + (all_chase_pulse_params[:,5] == 0).astype(int)) == 3).astype(bool)
    chase_accepted_indices_3 = (((accepted_chase_pulse_params[:,3] > 0).astype(int) + (accepted_chase_pulse_params[:,4] > 0).astype(int) + (accepted_chase_pulse_params[:,5] == 0).astype(int)) == 3).astype(bool)

    equal_death_prior = np.sum(chase_indices_1 > 0)
    old_death_prior = np.sum(chase_indices_2 > 0)
    replicative_death_prior = np.sum(chase_indices_3 > 0)
    equal_death_post = np.sum(chase_accepted_indices_1 > 0)/500
    old_death_post = np.sum(chase_accepted_indices_2 > 0)/500
    replicative_death_post = np.sum(chase_accepted_indices_3 > 0)/500
    plt.figure(0,figsize = (5,4))
    plt.bar([1,2,3], [equal_death_post, old_death_post, replicative_death_post], color = "orange")
    plt.xticks([1,2,3],["Equiprobable \n death", "Inactive \n death", "Active \n death"])
    plt.ylabel("Probability")
    plt.title("Degradation regime posterior")


def population_proportions(params):
    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params.transpose()
    mu_b = np.zeros(len(beta0))
    mu_b[mu_d_o != 0] = ((mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r)))[mu_d_o != 0]
    mu_b[mu_d_o == 0] = ((mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r))[mu_d_o == 0]

    f_y = 1/(1 + mu_b/(mu_d_r + p*mu_r) + mu_a/(mu_d_o + mu_rej))
    f_r = (mu_b/(mu_d_r + p*mu_r))*1/(1 + mu_b/(mu_d_r + p*mu_r) + mu_a/(mu_d_o + mu_rej))
    f_o = 1-f_r-f_y

    replicating_values = np.arange(0,0.2, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(f_r.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicating_values.reshape(-1,1))
    replicating_kde = np.exp(log_dens1)

    replicative_values = np.arange(0,0.8, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(f_y.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicative_values.reshape(-1,1))
    replicative_kde = np.exp(log_dens1)

    old_values = np.arange(0.4,1, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(f_o.reshape(-1,1))
    log_dens1 = kde1.score_samples(old_values.reshape(-1,1))
    old_kde = np.exp(log_dens1)

    plt.figure(figsize = (6,4))
    plt.fill_between(replicating_values,replicating_kde, color="y", alpha = 0.7, label = "Replicating")
    plt.fill_between(replicative_values,replicative_kde, color="#118002", alpha = 0.7, label = "Active")
    plt.fill_between(old_values,old_kde, color="grey", alpha = 0.7, label = "Inactive")
    plt.xlabel("Fraction")
    plt.ylabel("Probability density")
    plt.ylim(0,17)
    plt.legend()
    plt.tight_layout()


def parameter_coalescent_correlations(params):
    beta0, beta1, p, mu_d_r, mu_d_y, mu_d_o, mu_a, mu_r, mu_rej, c = params.transpose()
    mu_b = np.zeros(len(beta0))
    mu_b[mu_d_o != 0] = ((mu_d_r+p*mu_r)*(mu_d_y*mu_d_o + mu_d_y*mu_rej+mu_d_o*mu_a)/((mu_d_o+mu_rej)*(mu_r-mu_d_r)))[mu_d_o != 0]
    mu_b[mu_d_o == 0] = ((mu_d_r+p*mu_r)*mu_d_y/(mu_r-mu_d_r))[mu_d_o == 0]

    f_y = 1/(1 + mu_b/(mu_d_r + p*mu_r) + mu_a/(mu_d_o + mu_rej))
    f_r = (mu_b/(mu_d_r + p*mu_r))*1/(1 + mu_b/(mu_d_r + p*mu_r) + mu_a/(mu_d_o + mu_rej))
    f_o = 1-f_r-f_y
    pi_y = 1/(1 + (1+p)*mu_b*mu_r/(mu_d_r+p*mu_r)**2 + mu_a*mu_rej/(mu_d_o + mu_rej)**2)
    m = mu_rej/(mu_d_o+mu_rej)
    mu_coal =  2 * (pi_y/f_y)**2 * f_r * (mu_r + mu_d_r*p)/(mu_r*p + mu_d_r) * mu_r

    fig, ax = plt.subplots(1,5, figsize = (22/1.3,4/1.3))
    ax[0].scatter(m, np.log10(mu_coal))
    ax[1].scatter(f_y, np.log10(mu_coal), c = m)
    ax[2].scatter(np.log10(f_r*mu_r),np.log10(mu_coal), c = m)
    ax[3].scatter((p*mu_d_r+mu_r)/(mu_d_r+p*mu_r),np.log10(mu_coal), c = m)
    sc1 = ax[4].scatter(((1+p)*mu_r)/(mu_d_r+p*mu_r),np.log10(mu_coal), c = m)


    # 3. Create the colorbar by passing the mappable object and target axis
    # If you want it next to the 3rd plot:
    fig.colorbar(sc1, ax=ax[4], label = "$m_i$")

    ax[0].set_ylabel("$N\\mu_{coal}$")
    ax[0].set_xlabel("$m_i$")
    ax[1].set_xlabel("$f_a$")
    ax[2].set_xlabel("$\\mu_{turn}$ per-capita")
    ax[3].set_xlabel("$\\mathcal{F}_p$")
    ax[4].set_xlabel("$m_r$")

    for i in range(5):
        ax[i].set_yticks([],[])

    # Optional: Prevents labels and colorbars from overlapping
    #plt.tight_layout() 
    plt.show()

    fig, ax = plt.subplots(2,5, figsize = (15,6))
    ax[0,0].scatter(params.transpose()[2], np.log10(mu_coal), c = m)
    ax[0,1].scatter(extended_log10(params.transpose()[3]), np.log10(mu_coal), c = m)
    ax[0,2].scatter(extended_log10(params.transpose()[4]), np.log10(mu_coal), c = m)
    ax[0,3].scatter(extended_log10(params.transpose()[5]), np.log10(mu_coal), c = m)
    ax[0,4].scatter(extended_log10(params.transpose()[6]), np.log10(mu_coal), c = m)
    ax[1,0].scatter(1/(params.transpose()[7]), np.log10(mu_coal), c = m)
    ax[1,1].scatter(extended_log10_2(params.transpose()[8]), np.log10(mu_coal), c = m)
    ax[1,2].scatter(extended_log10(params.transpose()[9]), np.log10(mu_coal), c = m)
    sc = ax[1,3].scatter(extended_log10(mu_b), np.log10(mu_coal), c = m)

    ax[0,0].set_xlabel("$p$")
    ax[0,1].set_xlabel("$\\log_{10}(\\mu_{d,r})$")
    ax[0,2].set_xlabel("$\\log_{10}(\\mu_{d,a})$")
    ax[0,3].set_xlabel("$\\log_{10}(\\mu_{d,i})$")
    ax[0,4].set_xlabel("$\\log_{10}(\\mu_a)$")
    ax[1,0].set_xlabel("$1/\\mu_r$")
    ax[1,1].set_xlabel("$\\log_{10}(\\mu_a')$")
    ax[1,2].set_xlabel("$\\log_{10}(c)$")
    ax[1,3].set_xlabel("$\\log_{10}(\\mu_b)$")

    # 3. Create the colorbar by passing the mappable object and target axis
    # If you want it next to the 3rd plot:
    fig.colorbar(sc, ax=ax[1,4], label = "$m_i$")

    ax[0,0].set_ylabel("$N\\mu_{coal}$")
    ax[1,0].set_ylabel("$N\\mu_{coal}$")

    _ = ax[0,1].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,2].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,3].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[1,1].set_xticks([-8,-6,-4,-2, 0], ["$-\\infty$",-6,-4,-2, 0])

    ax[1,4].spines['top'].set_visible(False)
    ax[1,4].spines['right'].set_visible(False)
    ax[1,4].spines['bottom'].set_visible(False)
    ax[1,4].spines['left'].set_visible(False)
    ax[1,4].set_xticks([],[])
    ax[1,4].set_yticks([],[])

    plt.tight_layout()

    fig, ax = plt.subplots(2,5, figsize = (15,6))
    ax[0,0].scatter(params.transpose()[2], f_y)
    ax[0,1].scatter(extended_log10(params.transpose()[3]), f_y)
    ax[0,2].scatter(extended_log10(params.transpose()[4]), f_y)
    ax[0,3].scatter(extended_log10(params.transpose()[5]), f_y)
    ax[0,4].scatter(extended_log10(params.transpose()[6]), f_y)
    ax[1,0].scatter(1/(params.transpose()[7]), f_y)
    ax[1,1].scatter(extended_log10_2(params.transpose()[8]), f_y)
    ax[1,2].scatter(extended_log10(params.transpose()[9]), f_y)
    ax[1,3].scatter(extended_log10(mu_b), f_y)

    ax[0,0].set_ylabel("$f_a$")
    ax[1,0].set_ylabel("$f_a$")

    _ = ax[0,1].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,2].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,3].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[1,1].set_xticks([-8,-6,-4,-2, 0], ["$-\\infty$",-6,-4,-2, 0])

    ax[1,4].spines['top'].set_visible(False)
    ax[1,4].spines['right'].set_visible(False)
    ax[1,4].spines['bottom'].set_visible(False)
    ax[1,4].spines['left'].set_visible(False)
    ax[1,4].set_xticks([],[])
    ax[1,4].set_yticks([],[])

    ax[0,0].set_xlabel("$p$")
    ax[0,1].set_xlabel("$\\log_{10}(\\mu_{d,r})$")
    ax[0,2].set_xlabel("$\\log_{10}(\\mu_{d,a})$")
    ax[0,3].set_xlabel("$\\log_{10}(\\mu_{d,i})$")
    ax[0,4].set_xlabel("$\\log_{10}(\\mu_a)$")
    ax[1,0].set_xlabel("$1/\\mu_r$")
    ax[1,1].set_xlabel("$\\log_{10}(\\mu_a')$")
    ax[1,2].set_xlabel("$\\log_{10}(c)$")
    ax[1,3].set_xlabel("$\\log_{10}(\\mu_b)$")

    plt.tight_layout()

    fig, ax = plt.subplots(2,5, figsize = (15,6))
    ax[0,0].scatter(params.transpose()[2], m)
    ax[0,1].scatter(extended_log10(params.transpose()[3]), m)
    ax[0,2].scatter(extended_log10(params.transpose()[4]), m)
    ax[0,3].scatter(extended_log10(params.transpose()[5]), m)
    ax[0,4].scatter(extended_log10(params.transpose()[6]), m)
    ax[1,0].scatter(1/(params.transpose()[7]), m)
    ax[1,1].scatter(extended_log10_2(params.transpose()[8]), m)
    ax[1,2].scatter(extended_log10(params.transpose()[9]), m)
    ax[1,3].scatter(extended_log10(mu_b), m)

    ax[0,0].set_ylabel("$m_i$")
    ax[1,0].set_ylabel("$m_i$")

    _ = ax[0,1].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,2].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[0,3].set_xticks([-4,-2, -1], ["$-\\infty$",-2, -1])
    _ = ax[1,1].set_xticks([-8,-6,-4,-2, 0], ["$-\\infty$",-6,-4,-2, 0])

    ax[1,4].spines['top'].set_visible(False)
    ax[1,4].spines['right'].set_visible(False)
    ax[1,4].spines['bottom'].set_visible(False)
    ax[1,4].spines['left'].set_visible(False)
    ax[1,4].set_xticks([],[])
    ax[1,4].set_yticks([],[])

    ax[0,0].set_xlabel("$p$")
    ax[0,1].set_xlabel("$\\log_{10}(\\mu_{d,r})$")
    ax[0,2].set_xlabel("$\\log_{10}(\\mu_{d,a})$")
    ax[0,3].set_xlabel("$\\log_{10}(\\mu_{d,i})$")
    ax[0,4].set_xlabel("$\\log_{10}(\\mu_a)$")
    ax[1,0].set_xlabel("$1/\\mu_r$")
    ax[1,1].set_xlabel("$\\log_{10}(\\mu_a')$")
    ax[1,2].set_xlabel("$\\log_{10}(c)$")
    ax[1,3].set_xlabel("$\\log_{10}(\\mu_b)$")

    plt.tight_layout()


def single_posterior_predictive_pulse(nucleoid_trajectories, edu_trajectories, peak1, variance_statistics, mode = "full"):

    fig, ax = plt.subplots(1,3, figsize = (18,4))

    #Trajectories
    pulse_trajectory_time = np.arange(0,24.25,0.25)
    pulse_times = np.array([1,3,7,24])

    if mode == "full":
        edu_means = all_assays_edu_number_pulse_means
        edu_err = all_assays_edu_number_pulse_err
        dna_means = all_assays_dna_number_pulse_means
        dna_err = all_assays_dna_number_pulse_err
    
    elif mode == "validation":
        edu_means = validation_edu_number_pulse_means
        edu_err = validation_edu_number_pulse_err
        dna_means = validation_dna_number_pulse_means
        dna_err = validation_dna_number_pulse_err

    elif mode == "training":
        edu_means = training_edu_number_pulse_means
        edu_err = training_edu_number_pulse_err
        dna_means = training_dna_number_pulse_means
        dna_err = training_dna_number_pulse_err


    for traj in nucleoid_trajectories[:-1]:
        ax[0].plot(pulse_trajectory_time, traj, c ="mediumaquamarine", alpha=0.2)
    ax[0].plot(pulse_trajectory_time, nucleoid_trajectories[-1], c ="mediumaquamarine", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")
    ax[0].plot(pulse_trajectory_time, np.mean(nucleoid_trajectories, axis=0), c="#118002", linewidth=3, label = "Nucleoid Number Posterior Predictive Mean")

    for traj in edu_trajectories[:-1]:
        ax[0].plot(pulse_trajectory_time, traj, c ="plum", alpha=0.2)
    ax[0].plot(pulse_trajectory_time, edu_trajectories[-1], c ="plum", alpha=0.2, label = "EdU Number Posterior Predictive Draw")
    ax[0].plot(pulse_trajectory_time, np.mean(edu_trajectories, axis=0), c="#800080", linewidth=3, label = "EdU Number Posterior Predictive Mean")

    ax[0].errorbar(pulse_times, edu_means, edu_err, color='#800080', marker='o', ls = 'none', ecolor = '#800080', label = "Edu Number Data Mean + 95% Confidence Interval")
    ax[0].errorbar(pulse_times, dna_means, dna_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Nucleoid Number Data Mean + 95% Confidence Interval")
    ax[0].set_xlabel("Time (h)")
    ax[1].set_yticks([],[])
    ax[2].set_yticks([],[])
    ax[0].set_ylabel("Nucleoid number")
    ax[0].set_title("Nucleoid trajectories")
    ax[1].set_title("Singly tagged proportion")
    ax[2].set_title("Control strength statistic")

    legend_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior predictive mean'),
                    Line2D([0], [0], color='grey', lw=1.5, alpha = 0.4, label='Posterior predictive draw'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='grey', markersize=8),
                    Patch(facecolor='#800080', edgecolor='#800080', label='EdU number'),
                    Patch(facecolor='#118002', edgecolor='#118002', label='Nucleoid number')]

    ax[0].legend(handles=legend_elements)

    ax[1].hist(peak1, alpha = 0.7, density = True, label = "Posterior predictive") 
    ax[1].axvline(0.33, color='k', linestyle='dashed', linewidth=1)
    ax[1].axvline(0.39, color='k', linestyle='dashed', linewidth=1)
    ax[1].axvline(0.29, color='k', linestyle='dashed', linewidth=1, label = "Observed proportion for 3 assays")
    ax[1].set_xlim(0,1)
    ax[1].set_ylabel("Probability density")
    ax[1].set_xlabel("24 hour singly tagged proportion")

    ax[1].legend()

    ax[2].hist(variance_statistics, alpha = 0.7, color = "tab:purple", density = True, label = "Posterior predictive") 
    ax[2].axvline(0, color='k', linestyle='dashed', linewidth=1, label = "Observed")
    ax[2].set_ylabel("Probability density")
    ax[2].set_xlabel("$S_{cs}$")
    ax[2].legend()

def single_posterior_predictive_chase(nucleoid_trajectories, edu_trajectories, edu_proportions, peak1_0day, peak1_4day, variance_statistics, mode = "full"):

    if mode == "full":
        edu_means = all_assays_edu_number_chase_means
        edu_err = all_assays_edu_number_chase_err
        dna_means = all_assays_dna_number_chase_means
        dna_err = all_assays_dna_number_chase_err
        edu_prop = all_assays_edu_proportion_chase_means
        edu_prop_err = all_assays_edu_proportion_chase_err
    
    elif mode == "validation":
        edu_means = validation_edu_number_chase_means
        edu_err = validation_edu_number_chase_err
        dna_means = validation_dna_number_chase_means
        dna_err = validation_dna_number_chase_err
        edu_prop = validation_edu_proportion_chase_means
        edu_prop_err = validation_edu_proportion_chase_err

    elif mode == "training":
        edu_means = training_edu_number_chase_means
        edu_err = training_edu_number_chase_err
        dna_means = training_dna_number_chase_means
        dna_err = training_dna_number_chase_err
        edu_prop = training_edu_proportion_chase_means
        edu_prop_err = training_edu_proportion_chase_err

    fig, ax = plt.subplots(2,3, figsize = (18,8))

    #Trajectories
    chase_trajectory_time = np.arange(0,4*24+0.25,0.25) + 24
    chase_times = np.array([0,24,48,96])+24

    for traj in nucleoid_trajectories[:-1]:
        ax[0,0].plot(chase_trajectory_time, traj, c ="mediumaquamarine", alpha=0.2)
    ax[0,0].plot(chase_trajectory_time, nucleoid_trajectories[-1], c ="mediumaquamarine", alpha=0.2, label = "Nucleoid number posterior predictive draw")
    ax[0,0].plot(chase_trajectory_time, np.mean(nucleoid_trajectories, axis=0), c="#118002", linewidth=3, label = "Nucleoid number posterior predictive mean")

    for traj in edu_trajectories[:-1]:
        ax[0,0].plot(chase_trajectory_time, traj, c ="plum", alpha=0.2)
    ax[0,0].plot(chase_trajectory_time, edu_trajectories[-1], c ="plum", alpha=0.2, label = "EdU number posterior predictive draw")
    ax[0,0].plot(chase_trajectory_time, np.mean(edu_trajectories, axis=0), c="#800080", linewidth=3, label = "EdU number posterior predictive pean")

    ax[0,0].errorbar(chase_times, edu_means, edu_err, color='#800080', marker='o', ls = 'none', ecolor = '#800080', label = "Edu Number Data Mean + 95% Confidence Interval")
    ax[0,0].errorbar(chase_times, dna_means, dna_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Nucleoid Number Data Mean + 95% Confidence Interval")
    
    #ax[0,0].set_xticks([0,24,48,72,96], [0,1,2,3,4])
    ax[0,0].set_xlabel("Time (h)")

    ax[0,0].set_ylabel("Nucleoid number")
    ax[0,0].set_title("Nucleoid trajectories")
    ax[0,1].set_title("EdU/mtDNA trajectories")
    ax[0,1].set_ylabel("EdU/mtDNA")
    ax[0,2].set_title("Singly tagged proportion (24 h)")
    ax[0,2].set_ylabel("Probability Density")
    ax[1,0].set_title("Singly tagged proportion (120 h)")
    ax[1,0].set_ylabel("Probability density")
    ax[1,1].set_title("Control strength statistic")
    ax[1,1].set_ylabel("Probability density")

    legend_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior predictive mean'),
                    Line2D([0], [0], color='grey', lw=1.5, alpha = 0.4, label='Posterior predictive draw'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='grey', markersize=8),
                    Patch(facecolor='#800080', edgecolor='#800080', label='EdU number'),
                    Patch(facecolor='#118002', edgecolor='#118002', label='Nucleoid number')]

    ax[0,0].legend(handles=legend_elements)

    #Trajectories
    chase_trajectory_time = np.arange(0,4*24+0.25,0.25) + 24
    chase_times = np.array([0,24,48,96])+24

    for traj in edu_proportions[:-1]:
        ax[0,1].plot(chase_trajectory_time, traj, c ="mediumaquamarine", alpha=0.2)
    ax[0,1].plot(chase_trajectory_time, edu_proportions[-1], c ="mediumaquamarine", alpha=0.2, label = "Nucleoid number posterior predictive draw")
    ax[0,1].plot(chase_trajectory_time, np.mean(edu_proportions, axis=0), c="forestgreen", linewidth=3, label = "Nucleoid number posterior predictive mean")

    ax[0,1].errorbar(chase_times, edu_prop, edu_prop_err, fmt='go', ls = 'none', ecolor = 'g', label = "Edu Number Data Mean + 95% Confidence Interval")

    #ax[0,1].set_xticks([0,24,48,72,96], [0,1,2,3,4])
    ax[0,1].set_ylim(0,1)
    ax[0,1].set_xlabel("Time (h)")


    legend_elements = [Line2D([0], [0], color='forestgreen', lw=3, label='Posterior predictive mean'),
                    Line2D([0], [0], color='forestgreen', lw=1.5, alpha = 0.4, label='Posterior predictive draw'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='g', markersize=8)]

    ax[0,1].legend(handles=legend_elements)


    ax[0,2].hist(peak1_0day, alpha = 0.7, density = True, label = "Posterior predictive") 
    ax[0,2].axvline(0.33, color='k', linestyle='dashed', linewidth=1)
    ax[0,2].axvline(0.39, color='k', linestyle='dashed', linewidth=1)
    ax[0,2].axvline(0.29, color='k', linestyle='dashed', linewidth=1, label = "Observed proportion for 3 assays")
    ax[0,2].set_xlim(0.2,0.7)
    ax[0,2].legend()

    ax[1,0].hist(peak1_4day, alpha = 0.7, density = True, label = "Posterior predictive") 
    ax[1,0].axvline(0.67, color='k', linestyle='dashed', linewidth=1)
    ax[1,0].axvline(0.58, color='k', linestyle='dashed', linewidth=1)
    ax[1,0].axvline(0.43, color='k', linestyle='dashed', linewidth=1, label = "Observed proportion for 3 assays")
    ax[1,0].set_xlim(0.2,0.7)
    ax[1,0].set_xlabel("Singly tagged proportion")
    ax[1,0].legend()

    ax[1,1].hist(variance_statistics, alpha = 0.7, color = "tab:purple", density = True, label = "Posterior predictive") 
    ax[1,1].axvline(0, color='k', linestyle='dashed', linewidth=1, label = "Observed")
    ax[1,1].set_xlabel("$S_{cs}$")
    ax[1,1].legend()

    ax[1,2].spines['top'].set_visible(False)
    ax[1,2].spines['right'].set_visible(False)
    ax[1,2].spines['bottom'].set_visible(False)
    ax[1,2].spines['left'].set_visible(False)
    ax[1,2].set_xticks([],[])
    ax[1,2].set_yticks([],[])

    plt.tight_layout()



def ABC_validation_distances(pulse_train_distances, pulse_val_distances, chase_train_distances, chase_val_distances, number_sims = 500):
    fig, ax = plt.subplots(1,2, figsize = (8,10))
    ax[0].scatter(np.random.uniform(-0.1,0.1,number_sims), pulse_train_distances)
    ax[0].scatter(np.random.uniform(0.9,1.1,number_sims), pulse_val_distances)
    ax[0].set_xlim(-0.5,1.5)
    ax[0].set_xticks([0,1],["Training", "Validation"], fontsize = 21)
    ax[0].set_ylabel("ABC distance")
    ax[0].set_title("Pulse")

    ax[1].scatter(np.random.uniform(-0.1,0.1,number_sims), chase_train_distances)
    ax[1].scatter(np.random.uniform(0.9,1.1,number_sims), chase_val_distances)
    ax[1].set_xlim(-0.5,1.5)
    ax[1].set_xticks([0,1],["Training", "Validation"], fontsize = 21)
    ax[1].set_ylabel("ABC distance")
    ax[1].set_title("Pulse-chase")
    plt.tight_layout()