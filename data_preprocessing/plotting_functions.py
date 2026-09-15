import numpy as np
from .data_preprocessing import (all_assays_edu_number_chase_means, all_assays_edu_number_chase_err, all_assays_dna_number_chase_means,
                                 all_assays_dna_number_chase_err, all_assays_edu_proportion_chase_means, all_assays_edu_proportion_chase_err,
                                 validation_edu_number_chase_means, validation_edu_number_chase_err, validation_dna_number_chase_means,
                                 validation_dna_number_chase_err, validation_edu_proportion_chase_means, validation_edu_proportion_chase_err,
                                 all_assays_edu_number_pulse_means, all_assays_edu_number_pulse_err, all_assays_dna_number_pulse_means,
                                 all_assays_dna_number_pulse_err,
                                 validation_edu_number_pulse_means, validation_edu_number_pulse_err, validation_dna_number_pulse_means,
                                 validation_dna_number_pulse_err,
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

def extended_log10(x):
    return np.log10(np.maximum(x, 10e-9))

#Functions to plot priors and posteriors
def one_population_posteriors(all_params, accepted_params, mode = "pulse", dimensions = 1):
    if dimensions == 1:
        if mode == "pulse":
            fig, ax = plt.subplots(1,5, figsize=(18,9/4))
            ax[0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[0].set_title("$\\beta_0$")
            ax[0].set_ylabel("Density")

            ax[1].hist(np.transpose(all_params)[1], density = True, label = "prior", bins=50)
            ax[1].hist(np.transpose(accepted_params)[1], density = True, label = "posterior", bins=50, color = "orange")
            ax[1].set_title("$\\beta_1$")

            ax[2].hist(np.log10(np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
            ax[2].hist(np.log10(np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
            ax[2].set_title("$log_{10}(\mu)$")

            ax[3].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "Prior", bins=50)
            ax[3].hist(np.log10(np.transpose(accepted_params)[3]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[3].set_title("$log_{10}(c)$")

            ax[4].hist(np.array([]), label = "Prior")
            ax[4].hist(np.array([]), label = "Posterior", color = "orange")
            ax[4].legend(loc= "center")

            for i in range(5):
                ax[i].set_yticks([],[])

            ax[4].set_xticks([],[])
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].spines['left'].set_visible(False)

    if dimensions == 2:
        if mode == "pulse":
            
            plt.figure(figsize = (3,3))
            plt.scatter(np.log10(np.transpose(all_params)[2][:1000]),np.log10(np.transpose(all_params)[3][:1000]), label = "Prior")
            plt.scatter(np.log10(np.transpose(accepted_params)[2]),np.log10(np.transpose(accepted_params)[3]), color="orange", label = "Posterior")
            plt.legend()
            plt.xlabel('$log_{10}(\mu)$')
            plt.ylabel('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("one_pop_pulse_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,4, figsize=(12,4.5))
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, np.log10, np.log10]
            for i in range(4):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    else:
                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[i][:1000]), funcs[j](np.transpose(all_params)[j][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[i]), funcs[j](np.transpose(accepted_params)[j]), color="orange", label="Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])

            
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax2[1,1].legend(loc="center")

            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('$log_{10}(\mu_d)$')
            ax2[0,3].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("one_pop_pulse_betas_2d", dpi=300)

def two_population_posteriors(all_params, accepted_params, previous_params = 0, mode = "pulse", dimensions = 1, p=False, prel = False):
    #beta0, beta1, diffusion_prob_pulse, mu_d_pulse, true_birth_param, c
    if dimensions == 1:
        if mode == "pulse":
            if prel:
                s=1
            else:
                s=0
            fig, ax = plt.subplots(2,4-s, figsize=(18,5))
            if not prel:
                ax[0,0].hist(np.transpose(all_params)[2], density = True, label = "prior", bins=50)
                ax[0,0].hist(np.transpose(accepted_params)[2], density = True, label = "posterior", bins=50, color = "orange")
                ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            ax[0,1-s].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50)
            ax[0,1-s].hist(np.log10(np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1-s].set_title("$log_{10}(\mu_d)$")

            # ax[0,2-s].hist(np.log10(np.transpose(all_params)[3]*np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
            # ax[0,2-s].hist(np.log10(np.transpose(accepted_params)[3]*np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
            # if prel:
            #     ax[0,2-s].set_title("$log_{10}(\mu_b)$ ($\mu_b = \mu_d$)")
            # else:
            #     ax[0,2-s].set_title("$log_{10}(\mu_b)$ ($\mu_b = p\mu_d$)")

            ax[0,2-s].hist(1/(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,2-s].hist(1/(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2-s].set_title("$1/\mu_r$")

            ax[0,3-s].hist(np.log10(np.transpose(all_params)[5]), density = True, label = "Prior", bins=50)
            ax[0,3-s].hist(np.log10(np.transpose(accepted_params)[5]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,3-s].set_title("$log_{10}(c)$")

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "Prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            ax[1,2].hist(np.array([]), label = "Prior")
            ax[1,2].hist(np.array([]), label = "Posterior", color = "orange")
            ax[1,2].legend(loc= "center")
            for j in range(2):
                for i in range(4-s):
                    ax[j,i].set_yticks([],[])

            for j in range(2-s):
                ax[1,j+2].set_xticks([],[])
                ax[1,j+2].spines['top'].set_visible(False)
                ax[1,j+2].spines['right'].set_visible(False)
                ax[1,j+2].spines['bottom'].set_visible(False)
                ax[1,j+2].spines['left'].set_visible(False)

            plt.tight_layout()


        if mode == "chase":
            #beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, c
            fig, ax = plt.subplots(2,5, figsize=(18,6))

            ax[0,0].hist(np.transpose(previous_params)[2], density = True, color = "darkviolet")
            ax[0,0].hist(np.transpose(all_params)[3], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params)[3], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            ax[0,1].hist(np.log10(np.transpose(previous_params)[3]), density = True, color = "darkviolet")
            ax[0,1].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,1].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_d)$")

            # #ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50, color = "green")
            # ax[0,2].hist(np.log10(np.transpose(all_params)[4]*np.transpose(all_params)[3]), density = True, label = "prior", bins=50)
            # ax[0,2].hist(np.log10(np.transpose(accepted_params)[4]*np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            # ax[0,2].set_title("$log_{10}(\mu_b)$ ($\mu_b = p\mu_d$)")

            #ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "Pulse Prior", bins=50, color = "green")
            ax[0,2].hist(1/(np.transpose(previous_params)[4]), density = True, color = "darkviolet")
            ax[0,2].hist(1/(np.transpose(all_params)[5]), density = True, label = "Pulse Posterior/Chase Prior", bins=50)
            ax[0,2].hist(1/(np.transpose(accepted_params)[5]), density = True, label = "Chase Posterior", bins=50, color = "orange")
            ax[0,2].set_title("$1/\mu_r$")

            #ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Pulse Fitting Prior", bins=50, color = "green")
            ax[0,3].hist(np.log10(np.transpose(previous_params)[5]), density = True, color = "darkviolet")
            ax[0,3].hist(np.log10(np.transpose(all_params)[9]), density = True, label = "Prior", bins=50)
            ax[0,3].hist(np.log10(np.transpose(accepted_params)[9]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,3].set_title("$log_{10}(c)$")

            ax[0,4].set_xticks([],[])
            ax[0,4].set_yticks([],[])
            ax[0,4].spines['top'].set_visible(False)
            ax[0,4].spines['right'].set_visible(False)
            ax[0,4].spines['bottom'].set_visible(False)
            ax[0,4].spines['left'].set_visible(False)
            ax[0,4].hist(np.array([]), label = "Prior")
            ax[0,4].hist(np.array([]), label = "Posterior", color = "orange")
            ax[0,4].hist(np.array([]), label = "Original pulse data prior", color = "darkviolet")
            ax[0,4].legend(loc="center")

            ax[1,2].hist(np.log10(np.transpose(all_params)[8]), density = True, label = "prior", bins=50)
            ax[1,2].hist(np.log10(np.transpose(accepted_params)[8]), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,2].set_title('$log_{10}(\mu_d^{chase})$')

            all_adjusted_mu_bs = np.transpose(all_params)[6].copy()
            all_adjusted_mu_bs[all_adjusted_mu_bs == 0] = 10e-8
            accepted_adjusted_mu_bs = np.transpose(accepted_params)[6].copy()
            accepted_adjusted_mu_bs[accepted_adjusted_mu_bs == 0] = 10e-8

            ax[1,3].hist(np.log10(all_adjusted_mu_bs), density = True, label = "prior", bins=50)
            ax[1,3].hist(np.log10(accepted_adjusted_mu_bs), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,3].set_title("$log_{10}(\mu_b^{chase})$")
            _ = ax[1,3].set_xticks([-7,-6,-5,-4,-3,-2], ["$-\infty$", -6,-5,-4,-3,-2])

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            if p:
                ax[1,4].hist(np.transpose(all_params)[7], density = True, label = "prior", bins=50)
                ax[1,4].hist(np.transpose(accepted_params)[7], density = True, label = "posterior", bins=50, color = "orange")
                ax[1,4].set_title("$p^{chase}$")
                ax[1,4].legend(loc="center")

            else:
                ax[1,4].set_xticks([],[])
                ax[1,4].set_yticks([],[])
                ax[1,4].spines['top'].set_visible(False)
                ax[1,4].spines['right'].set_visible(False)
                ax[1,4].spines['bottom'].set_visible(False)
                ax[1,4].spines['left'].set_visible(False)

            for j in range(2):
                for i in range(5):
                    ax[j,i].set_yticks([],[]) 

            plt.tight_layout()           

    if dimensions == 2:
        if mode == "pulse":
            if prel:
                s=1
                figs = (9,7)
            else:
                s=0
                figs = (18,12)
            fig, ax = plt.subplots(4-s,4-s, figsize=figs)
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, lambda x: 1/x, np.log10]
            for i in range(4-s):
                for j in range(4-s):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        ax[j][i].scatter(funcs[i+2+s](np.transpose(all_params)[i+2+s][:1000]), funcs[j+2+s](np.transpose(all_params)[j+2+s][:1000]), label = "Prior")
                        ax[j][i].scatter(funcs[i+2+s](np.transpose(accepted_params)[i+2+s]), funcs[j+2+s](np.transpose(accepted_params)[j+2+s]), color = "orange", label = "Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=3 and i !=3) or (j!=2 and i ==3):
                        ax[j,i].set_xticks([],[])
        
            ax[2-s,2-s].scatter(np.array([]), np.array([]),label = "Prior")
            ax[2-s,2-s].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[2-s,2-s].legend(loc="center")

            if not prel:
                ax[0,1].set_ylabel('p')
                ax[1,0].set_title('p')
                ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
                ax[0,1].set_title('$log_{10}(\mu_d)$')

            else:
                ax[0,1].set_ylabel('$log_{10}(\mu_d)$')
                ax[1,0].set_title('$log_{10}(\mu_d)$')

            ax[2-s,0].set_ylabel('$1/\mu_r$')
            ax[0,2-s].set_title('$1/\mu_r$')
            ax[3-s,0].set_ylabel('$log_{10}(c)$')
            ax[0,3-s].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("prel_two_pop_pulse_2d", dpi=300)

            if prel:
                figs = (18,4.5)
            else:
                figs = (18,4.5)

            fig2, ax2 = plt.subplots(2,6-s, figsize=figs)
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, lambda x: 1/x, np.log10]
            for ii in range(6-s):
                for j in range(2):
                    if ii > 1:
                        i = ii + s
                    else:
                        i=ii
                    if ii == j:
                        ax2[j,ii].spines['top'].set_visible(False)
                        ax2[j,ii].spines['right'].set_visible(False)
                        ax2[j,ii].spines['bottom'].set_visible(False)
                        ax2[j,ii].spines['left'].set_visible(False)
                    else:
                        ax2[j][ii].scatter(funcs[i](np.transpose(all_params)[i][:1000]), funcs[j](np.transpose(all_params)[j][:1000]), label = "Prior")
                        ax2[j][ii].scatter(funcs[i](np.transpose(accepted_params)[i]), funcs[j](np.transpose(accepted_params)[j]), color="orange", label="Posterior")
                
                    if (ii != 0 and j !=0) or (ii!=1 and j ==0):
                        ax2[j,ii].set_yticks([],[])

                    if (j !=1 and ii !=1) or (j==1 and ii ==1):
                        ax2[j,ii].set_xticks([],[])
            
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax2[1,1].legend(loc="center")

            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            if not prel:
                ax2[0,2].set_title('p')
            ax2[0,3-s].set_title('$log_{10}(\mu_d)$')
            ax2[0,4-s].set_title('$1/\mu_r$')
            ax2[0,5-s].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("prel_two_pop_pulse_betas_2d", dpi=300)

        #beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, true_birth_pert, mu_b_chase, diffusion_prob_pulse_pert, mu_d_pulse_pert, c_pert
        if mode == "chase":
            if p:
                k =7
            else:
                k=6

            fig, ax = plt.subplots(k,k, figsize=(18,12))
            #beta0, beta1, p, mu_d, mu_r, c
            if p:
                funcs = [lambda x: x, np.log10, lambda x: 1/x, extended_log10, lambda x: x, np.log10, np.log10]
                indexes = [3,4,5,6,7,8,9]
            else:
                funcs = [lambda x: x, np.log10, lambda x: 1/x, extended_log10, np.log10, np.log10]
                indexes = [3,4,5,6,8,9]
            for i in range(k):
                for j in range(k):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        ax[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=k-1 and i !=k-1) or (j!=k-2 and i ==k-1):
                        ax[j,i].set_xticks([],[])

            _ = ax[3,0].set_yticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax[k-1,3].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])

            ax[3,3].scatter(np.array([]), np.array([]),label = "Prior")
            ax[3,3].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[3,3].legend(loc="center")
                
            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
            ax[0,1].set_title('$log_{10}(\mu_d)$')
            ax[2,0].set_ylabel('$1/\mu_r$')
            ax[0,2].set_title('$1/\mu_r$')
            ax[3,0].set_ylabel('$log_{10}(\mu_b^{chase})$')
            ax[0,3].set_title('$log_{10}(\mu_b^{chase})$')
            if p:
                ax[4,0].set_ylabel('$p^{chase}$')
                ax[0,4].set_title('$p^{chase}$')

            ax[k-2,0].set_ylabel('$log_{10}(\mu_d^{chase})$')
            ax[0,k-2].set_title('$log_{10}(\mu_d^{chase})$')
            ax[k-1,0].set_ylabel('$log_{10}(c)$')
            ax[0,k-1].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("two_pop_chase_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,k+2, figsize=(18,4.5))
            if p:
                funcs = [lambda x: x, lambda x: x,lambda x: x,np.log10, lambda x: 1/x, extended_log10, lambda x: x, np.log10, np.log10]
                indexes = [0,1,3,4,5,6,7,8,9]
            else:
                funcs = [lambda x: x,lambda x: x,lambda x: x, np.log10, lambda x: 1/x, extended_log10, np.log10, np.log10]
                indexes = [0,1,3,4,5,6,8,9]
            for i in range(k+2):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    
                    else:
                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])
                        
            _ = ax2[1,5].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax2[1,1].legend(loc="center")
            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('p')
            ax2[0,3].set_title('$log_{10}(\mu_d)$')
            ax2[0,4].set_title('$1/\mu_r$')
            ax2[0,5].set_title('$log_{10}(\mu_b^{chase})$')
            if p:
                ax2[0,6].set_title('$p^{chase}$')
            ax2[0,k].set_title('$log_{10}(\mu_d^{chase})$')
            ax2[0,k+1].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("two_pop_chase_betas_2d", dpi=300)

            one_zero_prior = np.sum(np.transpose(all_params)[-4] == 0)
            none_zero_prior = len(all_params) - one_zero_prior
            prior_spikes = np.array([one_zero_prior, none_zero_prior])/len(all_params)

            one_zero_post = np.sum(np.transpose(accepted_params)[-4] == 0)
            none_zero_post = len(accepted_params) - one_zero_post
            post_spikes = np.array([one_zero_post, none_zero_post])/len(accepted_params)

            plt.figure(3, figsize = (3,3))
            plt.bar(np.array([1,2])-0.25/2,prior_spikes, 0.25, label = "Prior")
            plt.bar(np.array([1,2])+0.25/2,post_spikes, 0.25, label = "Posterior", color = "orange")
            plt.xticks(np.array([1,2]), np.array(["$\mu_b^{chase} = 0$",  "$\mu_b^{chase} \\neq 0$"]))
            plt.ylabel("Probability")
            plt.legend()
            plt.tight_layout()
            #plt.savefig("two_pop_chase_spikes", dpi=300)

def three_population_posteriors(all_params, accepted_params, previous_params = 0, mode = "pulse", dimensions = 1, MAP = False):
    if dimensions == 1:
        if mode == "pulse":
            fig, ax = plt.subplots(2,5, figsize=(18,5))
            ax[0,0].hist(np.transpose(all_params)[2], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params)[2], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50)
            ax[0,1].hist(np.log10(np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_d)$")

            ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,2].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_b)$")

            # ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
            # ax[0,3].hist(np.log10(np.transpose(accepted_params)[4]/np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
            # ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            ax[0,3].hist(1/(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
            ax[0,3].hist(1/(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,3].set_title("$1/\mu_r$")

            ax[0,4].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Prior", bins=50)
            ax[0,4].hist(np.log10(np.transpose(accepted_params)[6]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,4].set_title("$log_{10}(c)$")

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "Prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            ax[1,2].hist(np.array([]), label = "Prior")
            ax[1,2].hist(np.array([]), label = "Posterior", color = "orange")
            ax[1,2].legend(loc= "center")
            for j in range(2):
                for i in range(5):
                    ax[j,i].set_yticks([],[])

            for j in range(3):
                ax[1,j+2].set_xticks([],[])
                ax[1,j+2].spines['top'].set_visible(False)
                ax[1,j+2].spines['right'].set_visible(False)
                ax[1,j+2].spines['bottom'].set_visible(False)
                ax[1,j+2].spines['left'].set_visible(False)
            
            plt.tight_layout()


        if mode == "chase":
            # - beta0 (float, beta0 >= 0)
            # - beta1 (float, beta1 >= 0)
            # - p (float, 0 <= p <= 1)
            # - mu_d (float, mu_d >= 0)
            # - mu_b (float, mu_b >= 0)
            # - mu_r (float, mu_r >= 0)
            # - c (float, c >= 0)
            fig, ax = plt.subplots(2,5, figsize=(18,6))

            ax[0,0].hist(np.transpose(previous_params)[2], density = True, color = "darkviolet")
            ax[0,0].hist(np.transpose(all_params)[3], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params)[3], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            #ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50, color = "green")
            ax[0,1].hist(np.log10(np.transpose(previous_params)[3]), density = True, color = "darkviolet")
            ax[0,1].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,1].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_d)$")

            #ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50, color = "green")
            ax[0,2].hist(np.log10(np.transpose(previous_params)[4]), density = True, color = "darkviolet")
            ax[0,2].hist(np.log10(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
            ax[0,2].hist(np.log10(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_b)$")

            # #ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50, color = "green")
            # ax[0,3].hist(np.log10(np.transpose(all_params)[5]/np.transpose(all_params)[3]), density = True, label = "posterior", bins=50)
            # ax[0,3].hist(np.log10(np.transpose(accepted_params)[5]/np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            # ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            #ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "Pulse Prior", bins=50, color = "green")
            ax[0,3].hist(1/(np.transpose(previous_params)[5]), density = True, color = "darkviolet")
            ax[0,3].hist(1/(np.transpose(all_params)[6]), density = True, label = "Pulse Posterior/Chase Prior", bins=50)
            ax[0,3].hist(1/(np.transpose(accepted_params)[6]), density = True, label = "Chase Posterior", bins=50, color = "orange")
            ax[0,3].set_title("$1/\mu_r$")

            #ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Pulse Fitting Prior", bins=50, color = "green")
            ax[0,4].hist(np.log10(np.transpose(previous_params)[6]), density = True, color = "darkviolet")
            ax[0,4].hist(np.log10(np.transpose(all_params)[11]), density = True, label = "Prior", bins=50)
            ax[0,4].hist(np.log10(np.transpose(accepted_params)[11]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,4].set_title("$log_{10}(c)$")

            all_adjusted_mu_bs = np.transpose(all_params)[7].copy()
            all_adjusted_mu_bs[all_adjusted_mu_bs == 0] = 10e-8
            accepted_adjusted_mu_bs = np.transpose(accepted_params)[7].copy()
            accepted_adjusted_mu_bs[accepted_adjusted_mu_bs == 0] = 10e-8

            ax[1,2].hist(np.log10(all_adjusted_mu_bs), density = True, label = "prior", bins=50)
            ax[1,2].hist(np.log10(accepted_adjusted_mu_bs), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,2].set_title("$log_{10}(\mu_b^{chase})$")
            _ = ax[1,2].set_xticks([-7,-6,-5,-4,-3,-2], ["$-\infty$", -6,-5,-4,-3,-2])

            all_adjusted_mu_as = np.transpose(all_params)[10].copy()
            all_adjusted_mu_as[all_adjusted_mu_as == 0] = 10e-8
            accepted_adjusted_mu_as = np.transpose(accepted_params)[10].copy()
            accepted_adjusted_mu_as[accepted_adjusted_mu_as == 0] = 10e-8

            ax[1,3].hist(np.log10(all_adjusted_mu_as), density = True, label = "prior", bins=50)
            ax[1,3].hist(np.log10(accepted_adjusted_mu_as), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,3].set_title("$log_{10}(\mu_a^{chase})$")
            _ = ax[1,3].set_xticks([-7,-5,-2.5,0], ["$-\infty$", -5,-2.5,0])

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            ax[1,4].set_xticks([],[])
            ax[1,4].set_yticks([],[])
            ax[1,4].spines['top'].set_visible(False)
            ax[1,4].spines['right'].set_visible(False)
            ax[1,4].spines['bottom'].set_visible(False)
            ax[1,4].spines['left'].set_visible(False)

            for j in range(2):
                for i in range(5):
                    ax[j,i].set_yticks([],[])

            ax[1,4].hist(np.array([]), label = "Prior")
            ax[1,4].hist(np.array([]), label = "Posterior", color = "orange")
            ax[1,4].hist(np.array([]), label = "Original pulse data prior", color = "darkviolet")
            ax[1,4].legend(loc="center")
            plt.tight_layout()

    if dimensions == 2:
        if mode == "pulse":
            fig, ax = plt.subplots(5,5, figsize=(18,12))
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10]
            for i in range(5):
                for j in range(5):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        a = ax[j][i].scatter(funcs[i+2](np.transpose(all_params)[i+2][:1000]), funcs[j+2](np.transpose(all_params)[j+2][:1000]), label = "Prior")
                        b = ax[j][i].scatter(funcs[i+2](np.transpose(accepted_params)[i+2]), funcs[j+2](np.transpose(accepted_params)[j+2]), color = "orange", label = "Posterior")
                        if type(MAP) != bool:
                            ax[j,i].scatter(funcs[i+2](MAP[i+2]), funcs[j+2](MAP[j+2]), color = "black", marker="x")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=4 and i !=4) or (j!=3 and i ==4):
                        ax[j,i].set_xticks([],[])
        
            ax[2,2].scatter(np.array([]), np.array([]),label = "Prior")
            ax[2,2].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[2,2].legend(loc="center")

            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
            ax[0,1].set_title('$log_{10}(\mu_d)$')
            ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
            ax[0,2].set_title('$log_{10}(\mu_b)$')
            ax[3,0].set_ylabel('$1/\mu_r$')
            ax[0,3].set_title('$1/\mu_r$')
            ax[4,0].set_ylabel('$log_{10}(c)$')
            ax[0,4].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("three_pop_pulse_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,7, figsize=(18,4.5))
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10]
            for i in range(7):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    else:
                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[i][:1000]), funcs[j](np.transpose(all_params)[j][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[i]), funcs[j](np.transpose(accepted_params)[j]), color="orange", label="Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])
            
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax2[1,1].legend(loc="center")

            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('p')
            ax2[0,3].set_title('$log_{10}(\mu_d)$')
            ax2[0,4].set_title('$log_{10}(\mu_b)$')
            ax2[0,5].set_title('$1/\mu_r$')
            ax2[0,6].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("three_pop_pulse_betas_2d", dpi=300)

        #beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_b_pulse_pert, true_birth_pert, mu_b_chase, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_a_chase, c_pert
        if mode == "chase":
            
            fig, ax = plt.subplots(7,7, figsize=(18,13))

            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, np.log10, np.log10, lambda x: 1/x, extended_log10, extended_log10, np.log10]
            indexes = [3,4,5,6,7,10,11]
            for i in range(7):
                for j in range(7):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        ax[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=6 and i !=6) or (j!=5 and i ==6):
                        ax[j,i].set_xticks([],[])

            ax[3,3].scatter(np.array([]), np.array([]),label = "Prior")
            ax[3,3].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[3,3].legend(loc="center")
                
            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
            ax[0,1].set_title('$log_{10}(\mu_d)$')
            ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
            ax[0,2].set_title('$log_{10}(\mu_b)$')
            ax[3,0].set_ylabel('$1/\mu_r$')
            ax[0,3].set_title('$1/\mu_r$')
            ax[4,0].set_ylabel('$log_{10}(\mu_b^{chase})$')
            ax[0,4].set_title('$log_{10}(\mu_b^{chase})$')
            ax[5,0].set_ylabel('$log_{10}(\mu_a^{chase})$')
            ax[0,5].set_title('$log_{10}(\mu_a^{chase})$')
            ax[6,0].set_ylabel('$log_{10}(c)$')
            ax[0,6].set_title('$log_{10}(c)$')

            _ = ax[4,0].set_yticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax[6,4].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax[5,0].set_yticks([-8,-6,-4,-2,0], ["$-\infty$", -6,-4,-2,0])
            _ = ax[6,5].set_xticks([-8,-6,-4,-2,0], ["$-\infty$", -6,-4,-2,0])

            plt.tight_layout()
            #plt.savefig("three_pop_chase_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,9, figsize=(18,4.5))
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, extended_log10, extended_log10, np.log10]
            indexes = [0,1,3,4,5,6,7,10,11]
            for i in range(9):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    
                    else:

                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])
                        
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax2[1,1].legend(loc="center")
            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('p')
            ax2[0,3].set_title('$log_{10}(\mu_d)$')
            ax2[0,4].set_title('$log_{10}(\mu_b)$')
            ax2[0,5].set_title('$1/\mu_r$')
            ax2[0,6].set_title('$log_{10}(\mu_b^{chase})$')
            ax2[0,7].set_title('$log_{10}(\mu_a^{chase})$')
            ax2[0,8].set_title('$log_{10}(c)$')

            _ = ax2[1,6].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax2[1,7].set_xticks([-8,-4,0], ["$-\infty$", -4,0])
            plt.tight_layout()
            #plt.savefig("three_pop_chase_betas_2d", dpi=300)

            both_zero_prior = np.sum(np.transpose(all_params)[indexes[-2]]==0)
            one_zero_prior = np.sum(np.transpose(all_params)[indexes[-3]] == 0) - both_zero_prior
            none_zero_prior = len(all_params) - both_zero_prior - one_zero_prior
            prior_spikes = np.array([both_zero_prior, one_zero_prior, none_zero_prior])/len(all_params)

            both_zero_post = np.sum(np.transpose(accepted_params)[indexes[-2]]==0)
            one_zero_post = np.sum(np.transpose(accepted_params)[indexes[-3]] == 0) - both_zero_post
            none_zero_post = len(accepted_params) - both_zero_post - one_zero_post
            post_spikes = np.array([both_zero_post, one_zero_post, none_zero_post])/len(accepted_params)

            plt.figure(3, figsize = (4.5,4.5))
            plt.bar(np.array([1,2,3])-0.25/2,prior_spikes, 0.25, label = "Prior")
            plt.bar(np.array([1,2,3])+0.25/2,post_spikes, 0.25, label = "Posterior", color = "orange")
            plt.xticks(np.array([1,2,3]), np.array(["$\mu_b^{chase} = 0$,\n $\mu_a^{chase} = 0$", "$\mu_b^{chase} = 0$, \n $\mu_a^{chase} \\neq 0$", "$\mu_b^{chase} \\neq 0$, \n $\mu_a^{chase} \\neq 0$"]))
            plt.ylabel("Probability")
            plt.legend()
            plt.tight_layout()
            #plt.savefig("three_pop_spikes", dpi=300)

def three_population_posteriors_comparison(all_params, accepted_params, mode = "pulse", dimensions = 1, MAP = False):
    if dimensions == 1:
        if mode == "pulse":
            fig, ax = plt.subplots(2,6, figsize=(18,18/4))
            ax[0,0].hist(np.transpose(all_params)[2], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params)[2], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50)
            ax[0,1].hist(np.log10(np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_d)$")

            ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,2].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_b)$")

            ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
            ax[0,3].hist(np.log10(np.transpose(accepted_params)[4]/np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
            ax[0,4].hist(1/(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,4].set_title("$1/\mu_r$")

            ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Prior", bins=50)
            ax[0,5].hist(np.log10(np.transpose(accepted_params)[6]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,5].set_title("$log_{10}(c)$")

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "Prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            ax[1,2].hist(np.array([]), label = "Logarithmic birth")
            ax[1,2].hist(np.array([]), label = "Dispersed logarithmic birth", color = "orange")
            ax[1,2].legend(loc= "center")
            for j in range(2):
                for i in range(6):
                    ax[j,i].set_yticks([],[])

            for j in range(4):
                ax[1,j+2].set_xticks([],[])
                ax[1,j+2].spines['top'].set_visible(False)
                ax[1,j+2].spines['right'].set_visible(False)
                ax[1,j+2].spines['bottom'].set_visible(False)
                ax[1,j+2].spines['left'].set_visible(False)


        if mode == "chase":
            fig, ax = plt.subplots(2,6, figsize=(18,18/4))

            ax[0,0].hist(np.transpose(all_params)[3], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params)[3], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            #ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50, color = "green")
            ax[0,1].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,1].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_d)$")

            #ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50, color = "green")
            ax[0,2].hist(np.log10(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
            ax[0,2].hist(np.log10(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_b)$")

            #ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50, color = "green")
            ax[0,3].hist(np.log10(np.transpose(all_params)[5]/np.transpose(all_params)[3]), density = True, label = "posterior", bins=50)
            ax[0,3].hist(np.log10(np.transpose(accepted_params)[5]/np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            #ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "Pulse Prior", bins=50, color = "green")
            ax[0,4].hist(1/(np.transpose(all_params)[6]), density = True, label = "Pulse Posterior/Chase Prior", bins=50)
            ax[0,4].hist(1/(np.transpose(accepted_params)[6]), density = True, label = "Chase Posterior", bins=50, color = "orange")
            ax[0,4].set_title("$1/\mu_r$")

            #ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Pulse Fitting Prior", bins=50, color = "green")
            ax[0,5].hist(np.log10(np.transpose(all_params)[11]), density = True, label = "Prior", bins=50)
            ax[0,5].hist(np.log10(np.transpose(accepted_params)[11]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,5].set_title("$log_{10}(c)$")

            all_adjusted_mu_bs = np.transpose(all_params)[7].copy()
            all_adjusted_mu_bs[all_adjusted_mu_bs == 0] = 10e-8
            accepted_adjusted_mu_bs = np.transpose(accepted_params)[7].copy()
            accepted_adjusted_mu_bs[accepted_adjusted_mu_bs == 0] = 10e-8

            ax[1,2].hist(np.log10(all_adjusted_mu_bs), density = True, label = "prior", bins=50)
            ax[1,2].hist(np.log10(accepted_adjusted_mu_bs), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,2].set_title("$log_{10}(\mu_b^{chase})$")
            _ = ax[1,2].set_xticks([-7,-6,-5,-4,-3,-2], ["$-\infty$", -6,-5,-4,-3,-2])

            all_adjusted_mu_as = np.transpose(all_params)[10].copy()
            all_adjusted_mu_as[all_adjusted_mu_as == 0] = 10e-8
            accepted_adjusted_mu_as = np.transpose(accepted_params)[10].copy()
            accepted_adjusted_mu_as[accepted_adjusted_mu_as == 0] = 10e-8

            ax[1,3].hist(np.log10(all_adjusted_mu_as), density = True, label = "prior", bins=50)
            ax[1,3].hist(np.log10(accepted_adjusted_mu_as), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,3].set_title("$log_{10}(\mu_a^{chase})$")
            _ = ax[1,3].set_xticks([-7,-5,-2.5,0], ["$-\infty$", -5,-2.5,0])

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            ax[1,4].set_xticks([],[])
            ax[1,4].set_yticks([],[])
            ax[1,4].spines['top'].set_visible(False)
            ax[1,4].spines['right'].set_visible(False)
            ax[1,4].spines['bottom'].set_visible(False)
            ax[1,4].spines['left'].set_visible(False)

            ax[1,5].set_xticks([],[])
            ax[1,5].set_yticks([],[])
            ax[1,5].spines['top'].set_visible(False)
            ax[1,5].spines['right'].set_visible(False)
            ax[1,5].spines['bottom'].set_visible(False)
            ax[1,5].spines['left'].set_visible(False)
            for j in range(2):
                for i in range(6):
                    ax[j,i].set_yticks([],[])

            ax[1,4].hist(np.array([]), label = "Logarithmic birth")
            ax[1,4].hist(np.array([]), label = "Dispersed logarithmic birth", color = "orange")
            ax[1,4].legend(loc="center")

    if dimensions == 2:
        if mode == "pulse":
            fig, ax = plt.subplots(5,5, figsize=(18,12))
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10]
            for i in range(5):
                for j in range(5):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        a = ax[j][i].scatter(funcs[i+2](np.transpose(all_params)[i+2][:1000]), funcs[j+2](np.transpose(all_params)[j+2][:1000]), label = "Prior")
                        b = ax[j][i].scatter(funcs[i+2](np.transpose(accepted_params)[i+2]), funcs[j+2](np.transpose(accepted_params)[j+2]), color = "orange", label = "Posterior")
                        if type(MAP) != bool:
                            ax[j,i].scatter(funcs[i+2](MAP[i+2]), funcs[j+2](MAP[j+2]), color = "black", marker="x")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=4 and i !=4) or (j!=3 and i ==4):
                        ax[j,i].set_xticks([],[])
        
            ax[2,2].scatter(np.array([]), np.array([]),label = "Logarithmic birth")
            ax[2,2].scatter(np.array([]), np.array([]),label = "Dispersed logarithmic birth", color = "orange")
            ax[2,2].legend(loc="center")

            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
            ax[0,1].set_title('$log_{10}(\mu_d)$')
            ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
            ax[0,2].set_title('$log_{10}(\mu_b)$')
            ax[3,0].set_ylabel('$1/\mu_r$')
            ax[0,3].set_title('$1/\mu_r$')
            ax[4,0].set_ylabel('$log_{10}(c)$')
            ax[0,4].set_title('$log_{10}(c)$')
            plt.tight_layout()       
            #plt.savefig("dispersed_three_pop_pulse_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,7, figsize=(18,4.5))
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10]
            for i in range(7):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    else:
                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[i][:1000]), funcs[j](np.transpose(all_params)[j][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[i]), funcs[j](np.transpose(accepted_params)[j]), color="orange", label="Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])
            
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Logarithmic birth")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Dispersed logarithmic birth", color = "orange")
            ax2[1,1].legend(loc="center")

            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('p')
            ax2[0,3].set_title('$log_{10}(\mu_d)$')
            ax2[0,4].set_title('$log_{10}(\mu_b)$')
            ax2[0,5].set_title('$1/\mu_r$')
            ax2[0,6].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("dispersed_three_pop_pulse_betas_2d", dpi=300)

        #beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_b_pulse_pert, true_birth_pert, mu_b_chase, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_a_chase, c_pert
        if mode == "chase":
            
            fig, ax = plt.subplots(7,7, figsize=(18,12))

            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, np.log10, np.log10, lambda x: 1/x, extended_log10, extended_log10, np.log10]
            indexes = [3,4,5,6,7,10,11]
            for i in range(7):
                for j in range(7):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        ax[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=6 and i !=6) or (j!=5 and i ==6):
                        ax[j,i].set_xticks([],[])

            ax[3,3].scatter(np.array([]), np.array([]),label = "Prior")
            ax[3,3].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[3,3].legend(loc="center")
                
            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
            ax[0,1].set_title('$log_{10}(\mu_d)$')
            ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
            ax[0,2].set_title('$log_{10}(\mu_b)$')
            ax[3,0].set_ylabel('$1/\mu_r$')
            ax[0,3].set_title('$1/\mu_r$')
            ax[4,0].set_ylabel('$log_{10}(\mu_b^{chase})$')
            ax[0,4].set_title('$log_{10}(\mu_b^{chase})$')
            ax[5,0].set_ylabel('$log_{10}(\mu_a^{chase})$')
            ax[0,5].set_title('$log_{10}(\mu_a^{chase})$')
            ax[6,0].set_ylabel('$log_{10}(c)$')
            ax[0,6].set_title('$log_{10}(c)$')

            _ = ax[4,0].set_yticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax[6,4].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax[5,0].set_yticks([-8,-6,-4,-2,0], ["$-\infty$", -6,-4,-2,0])
            _ = ax[6,5].set_xticks([-8,-6,-4,-2,0], ["$-\infty$", -6,-4,-2,0])

            plt.tight_layout()
            #plt.savefig("three_pop_chase_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,9, figsize=(18,4.5))
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, extended_log10, extended_log10, np.log10]
            indexes = [0,1,3,4,5,6,7,10,11]
            for i in range(9):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    
                    else:

                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])
                        
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax2[1,1].legend(loc="center")
            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('p')
            ax2[0,3].set_title('$log_{10}(\mu_d)$')
            ax2[0,4].set_title('$log_{10}(\mu_b)$')
            ax2[0,5].set_title('$1/\mu_r$')
            ax2[0,6].set_title('$log_{10}(\mu_b^{chase})$')
            ax2[0,7].set_title('$log_{10}(\mu_a^{chase})$')
            ax2[0,8].set_title('$log_{10}(c)$')

            _ = ax2[1,6].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax2[1,7].set_xticks([-8,-4,0], ["$-\infty$", -4,0])
            plt.tight_layout()
            #plt.savefig("three_pop_chase_betas_2d", dpi=300)

            both_zero_prior = np.sum(np.transpose(all_params)[indexes[-2]]==0)
            one_zero_prior = np.sum(np.transpose(all_params)[indexes[-3]] == 0) - both_zero_prior
            none_zero_prior = len(all_params) - both_zero_prior - one_zero_prior
            prior_spikes = np.array([both_zero_prior, one_zero_prior, none_zero_prior])/len(all_params)

            both_zero_post = np.sum(np.transpose(accepted_params)[indexes[-2]]==0)
            one_zero_post = np.sum(np.transpose(accepted_params)[indexes[-3]] == 0) - both_zero_post
            none_zero_post = len(accepted_params) - both_zero_post - one_zero_post
            post_spikes = np.array([both_zero_post, one_zero_post, none_zero_post])/len(accepted_params)

            plt.figure(3, figsize = (3,3))
            plt.bar(np.array([1,2,3])-0.25/2,prior_spikes, 0.25, label = "Prior")
            plt.bar(np.array([1,2,3])+0.25/2,post_spikes, 0.25, label = "Posterior", color = "orange")
            plt.xticks(np.array([1,2,3]), np.array(["$\mu_b = 0$,\n $\mu_a = 0$", "$\mu_b = 0$, \n $\mu_a \\neq 0$", "$\mu_b \\neq 0$, \n $\mu_a \\neq 0$"]))
            plt.ylabel("Probability")
            plt.legend()
            plt.tight_layout()

def dispersed_three_population_posteriors(all_params, accepted_params, mode = "pulse", dimensions = 1, MAP = False):
    if dimensions == 1:

        fig, ax = plt.subplots(2,6, figsize=(18,18/4))
        ax[0,0].hist(np.transpose(all_params)[2], density = True, label = "prior", bins=50)
        ax[0,0].hist(np.transpose(accepted_params)[2], density = True, label = "posterior", bins=50, color = "orange")
        ax[0,0].set_title("$p$")
        ax[0,0].set_ylabel("Density")

        ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50)
        ax[0,1].hist(np.log10(np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,1].set_title("$log_{10}(\mu_d)$")

        ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
        ax[0,2].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,2].set_title("$log_{10}(\mu_b)$")

        ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
        ax[0,3].hist(np.log10(np.transpose(accepted_params)[4]/np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

        ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
        ax[0,4].hist(1/(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,4].set_title("$1/\mu_r$")

        ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Prior", bins=50)
        ax[0,5].hist(np.log10(np.transpose(accepted_params)[6]), density = True, label = "Posterior", bins=50, color = "orange")
        ax[0,5].set_title("$log_{10}(c)$")

        ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
        ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
        ax[1,0].set_title('$\\beta_0$')
        ax[1,0].set_ylabel("Density")

        ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "Prior", bins=50)
        ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "Posterior", bins=50, color = "orange")
        ax[1,1].set_title('$\\beta_1$')

        ax[1,2].hist(np.transpose(all_params)[7], label = "Prior", bins=50, density = True)
        ax[1,2].hist(np.transpose(accepted_params)[7], label = "Posterior", color = "orange", bins=50, density = True)
        ax[1,2].set_title('$\\theta^{1/2}$')

        ax[1,3].hist(np.array([]), label = "Prior")
        ax[1,3].hist(np.array([]), label = "Posterior", color = "orange")
        ax[1,3].legend(loc= "center")
        for j in range(2):
            for i in range(6):
                ax[j,i].set_yticks([],[])

        for j in range(3):
            ax[1,j+3].set_xticks([],[])
            ax[1,j+3].spines['top'].set_visible(False)
            ax[1,j+3].spines['right'].set_visible(False)
            ax[1,j+3].spines['bottom'].set_visible(False)
            ax[1,j+3].spines['left'].set_visible(False)

        plt.tight_layout()


    if dimensions == 2:
        fig, ax = plt.subplots(6,6, figsize=(18,12))
        #beta0, beta1, p, mu_d, mu_b, mu_r, c
        funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10, lambda x: x]
        for i in range(6):
            for j in range(6):

                if i == j:
                    ax[j,i].spines['top'].set_visible(False)
                    ax[j,i].spines['right'].set_visible(False)
                    ax[j,i].spines['bottom'].set_visible(False)
                    ax[j,i].spines['left'].set_visible(False)
                else:

                    a = ax[j][i].scatter(funcs[i+2](np.transpose(all_params)[i+2][:1000]), funcs[j+2](np.transpose(all_params)[j+2][:1000]), label = "Prior")
                    b = ax[j][i].scatter(funcs[i+2](np.transpose(accepted_params)[i+2]), funcs[j+2](np.transpose(accepted_params)[j+2]), color = "orange", label = "Posterior")
                    if type(MAP) != bool:
                        ax[j,i].scatter(funcs[i+2](MAP[i+2]), funcs[j+2](MAP[j+2]), color = "black", marker="x")

                if (i != 0 and j !=0) or (i!=1 and j ==0):
                    ax[j,i].set_yticks([],[])

                if (j !=4 and i !=4) or (j!=3 and i ==4):
                    ax[j,i].set_xticks([],[])
    
        ax[2,2].scatter(np.array([]), np.array([]),label = "Prior")
        ax[2,2].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
        ax[2,2].legend(loc="center")

        ax[0,1].set_ylabel('p')
        ax[1,0].set_title('p')
        ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
        ax[0,1].set_title('$log_{10}(\mu_d)$')
        ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
        ax[0,2].set_title('$log_{10}(\mu_b)$')
        ax[3,0].set_ylabel('$1/\mu_r$')
        ax[0,3].set_title('$1/\mu_r$')
        ax[4,0].set_ylabel('$log_{10}(c)$')
        ax[0,4].set_title('$log_{10}(c)$')
        ax[5,0].set_ylabel('$\\theta^{1/2}$')
        ax[0,5].set_title('$\\theta^{1/2}$')
        plt.tight_layout()       
        #plt.savefig("three_pop_pulse_2d", dpi=300)

        fig2, ax2 = plt.subplots(2,8, figsize=(18,4.5))
        #beta0, beta1, p, mu_d, mu_b, mu_r, c
        funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10, lambda x: x]
        for i in range(8):
            for j in range(2):

                if i == j:
                    ax2[j,i].spines['top'].set_visible(False)
                    ax2[j,i].spines['right'].set_visible(False)
                    ax2[j,i].spines['bottom'].set_visible(False)
                    ax2[j,i].spines['left'].set_visible(False)
                else:
                    ax2[j][i].scatter(funcs[i](np.transpose(all_params)[i][:1000]), funcs[j](np.transpose(all_params)[j][:1000]), label = "Prior")
                    ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[i]), funcs[j](np.transpose(accepted_params)[j]), color="orange", label="Posterior")
            
                if (i != 0 and j !=0) or (i!=1 and j ==0):
                    ax2[j,i].set_yticks([],[])

                if (j !=1 and i !=1) or (j==1 and i ==1):
                    ax2[j,i].set_xticks([],[])
        
        ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
        ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
        ax2[1,1].legend(loc="center")

        ax2[0,1].set_ylabel('$\\beta_0$')
        ax2[1,0].set_title('$\\beta_0$')
        ax2[1,0].set_ylabel('$\\beta_1$')
        ax2[0,1].set_title('$\\beta_1$')
        ax2[0,2].set_title('p')
        ax2[0,3].set_title('$log_{10}(\mu_d)$')
        ax2[0,4].set_title('$log_{10}(\mu_b)$')
        ax2[0,5].set_title('$1/\mu_r$')
        ax2[0,6].set_title('$log_{10}(c)$')
        ax2[0,7].set_title('$\\theta^{1/2}$')
        plt.tight_layout()
        #plt.savefig("three_pop_pulse_betas_2d", dpi=300)

def ou_three_population_posteriors(all_params, accepted_params, mode = "pulse", dimensions = 1, MAP = False):
    if dimensions == 1:

        fig, ax = plt.subplots(2,6, figsize=(18,18/4))
        ax[0,0].hist(np.transpose(all_params)[2], density = True, label = "prior", bins=50)
        ax[0,0].hist(np.transpose(accepted_params)[2], density = True, label = "posterior", bins=50, color = "orange")
        ax[0,0].set_title("$p$")
        ax[0,0].set_ylabel("Density")

        ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50)
        ax[0,1].hist(np.log10(np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,1].set_title("$log_{10}(\mu_d)$")

        ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
        ax[0,2].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,2].set_title("$log_{10}(\mu_b)$")

        ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
        ax[0,3].hist(np.log10(np.transpose(accepted_params)[4]/np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

        ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
        ax[0,4].hist(1/(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
        ax[0,4].set_title("$1/\mu_r$")

        ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Prior", bins=50)
        ax[0,5].hist(np.log10(np.transpose(accepted_params)[6]), density = True, label = "Posterior", bins=50, color = "orange")
        ax[0,5].set_title("$log_{10}(c)$")

        ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
        ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
        ax[1,0].set_title('$\\beta_0$')
        ax[1,0].set_ylabel("Density")

        ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "Prior", bins=50)
        ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "Posterior", bins=50, color = "orange")
        ax[1,1].set_title('$\\beta_1$')

        ax[1,2].hist(np.log10(np.transpose(all_params)[7]), label = "Prior", bins=50, density = True)
        ax[1,2].hist(np.log10(np.transpose(accepted_params)[7]), label = "Posterior", color = "orange", bins=50, density = True)
        ax[1,2].set_title('$log_{10}(\\theta$)')

        ax[1,3].hist(np.transpose(all_params)[8], label = "Prior", bins=50, density = True)
        ax[1,3].hist(np.transpose(accepted_params)[8], label = "Posterior", color = "orange", bins=50, density = True)
        ax[1,3].set_title('$\sigma$')

        ax[1,4].hist(np.array([]), label = "Prior")
        ax[1,4].hist(np.array([]), label = "Posterior", color = "orange")
        ax[1,4].legend(loc= "center")
        for j in range(2):
            for i in range(6):
                ax[j,i].set_yticks([],[])

        for j in range(2):
            ax[1,j+4].set_xticks([],[])
            ax[1,j+4].spines['top'].set_visible(False)
            ax[1,j+4].spines['right'].set_visible(False)
            ax[1,j+4].spines['bottom'].set_visible(False)
            ax[1,j+4].spines['left'].set_visible(False)

        plt.tight_layout()


    if dimensions == 2:
        fig, ax = plt.subplots(7,7, figsize=(18,12))
        #beta0, beta1, p, mu_d, mu_b, mu_r, c
        funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10, lambda x: x, np.log10]
        for i in range(7):
            for j in range(7):

                if i == j:
                    ax[j,i].spines['top'].set_visible(False)
                    ax[j,i].spines['right'].set_visible(False)
                    ax[j,i].spines['bottom'].set_visible(False)
                    ax[j,i].spines['left'].set_visible(False)
                else:

                    a = ax[j][i].scatter(funcs[i+2](np.transpose(all_params)[i+2][:1000]), funcs[j+2](np.transpose(all_params)[j+2][:1000]), label = "Prior")
                    b = ax[j][i].scatter(funcs[i+2](np.transpose(accepted_params)[i+2]), funcs[j+2](np.transpose(accepted_params)[j+2]), color = "orange", label = "Posterior")
                    if type(MAP) != bool:
                        ax[j,i].scatter(funcs[i+2](MAP[i+2]), funcs[j+2](MAP[j+2]), color = "black", marker="x")

                if (i != 0 and j !=0) or (i!=1 and j ==0):
                    ax[j,i].set_yticks([],[])

                if (j !=4 and i !=4) or (j!=3 and i ==4):
                    ax[j,i].set_xticks([],[])
    
        ax[2,2].scatter(np.array([]), np.array([]),label = "Prior")
        ax[2,2].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
        ax[2,2].legend(loc="center")

        ax[0,1].set_ylabel('p')
        ax[1,0].set_title('p')
        ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
        ax[0,1].set_title('$log_{10}(\mu_d)$')
        ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
        ax[0,2].set_title('$log_{10}(\mu_b)$')
        ax[3,0].set_ylabel('$1/\mu_r$')
        ax[0,3].set_title('$1/\mu_r$')
        ax[4,0].set_ylabel('$log_{10}(c)$')
        ax[0,4].set_title('$log_{10}(c)$')
        ax[5,0].set_ylabel('$\\theta$')
        ax[0,5].set_title('$\\theta$')
        ax[6,0].set_ylabel('$log_{10}(\sigma)$')
        ax[0,6].set_title('$log_{10}(\sigma)$')
        plt.tight_layout()
        #plt.savefig("three_pop_pulse_2d", dpi=300)

        fig2, ax2 = plt.subplots(2,9, figsize=(18,4.5))
        #beta0, beta1, p, mu_d, mu_b, mu_r, c
        funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10, np.log10, lambda x: x]
        for i in range(9):
            for j in range(2):

                if i == j:
                    ax2[j,i].spines['top'].set_visible(False)
                    ax2[j,i].spines['right'].set_visible(False)
                    ax2[j,i].spines['bottom'].set_visible(False)
                    ax2[j,i].spines['left'].set_visible(False)
                else:
                    ax2[j][i].scatter(funcs[i](np.transpose(all_params)[i][:1000]), funcs[j](np.transpose(all_params)[j][:1000]), label = "Prior")
                    ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[i]), funcs[j](np.transpose(accepted_params)[j]), color="orange", label="Posterior")
            
                if (i != 0 and j !=0) or (i!=1 and j ==0):
                    ax2[j,i].set_yticks([],[])

                if (j !=1 and i !=1) or (j==1 and i ==1):
                    ax2[j,i].set_xticks([],[])
        
        ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
        ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
        ax2[1,1].legend(loc="center")

        ax2[0,1].set_ylabel('$\\beta_0$')
        ax2[1,0].set_title('$\\beta_0$')
        ax2[1,0].set_ylabel('$\\beta_1$')
        ax2[0,1].set_title('$\\beta_1$')
        ax2[0,2].set_title('p')
        ax2[0,3].set_title('$log_{10}(\mu_d)$')
        ax2[0,4].set_title('$log_{10}(\mu_b)$')
        ax2[0,5].set_title('$1/\mu_r$')
        ax2[0,6].set_title('$log_{10}(c)$')
        ax2[0,7].set_title('$log_{10}(\\theta)$')
        ax2[0,8].set_title('$\sigma$')
        plt.tight_layout()
        #plt.savefig("three_pop_pulse_betas_2d", dpi=300)

  
def three_population_posteriors_comparison(all_params, accepted_params, mode = "pulse", dimensions = 1, MAP = False):
    if dimensions == 1:
        if mode == "pulse":
            fig, ax = plt.subplots(2,6, figsize=(18,18/4))
            ax[0,0].hist(np.transpose(all_params)[2], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params)[2], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50)
            ax[0,1].hist(np.log10(np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_d)$")

            ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,2].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_b)$")

            ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50)
            ax[0,3].hist(np.log10(np.transpose(accepted_params)[4]/np.transpose(accepted_params)[2]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
            ax[0,4].hist(1/(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,4].set_title("$1/\mu_r$")

            ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Prior", bins=50)
            ax[0,5].hist(np.log10(np.transpose(accepted_params)[6]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,5].set_title("$log_{10}(c)$")

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "Prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "Posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            ax[1,2].hist(np.array([]), label = "Logarithmic birth")
            ax[1,2].hist(np.array([]), label = "Dispersed logarithmic birth", color = "orange")
            ax[1,2].legend(loc= "center")
            for j in range(2):
                for i in range(6):
                    ax[j,i].set_yticks([],[])

            for j in range(4):
                ax[1,j+2].set_xticks([],[])
                ax[1,j+2].spines['top'].set_visible(False)
                ax[1,j+2].spines['right'].set_visible(False)
                ax[1,j+2].spines['bottom'].set_visible(False)
                ax[1,j+2].spines['left'].set_visible(False)


        if mode == "chase":
            fig, ax = plt.subplots(2,6, figsize=(18,18/4))

            ax[0,0].hist(np.transpose(all_params)[3], density = True, label = "prior", bins=50)
            ax[0,0].hist(np.transpose(accepted_params)[3], density = True, label = "posterior", bins=50, color = "orange")
            ax[0,0].set_title("$p$")
            ax[0,0].set_ylabel("Density")

            #ax[0,1].hist(np.log10(np.transpose(all_params)[3]), density = True, label = "prior", bins=50, color = "green")
            ax[0,1].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50)
            ax[0,1].hist(np.log10(np.transpose(accepted_params)[4]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,1].set_title("$log_{10}(\mu_d)$")

            #ax[0,2].hist(np.log10(np.transpose(all_params)[4]), density = True, label = "prior", bins=50, color = "green")
            ax[0,2].hist(np.log10(np.transpose(all_params)[5]), density = True, label = "prior", bins=50)
            ax[0,2].hist(np.log10(np.transpose(accepted_params)[5]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,2].set_title("$log_{10}(\mu_b)$")

            #ax[0,3].hist(np.log10(np.transpose(all_params)[4]/np.transpose(all_params)[2]), density = True, label = "prior", bins=50, color = "green")
            ax[0,3].hist(np.log10(np.transpose(all_params)[5]/np.transpose(all_params)[3]), density = True, label = "posterior", bins=50)
            ax[0,3].hist(np.log10(np.transpose(accepted_params)[5]/np.transpose(accepted_params)[3]), density = True, label = "posterior", bins=50, color = "orange")
            ax[0,3].set_title("$log_{10}(\mu_a)$ ($\mu_a = \mu_b/p$)")

            #ax[0,4].hist(1/(np.transpose(all_params)[5]), density = True, label = "Pulse Prior", bins=50, color = "green")
            ax[0,4].hist(1/(np.transpose(all_params)[6]), density = True, label = "Pulse Posterior/Chase Prior", bins=50)
            ax[0,4].hist(1/(np.transpose(accepted_params)[6]), density = True, label = "Chase Posterior", bins=50, color = "orange")
            ax[0,4].set_title("$1/\mu_r$")

            #ax[0,5].hist(np.log10(np.transpose(all_params)[6]), density = True, label = "Pulse Fitting Prior", bins=50, color = "green")
            ax[0,5].hist(np.log10(np.transpose(all_params)[11]), density = True, label = "Prior", bins=50)
            ax[0,5].hist(np.log10(np.transpose(accepted_params)[11]), density = True, label = "Posterior", bins=50, color = "orange")
            ax[0,5].set_title("$log_{10}(c)$")

            all_adjusted_mu_bs = np.transpose(all_params)[7].copy()
            all_adjusted_mu_bs[all_adjusted_mu_bs == 0] = 10e-8
            accepted_adjusted_mu_bs = np.transpose(accepted_params)[7].copy()
            accepted_adjusted_mu_bs[accepted_adjusted_mu_bs == 0] = 10e-8

            ax[1,2].hist(np.log10(all_adjusted_mu_bs), density = True, label = "prior", bins=50)
            ax[1,2].hist(np.log10(accepted_adjusted_mu_bs), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,2].set_title("$log_{10}(\mu_b^{chase})$")
            _ = ax[1,2].set_xticks([-7,-6,-5,-4,-3,-2], ["$-\infty$", -6,-5,-4,-3,-2])

            all_adjusted_mu_as = np.transpose(all_params)[10].copy()
            all_adjusted_mu_as[all_adjusted_mu_as == 0] = 10e-8
            accepted_adjusted_mu_as = np.transpose(accepted_params)[10].copy()
            accepted_adjusted_mu_as[accepted_adjusted_mu_as == 0] = 10e-8

            ax[1,3].hist(np.log10(all_adjusted_mu_as), density = True, label = "prior", bins=50)
            ax[1,3].hist(np.log10(accepted_adjusted_mu_as), density = True, label = "posterior", bins=50, color = "orange")
            ax[1,3].set_title("$log_{10}(\mu_a^{chase})$")
            _ = ax[1,3].set_xticks([-7,-5,-2.5,0], ["$-\infty$", -5,-2.5,0])

            ax[1,0].hist(np.transpose(all_params)[0], density = True, label = "prior", bins=50)
            ax[1,0].hist(np.transpose(accepted_params)[0], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,0].set_title('$\\beta_0$')
            ax[1,0].set_ylabel("Density")

            ax[1,1].hist(np.transpose(all_params)[1], density = True, label = "prior", bins=50)
            ax[1,1].hist(np.transpose(accepted_params)[1], density = True, label = "posterior", bins=50, color = "orange")
            ax[1,1].set_title('$\\beta_1$')

            ax[1,4].set_xticks([],[])
            ax[1,4].set_yticks([],[])
            ax[1,4].spines['top'].set_visible(False)
            ax[1,4].spines['right'].set_visible(False)
            ax[1,4].spines['bottom'].set_visible(False)
            ax[1,4].spines['left'].set_visible(False)

            ax[1,5].set_xticks([],[])
            ax[1,5].set_yticks([],[])
            ax[1,5].spines['top'].set_visible(False)
            ax[1,5].spines['right'].set_visible(False)
            ax[1,5].spines['bottom'].set_visible(False)
            ax[1,5].spines['left'].set_visible(False)
            for j in range(2):
                for i in range(6):
                    ax[j,i].set_yticks([],[])

            ax[1,4].hist(np.array([]), label = "Logarithmic birth")
            ax[1,4].hist(np.array([]), label = "Dispersed logarithmic birth", color = "orange")
            ax[1,4].legend(loc="center")

    if dimensions == 2:
        if mode == "pulse":
            fig, ax = plt.subplots(5,5, figsize=(18,12))
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10]
            for i in range(5):
                for j in range(5):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        a = ax[j][i].scatter(funcs[i+2](np.transpose(all_params)[i+2][:1000]), funcs[j+2](np.transpose(all_params)[j+2][:1000]), label = "Prior")
                        b = ax[j][i].scatter(funcs[i+2](np.transpose(accepted_params)[i+2]), funcs[j+2](np.transpose(accepted_params)[j+2]), color = "orange", label = "Posterior")
                        if type(MAP) != bool:
                            ax[j,i].scatter(funcs[i+2](MAP[i+2]), funcs[j+2](MAP[j+2]), color = "black", marker="x")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=4 and i !=4) or (j!=3 and i ==4):
                        ax[j,i].set_xticks([],[])
        
            ax[2,2].scatter(np.array([]), np.array([]),label = "Logarithmic birth")
            ax[2,2].scatter(np.array([]), np.array([]),label = "Dispersed logarithmic birth", color = "orange")
            ax[2,2].legend(loc="center")

            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
            ax[0,1].set_title('$log_{10}(\mu_d)$')
            ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
            ax[0,2].set_title('$log_{10}(\mu_b)$')
            ax[3,0].set_ylabel('$1/\mu_r$')
            ax[0,3].set_title('$1/\mu_r$')
            ax[4,0].set_ylabel('$log_{10}(c)$')
            ax[0,4].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("dispersed_three_pop_pulse_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,7, figsize=(18,4.5))
            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, np.log10]
            for i in range(7):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    else:
                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[i][:1000]), funcs[j](np.transpose(all_params)[j][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[i]), funcs[j](np.transpose(accepted_params)[j]), color="orange", label="Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])
            
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Logarithmic birth")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Dispersed logarithmic birth", color = "orange")
            ax2[1,1].legend(loc="center")

            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('p')
            ax2[0,3].set_title('$log_{10}(\mu_d)$')
            ax2[0,4].set_title('$log_{10}(\mu_b)$')
            ax2[0,5].set_title('$1/\mu_r$')
            ax2[0,6].set_title('$log_{10}(c)$')
            plt.tight_layout()
            #plt.savefig("dispersed_three_pop_pulse_betas_2d", dpi=300)

        #beta0, beta1, sig, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_b_pulse_pert, true_birth_pert, mu_b_chase, diffusion_prob_pulse_pert, mu_d_pulse_pert, mu_a_chase, c_pert
        if mode == "chase":
            
            fig, ax = plt.subplots(7,7, figsize=(18,12))

            #beta0, beta1, p, mu_d, mu_b, mu_r, c
            funcs = [lambda x: x, np.log10, np.log10, lambda x: 1/x, extended_log10, extended_log10, np.log10]
            indexes = [3,4,5,6,7,10,11]
            for i in range(7):
                for j in range(7):

                    if i == j:
                        ax[j,i].spines['top'].set_visible(False)
                        ax[j,i].spines['right'].set_visible(False)
                        ax[j,i].spines['bottom'].set_visible(False)
                        ax[j,i].spines['left'].set_visible(False)
                    else:

                        ax[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")

                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax[j,i].set_yticks([],[])

                    if (j !=6 and i !=6) or (j!=5 and i ==6):
                        ax[j,i].set_xticks([],[])

            ax[3,3].scatter(np.array([]), np.array([]),label = "Prior")
            ax[3,3].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax[3,3].legend(loc="center")
                
            ax[0,1].set_ylabel('p')
            ax[1,0].set_title('p')
            ax[1,0].set_ylabel('$log_{10}(\mu_d)$')
            ax[0,1].set_title('$log_{10}(\mu_d)$')
            ax[2,0].set_ylabel('$log_{10}(\mu_b)$')
            ax[0,2].set_title('$log_{10}(\mu_b)$')
            ax[3,0].set_ylabel('$1/\mu_r$')
            ax[0,3].set_title('$1/\mu_r$')
            ax[4,0].set_ylabel('$log_{10}(\mu_b^{chase})$')
            ax[0,4].set_title('$log_{10}(\mu_b^{chase})$')
            ax[5,0].set_ylabel('$log_{10}(\mu_a^{chase})$')
            ax[0,5].set_title('$log_{10}(\mu_a^{chase})$')
            ax[6,0].set_ylabel('$log_{10}(c)$')
            ax[0,6].set_title('$log_{10}(c)$')

            _ = ax[4,0].set_yticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax[6,4].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax[5,0].set_yticks([-8,-6,-4,-2,0], ["$-\infty$", -6,-4,-2,0])
            _ = ax[6,5].set_xticks([-8,-6,-4,-2,0], ["$-\infty$", -6,-4,-2,0])

            plt.tight_layout()
            #plt.savefig("three_pop_chase_2d", dpi=300)

            fig2, ax2 = plt.subplots(2,9, figsize=(18,4.5))
            funcs = [lambda x: x, lambda x: x, lambda x: x, np.log10, np.log10, lambda x: 1/x, extended_log10, extended_log10, np.log10]
            indexes = [0,1,3,4,5,6,7,10,11]
            for i in range(9):
                for j in range(2):

                    if i == j:
                        ax2[j,i].spines['top'].set_visible(False)
                        ax2[j,i].spines['right'].set_visible(False)
                        ax2[j,i].spines['bottom'].set_visible(False)
                        ax2[j,i].spines['left'].set_visible(False)
                    
                    else:

                        ax2[j][i].scatter(funcs[i](np.transpose(all_params)[indexes[i]][:1000]), funcs[j](np.transpose(all_params)[indexes[j]][:1000]), label = "Prior")
                        ax2[j][i].scatter(funcs[i](np.transpose(accepted_params)[indexes[i]]), funcs[j](np.transpose(accepted_params)[indexes[j]]), color = "orange", label = "Posterior")
                
                    if (i != 0 and j !=0) or (i!=1 and j ==0):
                        ax2[j,i].set_yticks([],[])

                    if (j !=1 and i !=1) or (j==1 and i ==1):
                        ax2[j,i].set_xticks([],[])
                        
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Prior")
            ax2[1,1].scatter(np.array([]), np.array([]),label = "Posterior", color = "orange")
            ax2[1,1].legend(loc="center")
            ax2[0,1].set_ylabel('$\\beta_0$')
            ax2[1,0].set_title('$\\beta_0$')
            ax2[1,0].set_ylabel('$\\beta_1$')
            ax2[0,1].set_title('$\\beta_1$')
            ax2[0,2].set_title('p')
            ax2[0,3].set_title('$log_{10}(\mu_d)$')
            ax2[0,4].set_title('$log_{10}(\mu_b)$')
            ax2[0,5].set_title('$1/\mu_r$')
            ax2[0,6].set_title('$log_{10}(\mu_b^{chase})$')
            ax2[0,7].set_title('$log_{10}(\mu_a^{chase})$')
            ax2[0,8].set_title('$log_{10}(c)$')

            _ = ax2[1,6].set_xticks([-8,-6,-4,-2], ["$-\infty$", -6,-4,-2])
            _ = ax2[1,7].set_xticks([-8,-4,0], ["$-\infty$", -4,0])
            plt.tight_layout()
            #plt.savefig("three_pop_chase_betas_2d", dpi=300)

            both_zero_prior = np.sum(np.transpose(all_params)[indexes[-2]]==0)
            one_zero_prior = np.sum(np.transpose(all_params)[indexes[-3]] == 0) - both_zero_prior
            none_zero_prior = len(all_params) - both_zero_prior - one_zero_prior
            prior_spikes = np.array([both_zero_prior, one_zero_prior, none_zero_prior])/len(all_params)

            both_zero_post = np.sum(np.transpose(accepted_params)[indexes[-2]]==0)
            one_zero_post = np.sum(np.transpose(accepted_params)[indexes[-3]] == 0) - both_zero_post
            none_zero_post = len(accepted_params) - both_zero_post - one_zero_post
            post_spikes = np.array([both_zero_post, one_zero_post, none_zero_post])/len(accepted_params)

            plt.figure(3, figsize = (3,3))
            plt.bar(np.array([1,2,3])-0.25/2,prior_spikes, 0.25, label = "Prior")
            plt.bar(np.array([1,2,3])+0.25/2,post_spikes, 0.25, label = "Posterior", color = "orange")
            plt.xticks(np.array([1,2,3]), np.array(["$\mu_b = 0$,\n $\mu_a = 0$", "$\mu_b = 0$, \n $\mu_a \\neq 0$", "$\mu_b \\neq 0$, \n $\mu_a \\neq 0$"]))
            plt.ylabel("Probability")
            plt.legend()
            plt.tight_layout()


dirname = os.path.dirname(__file__)
def plot_posteriors(model = "three population", mode = "chase", dimensions = 2):
    """
    model: "three population", "two population", "one population"
    mode: "pulse" or "chase"
    dimensions: 1 or 2. Will plot 1 or 2 dimensional marginals
    variable_p: Bool. If model = "two population", mode = "chase", variable_p plots the variable p two population model, otherwise
                this parameter has no effect.
    preliminary: Bool. If model = "two population, mode = "pulse", preliminary plots the preliminary two population model (where p=1),
                otherwise this parameter has no effect.
    """

    if model == "three population":
        if mode == "pulse":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))
            previous_params = 0
        elif mode == "chase":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
            previous_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_params.npy"))
        three_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions)

    elif model == "two population":
        if mode == "pulse":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_accepted_params.npy"))
            previous_params = 0

        elif mode == "chase":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_chase_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_chase_accepted_params.npy"))
            previous_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_params.npy"))

        two_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions, p = False, prel = False)

    elif model == "preliminary two population":
        if mode == "pulse":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/preliminary/preliminary_two_population_pulse_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/preliminary/preliminary_two_population_pulse_accepted_params.npy"))
            previous_params = 0
            two_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions, p = False, prel = True)

    elif model == "variable p two population":
        if mode == "chase":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/variable_p/two_population_chase_variablep_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/variable_p/two_population_chase_variablep_accepted_params.npy"))
            previous_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_params.npy"))
            two_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions, p = True, prel = False)

    elif model == "one population":
        if mode == "pulse":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data/one_population_model/one_population_pulse_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data/one_population_model/one_population_pulse_accepted_params.npy"))
            one_population_posteriors(all_params, accepted_params, dimensions = dimensions)

    elif model == "stochastic three population":
        if mode == "pulse":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data_birth_rate_selection/stochastic_inhibition/three_population_pulse_stocinh_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data_birth_rate_selection/stochastic_inhibition/three_population_pulse_stocinh_accepted_params.npy"))
            ou_three_population_posteriors(all_params, accepted_params, dimensions = dimensions)      

    elif model == "dispersed three population":
        if mode == "pulse":
            all_params = np.load(os.path.join(dirname, "simulated_ABC_data_birth_rate_selection/logarithmic/dispersed_three_population_pulse_params.npy"))
            accepted_params = np.load(os.path.join(dirname, "simulated_ABC_data_birth_rate_selection/logarithmic/dispersed_three_population_pulse_accepted_params.npy"))
            dispersed_three_population_posteriors(all_params, accepted_params, dimensions = dimensions)          


def plot_posteriors_manual(all_params, accepted_params, model = "three population", mode = "chase", dimensions = 2):
    """
    model: "three population", "two population", "one population"
    mode: "pulse" or "chase"
    dimensions: 1 or 2. Will plot 1 or 2 dimensional marginals
    variable_p: Bool. If model = "two population", mode = "chase", variable_p plots the variable p two population model, otherwise
                this parameter has no effect.
    preliminary: Bool. If model = "two population, mode = "pulse", preliminary plots the preliminary two population model (where p=1),
                otherwise this parameter has no effect.
    """

    if model == "three population":
        previous_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_params.npy"))
        three_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions)

    elif model == "two population":
        previous_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_params.npy"))
        two_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions, p = False, prel = False)

    elif model == "preliminary two population":
        previous_params = 0
        two_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions, p = False, prel = True)

    elif model == "variable p two population":
        previous_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_params.npy"))
        two_population_posteriors(all_params, accepted_params, previous_params, mode = mode, dimensions = dimensions, p = True, prel = False)

    elif model == "one population":
        one_population_posteriors(all_params, accepted_params, dimensions = dimensions)

    elif model == "dispersed three population":
        dispersed_three_population_posteriors(all_params, accepted_params, dimensions = dimensions)

    elif model == "stochastic three population":
        ou_three_population_posteriors(all_params, accepted_params, dimensions = dimensions)
#Functions to plot posterior predictive trajectories of data already simulated

def chase_posterior_predictive_trajectories():

    edu_means = all_assays_edu_number_chase_means
    edu_err = all_assays_edu_number_chase_err
    dna_means = all_assays_dna_number_chase_means
    dna_err = all_assays_dna_number_chase_err
    edu_prop = all_assays_edu_proportion_chase_means
    edu_prop_err = all_assays_edu_proportion_chase_err

    nucleoid_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_nucleoid_trajectories.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/variable_p/chase_nucleoid_trajectories_variablep.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_nucleoid_trajectories.npy"))]

    edu_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_trajectories.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/variable_p/chase_edu_trajectories_variablep.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_trajectories.npy"))]

    edu_proportions_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_proportions.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/variable_p/chase_edu_proportions_variablep.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_proportions.npy"))]
    
    peak1_0day_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_peak1_0day.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/variable_p/chase_peak1_0day_variablep.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_peak1_0day.npy"))]
    
    peak1_4day_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_peak1_4day.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/variable_p/chase_peak1_4day_variablep.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_peak1_4day.npy"))]
    
    variance_statistics_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_variance_stat.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/variable_p/chase_variance_stat_variablep.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_variance_stat.npy"))]


    fig, ax = plt.subplots(1,3, figsize = (18,4))

    #Trajectories
    chase_trajectory_time = np.arange(0,4*24+0.25,0.25) + 24
    chase_times = np.array([0,24,48,96]) + 24
    for i in range(3):

        for traj in nucleoid_trajectories_list[i][:-1]:
            ax[i].plot(chase_trajectory_time, traj, c ="mediumaquamarine", alpha=0.2)
        ax[i].plot(chase_trajectory_time, nucleoid_trajectories_list[i][-1], c ="mediumaquamarine", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")
        ax[i].plot(chase_trajectory_time, np.mean(nucleoid_trajectories_list[i], axis=0), c="#118002", linewidth=3, label = "Nucleoid Number Posterior Predictive Mean")

        for traj in edu_trajectories_list[i][:-1]:
            ax[i].plot(chase_trajectory_time, traj, c ="plum", alpha=0.2)
        ax[i].plot(chase_trajectory_time, edu_trajectories_list[i][-1], c ="plum", alpha=0.2, label = "EdU Number Posterior Predictive Draw")
        ax[i].plot(chase_trajectory_time, np.mean(edu_trajectories_list[i], axis=0), c="#800080", linewidth=3, label = "EdU Number Posterior Predictive Mean")

        ax[i].errorbar(chase_times, edu_means, edu_err,  color='#800080', marker='o', ls = 'none', ecolor = '#800080', label = "Edu Number Data Mean + 95% Confidence Interval")
        ax[i].errorbar(chase_times, dna_means, dna_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Nucleoid Number Data Mean + 95% Confidence Interval")
        
        #ax[i].set_xticks([0,24,48,72,96], [0,1,2,3,4])
        ax[i].set_ylim(180,1400)
        ax[i].set_xlabel("Time (h)")
    ax[1].set_yticks([],[])
    ax[2].set_yticks([],[])
    ax[0].set_ylabel("Nucleoid number")
    ax[0].set_title("Three-population")
    ax[1].set_title("Variable-p two-population")
    ax[2].set_title("Two-population")

    legend_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior Predictive Mean'),
                    Line2D([0], [0], color='grey', lw=1.5, alpha = 0.4, label='Posterior Predictive Draw'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='grey', markersize=8),
                    Patch(facecolor='#800080', edgecolor='#800080', label='EdU Number'),
                    Patch(facecolor='#118002', edgecolor='#118002', label='Nucleoid Number')]

    ax[2].legend(handles=legend_elements)
    plt.tight_layout()
    #plt.savefig("chase_posterior_predictive_traj", dpi=300)

    fig11, ax11 = plt.subplots(1,3, figsize = (18,4))

    #Trajectories
    chase_trajectory_time = np.arange(0,4*24+0.25,0.25) + 24
    chase_times = np.array([0,24,48,96]) + 24
    for i in range(3):

        for traj in edu_proportions_list[i][:-1]:
            ax11[i].plot(chase_trajectory_time, traj, c ="mediumaquamarine", alpha=0.2)
        ax11[i].plot(chase_trajectory_time, edu_proportions_list[i][-1], c ="mediumaquamarine", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")
        ax11[i].plot(chase_trajectory_time, np.mean(edu_proportions_list[i], axis=0), c="#118002", linewidth=3, label = "Nucleoid Number Posterior Predictive Mean")

        ax11[i].errorbar(chase_times, edu_prop, edu_prop_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Edu Number Data Mean + 95% Confidence Interval")

        #ax11[i].set_xticks([0,24,48,72,96], [0,1,2,3,4])
        ax11[i].set_ylim(0,1)
        ax11[i].set_xlabel("Time (h)")
    ax11[1].set_yticks([],[])
    ax11[2].set_yticks([],[])
    ax11[0].set_ylabel("EdU/mtDNA")
    ax11[0].set_title("Three-population")
    ax11[1].set_title("Variable-p two-population")
    ax11[2].set_title("Two-population")

    legend_elements = [Line2D([0], [0], color='forestgreen', lw=3, label='Posterior predictive mean'),
                    Line2D([0], [0], color='forestgreen', lw=1.5, alpha = 0.4, label='Posterior predictive draw'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='g', markersize=8)]

    ax11[2].legend(handles=legend_elements)
    plt.tight_layout()
    #plt.savefig("chase_posterior_predictive_prop_traj", dpi=300)


    fig2, ax2 = plt.subplots(2,3, figsize = (18,6))
    #Peak 1 proportions
    for i in range(3):

        ax2[0,i].hist(peak1_0day_list[i], alpha = 0.7, density = True, label = "Posterior predictive") 
        ax2[0,i].axvline(0.33, color='k', linestyle='dashed', linewidth=1)
        ax2[0,i].axvline(0.39, color='k', linestyle='dashed', linewidth=1)
        ax2[0,i].axvline(0.29, color='k', linestyle='dashed', linewidth=1, label = "Observed proportion for 3 assays")
        ax2[0,i].set_xlim(0.2,0.7)
        ax2[0,i].set_yticks([],[])
        ax2[0,i].set_xticks([],[])

        ax2[1,i].hist(peak1_4day_list[i], alpha = 0.7, density = True) 
        ax2[1,i].axvline(0.67, color='k', linestyle='dashed', linewidth=1)
        ax2[1,i].axvline(0.58, color='k', linestyle='dashed', linewidth=1)
        ax2[1,i].axvline(0.43, color='k', linestyle='dashed', linewidth=1)
        ax2[1,i].set_xlim(0.2,0.7)
        ax2[1,i].set_yticks([],[])
        ax2[1,i].set_xlabel("Singly tagged proportion")

    ax2[0,0].set_ylabel("24 hours")
    ax2[1,0].set_ylabel("120 hours")
    ax2[0,0].set_title("Three-population")
    ax2[0,1].set_title("Variable-p two-population")
    ax2[0,2].set_title("Two-population")
    fig2.supylabel("Probability density")
    ax2[0,2].legend()
    plt.tight_layout()
    #plt.savefig("chase_posterior_predictive_singly_tagged", dpi=300)

    fig3, ax3 = plt.subplots(1,3, figsize = (18,3))
    for i in range(3):
        ax3[i].hist(variance_statistics_list[i], alpha = 0.7, color = "tab:purple", density = True, label = "Posterior predictive") 
        ax3[i].axvline(0, color='k', linestyle='dashed', linewidth=1, label = "Observed")
        ax3[i].set_xlim(-2,2)
        ax3[i].set_yticks([],[])
        ax3[i].set_xlabel("$S_{cs}$")
    ax3[0].set_title("Three-population")
    ax3[1].set_title("Variable-p two-population")
    ax3[2].set_title("Two-population")
    ax3[0].set_ylabel("Probability density")
    ax3[2].legend()
    plt.tight_layout()
    #plt.savefig("chase_posterior_predictive_control_strength", dpi=300)

def pulse_posterior_predictive_trajectories():

    edu_means = all_assays_edu_number_pulse_means
    edu_err = all_assays_edu_number_pulse_err
    dna_means = all_assays_dna_number_pulse_means
    dna_err = all_assays_dna_number_pulse_err

    nucleoid_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/pulse_nucleoid_trajectories.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/pulse_nucleoid_trajectories.npy")),                                 
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/preliminary/preliminary_pulse_nucleoid_trajectories.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/one_population/pulse_nucleoid_trajectories.npy"))]

    edu_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/pulse_edu_trajectories.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/pulse_edu_trajectories.npy")),                             
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/preliminary/preliminary_pulse_edu_trajectories.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/one_population/pulse_edu_trajectories.npy"))]

    peak1_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/pulse_peak1.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/pulse_peak1.npy")),                  
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/preliminary/preliminary_pulse_peak1.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/one_population/pulse_peak1.npy"))]
    
    variance_statistics_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/pulse_variance_stat.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/pulse_variance_stat.npy")),                                
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/preliminary/preliminary_pulse_variance_stat.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/one_population/pulse_variance_stat.npy"))]

    #fig, ax = plt.subplots(1,4, figsize = (23,5))
    fig, ax = plt.subplots(1,4, figsize = (18,4))
    #Trajectories
    pulse_trajectory_time = np.arange(0,24.25,0.25)
    pulse_times = np.array([1,3,7,24])
    for i in range(4):

        for traj in nucleoid_trajectories_list[i][:-1]:
            ax[i].plot(pulse_trajectory_time, traj, c ="mediumaquamarine", alpha=0.2)
        ax[i].plot(pulse_trajectory_time, nucleoid_trajectories_list[i][-1], c ="mediumaquamarine", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")
        ax[i].plot(pulse_trajectory_time, np.mean(nucleoid_trajectories_list[i], axis=0), c="#118002", linewidth=3, label = "Nucleoid Number Posterior Predictive Mean")

        for traj in edu_trajectories_list[i][:-1]:
            ax[i].plot(pulse_trajectory_time, traj, c ="plum", alpha=0.2)
        ax[i].plot(pulse_trajectory_time, edu_trajectories_list[i][-1], c ="plum", alpha=0.2, label = "EdU Number Posterior Predictive Draw")
        ax[i].plot(pulse_trajectory_time, np.mean(edu_trajectories_list[i], axis=0), c="#800080", linewidth=3, label = "EdU Number Posterior Predictive Mean")

        ax[i].errorbar(pulse_times, edu_means, edu_err,  color='#800080', marker='o', ls = 'none', ecolor = '#800080', label = "Edu Number Data Mean + 95% Confidence Interval")
        ax[i].errorbar(pulse_times, dna_means, dna_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Nucleoid Number Data Mean + 95% Confidence Interval")
        
        #ax[i].set_xticks([0,24,48,72,96], [0,1,2,3,4])
        ax[i].set_ylim(0,1200)
        ax[i].set_xlabel("Time (h)")
    ax[1].set_yticks([],[])
    ax[2].set_yticks([],[])
    ax[3].set_yticks([],[])
    ax[0].set_ylabel("Nucleoid number")
    ax[0].set_title("Three-population")
    ax[1].set_title("Two-population")
    ax[2].set_title("Preliminary two-population")
    ax[3].set_title("One-population")

    legend_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior Predictive Mean'),
                    Line2D([0], [0], color='grey', lw=1.5, alpha = 0.4, label='Posterior Predictive Draw'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='grey', markersize=8),
                    Patch(facecolor='#800080', edgecolor='#800080', label='EdU Number'),
                    Patch(facecolor='#118002', edgecolor='#118002', label='Nucleoid Number')]

    ax[3].legend(handles=legend_elements)
    plt.tight_layout()
    #plt.savefig("pulse_posterior_predictive_traj", dpi=300)

    fig2, ax2 = plt.subplots(1,4, figsize = (18,4))
    #Peak 1 proportions
    for i in range(4):

        ax2[i].hist(peak1_list[i], alpha = 0.7, density = True, label = "Posterior predictive") 
        ax2[i].axvline(0.33, color='k', linestyle='dashed', linewidth=1)
        ax2[i].axvline(0.39, color='k', linestyle='dashed', linewidth=1)
        ax2[i].axvline(0.29, color='k', linestyle='dashed', linewidth=1, label = "Observed proportion for 3 assays")
        ax2[i].set_xlim(0,1)
        ax2[i].set_yticks([],[])
        ax2[i].set_xlabel("24 hour singly tagged proportion")

    ax2[0].set_title("Three-population")
    ax2[1].set_title("Two-population")
    ax2[2].set_title("Preliminary two-population")
    ax2[3].set_title("One-population")
    ax2[0].set_ylabel("Probability density")
    ax2[3].legend()
    plt.tight_layout()
    #plt.savefig("pulse_posterior_predictive_singly_tagged_prop", dpi=300)

    fig3, ax3 = plt.subplots(1,4, figsize = (18,4))
    for i in range(4):
        ax3[i].hist(variance_statistics_list[i], alpha = 0.7, color = "tab:purple", density = True, label = "Posterior predictive") 
        ax3[i].axvline(0, color='k', linestyle='dashed', linewidth=1, label = "Observed")
        ax3[i].set_xlim(-2,2)
        ax3[i].set_yticks([],[])
        ax3[i].set_xlabel("$S_{cs}$")
    ax3[0].set_title("Three-population")
    ax3[1].set_title("Two-population")
    ax3[2].set_title("Preliminary two-population")
    ax3[3].set_title("One-population")
    ax3[0].set_ylabel("Probability density")
    ax3[3].legend()
    plt.tight_layout()
    #plt.savefig("pulse_posterior_predictive_control_strength", dpi=300)

def pulse_posterior_predictive_trajectories_poster():

    edu_means = all_assays_edu_number_pulse_means
    edu_err = all_assays_edu_number_pulse_err
    dna_means = all_assays_dna_number_pulse_means
    dna_err = all_assays_dna_number_pulse_err

    edu_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/preliminary/preliminary_pulse_edu_trajectories.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/one_population/pulse_edu_trajectories.npy"))]

    #fig, ax = plt.subplots(1,4, figsize = (23,5))
    plt.figure(figsize = (6,3))
    #Trajectories
    pulse_trajectory_time = np.arange(0,24.25,0.25)
    pulse_times = np.array([1,3,7,24])

    for k in range(len(edu_trajectories_list[0][:-1])):
        plt.plot(pulse_trajectory_time, edu_trajectories_list[1][k], c ="mediumaquamarine", alpha=0.2)
        plt.plot(pulse_trajectory_time, edu_trajectories_list[0][k], c ="plum", alpha=0.2)
    plt.plot(pulse_trajectory_time, edu_trajectories_list[0][-1], c ="plum", alpha=0.2, label = "EdU Number Posterior Predictive Draw")
    plt.plot(pulse_trajectory_time, np.mean(edu_trajectories_list[0], axis=0), c="#800080", linewidth=3, label = "EdU Number Posterior Predictive Mean")
    plt.plot(pulse_trajectory_time, edu_trajectories_list[1][-1], c ="mediumaquamarine", alpha=0.2, label = "EdU Number Posterior Predictive Draw")
    plt.plot(pulse_trajectory_time, np.mean(edu_trajectories_list[1], axis=0), c="#118002", linewidth=3, label = "EdU Number Posterior Predictive Mean")

    #ax[i].set_xticks([0,24,48,72,96], [0,1,2,3,4])
    plt.xlabel("Time (h)")
    plt.ylabel("DNA number")
    plt.title("Pulse data posterior predictive")

    legend_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior Predictive Mean'),
                    Line2D([0], [0], color='grey', lw=1.5, alpha = 0.4, label='Posterior Predictive Draw'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='black', markersize=8),
                    Patch(facecolor='#800080', edgecolor='#800080', label='Preliminary two pop'),
                    Patch(facecolor='#118002', edgecolor='#118002', label='One pop')]

    plt.legend(handles=legend_elements)
    plt.errorbar(pulse_times, edu_means, edu_err,  color='black', marker='o', ls = 'none', ecolor = 'black', label = "Edu Number Data Mean + 95% Confidence Interval")

def chase_trajectory_comparison1():

    edu_means = all_assays_edu_number_chase_means
    edu_err = all_assays_edu_number_chase_err
    dna_means = all_assays_dna_number_chase_means
    dna_err = all_assays_dna_number_chase_err

    edu_props = all_assays_edu_proportion_chase_means
    edu_props_err = all_assays_edu_proportion_chase_err

    nucleoid_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_nucleoid_trajectories.npy")),
                                  np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_nucleoid_trajectories.npy"))]
    
    edu_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_trajectories.npy")),
                             np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_trajectories.npy"))]
    
    edu_proportions_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_proportions.npy")),
                            np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_proportions.npy"))]

    chase_trajectory_time = np.arange(0,4*24+0.25,0.25)
    chase_times = np.array([0,24,48,96])
    fig, ax = plt.subplots(1,1, figsize = (4.5*0.9,3.5*0.9))
    ax.plot(chase_trajectory_time, np.mean(nucleoid_trajectories_list[0], axis=0), c="#118002", ls = "dashed", label = "Two Population")
    ax.plot(chase_trajectory_time, np.mean(nucleoid_trajectories_list[1], axis=0), c="#118002", label = "Three Population")
    ax.plot(chase_trajectory_time, np.mean(edu_trajectories_list[0], axis=0), c="#800080", ls = "dashed",  label = "Two Population")
    ax.plot(chase_trajectory_time, np.mean(edu_trajectories_list[1], axis=0), c="#800080", label = "Three Population")

    ax.errorbar(chase_times, edu_means, edu_err, color='#800080', marker='o', ls = 'none', ecolor = '#800080', label = "Edu Number Data Mean + 95% Confidence Interval")
    ax.errorbar(chase_times, dna_means, dna_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Nucleoid Number Data Mean + 95% Confidence Interval")

    legend_elements = [Line2D([0], [0], color='grey', label='Three population'),
                    Line2D([0], [0], color='grey', ls = "dashed", label="Two population"),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='grey', markersize=8)]
    ax.set_ylim(0,1500)

    ax.legend(handles=legend_elements, ncol = 1, loc= (0.2,0.7), fontsize = 16)

    ax.set_ylabel("Nucleoid number")
    ax.set_xlabel("Time (h)")
    ax.text(48, 450, "mtEdU", c = '#800080', fontsize=16)
    ax.text(0, 750, "mtDNA", c = '#118002', fontsize=16)

    ax.set_xticks([0,24,48,72,96], [0,24,48,72,96])
    #plt.tight_layout()

def chase_trajectory_comparison2():

    edu_means = all_assays_edu_number_chase_means
    edu_err = all_assays_edu_number_chase_err
    dna_means = all_assays_dna_number_chase_means
    dna_err = all_assays_dna_number_chase_err

    edu_props = all_assays_edu_proportion_chase_means
    edu_props_err = all_assays_edu_proportion_chase_err

    nucleoid_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_nucleoid_trajectories.npy")),
                                  np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_nucleoid_trajectories.npy"))]
    
    edu_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_trajectories.npy")),
                             np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_trajectories.npy"))]
    
    edu_proportions_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_proportions.npy")),
                            np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_proportions.npy"))]

    chase_trajectory_time = np.arange(0,4*24+0.25,0.25)
    chase_times = np.array([0,24,48,96])
    fig, ax = plt.subplots(1,1, figsize = (4.5*0.9,3.5*0.9))
    
    for i in range(len(edu_proportions_list[0])-1):
        ax.plot(chase_trajectory_time, edu_proportions_list[1][i], c ="bisque", alpha=0.2)
        ax.plot(chase_trajectory_time, edu_proportions_list[0][i], c ="lightskyblue", alpha=0.2)
    ax.plot(chase_trajectory_time, edu_proportions_list[1][-1], c ="bisque", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")
    ax.plot(chase_trajectory_time, edu_proportions_list[0][-1], c ="lightskyblue", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")

    #ax[1].plot(chase_trajectory_time, np.mean(edu_trajectories_list[0]/nucleoid_trajectories_list[0], axis=0), c="forestgreen", ls = "dashed", label = "Two Population")
    ax.plot(chase_trajectory_time, np.mean(edu_proportions_list[1], axis=0), c="darkorange", label = "Three Population")
    ax.plot(chase_trajectory_time, np.mean(edu_proportions_list[0], axis=0), c="mediumblue", label = "Three Population")
    
    ax.errorbar(chase_times, edu_props, edu_props_err, color = "black", marker = 'o', ls = 'none', ecolor = 'black', label = "")

    ax.set_ylim(0,1)

    # legend_elements = [Patch(facecolor='darkorange', edgecolor='darkorange', label='Three population'),
    #                 Patch(facecolor='mediumblue', edgecolor='mediumblue', label='Two population')]
    
    # legend2_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior predictive mean'),
    #         Line2D([0], [0], color='grey', lw=1.5, alpha = 0.4, label='Posterior predictive draw'),
    #         Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='black', markersize=8)]

    # legend1 = ax[1].legend(handles=  legend_elements, ncol=1, loc = "upper right")
    # ax[1].legend(handles = legend2_elements, loc = "lower right")
    # plt.gca().add_artist(legend1)

    legend_elements2 = [Line2D([0], [0], color='grey', lw=3, label='Posterior \npredictive'),
        Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='black', markersize=8)]
    
    ax.legend(handles=legend_elements2, ncol = 1, loc= (0.45,0.75))

    # legend1 = plt.legend(legend_elements, loc=1)
    # legend2 = plt.legend(legend2_elements, loc=4)
    # ax[1].add_artist(legend1)
    # ax[1].add_artist(legend2)
    ax.set_ylabel("mtEdU/mtDNA proportion")
    ax.set_xlabel("Time (h)")
    ax.set_xticks([0,24,48,72,96], [0,24,48,72,96])
    ax.text(30, 0.3, "Two population", c = "mediumblue", fontsize=16)
    ax.text(0, 0.65, "Three population", c = "darkorange", fontsize=16)
    plt.tight_layout()


def chase_trajectory_comparison_poster():

    edu_means = all_assays_edu_number_chase_means
    edu_err = all_assays_edu_number_chase_err
    dna_means = all_assays_dna_number_chase_means
    dna_err = all_assays_dna_number_chase_err

    edu_props = all_assays_edu_proportion_chase_means
    edu_props_err = all_assays_edu_proportion_chase_err

    nucleoid_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_nucleoid_trajectories.npy")),
                                  np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_nucleoid_trajectories.npy"))]
    
    edu_trajectories_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_trajectories.npy")),
                             np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_trajectories.npy"))]
    
    edu_proportions_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/chase_edu_proportions.npy")),
                            np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/chase_edu_proportions.npy"))]

    chase_trajectory_time = np.arange(0,4*24+0.25,0.25)
    chase_times = np.array([0,24,48,96])
    fig, ax = plt.subplots(1,2, figsize = (12,3))
    ax[0].plot(chase_trajectory_time, np.mean(nucleoid_trajectories_list[0], axis=0), c="#118002", ls = "dashed", label = "Two Population")
    ax[0].plot(chase_trajectory_time, np.mean(nucleoid_trajectories_list[1], axis=0), c="#118002", label = "Three Population")
    ax[0].plot(chase_trajectory_time, np.mean(edu_trajectories_list[0], axis=0), c="#800080", ls = "dashed",  label = "Two Population")
    ax[0].plot(chase_trajectory_time, np.mean(edu_trajectories_list[1], axis=0), c="#800080", label = "Three Population")

    ax[0].errorbar(chase_times, edu_means, edu_err, color='#800080', marker='o', ls = 'none', ecolor = '#800080', label = "Edu Number Data Mean + 95% Confidence Interval")
    ax[0].errorbar(chase_times, dna_means, dna_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Nucleoid Number Data Mean + 95% Confidence Interval")

    legend_elements = [Line2D([0], [0], color='grey', label='Three population'),
                    Line2D([0], [0], color='grey', ls = "dashed", label="Two population"),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='grey', markersize=8),
                    Patch(facecolor='#800080', edgecolor='#800080', label='mtEdU'),
                    Patch(facecolor='#118002', edgecolor='#118002', label='mtDNA')]

    ax[0].legend(handles=legend_elements, ncol = 1)

    ax[0].set_ylabel("DNA number")
    ax[0].set_xlabel("Time (h)")
    ax[0].set_title("Chase data posterior predictive")

    for i in range(len(edu_proportions_list[0])-1):
        ax[1].plot(chase_trajectory_time, edu_proportions_list[1][i], c ="bisque", alpha=0.2)
        ax[1].plot(chase_trajectory_time, edu_proportions_list[0][i], c ="lightskyblue", alpha=0.2)
    ax[1].plot(chase_trajectory_time, edu_proportions_list[1][-1], c ="bisque", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")
    ax[1].plot(chase_trajectory_time, edu_proportions_list[0][-1], c ="lightskyblue", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")

    #ax[1].plot(chase_trajectory_time, np.mean(edu_trajectories_list[0]/nucleoid_trajectories_list[0], axis=0), c="forestgreen", ls = "dashed", label = "Two Population")
    ax[1].plot(chase_trajectory_time, np.mean(edu_proportions_list[1], axis=0), c="darkorange", label = "Three Population")
    ax[1].plot(chase_trajectory_time, np.mean(edu_proportions_list[0], axis=0), c="mediumblue", label = "Three Population")
    
    ax[1].errorbar(chase_times, edu_props, edu_props_err, color = "black", marker = 'o', ls = 'none', ecolor = 'black', label = "")

    ax[1].set_ylim(0,1)

    legend_elements = [Patch(facecolor='darkorange', edgecolor='darkorange', label='Three population'),
                    Patch(facecolor='mediumblue', edgecolor='mediumblue', label='Two population')]
    
    legend2_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior predictive mean'),
            Line2D([0], [0], color='grey', lw=1.5, alpha = 0.4, label='Posterior predictive draw'),
            Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='black', markersize=8)]

    legend1 = ax[1].legend(handles=  legend_elements, ncol=1, loc = "upper right")
    ax[1].legend(handles = legend2_elements, loc = "lower right")
    plt.gca().add_artist(legend1)
    # legend1 = plt.legend(legend_elements, loc=1)
    # legend2 = plt.legend(legend2_elements, loc=4)
    # ax[1].add_artist(legend1)
    # ax[1].add_artist(legend2)
    ax[1].set_ylabel("mtEdU/mtDNA proportion")
    ax[1].set_xlabel("Time (h)")


def pulse_peak1_comparison():

    main_peak1_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/pulse_peak1.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/pulse_peak1.npy")),                  
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/preliminary/preliminary_pulse_peak1.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/one_population/pulse_peak1.npy"))]
    
    prel_twopop_values = np.arange(0.6,0.8, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.015).fit(main_peak1_list[2].reshape(-1,1))
    log_dens1 = kde1.score_samples(prel_twopop_values.reshape(-1,1))
    prel_twopop_kde = np.exp(log_dens1)
    print("prel_twopop_mean = " + str(np.mean(main_peak1_list[2])))
    print("prel_twopop_err = " + str(np.std(main_peak1_list[2])))

    one_pop_values = np.arange(0.6,0.8, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.015).fit(main_peak1_list[3].reshape(-1,1))
    log_dens1 = kde1.score_samples(one_pop_values.reshape(-1,1))
    one_pop_kde = np.exp(log_dens1)
    print("onepop_mean = " + str(np.mean(main_peak1_list[3])))
    print("onepop_err = " + str(np.std(main_peak1_list[3])))

    two_pop_values = np.arange(0.15,0.5, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(main_peak1_list[1].reshape(-1,1))
    log_dens1 = kde1.score_samples(two_pop_values.reshape(-1,1))
    two_pop_kde = np.exp(log_dens1)
    print("twopop_mean = " + str(np.mean(main_peak1_list[1])))
    print("twopop_err = " + str(np.std(main_peak1_list[1])))

    three_pop_values = np.arange(0.15,0.5, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(main_peak1_list[0].reshape(-1,1))
    log_dens1 = kde1.score_samples(three_pop_values.reshape(-1,1))
    three_pop_kde = np.exp(log_dens1)
    print("threepop_mean = " + str(np.mean(main_peak1_list[0])))
    print("threepop_err = " + str(np.std(main_peak1_list[0])))
    

    plt.figure(3, figsize = (5.25,4))
    plt.fill_between(three_pop_values,three_pop_kde, color="darkorange", alpha = 0.7, label = "Three population")
    plt.fill_between(two_pop_values,two_pop_kde, color="lightskyblue", alpha = 0.7, label = "Two population")
    plt.fill_between(prel_twopop_values,prel_twopop_kde, color="yellowgreen", alpha = 0.7, label = "Preliminary two population")
    plt.fill_between(one_pop_values,one_pop_kde, color="grey", alpha = 0.7, label = "One population")

    plt.axvline(0.33, color='k', linestyle='dashed', linewidth=1, label = "Observed proportion for 3 assays")
    plt.axvline(0.39, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(0.29, color='k', linestyle='dashed', linewidth=1)

    plt.xlabel("Single tagged proportion")
    plt.ylabel("Probability density")
    plt.ylim(0,30)
    plt.legend()
    plt.tight_layout()

def pulse_peak1_comparison_poster():

    main_peak1_list = [np.load(os.path.join(dirname, "simulated_posterior_predictive_data/three_population/pulse_peak1.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/pulse_peak1.npy")),                  
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/two_population/preliminary/preliminary_pulse_peak1.npy")),
                                np.load(os.path.join(dirname, "simulated_posterior_predictive_data/one_population/pulse_peak1.npy"))]
    
    prel_twopop_values = np.arange(0.6,0.8, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.015).fit(main_peak1_list[2].reshape(-1,1))
    log_dens1 = kde1.score_samples(prel_twopop_values.reshape(-1,1))
    prel_twopop_kde = np.exp(log_dens1)

    one_pop_values = np.arange(0.6,0.8, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.015).fit(main_peak1_list[3].reshape(-1,1))
    log_dens1 = kde1.score_samples(one_pop_values.reshape(-1,1))
    one_pop_kde = np.exp(log_dens1)

    two_pop_values = np.arange(0.15,0.5, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(main_peak1_list[1].reshape(-1,1))
    log_dens1 = kde1.score_samples(two_pop_values.reshape(-1,1))
    two_pop_kde = np.exp(log_dens1)

    three_pop_values = np.arange(0.15,0.5, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(main_peak1_list[0].reshape(-1,1))
    log_dens1 = kde1.score_samples(three_pop_values.reshape(-1,1))
    three_pop_kde = np.exp(log_dens1)

    plt.figure(3, figsize = (6,3))
    plt.fill_between(three_pop_values,three_pop_kde, color="darkorange", alpha = 0.7, label = "Three population")
    plt.fill_between(two_pop_values,two_pop_kde, color="lightskyblue", alpha = 0.7, label = "Two population")
    plt.fill_between(prel_twopop_values,prel_twopop_kde, color="yellowgreen", alpha = 0.7, label = "Preliminary two population")
    plt.fill_between(one_pop_values,one_pop_kde, color="grey", alpha = 0.7, label = "One population")

    plt.axvline(0.33, color='k', linestyle='dashed', linewidth=1, label = "Observed proportion for 3 assays")
    plt.axvline(0.39, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(0.29, color='k', linestyle='dashed', linewidth=1)

    plt.xlabel("Single tagged proportion")
    plt.ylabel("Probability density")
    plt.ylim(0,30)
    plt.legend()
    plt.title("24h single tagged proportion posterior predictive")

def global_turnover_rate(chase = True):
    #beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c
    pulse_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))
    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c = np.transpose(pulse_params)
    pulse_turnover_rate = 1/(1/true_birth_param + 1/mu_d_pulse + 1/(mu_b_pulse/diffusion_prob_pulse))
    print(np.mean(pulse_turnover_rate))
    print(np.std(pulse_turnover_rate)) 
    chase_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = np.transpose(chase_params)
    chase_turnover_rate = 1/(1/true_birth_param + 1/mu_d_pulse + 1/(mu_b_pulse/diffusion_prob_pulse))

    # plt.hist(pulse_turnover_rate, density = True)
    # plt.hist(chase_turnover_rate, density = True)

    pulse_values = np.arange(0.008,0.025, 0.0001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.0005).fit(pulse_turnover_rate.reshape(-1,1))
    log_dens1 = kde1.score_samples(pulse_values.reshape(-1,1))
    pulse_kde = np.exp(log_dens1)

    chase_values = np.arange(0.008,0.03, 0.0001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.0005).fit(chase_turnover_rate.reshape(-1,1))
    log_dens1 = kde1.score_samples(chase_values.reshape(-1,1))
    chase_kde = np.exp(log_dens1)

    plt.figure(1, figsize = (4.5,4))
    plt.fill_between(pulse_values,pulse_kde, color="b", alpha = 0.3, label = "Pulse Cells")
    if chase:
        plt.fill_between(chase_values,chase_kde, color="r", alpha = 0.3, label = "Pulse-Chase Cells")

    else:
        plt.axvline(x = np.mean(pulse_turnover_rate), ymin=0.05, ls = '--', color = "black")
        plt.text(0.015, 100, "Mean = 1.5%")
    plt.ylabel("Probability density")
    plt.xlabel("Birth rate per capita (h$^{\mathrm{-1}}$)")
    if chase:
        plt.legend()

def global_turnover_rate_different_models():
    #beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c
    three_pop_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))
    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c = np.transpose(three_pop_params)
    three_pop_turnover_rate = 1/(1/true_birth_param + 1/mu_d_pulse + 1/(mu_b_pulse/diffusion_prob_pulse))
    print("three population turnover rate = " + str(np.mean(three_pop_turnover_rate)) + "+/-" + str(np.std(three_pop_turnover_rate)))
    two_pop_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/two_population_pulse_accepted_params.npy"))
    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, true_birth_param, c = np.transpose(two_pop_params)
    two_pop_turnover_rate = 1/(1/true_birth_param + 1/mu_d_pulse)
    print("two population turnover rate = " + str(np.mean(two_pop_turnover_rate)) + "+/-" + str(np.std(two_pop_turnover_rate)))
    prel_two_pop_params = np.load(os.path.join(dirname, "simulated_ABC_data/two_population_model/preliminary/preliminary_two_population_pulse_accepted_params.npy"))
    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, true_birth_param, c = np.transpose(prel_two_pop_params)
    prel_two_pop_turnover_rate = 1/(1/true_birth_param + 1/mu_d_pulse)
    print("preliminary two population turnover rate = " + str(np.mean(prel_two_pop_turnover_rate)) + "+/-" + str(np.std(prel_two_pop_turnover_rate)))

    three_pop_values = np.arange(0.008,0.025, 0.0001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.0005).fit(three_pop_turnover_rate.reshape(-1,1))
    log_dens1 = kde1.score_samples(three_pop_values.reshape(-1,1))
    three_pop_kde = np.exp(log_dens1)

    two_pop_values = np.arange(0.008,0.025, 0.0001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.0005).fit(two_pop_turnover_rate.reshape(-1,1))
    log_dens1 = kde1.score_samples(two_pop_values.reshape(-1,1))
    two_pop_kde = np.exp(log_dens1)

    prel_two_pop_values = np.arange(0.004,0.015, 0.0001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.0005).fit(prel_two_pop_turnover_rate.reshape(-1,1))
    log_dens1 = kde1.score_samples(prel_two_pop_values.reshape(-1,1))
    prel_two_pop_kde = np.exp(log_dens1)

    plt.figure(1, figsize = (4.5,4))
    plt.fill_between(three_pop_values,three_pop_kde, color="darkorange", alpha = 0.3, label = "Three-population")
    plt.fill_between(two_pop_values,two_pop_kde, color="lightskyblue", alpha = 0.3, label = "Two-population")
    plt.fill_between(prel_two_pop_values,prel_two_pop_kde, color="yellowgreen", alpha = 0.3, label = "Preliminary two-population")

    plt.ylabel("Probability density")
    plt.xlabel("Turnover rate per capita (h$^{\mathrm{-1}}$)")
    plt.legend()


def global_turnover_rate_poster(chase = True):
    #beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c
    pulse_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_pulse_accepted_params.npy"))
    beta0, beta1, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, c = np.transpose(pulse_params)
    pulse_turnover_rate = 1/(1/true_birth_param + 1/mu_d_pulse + 1/(mu_b_pulse/diffusion_prob_pulse))

    chase_params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = np.transpose(chase_params)
    chase_turnover_rate = 1/(1/true_birth_param + 1/mu_d_pulse + 1/(mu_b_pulse/diffusion_prob_pulse))

    # plt.hist(pulse_turnover_rate, density = True)
    # plt.hist(chase_turnover_rate, density = True)

    pulse_values = np.arange(0.008,0.025, 0.0001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.0005).fit(pulse_turnover_rate.reshape(-1,1))
    log_dens1 = kde1.score_samples(pulse_values.reshape(-1,1))
    pulse_kde = np.exp(log_dens1)

    chase_values = np.arange(0.008,0.03, 0.0001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.0005).fit(chase_turnover_rate.reshape(-1,1))
    log_dens1 = kde1.score_samples(chase_values.reshape(-1,1))
    chase_kde = np.exp(log_dens1)

    plt.figure(1, figsize = (6,3))
    plt.fill_between(pulse_values,pulse_kde, color="#2F5597", alpha = 1, label = "Pulse Cells")
    if chase:
        plt.fill_between(chase_values,chase_kde, color="r", alpha = 0.3, label = "Pulse-Chase Cells")

    else:
        plt.axvline(x = np.mean(pulse_turnover_rate), ymin=0.05, ls = '--', color = "black")
        plt.text(0.015, 100, "Mean = 1.5%")
    plt.ylabel("Probability density")
    plt.xlabel("Birth and death rate per capita")
    plt.title("Birth rate posterior")
    if chase:
        plt.legend()


def analytical_old_young(params):
    _, _, _, p, mu_d, mu_b, mu_r, _, _, _, _, _ = params 
    mu_a = mu_b/p
    t_r = 2/((1+p)*(mu_r*mu_a))*(mu_b+mu_a + mu_r*(1+p)/2)
    t_y = 2/((1+p)*(mu_r*mu_a))*(mu_b+mu_r*(1+p)/2)
    denom = 1/mu_r + 1/mu_a
    t = 1/(mu_r*denom)*t_r + 1/(mu_a*denom)*t_y
    return t

def population_proportions():
    #beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c
    params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = np.transpose(params)
    mu_a = mu_b_pulse/diffusion_prob_pulse

    denom = 1/true_birth_param + 1/mu_d_pulse + 1/(mu_a)
    replicating_prop = 1/(true_birth_param*denom)
    young_prop = 1/(mu_a*denom)
    old_prop = 1/(mu_d_pulse*denom)
    print("replicating_mean = " + str(np.mean(replicating_prop)))
    print("replicating_err = " + str(np.std(replicating_prop)))
    print("young_mean = " + str(np.mean(young_prop)))
    print("young_err = " + str(np.std(young_prop)))
    print("old_mean = " + str(np.mean(old_prop)))
    print("old_err = " + str(np.std(old_prop)))
    print("replicative_mean = " + str(np.mean(replicating_prop + young_prop)))
    print("replicative_err = " + str(np.std(replicating_prop +young_prop)))


    replicating_values = np.arange(0,0.2, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(replicating_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicating_values.reshape(-1,1))
    replicating_kde = np.exp(log_dens1)

    young_values = np.arange(0,0.6, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(young_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(young_values.reshape(-1,1))
    young_kde = np.exp(log_dens1)

    old_values = np.arange(0.4,1, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(old_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(old_values.reshape(-1,1))
    old_kde = np.exp(log_dens1)

    plt.figure(figsize = (5.25,4))
    plt.fill_between(replicating_values,replicating_kde, color="y", alpha = 0.7, label = "Replicating")
    plt.fill_between(young_values,young_kde, color="#118002", alpha = 0.7, label = "Young")
    plt.fill_between(old_values,old_kde, color="grey", alpha = 0.7, label = "Old")
    plt.xlabel("Fraction")
    plt.ylabel("Probability density")
    plt.ylim(0,17)
    plt.legend()
    plt.tight_layout()


def population_proportions2():
    #beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c
    params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = np.transpose(params)
    mu_a = mu_b_pulse/diffusion_prob_pulse

    denom = 1/true_birth_param + 1/mu_d_pulse + 1/(mu_a)
    replicating_prop = 1/(true_birth_param*denom)
    young_prop = 1/(mu_a*denom)
    old_prop = 1/(mu_d_pulse*denom)
    replicative_prop = young_prop + replicating_prop

    replicating_values = np.arange(0,0.2, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(replicating_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicating_values.reshape(-1,1))
    replicating_kde = np.exp(log_dens1)

    replicative_values = np.arange(0,0.8, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(replicative_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicative_values.reshape(-1,1))
    replicative_kde = np.exp(log_dens1)

    old_values = np.arange(0.4,1, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(old_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(old_values.reshape(-1,1))
    old_kde = np.exp(log_dens1)

    plt.figure(figsize = (4.5,3.5))
    plt.fill_between(replicating_values,replicating_kde, color="y", alpha = 0.7, label = "Replicating")
    plt.fill_between(replicative_values,replicative_kde, color="#118002", alpha = 0.7, label = "Replicative")
    plt.fill_between(old_values,old_kde, color="grey", alpha = 0.7, label = "Old")
    plt.xlabel("Fraction")
    plt.ylabel("Probability density")
    plt.ylim(0,17)
    plt.legend()
    plt.tight_layout()

def dwell_times():
    #beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c
    params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = np.transpose(params)
    mu_a = mu_b_pulse/diffusion_prob_pulse

    replicating_time = 1/(true_birth_param)
    replicative_time = analytical_old_young(params.transpose())
    old_time = 1/(mu_d_pulse)

    replicating_values = np.arange(0,10, 0.1)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.4).fit(replicating_time.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicating_values.reshape(-1,1))
    replicating_kde = np.exp(log_dens1)

    replicative_values = np.arange(5,30, 0.1)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.9).fit(replicative_time.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicative_values.reshape(-1,1))
    replicative_kde = np.exp(log_dens1)

    old_values = np.arange(20,60, 0.1)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.9).fit(old_time.reshape(-1,1))
    log_dens1 = kde1.score_samples(old_values.reshape(-1,1))
    old_kde = np.exp(log_dens1)

    plt.figure(figsize = (4.5,3.5))
    plt.fill_between(replicating_values,replicating_kde, color="y", alpha = 0.7, label = "Replicating before replication")
    plt.fill_between(replicative_values,replicative_kde, color="#118002", alpha = 0.7, label = "Replicative before ageing")
    plt.fill_between(old_values,old_kde, color="grey", alpha = 0.7, label = "Old before death")
    plt.xlabel("Dwell time (h)")
    plt.ylabel("Probability density")
    plt.ylim(0,0.45)
    plt.legend(loc = (0,0.7))
    plt.tight_layout()

def population_proportions_poster():
    #beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c
    params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = np.transpose(params)
    mu_a = mu_b_pulse/diffusion_prob_pulse

    denom = 1/true_birth_param + 1/mu_d_pulse + 1/(mu_a)
    replicating_prop = 1/(true_birth_param*denom)
    young_prop = 1/(mu_a*denom)
    old_prop = 1/(mu_d_pulse*denom)

    replicating_values = np.arange(0,0.2, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.01).fit(replicating_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(replicating_values.reshape(-1,1))
    replicating_kde = np.exp(log_dens1)

    young_values = np.arange(0,0.6, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(young_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(young_values.reshape(-1,1))
    young_kde = np.exp(log_dens1)

    old_values = np.arange(0.4,1, 0.001)
    kde1 = KernelDensity(kernel='gaussian', bandwidth = 0.02).fit(old_prop.reshape(-1,1))
    log_dens1 = kde1.score_samples(old_values.reshape(-1,1))
    old_kde = np.exp(log_dens1)

    plt.figure(figsize = (6,3))
    plt.fill_between(replicating_values,replicating_kde, color="y", alpha = 0.7, label = "Replicating")
    plt.fill_between(young_values,young_kde, color="#118002", alpha = 0.7, label = "Young")
    plt.fill_between(old_values,old_kde, color="grey", alpha = 0.7, label = "Old")
    plt.xlabel("Fraction")
    plt.ylabel("Probability density")
    plt.title("Population proportion posteriors")
    plt.ylim(0,17)
    plt.legend()

def population_proportions_3d():
    #beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c
    params = np.load(os.path.join(dirname, "simulated_ABC_data/three_population_model/three_population_chase_accepted_params.npy"))
    beta0, beta1, sig, diffusion_prob_pulse, mu_d_pulse, mu_b_pulse, true_birth_param, mu_b_chase, diffusion_prob_chase, mu_d_chase, mu_a_chase, c = np.transpose(params)
    mu_a = mu_b_pulse/diffusion_prob_pulse

    denom = 1/true_birth_param + 1/mu_d_pulse + 1/(mu_a)
    replicating_prop = 1/(true_birth_param*denom)
    young_prop = 1/(mu_a*denom)
    old_prop = 1/(mu_d_pulse*denom)

    fig = go.Figure(data=[go.Scatter3d(x=young_prop, y=old_prop, z=replicating_prop,
                                    mode='markers',
                                    marker=dict(
                                                    size=4,
                                                    color=young_prop,                # set color to an array/list of desired values
                                                    colorscale='Viridis',   # choose a colorscale
                                                    opacity=0.8
                                                ))])
    fig.update_layout(scene = dict(
                        xaxis=dict(
                            title=dict(
                                text='Young Proportion'
                            )
                        ),
                        yaxis=dict(
                            title=dict(
                                text='Old Proportion'
                            )
                        ),
                        zaxis=dict(
                            title=dict(
                                text='Replicating Proportion'
                            )
                        ),
                        ),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10))  

    fig.show()

def single_posterior_predictive_pulse_traj(nucleoid_trajectories, edu_trajectories, mode = "full"):
    #fig, ax = plt.subplots(1,3, figsize = (23,5))
    #plt.figure(1, figsize = (4.5,3))
    plt.figure(1, figsize = (6,4))
    #fig, ax = plt.subplots(1,3, figsize = (18,4))

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

    for traj in nucleoid_trajectories[:-1]:
        plt.plot(pulse_trajectory_time, traj, c ="mediumaquamarine", alpha=0.2)
    plt.plot(pulse_trajectory_time, nucleoid_trajectories[-1], c ="mediumaquamarine", alpha=0.2, label = "Nucleoid Number Posterior Predictive Draw")
    plt.plot(pulse_trajectory_time, np.mean(nucleoid_trajectories, axis=0), c="#118002", linewidth=3, label = "Nucleoid Number Posterior Predictive Mean")

    for traj in edu_trajectories[:-1]:
        plt.plot(pulse_trajectory_time, traj, c ="plum", alpha=0.2)
    plt.plot(pulse_trajectory_time, edu_trajectories[-1], c ="plum", alpha=0.2, label = "EdU Number Posterior Predictive Draw")
    plt.plot(pulse_trajectory_time, np.mean(edu_trajectories, axis=0), c="#800080", linewidth=3, label = "EdU Number Posterior Predictive Mean")

    plt.errorbar(pulse_times, edu_means, edu_err, color='#800080', marker='o', ls = 'none', ecolor = '#800080', label = "Edu Number Data Mean + 95% Confidence Interval")
    plt.errorbar(pulse_times, dna_means, dna_err, color='#118002', marker='o', ls = 'none', ecolor = '#118002', label = "Nucleoid Number Data Mean + 95% Confidence Interval")
    plt.xlabel("Time (h)")
    plt.ylabel("Nucleoid number per cell")


    legend_elements = [Line2D([0], [0], color='grey', lw=3, label='Posterior predictive'),
                    Line2D([0], [0], marker='o', color='w', label='Observed',markerfacecolor='grey', markersize=8),
                    Patch(facecolor='#800080', edgecolor='#800080', label='MtEdU number'),
                    Patch(facecolor='#118002', edgecolor='#118002', label='Nucleoid number')]

    plt.legend(handles=legend_elements)


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

def single_cell_posterior_predictive_pulse(nucleoid_1hr, nucleoid_3hr, nucleoid_7hr, nucleoid_24hr, edu_1hr, edu_3hr, edu_7hr, edu_24hr):
    nucleoid_1hr_confints = []
    nucleoid_3hr_confints = []
    nucleoid_7hr_confints = []
    nucleoid_24hr_confints = []

    for cell in nucleoid_1hr.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_1hr_confints.append([cell[12], cell[-13]])

    for cell in nucleoid_3hr.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_3hr_confints.append([cell[12], cell[-13]])

    for cell in nucleoid_7hr.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_7hr_confints.append([cell[12], cell[-13]])

    for cell in nucleoid_24hr.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_24hr_confints.append([cell[12], cell[-13]])

    nucleoid_1hr_confints = np.array(nucleoid_1hr_confints)
    nucleoid_1hr_confints = nucleoid_1hr_confints[np.argsort(all_assays_mito_length_1hr), :]
    nucleoid_3hr_confints = np.array(nucleoid_3hr_confints)
    nucleoid_3hr_confints = nucleoid_3hr_confints[np.argsort(all_assays_mito_length_3hr), :]
    nucleoid_7hr_confints = np.array(nucleoid_7hr_confints)
    nucleoid_7hr_confints = nucleoid_7hr_confints[np.argsort(all_assays_mito_length_7hr), :]
    nucleoid_24hr_confints = np.array(nucleoid_24hr_confints)
    nucleoid_24hr_confints = nucleoid_24hr_confints[np.argsort(all_assays_mito_length_24hr), :]

    edu_1hr_confints = []
    edu_3hr_confints = []
    edu_7hr_confints = []
    edu_24hr_confints = []

    for cell in edu_1hr.transpose():
        cell = cell.copy()
        cell.sort()
        edu_1hr_confints.append([cell[12], cell[-13]])

    for cell in edu_3hr.transpose():
        cell = cell.copy()
        cell.sort()
        edu_3hr_confints.append([cell[12], cell[-13]])

    for cell in edu_7hr.transpose():
        cell = cell.copy()
        cell.sort()
        edu_7hr_confints.append([cell[12], cell[-13]])

    for cell in edu_24hr.transpose():
        cell = cell.copy()
        cell.sort()
        edu_24hr_confints.append([cell[12], cell[-13]])

    edu_1hr_confints = np.array(edu_1hr_confints)
    edu_1hr_confints = edu_1hr_confints[np.argsort(all_assays_mito_length_1hr), :]
    edu_3hr_confints = np.array(edu_3hr_confints)
    edu_3hr_confints = edu_3hr_confints[np.argsort(all_assays_mito_length_3hr), :]
    edu_7hr_confints = np.array(edu_7hr_confints)
    edu_7hr_confints = edu_7hr_confints[np.argsort(all_assays_mito_length_7hr), :]
    edu_24hr_confints = np.array(edu_24hr_confints)
    edu_24hr_confints = edu_24hr_confints[np.argsort(all_assays_mito_length_24hr), :]

    observed_nucleoid_values_1hr = all_assays_dna_number_1hr[np.argsort(all_assays_mito_length_1hr)]
    observed_nucleoid_values_3hr = all_assays_dna_number_3hr[np.argsort(all_assays_mito_length_3hr)]
    observed_nucleoid_values_7hr = all_assays_dna_number_7hr[np.argsort(all_assays_mito_length_7hr)]
    observed_nucleoid_values_24hr = all_assays_dna_number_24hr[np.argsort(all_assays_mito_length_24hr)]

    observed_edu_values_1hr = all_assays_edu_number_1hr[np.argsort(all_assays_mito_length_1hr)]
    observed_edu_values_3hr = all_assays_edu_number_3hr[np.argsort(all_assays_mito_length_3hr)]
    observed_edu_values_7hr = all_assays_edu_number_7hr[np.argsort(all_assays_mito_length_7hr)]
    observed_edu_values_24hr = all_assays_edu_number_24hr[np.argsort(all_assays_mito_length_24hr)]

    observed_mito_values_1hr = all_assays_mito_length_1hr[np.argsort(all_assays_mito_length_1hr)]
    observed_mito_values_3hr = all_assays_mito_length_3hr[np.argsort(all_assays_mito_length_3hr)]
    observed_mito_values_7hr = all_assays_mito_length_7hr[np.argsort(all_assays_mito_length_7hr)]
    observed_mito_values_24hr = all_assays_mito_length_24hr[np.argsort(all_assays_mito_length_24hr)]
    fig, ax = plt.subplots(1,4, figsize = (18,3))

    ax[0].scatter(observed_mito_values_1hr, observed_nucleoid_values_1hr,s=1, color = "black", label = "Individual cells")
    ax[0].fill_between(observed_mito_values_1hr, nucleoid_1hr_confints.transpose()[0], nucleoid_1hr_confints.transpose()[1], color = "#118002", alpha = 0.2, label = "Pointwise 95% credible interval")
    ax[0].set_ylim(0,5200)
    ax[0].set_ylabel("Nucleoid Number")
    ax[0].legend()
    ax[0].set_title("1 hour")
    ax[0].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[1].scatter(observed_mito_values_3hr, observed_nucleoid_values_3hr,s=1, color = "black")
    ax[1].fill_between(observed_mito_values_3hr, nucleoid_3hr_confints.transpose()[0], nucleoid_3hr_confints.transpose()[1], color = "#118002", alpha = 0.2)
    ax[1].set_ylim(0,5200)
    ax[1].set_yticks([],[])
    ax[1].set_title("3 hours")
    ax[1].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[2].scatter(observed_mito_values_7hr, observed_nucleoid_values_7hr,s=1, color = "black")
    ax[2].fill_between(observed_mito_values_7hr, nucleoid_7hr_confints.transpose()[0], nucleoid_7hr_confints.transpose()[1], color = "#118002", alpha = 0.2)
    ax[2].set_ylim(0,5200)
    ax[2].set_yticks([],[])
    ax[2].set_title("7 hours")
    ax[2].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[3].scatter(observed_mito_values_24hr, observed_nucleoid_values_24hr,s=1, color = "black")
    ax[3].fill_between(observed_mito_values_24hr, nucleoid_24hr_confints.transpose()[0], nucleoid_24hr_confints.transpose()[1], color = "#118002", alpha = 0.2)
    ax[3].set_ylim(0,5200)
    ax[3].set_yticks([],[])
    ax[3].set_title("24 hours")
    ax[3].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    plt.tight_layout()

    nuc_in_confint_1hr = (nucleoid_1hr_confints.transpose()[0] < observed_nucleoid_values_1hr) & (nucleoid_1hr_confints.transpose()[1] > observed_nucleoid_values_1hr)
    nuc_in_confint_3hr = (nucleoid_3hr_confints.transpose()[0] < observed_nucleoid_values_3hr) & (nucleoid_3hr_confints.transpose()[1] > observed_nucleoid_values_3hr)
    nuc_in_confint_7hr = (nucleoid_7hr_confints.transpose()[0] < observed_nucleoid_values_7hr) & (nucleoid_7hr_confints.transpose()[1] > observed_nucleoid_values_7hr)
    nuc_in_confint_24hr = (nucleoid_24hr_confints.transpose()[0] < observed_nucleoid_values_24hr) & (nucleoid_24hr_confints.transpose()[1] > observed_nucleoid_values_24hr)
    within_confint = np.sum(nuc_in_confint_1hr) + np.sum(nuc_in_confint_3hr) + np.sum(nuc_in_confint_7hr) + np.sum(nuc_in_confint_24hr)
    total = len(nuc_in_confint_1hr) + len(nuc_in_confint_3hr) + len(nuc_in_confint_7hr) + len(nuc_in_confint_24hr)
    print("Number of cells with nucleoid number within credible interval = " + str(within_confint))
    print("Number of total cells = " + str(total))
    print("Proportion of cells with nucleoid number within credible intervals = " + str(within_confint/total))
    fig, ax = plt.subplots(1,4, figsize = (18,3))

    ax[0].scatter(observed_mito_values_1hr, observed_edu_values_1hr,s=1, color = "black", label = "Individual cells")
    ax[0].fill_between(observed_mito_values_1hr, edu_1hr_confints.transpose()[0], edu_1hr_confints.transpose()[1], color = "#800080", alpha = 0.2, label = "Pointwise 95% credible interval")
    ax[0].set_ylim(0,2000)
    ax[0].set_ylabel("MtEdU Number")
    ax[0].legend()
    ax[0].set_title("1 hour")
    ax[0].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[1].scatter(observed_mito_values_3hr, observed_edu_values_3hr,s=1, color = "black")
    ax[1].fill_between(observed_mito_values_3hr, edu_3hr_confints.transpose()[0], edu_3hr_confints.transpose()[1], color = "#800080", alpha = 0.2)
    ax[1].set_ylim(0,2000)
    ax[1].set_yticks([],[])
    ax[1].set_title("3 hours")
    ax[1].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[2].scatter(observed_mito_values_7hr, observed_edu_values_7hr,s=1, color = "black")
    ax[2].fill_between(observed_mito_values_7hr, edu_7hr_confints.transpose()[0], edu_7hr_confints.transpose()[1], color = "#800080", alpha = 0.2)
    ax[2].set_ylim(0,2000)
    ax[2].set_yticks([],[])
    ax[2].set_title("7 hours")
    ax[2].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[3].scatter(observed_mito_values_24hr, observed_edu_values_24hr,s=1, color = "black")
    ax[3].fill_between(observed_mito_values_24hr, edu_24hr_confints.transpose()[0], edu_24hr_confints.transpose()[1], color = "#800080", alpha = 0.2)
    ax[3].set_ylim(0,2000)
    ax[3].set_yticks([],[])
    ax[3].set_title("24 hours")
    ax[3].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    plt.tight_layout()

    edu_in_confint_1hr = (edu_1hr_confints.transpose()[0] < observed_edu_values_1hr) & (edu_1hr_confints.transpose()[1] > observed_edu_values_1hr)
    edu_in_confint_3hr = (edu_3hr_confints.transpose()[0] < observed_edu_values_3hr) & (edu_3hr_confints.transpose()[1] > observed_edu_values_3hr)
    edu_in_confint_7hr = (edu_7hr_confints.transpose()[0] < observed_edu_values_7hr) & (edu_7hr_confints.transpose()[1] > observed_edu_values_7hr)
    edu_in_confint_24hr = (edu_24hr_confints.transpose()[0] < observed_edu_values_24hr) & (edu_24hr_confints.transpose()[1] > observed_edu_values_24hr)
    within_confint = np.sum(edu_in_confint_1hr) + np.sum(edu_in_confint_3hr) + np.sum(edu_in_confint_7hr) + np.sum(edu_in_confint_24hr)
    total = len(edu_in_confint_1hr) + len(edu_in_confint_3hr) + len(edu_in_confint_7hr) + len(edu_in_confint_24hr)
    print("Number of cells with mtEdU number within credible interval = " + str(within_confint))
    print("Number of total cells = " + str(total))
    print("Proportion of cells with mtEdU number within credible intervals = " + str(within_confint/total))

def single_cell_posterior_predictive_chase(nucleoid_0dy, nucleoid_1dy, nucleoid_2dy, nucleoid_4dy, edu_0dy, edu_1dy, edu_2dy, edu_4dy):
    nucleoid_0dy_confints = []
    nucleoid_1dy_confints = []
    nucleoid_2dy_confints = []
    nucleoid_4dy_confints = []

    for cell in nucleoid_0dy.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_0dy_confints.append([cell[12], cell[-13]])

    for cell in nucleoid_1dy.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_1dy_confints.append([cell[12], cell[-13]])

    for cell in nucleoid_2dy.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_2dy_confints.append([cell[12], cell[-13]])

    for cell in nucleoid_4dy.transpose():
        cell = cell.copy()
        cell.sort()
        nucleoid_4dy_confints.append([cell[12], cell[-13]])

    nucleoid_0dy_confints = np.array(nucleoid_0dy_confints)
    nucleoid_0dy_confints = nucleoid_0dy_confints[np.argsort(all_assays_mito_length_0dy), :]
    nucleoid_1dy_confints = np.array(nucleoid_1dy_confints)
    nucleoid_1dy_confints = nucleoid_1dy_confints[np.argsort(all_assays_mito_length_1dy), :]
    nucleoid_2dy_confints = np.array(nucleoid_2dy_confints)
    nucleoid_2dy_confints = nucleoid_2dy_confints[np.argsort(all_assays_mito_length_2dy), :]
    nucleoid_4dy_confints = np.array(nucleoid_4dy_confints)
    nucleoid_4dy_confints = nucleoid_4dy_confints[np.argsort(all_assays_mito_length_4dy), :]

    edu_0dy_confints = []
    edu_1dy_confints = []
    edu_2dy_confints = []
    edu_4dy_confints = []

    for cell in edu_0dy.transpose():
        cell = cell.copy()
        cell.sort()
        edu_0dy_confints.append([cell[12], cell[-13]])

    for cell in edu_1dy.transpose():
        cell = cell.copy()
        cell.sort()
        edu_1dy_confints.append([cell[12], cell[-13]])

    for cell in edu_2dy.transpose():
        cell = cell.copy()
        cell.sort()
        edu_2dy_confints.append([cell[12], cell[-13]])

    for cell in edu_4dy.transpose():
        cell = cell.copy()
        cell.sort()
        edu_4dy_confints.append([cell[12], cell[-13]])

    edu_0dy_confints = np.array(edu_0dy_confints)
    edu_0dy_confints = edu_0dy_confints[np.argsort(all_assays_mito_length_0dy), :]
    edu_1dy_confints = np.array(edu_1dy_confints)
    edu_1dy_confints = edu_1dy_confints[np.argsort(all_assays_mito_length_1dy), :]
    edu_2dy_confints = np.array(edu_2dy_confints)
    edu_2dy_confints = edu_2dy_confints[np.argsort(all_assays_mito_length_2dy), :]
    edu_4dy_confints = np.array(edu_4dy_confints)
    edu_4dy_confints = edu_4dy_confints[np.argsort(all_assays_mito_length_4dy), :]

    observed_nucleoid_values_0dy = all_assays_dna_number_0dy[np.argsort(all_assays_mito_length_0dy)]
    observed_nucleoid_values_1dy = all_assays_dna_number_1dy[np.argsort(all_assays_mito_length_1dy)]
    observed_nucleoid_values_2dy = all_assays_dna_number_2dy[np.argsort(all_assays_mito_length_2dy)]
    observed_nucleoid_values_4dy = all_assays_dna_number_4dy[np.argsort(all_assays_mito_length_4dy)]

    observed_edu_values_0dy = all_assays_edu_number_0dy[np.argsort(all_assays_mito_length_0dy)]
    observed_edu_values_1dy = all_assays_edu_number_1dy[np.argsort(all_assays_mito_length_1dy)]
    observed_edu_values_2dy = all_assays_edu_number_2dy[np.argsort(all_assays_mito_length_2dy)]
    observed_edu_values_4dy = all_assays_edu_number_4dy[np.argsort(all_assays_mito_length_4dy)]

    observed_mito_values_0dy = all_assays_mito_length_0dy[np.argsort(all_assays_mito_length_0dy)]
    observed_mito_values_1dy = all_assays_mito_length_1dy[np.argsort(all_assays_mito_length_1dy)]
    observed_mito_values_2dy = all_assays_mito_length_2dy[np.argsort(all_assays_mito_length_2dy)]
    observed_mito_values_4dy = all_assays_mito_length_4dy[np.argsort(all_assays_mito_length_4dy)]

    fig, ax = plt.subplots(1,4, figsize = (18,3))

    ax[0].scatter(observed_mito_values_0dy, observed_nucleoid_values_0dy,s=1, color = "black", label = "Individual cells")
    ax[0].fill_between(observed_mito_values_0dy, nucleoid_0dy_confints.transpose()[0], nucleoid_0dy_confints.transpose()[1], color = "#118002", alpha = 0.2, label = "Pointwise 95% credible interval")
    ax[0].set_ylim(0,5200)
    ax[0].set_ylabel("Nucleoid Number")
    ax[0].legend()
    ax[0].set_title("24 hours")
    ax[0].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[1].scatter(observed_mito_values_1dy, observed_nucleoid_values_1dy,s=1, color = "black")
    ax[1].fill_between(observed_mito_values_1dy, nucleoid_1dy_confints.transpose()[0], nucleoid_1dy_confints.transpose()[1], color = "#118002", alpha = 0.2)
    ax[1].set_ylim(0,5200)
    ax[1].set_yticks([],[])
    ax[1].set_title("48 hours")
    ax[1].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[2].scatter(observed_mito_values_2dy, observed_nucleoid_values_2dy,s=1, color = "black")
    ax[2].fill_between(observed_mito_values_2dy, nucleoid_2dy_confints.transpose()[0], nucleoid_2dy_confints.transpose()[1], color = "#118002", alpha = 0.2)
    ax[2].set_ylim(0,5200)
    ax[2].set_yticks([],[])
    ax[2].set_title("72 hours")
    ax[2].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[3].scatter(observed_mito_values_4dy, observed_nucleoid_values_4dy,s=1, color = "black")
    ax[3].fill_between(observed_mito_values_4dy, nucleoid_4dy_confints.transpose()[0], nucleoid_4dy_confints.transpose()[1], color = "#118002", alpha = 0.2)
    ax[3].set_ylim(0,5200)
    ax[3].set_yticks([],[])
    ax[3].set_title("120 hours")
    ax[3].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    nuc_in_confint_0dy = (nucleoid_0dy_confints.transpose()[0] < observed_nucleoid_values_0dy) & (nucleoid_0dy_confints.transpose()[1] > observed_nucleoid_values_0dy)
    nuc_in_confint_1dy = (nucleoid_1dy_confints.transpose()[0] < observed_nucleoid_values_1dy) & (nucleoid_1dy_confints.transpose()[1] > observed_nucleoid_values_1dy)
    nuc_in_confint_2dy = (nucleoid_2dy_confints.transpose()[0] < observed_nucleoid_values_2dy) & (nucleoid_2dy_confints.transpose()[1] > observed_nucleoid_values_2dy)
    nuc_in_confint_4dy = (nucleoid_4dy_confints.transpose()[0] < observed_nucleoid_values_4dy) & (nucleoid_4dy_confints.transpose()[1] > observed_nucleoid_values_4dy)
    within_confint = np.sum(nuc_in_confint_0dy) + np.sum(nuc_in_confint_1dy) + np.sum(nuc_in_confint_2dy) + np.sum(nuc_in_confint_4dy)
    total = len(nuc_in_confint_0dy) + len(nuc_in_confint_1dy) + len(nuc_in_confint_2dy) + len(nuc_in_confint_4dy)
    print("Number of cells with nucleoid number within credible interval = " + str(within_confint))
    print("Number of total cells = " + str(total))
    print("Proportion of cells with nucleoid number within credible intervals = " + str(within_confint/total))

    plt.tight_layout()
    fig, ax = plt.subplots(1,4, figsize = (18,3))

    ax[0].scatter(observed_mito_values_0dy, observed_edu_values_0dy,s=1, color = "black", label = "Individual cells")
    ax[0].fill_between(observed_mito_values_0dy, edu_0dy_confints.transpose()[0], edu_0dy_confints.transpose()[1], color = "#800080", alpha = 0.2, label = "Pointwise 95% credible interval")
    ax[0].set_ylim(0,3000)
    ax[0].set_ylabel("MtEdU Number")
    ax[0].legend()
    ax[0].set_title("24 hours")
    ax[0].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[1].scatter(observed_mito_values_1dy, observed_edu_values_1dy,s=1, color = "black")
    ax[1].fill_between(observed_mito_values_1dy, edu_1dy_confints.transpose()[0], edu_1dy_confints.transpose()[1], color = "#800080", alpha = 0.2)
    ax[1].set_ylim(0,3000)
    ax[1].set_yticks([],[])
    ax[1].set_title("48 hours")
    ax[1].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[2].scatter(observed_mito_values_2dy, observed_edu_values_2dy,s=1, color = "black")
    ax[2].fill_between(observed_mito_values_2dy, edu_2dy_confints.transpose()[0], edu_2dy_confints.transpose()[1], color = "#800080", alpha = 0.2)
    ax[2].set_ylim(0,3000)
    ax[2].set_yticks([],[])
    ax[2].set_title("72 hours")
    ax[2].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    ax[3].scatter(observed_mito_values_4dy, observed_edu_values_4dy,s=1, color = "black")
    ax[3].fill_between(observed_mito_values_4dy, edu_4dy_confints.transpose()[0], edu_4dy_confints.transpose()[1], color = "#800080", alpha = 0.2)
    ax[3].set_ylim(0,3000)
    ax[3].set_yticks([],[])
    ax[3].set_title("120 hours")
    ax[3].set_xlabel("Mitochondrial volume $(\mu m^3)$")

    edu_in_confint_0dy = (edu_0dy_confints.transpose()[0] < observed_edu_values_0dy) & (edu_0dy_confints.transpose()[1] > observed_edu_values_0dy)
    edu_in_confint_1dy = (edu_1dy_confints.transpose()[0] < observed_edu_values_1dy) & (edu_1dy_confints.transpose()[1] > observed_edu_values_1dy)
    edu_in_confint_2dy = (edu_2dy_confints.transpose()[0] < observed_edu_values_2dy) & (edu_2dy_confints.transpose()[1] > observed_edu_values_2dy)
    edu_in_confint_4dy = (edu_4dy_confints.transpose()[0] < observed_edu_values_4dy) & (edu_4dy_confints.transpose()[1] > observed_edu_values_4dy)
    within_confint = np.sum(edu_in_confint_0dy) + np.sum(edu_in_confint_1dy) + np.sum(edu_in_confint_2dy) + np.sum(edu_in_confint_4dy)
    total = len(edu_in_confint_0dy) + len(edu_in_confint_1dy) + len(edu_in_confint_2dy) + len(edu_in_confint_4dy)
    print("Number of cells with mtEdU number within credible interval = " + str(within_confint))
    print("Number of total cells = " + str(total))
    print("Proportion of cells with mtEdU number within credible intervals = " + str(within_confint/total))

    plt.tight_layout()

def higher_order_pulse_moments(nucleoid_1hr, nucleoid_3hr, nucleoid_7hr, nucleoid_24hr, edu_1hr, edu_3hr, edu_7hr, edu_24hr):
    nucleoid_mean_1hr = np.mean(nucleoid_1hr, axis=1)
    nucleoid_mean_3hr = np.mean(nucleoid_3hr, axis=1)
    nucleoid_mean_7hr = np.mean(nucleoid_7hr, axis=1)
    nucleoid_mean_24hr = np.mean(nucleoid_24hr, axis=1)

    edu_mean_1hr = np.mean(edu_1hr, axis=1)
    edu_mean_3hr = np.mean(edu_3hr, axis=1)
    edu_mean_7hr = np.mean(edu_7hr, axis=1)
    edu_mean_24hr = np.mean(edu_24hr, axis=1)

    nucleoid_var_1hr = np.var(nucleoid_1hr, axis=1)
    nucleoid_var_3hr = np.var(nucleoid_3hr, axis=1)
    nucleoid_var_7hr = np.var(nucleoid_7hr, axis=1)
    nucleoid_var_24hr = np.var(nucleoid_24hr, axis=1)

    edu_var_1hr = np.var(edu_1hr, axis=1)
    edu_var_3hr = np.var(edu_3hr, axis=1)
    edu_var_7hr = np.var(edu_7hr, axis=1)
    edu_var_24hr = np.var(edu_24hr, axis=1)

    nuc_edu_cov_1hr = np.zeros(500)
    nuc_edu_cov_3hr = np.zeros(500)
    nuc_edu_cov_7hr = np.zeros(500)
    nuc_edu_cov_24hr = np.zeros(500)

    for i in range(500):
        nuc_edu_cov_1hr[i] = np.cov(nucleoid_1hr[i], edu_1hr[i])[0,1]
        nuc_edu_cov_3hr[i] = np.cov(nucleoid_3hr[i], edu_3hr[i])[0,1]
        nuc_edu_cov_7hr[i] = np.cov(nucleoid_7hr[i], edu_7hr[i])[0,1]
        nuc_edu_cov_24hr[i] = np.cov(nucleoid_24hr[i], edu_24hr[i])[0,1]

    nucleoid_mean_1hr.sort()
    nucleoid_mean_1hr_confint = [nucleoid_mean_1hr[12], nucleoid_mean_1hr[-13]]
    nucleoid_mean_3hr.sort()
    nucleoid_mean_3hr_confint = [nucleoid_mean_3hr[12], nucleoid_mean_3hr[-13]] 
    nucleoid_mean_7hr.sort()
    nucleoid_mean_7hr_confint = [nucleoid_mean_7hr[12], nucleoid_mean_7hr[-13]] 
    nucleoid_mean_24hr.sort()
    nucleoid_mean_24hr_confint = [nucleoid_mean_24hr[12], nucleoid_mean_24hr[-13]] 

    edu_mean_1hr.sort()
    edu_mean_1hr_confint = [edu_mean_1hr[12], edu_mean_1hr[-13]] 
    edu_mean_3hr.sort()
    edu_mean_3hr_confint = [edu_mean_3hr[12], edu_mean_3hr[-13]] 
    edu_mean_7hr.sort()
    edu_mean_7hr_confint = [edu_mean_7hr[12], edu_mean_7hr[-13]] 
    edu_mean_24hr.sort()
    edu_mean_24hr_confint = [edu_mean_24hr[12], edu_mean_24hr[-13]] 

    nucleoid_var_1hr.sort()
    nucleoid_var_1hr_confint = [nucleoid_var_1hr[12], nucleoid_var_1hr[-13]] 
    nucleoid_var_3hr.sort()
    nucleoid_var_3hr_confint = [nucleoid_var_3hr[12], nucleoid_var_3hr[-13]] 
    nucleoid_var_7hr.sort()
    nucleoid_var_7hr_confint = [nucleoid_var_7hr[12], nucleoid_var_7hr[-13]] 
    nucleoid_var_24hr.sort()
    nucleoid_var_24hr_confint = [nucleoid_var_24hr[12], nucleoid_var_24hr[-13]] 

    edu_var_1hr.sort()
    edu_var_1hr_confint = [edu_var_1hr[12], edu_var_1hr[-13]] 
    edu_var_3hr.sort()
    edu_var_3hr_confint = [edu_var_3hr[12], edu_var_3hr[-13]]
    edu_var_7hr.sort()
    edu_var_7hr_confint = [edu_var_7hr[12], edu_var_7hr[-13]]
    edu_var_24hr.sort()
    edu_var_24hr_confint = [edu_var_24hr[12], edu_var_24hr[-13]]

    nuc_edu_cov_1hr.sort()
    nuc_edu_cov_1hr_confint = [nuc_edu_cov_1hr[12], nuc_edu_cov_1hr[-13]] 
    nuc_edu_cov_3hr.sort()
    nuc_edu_cov_3hr_confint = [nuc_edu_cov_3hr[12], nuc_edu_cov_3hr[-13]] 
    nuc_edu_cov_7hr.sort()
    nuc_edu_cov_7hr_confint = [nuc_edu_cov_7hr[12], nuc_edu_cov_7hr[-13]] 
    nuc_edu_cov_24hr.sort()
    nuc_edu_cov_24hr_confint = [nuc_edu_cov_24hr[12], nuc_edu_cov_24hr[-13]] 

    observed_nucleoid_mean_1hr = np.mean(all_assays_dna_number_1hr)
    observed_nucleoid_mean_3hr = np.mean(all_assays_dna_number_3hr)
    observed_nucleoid_mean_7hr = np.mean(all_assays_dna_number_7hr)
    observed_nucleoid_mean_24hr = np.mean(all_assays_dna_number_24hr)

    observed_edu_mean_1hr = np.mean(all_assays_edu_number_1hr)
    observed_edu_mean_3hr = np.mean(all_assays_edu_number_3hr)
    observed_edu_mean_7hr = np.mean(all_assays_edu_number_7hr)
    observed_edu_mean_24hr = np.mean(all_assays_edu_number_24hr)

    observed_nucleoid_var_1hr = np.var(all_assays_dna_number_1hr)
    observed_nucleoid_var_3hr = np.var(all_assays_dna_number_3hr)
    observed_nucleoid_var_7hr = np.var(all_assays_dna_number_7hr)
    observed_nucleoid_var_24hr = np.var(all_assays_dna_number_24hr)

    observed_edu_var_1hr = np.var(all_assays_edu_number_1hr)
    observed_edu_var_3hr = np.var(all_assays_edu_number_3hr)
    observed_edu_var_7hr = np.var(all_assays_edu_number_7hr)
    observed_edu_var_24hr = np.var(all_assays_edu_number_24hr)

    observed_nuc_edu_cov_1hr = np.cov(all_assays_dna_number_1hr, all_assays_edu_number_1hr)[0,1]
    observed_nuc_edu_cov_3hr = np.cov(all_assays_dna_number_3hr, all_assays_edu_number_3hr)[0,1]
    observed_nuc_edu_cov_7hr = np.cov(all_assays_dna_number_7hr, all_assays_edu_number_7hr)[0,1]
    observed_nuc_edu_cov_24hr = np.cov(all_assays_dna_number_24hr, all_assays_edu_number_24hr)[0,1]

    pulse_observed_nucleoid_means = np.array([observed_nucleoid_mean_1hr, observed_nucleoid_mean_3hr, observed_nucleoid_mean_7hr, observed_nucleoid_mean_24hr])
    pulse_observed_edu_means = np.array([observed_edu_mean_1hr, observed_edu_mean_3hr, observed_edu_mean_7hr, observed_edu_mean_24hr])

    pulse_observed_nucleoid_vars = np.array([observed_nucleoid_var_1hr, observed_nucleoid_var_3hr, observed_nucleoid_var_7hr, observed_nucleoid_var_24hr])
    pulse_observed_edu_vars = np.array([observed_edu_var_1hr, observed_edu_var_3hr, observed_edu_var_7hr, observed_edu_var_24hr])

    pulse_observed_nuc_edu_covs = np.array([observed_nuc_edu_cov_1hr, observed_nuc_edu_cov_3hr, observed_nuc_edu_cov_7hr, observed_nuc_edu_cov_24hr])

    times = np.array([1,3,7,24])
    fig, ax = plt.subplots(1,3, figsize = (18,4.5))
    ax[0].plot(np.array([1,1]), nucleoid_mean_1hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[0].plot(np.array([3,3]), nucleoid_mean_3hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[0].plot(np.array([7,7]), nucleoid_mean_7hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[0].plot(np.array([24,24]), nucleoid_mean_24hr_confint, linewidth=5, alpha=0.3, color = "#118002", label = "Nucleoid 95% credible interval")
    ax[0].scatter(times, pulse_observed_nucleoid_means, marker="x", color = "#118002", label = "Observed nucleoids")

    ax[0].plot(np.array([1,1]), edu_mean_1hr_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[0].plot(np.array([3,3]), edu_mean_3hr_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[0].plot(np.array([7,7]), edu_mean_7hr_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[0].plot(np.array([24,24]), edu_mean_24hr_confint, linewidth=5, alpha=0.3, color = "#800080", label = "MtEdU 95% credible interval")
    ax[0].scatter(times, pulse_observed_edu_means, marker="x", color = "#800080", label = "Observed mtEdU")

    ax[1].plot(np.array([1,1]), nucleoid_var_1hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[1].plot(np.array([3,3]), nucleoid_var_3hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[1].plot(np.array([7,7]), nucleoid_var_7hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[1].plot(np.array([24,24]), nucleoid_var_24hr_confint, linewidth=5, alpha=0.3, color = "#118002", label = "Nucleoid 95% credible interval")
    ax[1].scatter(times, pulse_observed_nucleoid_vars, marker="x", color = "#118002", label = "Observed nucleoids")

    ax[1].plot(np.array([1,1]), edu_var_1hr_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[1].plot(np.array([3,3]), edu_var_3hr_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[1].plot(np.array([7,7]), edu_var_7hr_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[1].plot(np.array([24,24]), edu_var_24hr_confint, linewidth=5, alpha=0.3, color = "#800080", label = "MtEdU 95% credible interval")
    ax[1].scatter(times, pulse_observed_edu_vars, marker="x", color = "#800080", label = "Observed mtEdU")

    ax[2].plot(np.array([1,1]), nuc_edu_cov_1hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[2].plot(np.array([3,3]), nuc_edu_cov_3hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[2].plot(np.array([7,7]), nuc_edu_cov_7hr_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[2].plot(np.array([24,24]), nuc_edu_cov_24hr_confint, linewidth=5, alpha=0.3, color = "#118002", label = "95% credible interval")
    ax[2].scatter(times, pulse_observed_nuc_edu_covs, marker="x", color = "#118002", label = "Observed")

    ax[0].set_xlabel("Time (h)")
    ax[1].set_xlabel("Time (h)")
    ax[2].set_xlabel("Time (h)")

    ax[0].set_ylabel("Mean")
    ax[1].set_ylabel("Variance")
    ax[2].set_ylabel("Covariance")

    ax[0].set_title("Copy number mean")
    ax[1].set_title("Copy number variance")
    ax[2].set_title("Nucleoid-mtEdU number covariance")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.tight_layout()

def higher_order_chase_moments(nucleoid_0dy, nucleoid_1dy, nucleoid_2dy, nucleoid_4dy, edu_0dy, edu_1dy, edu_2dy, edu_4dy):
    nucleoid_mean_0dy = np.mean(nucleoid_0dy, axis=1)
    nucleoid_mean_1dy = np.mean(nucleoid_1dy, axis=1)
    nucleoid_mean_2dy = np.mean(nucleoid_2dy, axis=1)
    nucleoid_mean_4dy = np.mean(nucleoid_4dy, axis=1)

    edu_mean_0dy = np.mean(edu_0dy, axis=1)
    edu_mean_1dy = np.mean(edu_1dy, axis=1)
    edu_mean_2dy = np.mean(edu_2dy, axis=1)
    edu_mean_4dy = np.mean(edu_4dy, axis=1)

    nucleoid_var_0dy = np.var(nucleoid_0dy, axis=1)
    nucleoid_var_1dy = np.var(nucleoid_1dy, axis=1)
    nucleoid_var_2dy = np.var(nucleoid_2dy, axis=1)
    nucleoid_var_4dy = np.var(nucleoid_4dy, axis=1)

    edu_var_0dy = np.var(edu_0dy, axis=1)
    edu_var_1dy = np.var(edu_1dy, axis=1)
    edu_var_2dy = np.var(edu_2dy, axis=1)
    edu_var_4dy = np.var(edu_4dy, axis=1)

    nuc_edu_cov_0dy = np.zeros(500)
    nuc_edu_cov_1dy = np.zeros(500)
    nuc_edu_cov_2dy = np.zeros(500)
    nuc_edu_cov_4dy = np.zeros(500)

    for i in range(500):
        nuc_edu_cov_0dy[i] = np.cov(nucleoid_0dy[i], edu_0dy[i])[0,1]
        nuc_edu_cov_1dy[i] = np.cov(nucleoid_1dy[i], edu_1dy[i])[0,1]
        nuc_edu_cov_2dy[i] = np.cov(nucleoid_2dy[i], edu_2dy[i])[0,1]
        nuc_edu_cov_4dy[i] = np.cov(nucleoid_4dy[i], edu_4dy[i])[0,1]

    nucleoid_mean_0dy.sort()
    nucleoid_mean_0dy_confint = [nucleoid_mean_0dy[12], nucleoid_mean_0dy[-13]]
    nucleoid_mean_1dy.sort()
    nucleoid_mean_1dy_confint = [nucleoid_mean_1dy[12], nucleoid_mean_1dy[-13]] 
    nucleoid_mean_2dy.sort()
    nucleoid_mean_2dy_confint = [nucleoid_mean_2dy[12], nucleoid_mean_2dy[-13]] 
    nucleoid_mean_4dy.sort()
    nucleoid_mean_4dy_confint = [nucleoid_mean_4dy[12], nucleoid_mean_4dy[-13]] 

    edu_mean_0dy.sort()
    edu_mean_0dy_confint = [edu_mean_0dy[12], edu_mean_0dy[-13]] 
    edu_mean_1dy.sort()
    edu_mean_1dy_confint = [edu_mean_1dy[12], edu_mean_1dy[-13]] 
    edu_mean_2dy.sort()
    edu_mean_2dy_confint = [edu_mean_2dy[12], edu_mean_2dy[-13]] 
    edu_mean_4dy.sort()
    edu_mean_4dy_confint = [edu_mean_4dy[12], edu_mean_4dy[-13]] 

    nucleoid_var_0dy.sort()
    nucleoid_var_0dy_confint = [nucleoid_var_0dy[12], nucleoid_var_0dy[-13]] 
    nucleoid_var_1dy.sort()
    nucleoid_var_1dy_confint = [nucleoid_var_1dy[12], nucleoid_var_1dy[-13]] 
    nucleoid_var_2dy.sort()
    nucleoid_var_2dy_confint = [nucleoid_var_2dy[12], nucleoid_var_2dy[-13]] 
    nucleoid_var_4dy.sort()
    nucleoid_var_4dy_confint = [nucleoid_var_4dy[12], nucleoid_var_4dy[-13]] 

    edu_var_0dy.sort()
    edu_var_0dy_confint = [edu_var_0dy[12], edu_var_0dy[-13]] 
    edu_var_1dy.sort()
    edu_var_1dy_confint = [edu_var_1dy[12], edu_var_1dy[-13]]
    edu_var_2dy.sort()
    edu_var_2dy_confint = [edu_var_2dy[12], edu_var_2dy[-13]]
    edu_var_4dy.sort()
    edu_var_4dy_confint = [edu_var_4dy[12], edu_var_4dy[-13]]

    nuc_edu_cov_0dy.sort()
    nuc_edu_cov_0dy_confint = [nuc_edu_cov_0dy[12], nuc_edu_cov_0dy[-13]] 
    nuc_edu_cov_1dy.sort()
    nuc_edu_cov_1dy_confint = [nuc_edu_cov_1dy[12], nuc_edu_cov_1dy[-13]] 
    nuc_edu_cov_2dy.sort()
    nuc_edu_cov_2dy_confint = [nuc_edu_cov_2dy[12], nuc_edu_cov_2dy[-13]] 
    nuc_edu_cov_4dy.sort()
    nuc_edu_cov_4dy_confint = [nuc_edu_cov_4dy[12], nuc_edu_cov_4dy[-13]] 

    observed_nucleoid_mean_0dy = np.mean(all_assays_dna_number_0dy)
    observed_nucleoid_mean_1dy = np.mean(all_assays_dna_number_1dy)
    observed_nucleoid_mean_2dy = np.mean(all_assays_dna_number_2dy)
    observed_nucleoid_mean_4dy = np.mean(all_assays_dna_number_4dy)

    observed_edu_mean_0dy = np.mean(all_assays_edu_number_0dy)
    observed_edu_mean_1dy = np.mean(all_assays_edu_number_1dy)
    observed_edu_mean_2dy = np.mean(all_assays_edu_number_2dy)
    observed_edu_mean_4dy = np.mean(all_assays_edu_number_4dy)

    observed_nucleoid_var_0dy = np.var(all_assays_dna_number_0dy)
    observed_nucleoid_var_1dy = np.var(all_assays_dna_number_1dy)
    observed_nucleoid_var_2dy = np.var(all_assays_dna_number_2dy)
    observed_nucleoid_var_4dy = np.var(all_assays_dna_number_4dy)

    observed_edu_var_0dy = np.var(all_assays_edu_number_0dy)
    observed_edu_var_1dy = np.var(all_assays_edu_number_1dy)
    observed_edu_var_2dy = np.var(all_assays_edu_number_2dy)
    observed_edu_var_4dy = np.var(all_assays_edu_number_4dy)

    observed_nuc_edu_cov_0dy = np.cov(all_assays_dna_number_0dy, all_assays_edu_number_0dy)[0,1]
    observed_nuc_edu_cov_1dy = np.cov(all_assays_dna_number_1dy, all_assays_edu_number_1dy)[0,1]
    observed_nuc_edu_cov_2dy = np.cov(all_assays_dna_number_2dy, all_assays_edu_number_2dy)[0,1]
    observed_nuc_edu_cov_4dy = np.cov(all_assays_dna_number_4dy, all_assays_edu_number_4dy)[0,1]

    chase_observed_nucleoid_means = np.array([observed_nucleoid_mean_0dy, observed_nucleoid_mean_1dy, observed_nucleoid_mean_2dy, observed_nucleoid_mean_4dy])
    chase_observed_edu_means = np.array([observed_edu_mean_0dy, observed_edu_mean_1dy, observed_edu_mean_2dy, observed_edu_mean_4dy])

    chase_observed_nucleoid_vars = np.array([observed_nucleoid_var_0dy, observed_nucleoid_var_1dy, observed_nucleoid_var_2dy, observed_nucleoid_var_4dy])
    chase_observed_edu_vars = np.array([observed_edu_var_0dy, observed_edu_var_1dy, observed_edu_var_2dy, observed_edu_var_4dy])

    chase_observed_nuc_edu_covs = np.array([observed_nuc_edu_cov_0dy, observed_nuc_edu_cov_1dy, observed_nuc_edu_cov_2dy, observed_nuc_edu_cov_4dy])
    times = np.array([24,48,72,120])
    fig, ax = plt.subplots(1,3, figsize = (18,4.5))
    ax[0].plot(np.array([24,24]), nucleoid_mean_0dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[0].plot(np.array([48,48]), nucleoid_mean_1dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[0].plot(np.array([72,72]), nucleoid_mean_2dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[0].plot(np.array([120,120]), nucleoid_mean_4dy_confint, linewidth=5, alpha=0.3, color = "#118002", label = "Nucleoid 95% credible interval")
    ax[0].scatter(times, chase_observed_nucleoid_means, marker="x", color = "#118002", label = "Observed nucleoids")

    ax[0].plot(np.array([24,24]), edu_mean_0dy_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[0].plot(np.array([48,48]), edu_mean_1dy_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[0].plot(np.array([72,72]), edu_mean_2dy_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[0].plot(np.array([120,120]), edu_mean_4dy_confint, linewidth=5, alpha=0.3, color = "#800080", label = "MtEdU 95% credible interval")
    ax[0].scatter(times, chase_observed_edu_means, marker="x", color = "#800080", label = "Observed mtEdU")

    ax[1].plot(np.array([24,24]), nucleoid_var_0dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[1].plot(np.array([48,48]), nucleoid_var_1dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[1].plot(np.array([72,72]), nucleoid_var_2dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[1].plot(np.array([120,120]), nucleoid_var_4dy_confint, linewidth=5, alpha=0.3, color = "#118002", label = "Nucleoid 95% credible interval")
    ax[1].scatter(times, chase_observed_nucleoid_vars, marker="x", color = "#118002", label = "Observed nucleoids")

    ax[1].plot(np.array([24,24]), edu_var_0dy_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[1].plot(np.array([48,48]), edu_var_1dy_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[1].plot(np.array([72,72]), edu_var_2dy_confint, linewidth=5, alpha=0.3, color = "#800080")
    ax[1].plot(np.array([120,120]), edu_var_4dy_confint, linewidth=5, alpha=0.3, color = "#800080", label = "MtEdU 95% credible interval")
    ax[1].scatter(times, chase_observed_edu_vars, marker="x", color = "#800080", label = "Observed mtEdU")

    ax[2].plot(np.array([24,24]), nuc_edu_cov_0dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[2].plot(np.array([48,48]), nuc_edu_cov_1dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[2].plot(np.array([72,72]), nuc_edu_cov_2dy_confint, linewidth=5, alpha=0.3, color = "#118002")
    ax[2].plot(np.array([120,120]), nuc_edu_cov_4dy_confint, linewidth=5, alpha=0.3, color = "#118002", label = "95% credible interval")
    ax[2].scatter(times, chase_observed_nuc_edu_covs, marker="x", color = "#118002", label = "Observed")

    ax[0].set_xlabel("Time (h)")
    ax[1].set_xlabel("Time (h)")
    ax[2].set_xlabel("Time (h)")

    ax[0].set_ylabel("Mean")
    ax[1].set_ylabel("Variance")
    ax[2].set_ylabel("Covariance")

    ax[0].set_title("Copy number mean")
    ax[1].set_title("Copy number variance")
    ax[2].set_title("Nucleoid-mtEdU number covariance")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.tight_layout()

def model_probabilities(prior, posterior, names, colors):
    population_num = len(names)
    prior_bars = np.zeros(population_num)
    posterior_bars = np.zeros(population_num)
    offsets = np.linspace(-0.25, 0.25, population_num)
    if population_num ==2:
        offsets = np.array([-0.2,0.2])
    plt.figure(figsize = (4.5,3.5))
    for i in range(population_num):
        prior_bars[i] = np.sum(prior == i+1)/len(prior)
        posterior_bars[i] = np.sum(posterior == i+1)/len(posterior)
        plt.bar(np.array([0+offsets[i],1+offsets[i]]), np.array([prior_bars[i], posterior_bars[i]]), 0.4/(population_num-1), color = colors[i], label = names[i])
    plt.xticks([0,1], ["Prior", "Posterior"], fontsize=18)
    plt.ylabel("Probability")
    plt.legend(loc = (0, 0.7))
    plt.tight_layout()


def plot_mean_pulse_traj(nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr):
    plt.figure(figsize = (4.5,3.5))
    mean_observed_edu = np.array([np.mean(all_assays_edu_number_1hr), np.mean(all_assays_edu_number_3hr), np.mean(all_assays_edu_number_7hr), np.mean(all_assays_edu_number_24hr)])
    err_observed_edu = np.array([np.std(all_assays_edu_number_1hr)/np.sqrt(len(all_assays_edu_number_1hr)),
                        np.std(all_assays_edu_number_3hr)/np.sqrt(len(all_assays_edu_number_3hr)),
                        np.std(all_assays_edu_number_7hr)/np.sqrt(len(all_assays_edu_number_7hr)),
                        np.std(all_assays_edu_number_24hr)/np.sqrt(len(all_assays_edu_number_24hr))])

    mean_simulated_edu = np.array([np.mean(tagged_num_1hr), np.mean(tagged_num_3hr), np.mean(tagged_num_7hr), np.mean(tagged_num_24hr)])

    mean_observed_dna = np.array([np.mean(all_assays_dna_number_1hr), np.mean(all_assays_dna_number_3hr), np.mean(all_assays_dna_number_7hr), np.mean(all_assays_dna_number_24hr)])
    err_observed_dna = np.array([np.std(all_assays_dna_number_1hr)/np.sqrt(len(all_assays_edu_number_1hr)),
                        np.std(all_assays_dna_number_3hr)/np.sqrt(len(all_assays_edu_number_3hr)),
                        np.std(all_assays_dna_number_7hr)/np.sqrt(len(all_assays_edu_number_7hr)),
                        np.std(all_assays_dna_number_24hr)/np.sqrt(len(all_assays_edu_number_24hr))])

    mean_simulated_dna = np.array([np.mean(nucleoid_num_1hr), np.mean(nucleoid_num_3hr), np.mean(nucleoid_num_7hr), np.mean(nucleoid_num_24hr)])

    plt.errorbar(np.array([1,3,7,24]), mean_observed_edu, err_observed_edu, fmt = "o", color = '#800080',capsize=5, label = "Observed mtEdU")
    plt.plot(np.array([1,3,7,24]), mean_simulated_edu, "x-", color = '#800080', label = "Simulated mtEdU")
    plt.errorbar(np.array([1,3,7,24]), mean_observed_dna, err_observed_dna, fmt = "o", color = "#118002",capsize=5, label = "Observed mtDNA")
    plt.plot(np.array([1,3,7,24]), mean_simulated_dna, "x-", color = "#118002", label = "Simulated mtDNA")
    plt.ylabel("Nucleoid number")
    plt.xlabel("Time (h)")
    plt.title("Mean simulated and observed trajectories")
    plt.legend()
    plt.tight_layout()

def plot_stochastic_pulse_traj(nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr):

    fig, ax = plt.subplots(3,2, figsize = (10,12))
    ax[0,0].scatter(all_assays_mito_lengths, all_assays_dna_numbers,label = "Observed")
    ax[0,0].scatter(all_assays_mito_lengths,np.concatenate([nucleoid_num_1hr,nucleoid_num_3hr,nucleoid_num_7hr,nucleoid_num_24hr]), label = "Simulated")
    ax[0,0].set_xlabel("Mitochondrial volume (μm³)")
    ax[0,0].set_ylabel("Nucleoid number")
    ax[0,0].set_title("Nucleoid number")
    ax[0,0].legend()

    ax[0,1].scatter(mtvolume_1hr, all_assays_edu_number_1hr,label = "data")
    ax[0,1].scatter(mtvolume_1hr,tagged_num_1hr, label = "sim")
    ax[0,1].set_xlabel("Mitochondrial volume (μm³)")
    ax[0,1].set_ylabel("MtEdU number")
    ax[0,1].set_title("1 hr mtEdU")
    ax[0,1].set_ylim(0,2200)

    ax[1,0].scatter(mtvolume_3hr, all_assays_edu_number_3hr,label = "data")
    ax[1,0].scatter(mtvolume_3hr,tagged_num_3hr, label = "sim")
    ax[1,0].set_xlabel("Mitochondrial volume (μm³)")
    ax[1,0].set_ylabel("MtEdU number")
    ax[1,0].set_title("3 hr mtEdU")
    ax[1,0].set_ylim(0,2200)

    ax[1,1].scatter(mtvolume_7hr,all_assays_edu_number_7hr,  label = "data")
    ax[1,1].scatter(mtvolume_7hr, tagged_num_7hr,label = "sim")
    ax[1,1].set_xlabel("Mitochondrial volume (μm³)")
    ax[1,1].set_ylabel("MtEdU number")
    ax[1,1].set_title("7 hr mtEdU")
    ax[1,1].set_ylim(0,2200)

    ax[2,0].scatter(mtvolume_24hr,all_assays_edu_number_24hr,  label = "data")
    ax[2,0].scatter(mtvolume_24hr,tagged_num_24hr, label = "sim")
    ax[2,0].set_xlabel("Mitochondrial volume (μm³)")
    ax[2,0].set_ylabel("MtEdU number")
    ax[2,0].set_title("24 hr mtEdU")
    ax[2,0].set_ylim(0,2200)

    ax[2,1].set_yticks([],[])
    ax[2,1].set_xticks([],[])
    ax[2,1].spines['top'].set_visible(False)
    ax[2,1].spines['right'].set_visible(False)
    ax[2,1].spines['bottom'].set_visible(False)
    ax[2,1].spines['left'].set_visible(False)

    plt.tight_layout()

def plot_mean_chase_traj(nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy):
    
    plt.figure(figsize = (8,6))
    mean_observed_edu = np.array([np.mean(all_assays_edu_number_0dy), np.mean(all_assays_edu_number_1dy), np.mean(all_assays_edu_number_2dy), np.mean(all_assays_edu_number_4dy)])
    err_observed_edu = np.array([np.std(all_assays_edu_number_0dy)/np.sqrt(len(all_assays_edu_number_0dy)),
                        np.std(all_assays_edu_number_1dy)/np.sqrt(len(all_assays_edu_number_1dy)),
                        np.std(all_assays_edu_number_2dy)/np.sqrt(len(all_assays_edu_number_2dy)),
                        np.std(all_assays_edu_number_4dy)/np.sqrt(len(all_assays_edu_number_4dy))])

    mean_simulated_edu = np.array([np.mean(tagged_num_0dy), np.mean(tagged_num_1dy), np.mean(tagged_num_2dy), np.mean(tagged_num_4dy)])

    mean_observed_dna = np.array([np.mean(all_assays_dna_number_0dy), np.mean(all_assays_dna_number_1dy), np.mean(all_assays_dna_number_2dy), np.mean(all_assays_dna_number_4dy)])
    err_observed_dna = np.array([np.std(all_assays_dna_number_0dy)/np.sqrt(len(all_assays_edu_number_0dy)),
                        np.std(all_assays_dna_number_1dy)/np.sqrt(len(all_assays_edu_number_1dy)),
                        np.std(all_assays_dna_number_2dy)/np.sqrt(len(all_assays_edu_number_2dy)),
                        np.std(all_assays_dna_number_4dy)/np.sqrt(len(all_assays_edu_number_4dy))])

    mean_simulated_dna = np.array([np.mean(nucleoid_num_0dy), np.mean(nucleoid_num_1dy), np.mean(nucleoid_num_2dy), np.mean(nucleoid_num_4dy)])

    plt.errorbar(np.array([1,3,7,24]), mean_observed_edu, err_observed_edu, fmt = "o", color = '#800080',capsize=5, label = "Observed mtEdU")
    plt.plot(np.array([1,3,7,24]), mean_simulated_edu, "x-", color = '#800080', label = "Simulated mtEdU")
    plt.errorbar(np.array([1,3,7,24]), mean_observed_dna, err_observed_dna, fmt = "o", color = "#118002",capsize=5, label = "Observed mtDNA")
    plt.plot(np.array([1,3,7,24]), mean_simulated_dna, "x-", color = "#118002", label = "Simulated mtDNA")
    plt.ylabel("Nucleoid number")
    plt.xlabel("Time (h)")
    plt.title("Mean simulated and observed trajectories")
    plt.legend()
    plt.tight_layout()

def plot_stochastic_chase_traj(nucleoid_num_0dy, tagged_num_0dy, mtvolume_0dy, nucleoid_num_1dy, tagged_num_1dy, mtvolume_1dy,
           nucleoid_num_2dy, tagged_num_2dy, mtvolume_2dy, nucleoid_num_4dy, tagged_num_4dy, mtvolume_4dy):
    
    fig, ax = plt.subplots(4,2, figsize = (10,12))

    ax[0,0].scatter(mtvolume_0dy, all_assays_dna_number_0dy,label = "Observed")
    ax[0,0].scatter(mtvolume_0dy,nucleoid_num_0dy, label = "Simulated")
    ax[0,0].legend()
    ax[0,0].set_xlabel("Mitochondrial volume (μm³)")
    ax[0,0].set_ylabel("Nucleoid number")
    ax[0,0].set_title("24h nucleoid number")
    ax[0,0].set_ylim(0,6000)
    ax[0,0].legend()

    ax[0,1].scatter(mtvolume_1dy, all_assays_dna_number_1dy,label = "data")
    ax[0,1].scatter(mtvolume_1dy,nucleoid_num_1dy, label = "sim")
    ax[0,1].set_xlabel("Mitochondrial volume (μm³)")
    ax[0,1].set_ylabel("Nucleoid number")
    ax[0,1].set_title("48h nucleoid number")
    ax[0,1].set_ylim(0,6000)

    ax[1,0].scatter(mtvolume_2dy, all_assays_dna_number_2dy,  label = "data")
    ax[1,0].scatter(mtvolume_2dy, nucleoid_num_2dy,label = "sim")
    ax[1,0].set_xlabel("Mitochondrial volume (μm³)")
    ax[1,0].set_ylabel("Nucleoid number")
    ax[1,0].set_title("72h nucleoid number")
    ax[1,0].set_ylim(0,6000)

    ax[1,1].scatter(mtvolume_4dy,all_assays_dna_number_4dy,  label = "data")
    ax[1,1].scatter(mtvolume_4dy,nucleoid_num_4dy, label = "sim")
    ax[1,1].set_xlabel("Mitochondrial volume (μm³)")
    ax[1,1].set_ylabel("Nucleoid number")
    ax[1,1].set_title("120h nucleoid number")
    ax[1,1].set_ylim(0,6000)

    ax[2,0].scatter(mtvolume_0dy, all_assays_edu_number_0dy,label = "Observed")
    ax[2,0].scatter(mtvolume_0dy,tagged_num_0dy, label = "Simulated")
    ax[2,0].set_xlabel("Mitochondrial volume (μm³)")
    ax[2,0].set_ylabel("MtEdU number")
    ax[2,0].set_title("24h mtEdU")
    ax[2,0].set_ylim(0,2100)

    ax[2,1].scatter(mtvolume_1dy, all_assays_edu_number_1dy,label = "data")
    ax[2,1].scatter(mtvolume_1dy,tagged_num_1dy, label = "sim")
    ax[2,1].set_xlabel("Mitochondrial volume (μm³)")
    ax[2,1].set_ylabel("MtEdU number")
    ax[2,1].set_title("48h mtEdU")
    ax[2,1].set_ylim(0,2100)

    ax[3,0].scatter(mtvolume_2dy,all_assays_edu_number_2dy,  label = "data")
    ax[3,0].scatter(mtvolume_2dy, tagged_num_2dy,label = "sim")
    ax[3,0].set_xlabel("Mitochondrial volume (μm³)")
    ax[3,0].set_ylabel("MtEdU number")
    ax[3,0].set_title("72h mtEdU")
    ax[3,0].set_ylim(0,2100)

    ax[3,1].scatter(mtvolume_4dy,all_assays_edu_number_4dy,  label = "data")
    ax[3,1].scatter(mtvolume_4dy,tagged_num_4dy, label = "sim")
    ax[3,1].set_xlabel("Mitochondrial volume (μm³)")
    ax[3,1].set_ylabel("MtEdU number")
    ax[3,1].set_title("120h mtEdU")
    ax[3,1].set_ylim(0,2100)

    plt.tight_layout()

def plot_nucleoid_mito(nucleoid_num_1hr, tagged_num_1hr, mtvolume_1hr, nucleoid_num_3hr, tagged_num_3hr, mtvolume_3hr, 
           nucleoid_num_7hr, tagged_num_7hr, mtvolume_7hr, nucleoid_num_24hr, tagged_num_24hr, mtvolume_24hr):
    plt.scatter(all_assays_mito_lengths, all_assays_dna_numbers,label = "Observed")
    plt.scatter(all_assays_mito_lengths,np.concatenate([nucleoid_num_1hr,nucleoid_num_3hr,nucleoid_num_7hr,nucleoid_num_24hr]), label = "Simulated")
    plt.legend()
    plt.xlabel("Mitochondrial volume (μm³)")
    plt.ylabel("Nucleoid number")