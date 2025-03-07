import math
import random
import numpy as np
import pandas as pd
import scipy
import emcee
from multiprocessing import Pool

#Note: Most (meaning nearly all) of the following lines have been taken / combined from:
#https://emcee.readthedocs.io/en/stable/

#This is a development version of the emcee fitter...

class MCMC_Fitter_DEV(object):
      
    def __init__(self):
          self.data = []

    #Define K-matrix and its components here:
    #*******************************************************************
    #Get the difference matrix (between each point of the 1D DP):
    def get_Diff_Matrix(self,global_bins):
        n_dim = global_bins.shape[0]
        max_bin = np.max(global_bins)
        diff_matrix = np.fromfunction(lambda i, j: (global_bins[i]-global_bins[j])/max_bin, (n_dim, n_dim), dtype=int)
    
        return diff_matrix

    #-----------------------------

    #Get the rbf:
    def get_RBF(self,d,length_scale):
        arg = d / length_scale
        return math.exp(-0.5*arg*arg)

    #-----------------------------

    #Get error matrix:
    def get_Error_Matrix(self,errors):
        n_dim = errors.shape[0]
        err_matrix = np.fromfunction(lambda i, j: errors[i]*errors[j], (n_dim, n_dim), dtype=int)
        
        return err_matrix

    #-----------------------------

    #Get the entire K-Matrix:
    def get_K_Matrix(self,err_matrix,diff_matrix,length_scale,noise_scale,y_fit):
        K_Matrix = np.vectorize(self.get_RBF)(diff_matrix,length_scale)*err_matrix + (noise_scale*noise_scale*np.diag(y_fit**2))
        return K_Matrix
    #*******************************************************************

    #Different approach: loop over pairs:
    #*******************************************************************
    #Get the correlation:
    def get_rho(self,i,x,y,length_scale,cut_off,use_sign):
        sign = 1.0
        if use_sign:
           sign = np.sign(y[i] - y)
        arg = (x[i] - x) / length_scale
        rho = sign*np.exp(-0.5*arg*arg)
        rho[i] = 0.0

        return np.where(np.abs(rho) >= cut_off,rho,0.0)

    #--------------------------------------------

    #Get the likelihood for comparing two points:
    def get_pair_logL(self,i,global_bins,y_data,y_err,y_fit,length_scale,cut_off,use_sign,noise_scale):
        rho_ij = self.get_rho(i,global_bins,y_data,length_scale,cut_off,use_sign)
        norm = 1.0-rho_ij**2
        norm[norm < 0.0] = 0.001
        white_noise = y_fit**2 * np.exp(2*noise_scale)

        diff = y_data-y_fit
        err = np.sqrt(y_err**2 + white_noise)
        arg = ( (diff[i]/err[i])**2 -2.0*rho_ij*(diff[i]*diff) / (err*err[i])  + diff/(err)**2 )/norm 
        norm_2 = 2.0*np.log(2.0*math.pi*err[i]*err*np.sqrt(norm))
        log_L = -0.5*( arg + norm_2 )
        log_L[i] = 0.0

        return np.sum(log_L)
       
    
    #Get the entire likelihood:
    def get_full_logL(self,index_list,global_bins,y_data,y_err,y_fit,length_scale,cut_off,use_sign,noise_scale):
        log_L = np.fromiter(( self.get_pair_logL(i,global_bins,y_data,y_err,y_fit,length_scale,cut_off,use_sign,noise_scale) for i in index_list ),index_list.dtype)
    
        return np.sum(log_L)
    #*******************************************************************s

    #Define basic functions for the fitter to minimize:
    #*******************************************************************
    #Get the 1D Dalitz Plot function:
    def get_oneD_Dalitz_Function(self,theta,x):
        norm = theta[0]
        parA = theta[1]
        parB = theta[2]
        parC = theta[3]
        parD = theta[4]
        parE = theta[5]
        parF = theta[6]
        parG = theta[7]
        parH = theta[8]
        parL = theta[9]

        DP_X = x[:,0]
        DP_Y = x[:,1]

        return norm*(1.0 + parA*DP_Y + parB*DP_Y*DP_Y + parC*DP_X + parD*DP_X*DP_X + parE*DP_X*DP_Y + parF*DP_Y*DP_Y*DP_Y + parG*DP_X*DP_X*DP_Y + parH*DP_X*DP_Y*DP_Y + parL*DP_X*DP_X*DP_X)

    #----------------------------------

    #Define the log likelihood:
    #def log_likelihood(self,theta,x,y,err_matrix,diff_matrix):
    def log_likelihood(self,theta,x,y,y_err,cut_off,use_sign):
        N_fit = self.get_oneD_Dalitz_Function(theta,x)
        
        length_scale = 3.0
        noise_level = 0.0
        gbin = x[:,2]
        
        index = random.randint(0,gbin.shape[0]-1)
        log_L = self.get_pair_logL(index,gbin,y,y_err,N_fit,length_scale,cut_off,use_sign,noise_level)

        if log_L <= 0.0:
            
            return log_L
        return -np.inf
   
    #----------------------------------

    #Put constraint on efficiency model:
    def log_prior_kernels(self,theta,length_limits,noise_limits):
        if theta[10] > length_limits[0] and theta[10] < length_limits[1] and theta[11] > noise_limits[0] and theta[11] < noise_limits[1]:
            return 0.0
        else:
            return -np.inf

    #----------------------------------

    #Include constraints on c-violating parameters:
    def log_prior(self,theta,constr_err):
        arg_c = theta[3]/constr_err[0]
        arg_e = theta[5]/constr_err[1]
        arg_h = theta[8]/constr_err[2]
        arg_l = theta[9]/constr_err[3]

        norm = 0.0
        #++++++++++++++++++++++++
        for err in constr_err:
            norm += math.log(2.0*math.pi*err*err)
        #++++++++++++++++++++++++

        log_P = -0.5*(arg_c*arg_c + arg_e*arg_e * arg_h*arg_h + arg_l*arg_l + norm)
        return log_P

    #----------------------------------

    def log_probability(self,theta,x,y,y_err,constr_err,length_limits,cut_off,use_sign,noise_limits):
        #prob = self.log_likelihood(theta,x,y,y_err,cut_off,use_sign) + self.log_prior(theta,constr_err) + self.log_prior_kernels(theta,length_limits,noise_limits)
        prob = self.log_likelihood(theta,x,y,y_err,cut_off,use_sign) + self.log_prior(theta,constr_err)
        if not np.isfinite(prob):
            return -np.inf
        return prob
    #*******************************************************************

    #Get start values for the fitter:
    #*******************************************************************
    def get_start_values(self,minVals,maxVals,kernel_inits=[]):
        start_values = []
        #+++++++++++++++++++++++++++++
        for a,b in zip(minVals,maxVals):
            start_values.append(random.uniform(a,b))
        #+++++++++++++++++++++++++++++

        if len(kernel_inits) > 0.0:
            start_values.append(random.uniform(kernel_inits[0][0],kernel_inits[0][1]))
            start_values.append(random.uniform(kernel_inits[1][0],kernel_inits[1][1]))

        return start_values
    #*******************************************************************

    #Get mover, define the ones you are interested here:
    #*******************************************************************
    #Gaussian mover:
    def get_gaussian_mover(self,cov_matrix,parameter_update_mode):
        return emcee.moves.GaussianMove(cov=cov_matrix,mode=parameter_update_mode)

    #Stretch move:
    def get_stretch_mover(self,par_a):
        return emcee.moves.StretchMove(a=par_a)

    #Walk move:
    def get_walk_mover(self,par_s):
        return emcee.moves.WalkMove(s=par_s)

    #Kernel density move:
    def get_kde_mover(self,bandwidth_method):
        return emcee.moves.KDEMove(bw_method=bandwidth_method)

    #DE Move:
    def get_de_mover(self,par_sigma,par_gamma):
        return emcee.moves.DEMove(sigma=par_sigma, gamma0=par_gamma)

    #DE Snooker move:
    def get_de_snooker_mover(self,par_gamma):
        return emcee.moves.DESnookerMove(gammas=par_gamma)
    #*******************************************************************

    #Run the sampling
    #*******************************************************************
    def run_sampling(self,walkers,start_values,nIterations,nAddIterations,crit_tau_factor,monitor_tau,tolerance,monitor_iteration,show_progress):
        autocorr_time = None
        max_autocorr_time = None

        #---------------------------
        if monitor_tau == False:
           walkers.run_mcmc(start_values,nIterations,progress=show_progress)
           autocorr_time = walkers.get_autocorr_time()

           return [nIterations,nIterations,np.mean(autocorr_time),np.max(autocorr_time)]
        else:
           autocorr_time = np.empty(nIterations)
           max_autocorr_time = np.empty(nIterations)
           old_tau = np.inf
           sufficient_chain_length = False
           index = 0

           while sufficient_chain_length == False:

              max_iter = nIterations
              if index > 0:
                   n_iterations_run  = index*monitor_iteration
                   print("  ")
                   print("   >>> Did not converge after " + str(n_iterations_run) + " iterations!<<<")
                   print("   >>> Going to run another " + str(nAddIterations) + " iterations<<<")
                   print("  ")
                   max_iter = nAddIterations

              #+++++++++++++++++++++++++++++++++++
              for _ in walkers.sample(start_values, iterations=max_iter, progress=show_progress):
                   if walkers.iteration % monitor_iteration:
                      continue

                   tau = walkers.get_autocorr_time(tol=0)
                   autocorr_time[index] = np.mean(tau)
                   max_autocorr_time[index] = np.max(tau)
                   index += 1

                   sufficient_chain_length = np.all(tau * crit_tau_factor < walkers.iteration) #We want to ensure that the chains have a sufficient length!
                   converged = np.all(np.abs(old_tau - tau) / tau < tolerance)
                   if sufficient_chain_length and converged:
                      break
                   old_tau = tau
              #+++++++++++++++++++++++++++++++++++

              start_values = walkers.get_last_sample()

           return [index,monitor_iteration,autocorr_time,max_autocorr_time]
    #*******************************************************************

    #Run the fitter:
    #*******************************************************************
    def run_mcmc_fitter(self,nWalkers,minVals,maxVals,ordered_DP_Data,constr_err,nIterations,moves,length_limits,cut_off,use_sign,noise_limits,kernel_inits,nAddIterations=1000,crit_tau_factor=50,monitor_tau=False,tolerance=0.001,monitor_iteration=100,show_progress=True,use_multiprocessing=False,resultFileName=None,continue_existing_chain=False):

        start_list = []
        #+++++++++++++++++++++++++
        for _ in range(nWalkers):
            start_list.append(self.get_start_values(minVals,maxVals,kernel_inits))
        #+++++++++++++++++++++++++

        start_values = np.array(start_list)
        n_pars = start_values.shape[1]

        x = ordered_DP_Data[:,[0,1,4]]
        y = ordered_DP_Data[:,2]
        yerr = ordered_DP_Data[:,3]
    
        walker_backend = None
        #---------------------------------------------------------------------
        if resultFileName is not None:
            walker_backend = emcee.backends.HDFBackend(resultFileName + '.h5')
            if continue_existing_chain == False:
               walker_backend.reset(nWalkers,n_pars)
               np.save(resultFileName + '_StartValues.npy',start_values)
            else:
               start_values = walker_backend.get_last_sample()
        #---------------------------------------------------------------------

        walk_results = None
        #-------------------------------- 
        if use_multiprocessing:
           with Pool() as pool:
                sampler = emcee.EnsembleSampler(nWalkers,n_pars,self.log_probability,args=(x,y,yerr,constr_err,length_limits,cut_off,use_sign,noise_limits),pool=pool,backend=walker_backend)
                walk_results = self.run_sampling(sampler,start_values,nIterations,nAddIterations,crit_tau_factor,monitor_tau,tolerance,monitor_iteration,show_progress)
        else:
           sampler = emcee.EnsembleSampler(nWalkers,n_pars,self.log_probability,args=(x,y,yerr,constr_err,length_limits,cut_off,use_sign,noise_limits),backend=walker_backend)
           walk_results = self.run_sampling(sampler,start_values,nIterations,nAddIterations,crit_tau_factor,monitor_tau,tolerance,monitor_iteration,show_progress)
        #-------------------------------- 

        return walk_results
    #*******************************************************************

    #Get a nice intro
    #*******************************************************************
    def show_intro(self):
        print("  ")
        print("**************************************")
        print("*                                    *")
        print("*   Random Walk Dalitz Plot Fitter   *")
        print("*                                    *")
        print("**************************************")
        print("  ")
    #*******************************************************************

    