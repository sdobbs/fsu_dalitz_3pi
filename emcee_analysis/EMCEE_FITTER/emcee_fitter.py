import math
import random
import numpy as np
import pandas as pd
import emcee
from multiprocessing import Pool
from FIT_UTILS.fitter_utils import fitter_utils

#Note: Most (meaning nearly all) of the following lines have been taken / combined from:
#https://emcee.readthedocs.io/en/stable/

class MCMC_Fitter(object):
      
    def __init__(self):
          self.data = []
          
    def setup_fitter(self,kinematic_acceptance=None,kinematic_acceptance_limits=None):
        self.kinematic_acceptance = kinematic_acceptance
        self.kinematic_acceptance_limits = kinematic_acceptance_limits
        self.fit_manager = fitter_utils(kinematic_acceptance)

    #Define basic functions for the fitter to minimize:
    #*******************************************************************
    #Get the 1D Dalitz Plot function --> The model:
    def oneD_Dalitz_Function(self,theta,x):
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

    #Same log-prior as above, but include limits for the kinematic acceptance cut:
    def log_prior_kin_acc(self,theta,const_err,kin_acc_limits):
        kin_acc_cut = theta[10]

        if kin_acc_cut >= kin_acc_limits[0] and kin_acc_cut <= kin_acc_limits[1]:
            return self.log_prior(theta,const_err)

        return -np.inf

    #----------------------------------

    def get_log_prior_func(self,constr_err):
        log_p_f = None
         
        #----------------------------------------------------------------------------------- 
        if self.kinematic_acceptance is None or self.kinematic_acceptance_limits is None:
           def log_p_f(theta):
               return self.log_prior(theta,constr_err) 
        else:
            def log_p_f(theta):
               return self.log_prior_kin_acc(theta,constr_err,self.kinematic_acceptance_limits)
        #----------------------------------------------------------------------------------- 

        return log_p_f
    
    #----------------------------------

    #Now get the entire objective function:
    def get_emcee_objective_function(self,func_str,func_par,constr_err):
        prior_func = self.get_log_prior_func(constr_err)

        return self.fit_manager.get_objective_function(self.oneD_Dalitz_Function,func_str,func_par,prior_func)
    #*******************************************************************

    

    #Get start values for the fitter:
    #*******************************************************************
    def get_start_values(self,minVals,maxVals):
        start_values = []
        #+++++++++++++++++++++++++++++
        for a,b in zip(minVals,maxVals):
            start_values.append(random.uniform(a,b))
        #+++++++++++++++++++++++++++++

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
              for _ in walkers.sample(start_values, iterations=max_iter, progress=show_progress,store=True):
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
    def run_mcmc_fitter(self,nWalkers,minVals,maxVals,ordered_DP_Data,nIterations,movers,constr_error=[0.01,0.01,0.01,0.01],loss_func_str='linear',loss_func_par=1.0,nAddIterations=1000,crit_tau_factor=50,monitor_tau=False,tolerance=0.001,monitor_iteration=100,show_progress=True,use_multiprocessing=False,resultFileName=None,continue_existing_chain=False,kinematic_acceptance_start_values=None):
        objective_function = self.get_emcee_objective_function(loss_func_str,loss_func_par,constr_error)

        if self.kinematic_acceptance is not None and self.kinematic_acceptance_limits is not None and kinematic_acceptance_start_values is not None:
            minVals.append(kinematic_acceptance_start_values[0])
            maxVals.append(kinematic_acceptance_start_values[1])

        start_list = []
        #+++++++++++++++++++++++++
        for _ in range(nWalkers):
            start_list.append(self.get_start_values(minVals,maxVals))
        #+++++++++++++++++++++++++

        start_values = np.array(start_list)
        n_pars = start_values.shape[1]

        x = ordered_DP_Data[:,[0,1]]
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
                   sampler = emcee.EnsembleSampler(nWalkers,n_pars,objective_function,args=(x,y,yerr),pool=pool,backend=walker_backend,moves=movers)
                   walk_results = self.run_sampling(sampler,start_values,nIterations,nAddIterations,crit_tau_factor,monitor_tau,tolerance,monitor_iteration,show_progress)
        else:
               sampler = emcee.EnsembleSampler(nWalkers,n_pars,objective_function,args=(x,y,yerr),backend=walker_backend,moves=movers)
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

    