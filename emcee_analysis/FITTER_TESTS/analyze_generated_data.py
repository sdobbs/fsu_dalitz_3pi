#!/usr/bin/env python3

import sys
import numpy as np
import copy
from UTILS.analysis_utils import DP_Analysis_Utils
from UTILS.chisquare_fitter import ChiSquare_Fitter
from EMCEE_FITTER.emcee_fitter import MCMC_Fitter

#Basic settings and definitions:
#**********************************************
ana_utils = DP_Analysis_Utils()
init_fitter = ChiSquare_Fitter()
mcmc_fitter = MCMC_Fitter()

in_name = sys.argv[1]
out_name = sys.argv[2]

#If you do not run an initial chi2 fit, you need to provide a covariance matrix
mcmc_cov_matrix = None
mcmc_parameter_update = 'random'
run_init_fit = True
init_fit_res_name = None #--> Just change if you want to store the fit results as a .png-file
init_fit_name = 'Initial Fit'
use_diag_cov = True

if use_diag_cov == False:
    mcmc_parameter_update = 'vector'

show_init_fit_results = False

#Parameters for the fitter:
nWalkers = 150
nIterations = 2000
constr_error = [0.035]*4
loss_function = 'linear'
n_sigma_start = 10.0
scan_tau = True
n_monitoring_steps = 50
store_chisqaure_results = True

#Start values:
valueNames = ['norm','a','b','c','d','e','f','g','h','l']
minDPVals = [400000,-1.2,0.05,-0.01,0.05,-0.01,0.1,-0.01,-0.01,-0.01]
maxDPVals = [800000,-1.0,0.2,0.01,0.1,0.01,0.2,0.01,0.01,0.01]
initFit_start_values = mcmc_fitter.get_start_values(minDPVals,maxDPVals)
#**********************************************

#Director where npy.data is stored and where 
#analysis results will be stored:
#**********************************************
load_npy_dir = '../../npy_dalitz_data/'
store_mcmc_dir = '../../mcmc_dalitz_data'
store_mcmc_pandas_dir = '../../pandas_mcmc_dalitz_data'
chisqure_result_dir = '../../Chisquare_Fitter_Results'

fullSaveName = store_mcmc_dir + '/' + out_name
#**********************************************

#NO CHANGES SHOULD BE DONE BELOW THIS LINE!!!!!
##################################################################################

#Show the intro:
#**********************************************
mcmc_fitter.show_intro()
#**********************************************

#Load the data and rebin it:
#**********************************************
print("Load and prepare data...")

Gen_DP_Data = np.load(load_npy_dir + in_name + '.npy')
rebinned_data = ana_utils.rebin_DP(Gen_DP_Data,include_raw_yields=False)

print("...done!")
print("  ")
#**********************************************

#Get initial values:
#**********************************************
if run_init_fit:
    print("Run chi2 fit to determine start parameters for the random walk fitter...")
    
    init_fit_values, init_fit_cov, init_fit_errs, _ = init_fitter.run_initial_fitter(rebinned_data,initFit_start_values,valueNames,'Gen Data',init_fit_name,show_results=show_init_fit_results,figSaveName=init_fit_res_name)
    
    if store_chisqaure_results:
       np.save(chisqure_result_dir + '/' + out_name + '_chisqFit_DP_pars.npy',init_fit_values)
       np.save(chisqure_result_dir + '/' + out_name + '_chisqFit_DP_errs.npy',init_fit_values)


    #----------------------
    if use_diag_cov:
        mcmc_cov_matrix = init_fit_errs
    else:
        mcmc_cov_matrix = init_fit_cov
    #----------------------

    minDPVals,maxDPVals = init_fitter.get_start_values_from_init_fit(init_fit_values,init_fit_errs,n_sigma_start)

    print("...done!")
    print("  ")  
#**********************************************

#Run the mcmc fitter:
#**********************************************
print("Run random walk analysis with " + str(nWalkers) + " walkers and " + str(nIterations) + " iterations each...")
print(" ")

if run_init_fit == False:
    print("   >>> Use pre-defined start values <<<")
    print("  ")


mover = mcmc_fitter.get_gaussian_mover(mcmc_cov_matrix,mcmc_parameter_update)
index, mon_iterations, tau, max_tau = mcmc_fitter.run_mcmc_fitter(nWalkers,minDPVals,maxDPVals,rebinned_data,nIterations,mover,loss_func_str=loss_function,constr_error=constr_error,tolerance=0.1,monitor_tau=scan_tau,resultFileName=fullSaveName,monitor_iteration=n_monitoring_steps)

if fullSaveName is not None:
    print("  ")
    print("   >>> Results will be stored at: " + fullSaveName + " <<<")
      
    #-------------------------------------
    if scan_tau:
         n = mon_iterations * np.arange(1, index + 1)
         y = tau[:index]
         y_max = max_tau[:index]

         np.save(fullSaveName + '_scanned_tau_iterations.npy',n)
         np.save(fullSaveName + '_scanned_tau_mean_values.npy',y)
         np.save(fullSaveName + '_scanned_tau_max_values.npy',y_max)
    else:
         np.save(fullSaveName + '_tau_values.npy',tau)  
   #-------------------------------------
      
print(" ")
print("...done! Have a great day!")
print(" ")
#**********************************************
