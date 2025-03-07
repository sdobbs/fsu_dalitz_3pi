#!/usr/bin/env python3

import sys
import numpy as np
import copy
from UTILS.analysis_utils import DP_Analysis_Utils
from UTILS.chisquare_fitter import ChiSquare_Fitter
from EMCEE_FITTER.emcee_fitter import MCMC_Fitter

#Basic definitions:
#**********************************************
data_set = int(sys.argv[1])
fileInitName = sys.argv[2]
ana_name = sys.argv[3]
out_name = sys.argv[4]

#kin_acc_limits = None
kin_acc_limits = [-2.5,2.5]
kin_acc_start_values = [-2.0,2.0]
kin_acc = None
kin_acc_step = 0.025

ana_utils = DP_Analysis_Utils()
mcmc_fitter = MCMC_Fitter()

data_dict = {
   0: ['17','2017','GlueX 2017','GlueX-2017'],
   1: ['18S','2018S','GlueX 2018-01','GlueX-2018-01'],
   2: ['18F','2018F','GlueX 2018-08','GlueX-2018-08'],
   3: ['All','All','Full GlueX-I','Full GlueX-I']
}

addName = data_dict[data_set][1] + '_' + ana_name

npyDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/npy_dalitz_data'
resultDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/mcmc_dalitz_data'

fullSaveName = resultDir + '/' + out_name + '_' + addName
dataSetName = data_dict[data_set][2]

#Start values:
valueNames = ['norm','a','b','c','d','e','f','g','h','l']

# minDPVals = [10000,-1.2,0.05,-0.01,0.05,-0.01,0.1,-0.01,-0.01,-0.01]
# maxDPVals = [30000,-1.0,0.2,0.01,0.1,0.01,0.2,0.01,0.01,0.01]

minDPVals = [1.0,-1.2,0.05,-0.01,0.05,-0.01,0.1,-0.01,-0.01,-0.01]
maxDPVals = [5.0,-1.0,0.2,0.01,0.1,0.01,0.2,0.01,0.01,0.01]
initFit_start_values = mcmc_fitter.get_start_values(minDPVals,maxDPVals)

#If you do not run an initial chi2 fit, you need to provide a covariance matrix
mcmc_cov_matrix = None
mcmc_parameter_update = 'random'
run_init_fit = True
init_fit_res_name = None #--> Just change if you want to store the fit results as a .png-file
init_fit_name = 'Initial Fit'
use_diag_cov = True

if use_diag_cov == False:
    mcmc_parameter_update = 'vector'

show_init_fit_results = True

#Parameters for the fitter:
nWalkers = 150 # 150
nIterations = 2000
constr_error = [0.035]*4
loss_function = 'huber'
n_sigma_start = 25.0
scan_tau = True
n_monitoring_steps = 50
#**********************************************

#Show the intro:
#**********************************************
mcmc_fitter.show_intro()
#**********************************************

#Load the data and rebin it:
#**********************************************
print("Load and prepare data...")

#-------------------------------
if data_set < 3:
   dataFileName = fileInitName + addName
   fullLoadName = npyDir + '/' + dataFileName + '.npy'
   npy_dalitz_data = np.load(fullLoadName)
else:
   data_collection = [] 
   #++++++++++++++++++++++++++
   for d in range(3):
       addName = data_dict[d][1] + '_' + ana_name
       dataFileName = fileInitName + addName
       fullLoadName = npyDir + '/' + dataFileName + '.npy'

       data_collection.append(np.load(fullLoadName))
   #++++++++++++++++++++++++++

   npy_dalitz_data = ana_utils.add_data(data_collection)
#-------------------------------

rebinned_data = ana_utils.rebin_DP(npy_dalitz_data)

if kin_acc_limits is not None:
   kin_acc = rebinned_data[:,5]


print("...done!")
print("  ")
#**********************************************

#**********************************************
print("Set up fitter...")

mcmc_fitter.setup_fitter(kin_acc,kin_acc_limits)
init_fitter = ChiSquare_Fitter(kinematic_acceptance=kin_acc)

print("...done!")
print("  ")
#**********************************************

# eff_M = None
# d_eff_M = None
# if use_eff_m:
#    #**********************************************
#    print("Load and prepare efficiency matrix...")

#    eff_M_raw = np.load(npyDir + '/' + fileInitName + addName + '_effMatrix.npy')
#    d_eff_M_raw = np.load(npyDir + '/' + fileInitName + addName + '_deffMatrix.npy')

#    eff_M, d_eff_M = ana_utils.resize_efficiency_matrix(rebinned_data,eff_M_raw,d_eff_M_raw)

#    print("...done!")
#    print("  ")
#    #**********************************************

#Get initial values:
#**********************************************
if run_init_fit:
    print("Run chi2 fit to determine start parameters for the random walk fitter...")
    
    init_fit_values, init_fit_cov, init_fit_errs, _ = init_fitter.run_initial_fitter(rebinned_data,initFit_start_values,valueNames,dataSetName,init_fit_name,show_results=show_init_fit_results,figSaveName=init_fit_res_name)
    
    #----------------------
    if use_diag_cov:
        mcmc_cov_matrix = init_fit_errs

        if kin_acc is not None:
            mcmc_cov_matrix = np.append(init_fit_errs,kin_acc_step)
    else:
        mcmc_cov_matrix = init_fit_cov

        if kin_acc is not None:
            n_el = init_fit_cov.shape[0] + 1
            mcmc_cov_matrix = np.zeros((n_el,n_el))

            #++++++++++++++++++++++++
            for a in range(n_el):
                #++++++++++++++++++++++++
                for b in range(n_el):
                    #-------------------------------
                    if a < n_el-1 and b < n_el -1:
                        mcmc_cov_matrix[a][b] = init_fit_cov[a][b]
                    else:
                        mcmc_cov_matrix[a][b] = 0.0
                    #-------------------------------
                #++++++++++++++++++++++++ 
            #++++++++++++++++++++++++

            mcmc_cov_matrix[n_el-1][n_el-1] = kin_acc_step

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


#mover = mcmc_fitter.get_gaussian_mover(mcmc_cov_matrix,mcmc_parameter_update)
#mover = mcmc_fitter.get_stretch_mover(2.0)
#mover = mcmc_fitter.get_walk_mover(None)
mover = mcmc_fitter.get_kde_mover(None)
#mover = mcmc_fitter.get_de_mover(1e-5,None)
#mover = mcmc_fitter.get_de_snooker_mover(1.7)


index, mon_iterations, tau, max_tau = mcmc_fitter.run_mcmc_fitter(nWalkers,minDPVals,maxDPVals,rebinned_data,nIterations,movers=mover,loss_func_str=loss_function,constr_error=constr_error,tolerance=0.1,monitor_tau=scan_tau,resultFileName=fullSaveName,monitor_iteration=n_monitoring_steps,kinematic_acceptance_start_values=kin_acc_start_values)

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

