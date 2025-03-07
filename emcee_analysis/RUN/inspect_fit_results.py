#!/usr/bin/env python3
import numpy as np
import sys

from UTILS.result_evaluator import Fit_Result_Evaluator
from UTILS.analysis_utils import DP_Analysis_Utils

#Basic definitions:
#**********************************************
data_set = int(sys.argv[1])
fileInitName = sys.argv[2]
ana_name = sys.argv[3]
out_name = sys.argv[4]

evaluator = Fit_Result_Evaluator()
ana_utils = DP_Analysis_Utils()

data_dict = {
   0: ['17','2017','GlueX 2017','GlueX-2017'],
   1: ['18S','2018S','GlueX 2018-01','GlueX-2018-01'],
   2: ['18F','2018F','GlueX 2018-08','GlueX-2018-08'],
   3: ['All','All','Full GlueX-I','Full GlueX-I']
}

addName = data_dict[data_set][1] + '_' + ana_name
dataSetName = data_dict[data_set][3]

resultDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/mcmc_dalitz_data'
npyDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/npy_dalitz_data'

fullLoadName = resultDir + '/' + out_name + '_' + addName


save_autocorr_name = None
parNames = ['norm','a','b','c','d','e','f','g','h','l','acc_cut']
#parNames = ['norm','a','b','c','d','e','f','g','h','l']
parameter_selection_mode = 'default'

save_results_name = None

save_dp_pars = None
show_dp_pars = True

do_comparison = True
save_comparison = None

show_burnin = False

#sys_errors = [0.046,0.048,0.009,0.056,0.017,0.116,0.121,0.04,0.016]
sys_errors = None
#**********************************************

print(" ")
#************************************************************
print("Load Dalitz Plot data and MCMC reader...")

#-------------------------------
if data_set < 3:
   DP_Data_Name = npyDir + '/' + fileInitName + addName + '.npy'
   DP_Data = ana_utils.rebin_DP(np.load(DP_Data_Name)) 
else:
   data_collection = [] 
   #++++++++++++++++++++++++++
   for d in range(3):
       addName = data_dict[d][1] + '_' + ana_name
       dataFileName = fileInitName + addName
       current_data_name = npyDir + '/' + dataFileName + '.npy'

       data_collection.append(np.load(current_data_name))
   #++++++++++++++++++++++++++

   DP_Data = ana_utils.rebin_DP(ana_utils.add_data(data_collection)) 
#-------------------------------

mcmc_reader = evaluator.get_reader(fullLoadName)

print("...done!")
print(" ")
#***********************************************************

#Show the autocorrelatio time first, you will notice here right away 
#if something went wrong:
#***********************************************************
print("Inspect mean autocorrelation time...")

evaluator.show_autocorr_time(fullLoadName,figSaveName=save_autocorr_name)

print("...done!")
print(" ")
#***********************************************************

#Get chains from walker:
#***********************************************************
print("Get chains from walkers...")

DP_Chains = evaluator.get_chains(mcmc_reader,parNames)

print("...done!")
print(" ")
#***********************************************************

if show_burnin:
   #***********************************************************
   print("Display burn-in phase...")

   evaluator.show_chains(mcmc_reader,parNames)

   print("...done!")
   print(" ")
   #***********************************************************

#Determine DP parameters:
#************************************************************
print("Determine Dalitz Plot parameters and errrors...")

DP_Results = evaluator.get_and_show_DP_pars(DP_Chains,parNames,dataSetName,mode=parameter_selection_mode,draw_parameters=show_dp_pars,figSaveName=save_dp_pars)

print("...done!")
print(" ")
#************************************************************

#Take a look at the fit results:
#************************************************************
print("Visualize fit results...")

evaluator.show_DP_fit_results(DP_Results,DP_Data,parNames,dataSetName,saveResults=save_results_name,fontSize=20,labelFontSize=20,plot_label='MCMC-Fit')

print("...done!")
print(" ")
#************************************************************

#Compare with other experiments, if wanted:
#************************************************************
if do_comparison:
    print("Show comparison to other experiments...")
  
    in_pars = DP_Results[0]
    in_errs = DP_Results[1]

    evaluator.run_comparison(in_pars,in_errs,['norm','a','b','c','d','e','f','g','h','l'],dataSetName,save_comparison,fontSize=20,labelFontSize=20,GlueX_sys_errors=sys_errors)

    print("...done!")
    print(" ")
#************************************************************

