#!/usr/bin/env python3
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

from UTILS.result_evaluator import Fit_Result_Evaluator
from UTILS.analysis_utils import DP_Analysis_Utils

#Basic definitions:
#**********************************************
ana_name = sys.argv[1]
gen_data_name = sys.argv[2]
gen_pars_idx = int(sys.argv[3])

evaluator = Fit_Result_Evaluator()
ana_utils = DP_Analysis_Utils()

resultDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/mcmc_dalitz_data'
npyDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/npy_dalitz_data'
chisqFit_Dir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/Chisquare_Fitter_Results'

fullLoadName = resultDir + '/' + ana_name
loadGenName = npyDir + '/' + gen_data_name

show_autocorr_time = True
save_autocorr_name = None
parNames = ['norm','a','b','c','d','e','f','g','h','l']
parameter_selection_mode = 'default'
dataSetName = 'Gen Data'

save_results_name = None

save_dp_pars = None
show_dp_pars = True

do_comparison = True
save_comparison = None

sys_errors = None
#**********************************************

#Define custom parameters here!
#**************************************
Gen_DP_Pars = [-1.0,0.1,0.0,0.08,0.0,0.07,0.0,0.0,0.0]
Gen_DP_Errs = [0.02]*len(Gen_DP_Pars)

if gen_pars_idx == 0:
    Gen_DP_Pars, Gen_DP_Errs, _ = evaluator.get_KLOE_results()
    Gen_DP_Pars[0] = - Gen_DP_Pars[0]
elif gen_pars_idx == 1:
    Gen_DP_Pars, Gen_DP_Errs, _ = evaluator.get_WASA_results()
    Gen_DP_Pars[0] = - Gen_DP_Pars[0]
#**************************************

#NO CHANGES SHOULD BE DONE BELOW THIS LINE!!!!!
##################################################################################

print(" ")
#************************************************************
print("Load Dalitz Plot data, MCMC reader and chi2 fit resutls...")

Gen_DP_Data = np.load(loadGenName + '.npy')
DP_Data = ana_utils.rebin_DP(Gen_DP_Data,include_raw_yields=False)

mcmc_reader = evaluator.get_reader(fullLoadName)

DP_Chisq_Values = np.load(chisqFit_Dir + '/' + ana_name + '_chisqFit_DP_pars.npy')
DP_Chisq_Errors = np.load(chisqFit_Dir + '/' + ana_name + '_chisqFit_DP_pars.npy')

print("...done!")
print(" ")
#***********************************************************

if show_autocorr_time:
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

#Determine DP parameters:
#************************************************************
print("Determine Dalitz Plot parameters and errrors...")

DP_Results = evaluator.get_and_show_DP_pars(DP_Chains,parNames,dataSetName,mode=parameter_selection_mode,draw_parameters=show_dp_pars,figSaveName=save_dp_pars)

DP_MCMC_Values = DP_Results[0]
DP_MCMC_Errors = DP_Results[1]

print("...done!")
print(" ")
#************************************************************

#Take a look at the fit results:
#************************************************************
print("Visualize fit results...")

evaluator.show_DP_fit_results(DP_Results,DP_Data,parNames,dataSetName,saveResults=save_results_name,fontSize=20,labelFontSize=20,pars_from_chisqFit=DP_Chisq_Values)

print("...done!")
print(" ")
#************************************************************

#Compare results to input parameters:
#************************************************************
print("Run comparison to input parameters...")

diff_mcmc = evaluator.calculate_differences(DP_MCMC_Values[1:],DP_MCMC_Errors[1:],Gen_DP_Pars,Gen_DP_Errs)
diff_chisq = evaluator.calculate_differences(DP_Chisq_Values[1:],DP_Chisq_Errors[1:],Gen_DP_Pars,Gen_DP_Errs)

plt.rcParams.update({'font.size': 20})
figc, axc = plt.subplots(1,2,figsize=(16,8))
figc.subplots_adjust(wspace=0.5)

Gen_DP_Pars[0] = - Gen_DP_Pars[0]
x_values = np.arange(len(Gen_DP_Pars))
x_labels = copy.copy(parNames[1:])    
x_labels[0] = '-a'

DP_MCMC_Values[1] = -DP_MCMC_Values[1]
DP_Chisq_Values[1] = -DP_Chisq_Values[1]

axc[0].errorbar(x_values,Gen_DP_Pars,Gen_DP_Errs,fmt='ko',label='Input Parameters')
axc[0].errorbar(x_values,DP_Chisq_Values[1:],DP_Chisq_Errors[1:],fmt='bd',label=r'$\chi^{2}$' + '-Fit')
axc[0].errorbar(x_values,DP_MCMC_Values[1:],DP_MCMC_Errors[1:],fmt='rs',label='MCMC-Fit')
axc[0].set_xticks(x_values)
axc[0].set_xticklabels(tuple(x_labels))
axc[0].set_ylabel('Parameter Value')

axc[0].legend()
axc[0].grid(True)

axc[1].errorbar(x_values,diff_chisq[0],diff_chisq[1],fmt='bd',label=r'$\chi^{2}$' + '-Fit')
axc[1].errorbar(x_values,diff_mcmc[0],diff_mcmc[1],fmt='rs',label='MCMC-Fit')
axc[1].set_xticks(x_values)
axc[1].set_xticklabels(tuple(x_labels))
axc[1].set_ylabel('Difference')
axc[1].plot([0.0,8.0],[0.0,0.0],'k--',linewidth=2.0)

axc[1].legend()
axc[1].grid(True)

plt.show()

print("...done!")
print(" ")
#************************************************************





