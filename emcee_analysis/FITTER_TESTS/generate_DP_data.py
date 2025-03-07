#!/usr/bin/env python3

import numpy as np
import sys

from UTILS.analysis_utils import DP_Analysis_Utils
from UTILS.result_evaluator import Fit_Result_Evaluator

#Read in user-input:
#**************************************
ana_utils = DP_Analysis_Utils()
fit_eval = Fit_Result_Evaluator()

n_events = int(sys.argv[1])
n_dp_bins = int(sys.argv[2])
g_noise = float(sys.argv[3])
f_noise = float(sys.argv[4])
gen_pars_idx = int(sys.argv[5])
out_name = sys.argv[6]

#Make sure, that some parameters are used:
if gen_pars_idx < 0 or gen_pars_idx > 2:
    gen_pars_idx = 2

par_dict = {
    0: 'KLOE DP parameters',
    1: 'WASA DP parameters',
    2: 'Custom DP parameters'
}
#**************************************

#Define custom parameters here!
#**************************************
Gen_DP_Pars = [-1.0,0.1,0.0,0.08,0.0,0.07,0.0,0.0,0.0]
Gen_DP_Errs = [0.02]*len(Gen_DP_Pars)

if gen_pars_idx == 0:
    Gen_DP_Pars, Gen_DP_Errs, _ = fit_eval.get_KLOE_results()
    Gen_DP_Pars[0] = - Gen_DP_Pars[0]
elif gen_pars_idx == 1:
    Gen_DP_Pars, Gen_DP_Errs, _ = fit_eval.get_WASA_results()
    Gen_DP_Pars[0] = - Gen_DP_Pars[0]
#**************************************

#Important directories:
#**************************************
store_npy_dir = '../../npy_dalitz_data/'
#**************************************

#NO CHANGES SHOULD BE DONE BELOW THIS LINE!!!!!
##################################################################################

#Get a nice intro:
#**************************************
print("  ")
print("***********************************************")
print("*                                             *")
print("* Run 1D eta->pi+pi-pi0 Dalitz Plot Generator *")
print("*                                             *")
print("***********************************************")
print("  ")
#**************************************

#Set up and run the generator:
#**************************************
print("Generate 1D Dalitz Plot with " + str(n_dp_bins*n_dp_bins) + " bins and " + str(n_events) + " events...")
print("   >>> Apply gaussian noise: " + str(g_noise) + " <<< ")
print("   >>> Apply flat noise: " + str(f_noise) + " <<< ")
print("   >>> Use " + par_dict[gen_pars_idx] + " <<< ")

Gen_DP_Data = ana_utils.gen_DP(Gen_DP_Pars,Gen_DP_Errs,n_dp_bins,n_events,add_gaussian_noise=g_noise,add_flat_noise=f_noise)

print("...done!")
print("  ")
#**************************************

#Store the data:
#**************************************
full_save_name = store_npy_dir + out_name + '.npy'

print("Store generated data at: " + full_save_name + '...')

np.save(full_save_name,Gen_DP_Data)

print("...done!")
print("  ")
#**************************************



