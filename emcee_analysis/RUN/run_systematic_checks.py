#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sys
from SYSERR_ESTIMATION.error_estimator import Error_Estimator
from UTILS.result_evaluator import Fit_Result_Evaluator

err_estimator = Error_Estimator()
evaluator = Fit_Result_Evaluator()

# kloe_results = evaluator.get_KLOE_results()[0]
# wasa_results = evaluator.get_WASA_results()[0]

data_set = int(sys.argv[1])
sys_var = sys.argv[2]
out_name = sys.argv[3]
result_df_name = sys.argv[4]

kfit_ref = 3
imgg_ref = 1
ebeam_ref = 11 #was 11
data_ref = 3
yield_sys_ref = 13
fit_sys_ref = 7 # was 11

data_dict = {
   0: ['17','2017','GlueX 2017','GlueX-2017'],
   1: ['18S','2018S','GlueX 2018-01','GlueX-2018-01'],
   2: ['18F','2018F','GlueX 2018-08','GlueX-2018-08'],
   3: ['All','All','GlueX-I','GlueX-I']
}

sys_dict = {
    'kfit': [[0,1,2,3,4,5],kfit_ref,'KFit Cut',True],
    'imgg': [[0,1,2],imgg_ref,'IM(gg) Cut',True],
    'ebeam': [[i for i in range(ebeam_ref+1)],ebeam_ref,'Ebeam Range',False],
    'data': [[0,1,2,3],data_ref,'Data Set',False],
    'yield_sys': [[i for i in range(yield_sys_ref+1)],yield_sys_ref,'Yield Extraction',True],
    'fit_sys': [[i for i in range(fit_sys_ref+1)],fit_sys_ref,'Fitter Settings',True]
}

corr_stats = sys_dict[sys_var][3]

dataDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/pandas_mcmc_dalitz_data' 
parNames = ['norm','a','b','c','d','e','f','g','h','l']
finalResultDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/final_results'

print("  ")
#************************************************************
print("Set up DataFrame to store results...")

result_df = err_estimator.load_result_df(finalResultDir,result_df_name)

print("...done!")
print(" ")
#************************************************************

#************************************************************
print("Load and combine dataframes...")

yield_sys_names = [
    "BkgPol2_",
    "BkgPol4_",
    "IntR2_",
    "IntR4_",
    "IntR5_",
    "NFit3_",
    "NFit7_",
    "FitBkg_",
    "FitYield_",
    "nbins9_",
    "nbins10_",
    "nbins12_",
    "nbins13_"
]

fit_sys_names = [
   # "nonDiagCov_",
    "stretchMover_",
    "walkMover_",
  #  "kdeMover_",
   # "deMover_",
   # "deSnookerMover_",
    "100Walkers_",
    "300Walkers_",
    "linearLoss_",
    "softL1Loss_",
    "cauchyLoss_"
]

sys_indices = sys_dict[sys_var][0]
sys_dfs = []
current_name = ""
addName = "_no_acc_cut_df"
#++++++++++++++++++++++++++++++++++++
for i in sys_indices:
    if sys_var == "kfit":
        current_name = out_name + "_" + data_dict[data_set][1] + "_nbins11_kfit_cut" + str(i) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName
    elif sys_var == "imgg":
        current_name = out_name + "_" + data_dict[data_set][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(i) + "_zv_cut1" + addName
    elif sys_var == "ebeam":
        if i < ebeam_ref:
           current_name = out_name + "_" + data_dict[data_set][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1_ebeam" + str(i) + addName
        else:
           current_name = out_name + "_" + data_dict[3][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName #--> change back!
    elif sys_var == "data":
        current_name = out_name + "_" + data_dict[i][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName
    elif sys_var == "yield_sys":
        if i < 9:
           current_name = yield_sys_names[i] + out_name + "_" + data_dict[3][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName
        elif i < yield_sys_ref:
           current_name = out_name + "_" + data_dict[3][1] + "_" + yield_sys_names[i] + "kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName
        else:
           current_name = out_name + "_" + data_dict[3][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName
    elif sys_var == "fit_sys":
        if i < fit_sys_ref:
           current_name = fit_sys_names[i] + out_name + "_" + data_dict[3][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName
        else:
           current_name = out_name + "_" + data_dict[3][1] + "_nbins11_kfit_cut" + str(kfit_ref) + "_imgg_cut" + str(imgg_ref) + "_zv_cut1" + addName
          
    current_df = pd.read_csv(dataDir + '/' + current_name + ".csv")
    current_df[sys_var] = i
    sys_dfs.append(current_df)
#++++++++++++++++++++++++++++++++++++

full_df = pd.concat(sys_dfs)
full_df = shuffle(full_df,random_state=2)

print("...done!")
print(" ")
#************************************************************

#Determine parameters:
#************************************************************
print("Determine Dalitz Plot parameters from dataframe...")

dp_par_val_list = []
dp_par_err_list = []
#+++++++++++++++++++++++++++++++++
for i in sys_indices:
    dp_par_val_list.append(full_df[full_df[sys_var]==i][parNames].mean().values)
    dp_par_err_list.append(full_df[full_df[sys_var]==i][parNames].std().values)
#+++++++++++++++++++++++++++++++++

dp_par_values = np.array(dp_par_val_list)
dp_par_errors = np.array(dp_par_err_list)

print("...done!")
print(" ")
#************************************************************

#Prepare errors according to the Barlow paper:
#************************************************************
print("Handle errors for stat. correlated data sets...")
ref_index = sys_dict[sys_var][1]
new_dp_par_errors = np.zeros_like(dp_par_errors)
#+++++++++++++++++++++++++++++++
for p in range(len(parNames)):
      current_errs = dp_par_errors[:,p]
      new_current_errs = err_estimator.subtract_errors(dp_par_errors[ref_index,p],current_errs)

      #+++++++++++++++++++++++++++++++
      for  a in range(dp_par_errors.shape[0]):
        if corr_stats:    
           new_dp_par_errors[a][p] = new_current_errs[a]
        else:
           new_dp_par_errors[a][p] = current_errs[a]
      #+++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++

print("...done!")
print(" ")
#************************************************************

#Calculate systematic errors (very experimental!):
#************************************************************
print("Calculate systematic errors...")

low_errs = np.zeros(len(parNames)-1)
high_errs = np.zeros(len(parNames)-1)
center_values = np.zeros(len(parNames)-1)
low_values = np.zeros(len(parNames)-1)
high_values = np.zeros(len(parNames)-1)

#++++++++++++++++++++++++++++++++++++++++
for p in range(1,len(parNames)):
    low_errs[p-1],high_errs[p-1],center_values[p-1],low_values[p-1],high_values[p-1] = err_estimator.calc_sys_error_from_linear_fit(dp_par_values,new_dp_par_errors,ref_index,p)
#++++++++++++++++++++++++++++++++++++++++

print("...done!")
print(" ")
#************************************************************

#************************************************************
print("Visualize results...")

plt.rcParams.update({'font.size': 20})

fig_res,ax_res = plt.subplots(2,5,sharex=True)
fig_res.set_size_inches((22,10))
fig_res.subplots_adjust(wspace=1.5)

x_values = np.array(sys_dict[sys_var][0])
x_min = min(sys_dict[sys_var][0])
x_max = max(sys_dict[sys_var][0])

y_names = ['Norm','Parameter a','Parameter b','Parameter c','Parameter d','Parameter e','Parameter f','Parameter g','Parameter h','Parameter l']

#+++++++++++++++++++++++++
for p in range(5):
   ax_res[0][p].errorbar(x_values,dp_par_values[:,p],new_dp_par_errors[:,p],fmt='ko')
   ax_res[1][p].errorbar(x_values,dp_par_values[:,p+5],new_dp_par_errors[:,p+5],fmt='ko')

   if p > 0:
      ax_res[0][p].plot([x_min,x_max],[center_values[p-1],center_values[p-1]],'r-',linewidth=2.0)
      ax_res[0][p].plot([x_min,x_max],[low_values[p-1],low_values[p-1]],'b--',linewidth=2.0)
      ax_res[0][p].plot([x_min,x_max],[high_values[p-1],high_values[p-1]],'b--',linewidth=2.0)
    
   ax_res[1][p].plot([x_min,x_max],[center_values[p+4],center_values[p+4]],'r-',linewidth=2.0)
   ax_res[1][p].plot([x_min,x_max],[low_values[p+4],low_values[p+4]],'b--',linewidth=2.0)
   ax_res[1][p].plot([x_min,x_max],[high_values[p+4],high_values[p+4]],'b--',linewidth=2.0)

   ax_res[0][p].grid(True)
   ax_res[1][p].grid(True)
   ax_res[1][p].set_xlabel(sys_dict[sys_var][2])
   ax_res[1][p].set_xticks(x_values)

   ax_res[0][p].set_ylabel(y_names[p])
   ax_res[1][p].set_ylabel(y_names[5+p])

   if sys_var == "ebeam" or sys_var == "yield_sys" or sys_var == "fit_sys":
      if sys_var == "ebeam": 
         every_nth = 2
      if sys_var == "yield_sys" or "fit_sys": 
         every_nth = 3

      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      for n, label in enumerate(ax_res[1][p].xaxis.get_ticklabels()):
        if n % every_nth != 0:
           label.set_visible(False)
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++

plt.show()

print("...done!")
print(" ")
#************************************************************

dp_values_set = False
dp_values = {}
dp_stat_errs = {}
dp_sys_errs = {}

value_check = result_df[result_df['flag'].str.contains('dp_values')]

if len(value_check) > 0:
   dp_values_set = True

#++++++++++++++++++++++++++++++++++++++
for p in range(len(parNames)):
    if p > 0:
        print("  ")
        print("Parameter: " + parNames[p])
        print("------------------")
        print("Value: " + str(center_values[p-1]))
        print("stat. error: " + str(dp_par_errors[ref_index][p]))
        print("low. error: " + str(low_errs[p-1]))
        print("high. error: " + str(high_errs[p-1]))
        print("------------------")

        if dp_values_set == False:
           dp_values[parNames[p]] = [center_values[p-1]]
           dp_stat_errs[parNames[p]] = [dp_par_errors[ref_index][p]]

        dp_sys_errs[parNames[p]] = [0.5*(math.fabs(low_errs[p-1]) + math.fabs(high_errs[p-1]))]
#++++++++++++++++++++++++++++++++++++++

if dp_values_set == False:
   dp_values['flag'] = 'dp_values'
   dp_stat_errs['flag'] = 'dp_stat_errs'

   result_df = result_df.append(pd.DataFrame(dp_values),ignore_index=True)
   result_df = result_df.append(pd.DataFrame(dp_stat_errs),ignore_index=True)

dp_sys_errs['flag'] = sys_var
result_df = result_df.append(pd.DataFrame(dp_sys_errs),ignore_index=True)

result_df.to_csv(path_or_buf=finalResultDir + '/' + result_df_name + '.csv',index=False)

#print(result_df)

print("  ")
