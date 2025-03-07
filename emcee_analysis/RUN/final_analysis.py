#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from UTILS.result_evaluator import Fit_Result_Evaluator
import sys

result_df_name = sys.argv[1]

sys_flags = ['kfit','imgg','ebeam','yield_sys','fit_sys','data']
stat_err_flag = 'dp_stat_errs'
rec_err_flag = ['kfit','imgg']

color_dict = {
    'rec': 'm',
    'ebeam': 'g',
    'yield_sys': 'b',
    'data': 'r',
    'fit_sys': 'orange',
    'all': 'c'
}

line_dict = {
    'rec': '-',
    'ebeam': '-.',
    'yield_sys': ':',
    'data': '--',
    'fit_sys': '-.',
    'all': '-'
}

marker_dict = {
    'rec': 'o',
    'ebeam': 'd',
    'yield_sys': 'p',
    'data': 's',
    'fit_sys': '*',
    'all': 'P'
}

label_dict = {
    'rec': r'$\sigma_{rec}$',
    'ebeam': r'$\sigma_{E_{\gamma}}$',
    'yield_sys': r'$\sigma_{yield}$',
    'data': r'$\sigma_{run}$',
    'fit_sys': r'$\sigma_{fit}$',
    'all': r'$\sigma_{sys}$'
}



#>>> NO CHANGES BELOW THIS LINE <<<<

#/////////////////////////////////////////////////////////////////////////

evaluator = Fit_Result_Evaluator()

data_dir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/final_results'
result_df = pd.read_csv(data_dir + '/' + result_df_name + '.csv')

#*****************************************
def get_err(df,err_flag):
    return df[df['flag'] == err_flag].to_numpy()[0][:9]

#---------------------------

def calc_R(a,b):
    return a/b    

#---------------------------

def run_analysis(df,stat_err_flag,sys_err_flags,rec_err_flags):
    #Get the individual error contributions:
    
    stat_errs = get_err(df,stat_err_flag)
    total_errs = 0
    rec_errs = 0
    ratios = {}
    #++++++++++++++++++++++++
    for err_flag in sys_err_flags:
        current_err = get_err(df,err_flag)

        total_errs += current_err**2

        #+++++++++++++++++++++
        for rec_err in rec_err_flags:
            if err_flag == rec_err:
                rec_errs += current_err**2
            else:
                ratios[err_flag] = calc_R(current_err,stat_errs)
        #+++++++++++++++++++++

    #++++++++++++++++++++++++

    rec_errs = np.sqrt(np.array(rec_errs,dtype=np.float32))
    ratios['rec'] = calc_R(rec_errs,stat_errs) 

    total_errs = np.sqrt(np.array(total_errs,dtype=np.float32))

    ratios['all'] = calc_R(total_errs,stat_errs) 

    return stat_errs, total_errs, ratios
#*****************************************

sigma_stat, sigma_sys, sigma_r =  run_analysis(result_df,stat_err_flag,sys_flags,rec_err_flag)    

sigma_stat = np.array(sigma_stat,dtype=np.float32)
sigma_all = np.sqrt(sigma_stat**2 + sigma_sys**2)

x_values = np.linspace(1,9,9)
x_names = ['a','b','c','d','e','f','g','h','l']
par_names = ['a','b','c','d','e','f','g','h','l']

plt.rcParams.update({'font.size': 30})

fig,ax = plt.subplots(figsize=(12,9))
fig.suptitle('Systematic Error Summary Sept. 2022')

#+++++++++++++++++++++++++
for el in color_dict:
    ax.plot(x_values,sigma_r[el],color=color_dict[el],linestyle=line_dict[el],marker=marker_dict[el],label=label_dict[el],markersize=12,linewidth=3.0)
#+++++++++++++++++++++++++
ax.plot(x_values,np.ones(9),"k--",linewidth=3.0,label=r'$\sigma_{stat}$')

ax.set_ylabel(r'$\sigma / \sigma_{stat}$')
ax.set_xticks(x_values)
ax.set_xticklabels(par_names)
ax.set_xlabel('Dalitz Parameter')

ax.legend(prop={"size":20})
ax.set_ylim(-0.05,7.5)
ax.grid(True)

plt.show()

dp_values = result_df[result_df['flag']=='dp_values'].to_numpy()[0][:9]

WASA_values = [1.144,0.219,0.086,0.115,0.0]
WASA_errors = [0.018,0.066,0.033,0.037,0.0]
WASA_stat_errors = [0.018,0.019,0.018,0.037]

# KLOE_values = [1.104,0.142,0.0,0.073,0.0,0.154,0.0,0.0,0.0]
# KLOE_errors = [0.005,0.011,0.0,0.007,0.0,0.011,0.0,0.0,0.0]
# KLOE_stat_errors = [0.003,0.003,0.003,0.006]

KLOE_values = [1.095,0.145,0.081,0.141,-0.044]
KLOE_errors = [0.006,0.008,0.008,0.014,0.0021]
KLOE_stat_errors = [0.003,0.003,0.003,0.007]

amptool_values = [1.059,0.178,0.003,0.065,0.017,0.182,0.003,0.016,0.007]
amptool_errors = [0.007,0.007,0.005,0.007,0.007,0.012,0.01,0.011,0.01]

#*****************************************
def calc_diff(y_1,dy_1,y_2,dy_2):
    n_points= len(y_1)
    diff = []
    diff_err = []

    #++++++++++++++++++++++++++
    for i in range(n_points):
        diff.append(y_1[i] - y_2[i])
        diff_err.append(math.sqrt(dy_1[i]*dy_1[i] + dy_2[i]*dy_2[i]))
    #++++++++++++++++++++++++++

    return diff, diff_err
#*****************************************

y_values = np.linspace(1,9,9)
par_names = ['a','b','d','f','g']

dp_values[0] *= -1.0 

GlueX_values = [dp_values[0],dp_values[1],dp_values[3],dp_values[5],dp_values[6]]
GlueX_errors = [sigma_all[0],sigma_all[1],sigma_all[3],sigma_all[5],sigma_all[6]]
GlueX_stat_errors = [sigma_stat[0],sigma_stat[1],sigma_stat[3],sigma_stat[5],sigma_stat[6]]


#diff_wasa_kloe, diff_err_wasa_kloe = calc_diff(WASA_values,WASA_errors,KLOE_values,KLOE_errors)
diff_gluex_kloe, diff_err_gluex_kloe = calc_diff(GlueX_values,GlueX_errors,KLOE_values,KLOE_errors)
diff_gluex_kloe_stat, diff_err_gluex_kloe_stat = calc_diff(GlueX_values,GlueX_stat_errors,KLOE_values,KLOE_errors)


diff_gluex_wasa, diff_err_gluex_wasa = calc_diff(GlueX_values,GlueX_errors,WASA_values,WASA_errors)
diff_gluex_wasa_stat, diff_err_gluex_wasa_stat = calc_diff(GlueX_values,GlueX_stat_errors,WASA_values,WASA_errors)
#diff_gluex_amptool, diff_err_gluex_amptool = calc_diff(dp_values,sigma_all,amptool_values,amptool_errors)

par_plot_dict = {}
n_steps = 11
#++++++++++++++++++++
for j in range(5):
    par_plot_dict[par_names[j]] = [j,np.linspace(1+j*n_steps,1+(j+1)*n_steps,n_steps)]
#++++++++++++++++++++


def show_one_par(ax,par_name,res_1,res_2,res_1_err,res_2_err,exp):
    y_values = par_plot_dict[par_name][1]
    idx = par_plot_dict[par_name][0]

    if exp == 'kloe':
        ax.errorbar(x=res_1[idx],y=y_values[5],xerr=res_1_err[idx],fmt='ro',ecolor='r',elinewidth =5, capsize=10,label='Stat. + Sys. Errs')
        ax.errorbar(x=res_2[idx],y=y_values[2],xerr=res_2_err[idx],fmt='ko',ecolor='k',elinewidth =5, capsize=10,label='Stat. Errs')
    elif exp == 'wasa':
        ax.errorbar(x=res_1[idx],y=y_values[5],xerr=res_1_err[idx],fmt='bo',ecolor='b',elinewidth =5, capsize=10,label='Stat. + Sys. Errs')
        ax.errorbar(x=res_2[idx],y=y_values[2],xerr=res_2_err[idx],fmt='ko',ecolor='k',elinewidth =5, capsize=10,label='Stat. Errs')

    # ax.errorbar(x=res_1[idx],y=y_values[19],xerr=res_1_err[idx],fmt='mo',ecolor='m',elinewidth =5, capsize=10,label='WASA - KLOE')

    # ax.errorbar(x=res_2[idx],y=y_values[14],xerr=res_2_err[idx],fmt='ro',ecolor='r',elinewidth =5, capsize=10,label='GlueX-I - KLOE')

    # ax.errorbar(x=res_3[idx],y=y_values[9],xerr=res_3_err[idx],fmt='bo',ecolor='b',elinewidth =5, capsize=10, label='GlueX-I - WASA')

    # ax.errorbar(x=res_4[idx],y=y_values[4],xerr=res_4_err[idx],fmt='go',ecolor='g',elinewidth =5, capsize=10, label='GlueX-I - AmpTool')

    ax.plot([0.0]*n_steps,y_values,'k--',linewidth=3.0)

    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_ylabel(par_name)



plt.rcParams.update({'font.size': 25})
fig, ax = plt.subplots(5,1,figsize=(12,8),sharex=True)
fig.subplots_adjust(hspace=0)
#fig.suptitle('Comparing GlueX-I and KLOE Dalitz Parameters')

fig.suptitle('Comparing GlueX-I and KLOE Dalitz Parameters')

#++++++++++++++++
for i in range(5):
#     show_one_par(ax[4-i],par_names[i],diff_gluex_wasa,diff_gluex_wasa_stat,
#       diff_err_gluex_wasa,diff_err_gluex_wasa_stat,
#       'wasa'
#    )

    show_one_par(ax[4-i],par_names[i],diff_gluex_kloe,diff_gluex_kloe_stat,
      diff_err_gluex_kloe,diff_err_gluex_kloe_stat,
      'kloe'
   )
#++++++++++++++++

ax[4].set_xlabel('Difference in Dalitz Parameter')
ax[4].set_xlim(-0.2,0.2)
ax[1].legend(bbox_to_anchor=(1.1, 2.25))

plt.show()

