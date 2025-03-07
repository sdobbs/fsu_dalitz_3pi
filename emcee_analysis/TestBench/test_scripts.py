#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import random

def one_DP_function(dp_pars,dp_errs,dp_x,dp_y):
        parA = random.gauss(dp_pars[0],dp_errs[0])
        parB = random.gauss(dp_pars[1],dp_errs[1])
        parC = random.gauss(dp_pars[2],dp_errs[2])
        parD = random.gauss(dp_pars[3],dp_errs[3])
        parE = random.gauss(dp_pars[4],dp_errs[4])
        parF = random.gauss(dp_pars[5],dp_errs[5])
        parG = random.gauss(dp_pars[6],dp_errs[6])
        parH = random.gauss(dp_pars[7],dp_errs[7])
        parL = random.gauss(dp_pars[8],dp_errs[8])

        arg = (1.0 + parA*dp_y + parB*dp_y*dp_y + parC*dp_x + parD*dp_x*dp_x + parE*dp_x*dp_y + parF*dp_y*dp_y*dp_y + parG*dp_x*dp_x*dp_y + parH*dp_x*dp_y*dp_y + parL*dp_x*dp_x*dp_x)
        return arg

par_vals = [-1.035,0.169,0.003,0.116,0.012,0.076,-0.109,-0.021,-0.014]
par_errs = [0.013,0.013,0.011,0.016,0.016,0.026,0.043,0.033,0.02]

KLOE_values = [-1.104,0.142,0.0,0.073,0.0,0.154,0.0,0.0,0.0]
KLOE_errors = [0.005,0.011,0.0,0.007,0.0,0.011,0.0,0.0,0.0]

def gen_DP(dp_pars,dp_errs,n_dp_bins,n_events,DP_Range=[-1.1,1.1],add_gaussian_noise=0.0,add_flat_noise=0.0):
    dp_bins = np.linspace(DP_Range[0],DP_Range[1],n_dp_bins)
    X,Y = np.meshgrid(dp_bins,dp_bins)
    XY = np.array([X.flatten(),Y.flatten()]).T

    DP_X = XY[:,0]
    DP_Y = XY[:,1]

    global_bin = np.arange(0,n_dp_bins*n_dp_bins,1)

    N_DP_Gen = one_DP_function(dp_pars,dp_errs,DP_X,DP_Y)
    #+++++++++++++++++++++++++++
    for _ in range(n_events-1):
        N_DP_Gen += one_DP_function(dp_pars,dp_errs,DP_X,DP_Y)
    #+++++++++++++++++++++++++++

    DN_DP_Gen = np.sqrt(N_DP_Gen)

    if add_gaussian_noise != 0.0:
        gaussian_noise = np.random.normal(1.0,add_gaussian_noise,n_dp_bins*n_dp_bins)
            
        N_DP_Gen = np.multiply(N_DP_Gen,gaussian_noise)

    if add_flat_noise != 0.0:
        flat_noise = np.random.uniform(1.0-add_flat_noise,1.0+add_flat_noise,n_dp_bins*n_dp_bins)
        N_DP_Gen = np.multiply(N_DP_Gen,flat_noise)

    result_vec = np.vstack(( 
        DP_X,
        DP_Y,
        N_DP_Gen,
        DN_DP_Gen,
        global_bin
    ))
    
    return np.transpose(result_vec)


gen_data = gen_DP(KLOE_values,KLOE_errors,11,1000000)

N = gen_data[:,2]
DN = gen_data[:,3]
gbin = gen_data[:,4]

plt.errorbar(gbin,N,DN,fmt='ko')
plt.show()