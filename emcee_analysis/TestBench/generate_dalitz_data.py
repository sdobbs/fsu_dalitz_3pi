import numpy as np
import math
import random
from ROOT import TH2F, TCanvas

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#********************************************************
def one_DP_function(dp_pars,dp_x,dp_y):
        parA = dp_pars[0]
        parB = dp_pars[1]
        parC = dp_pars[2]
        parD = dp_pars[3]
        parE = dp_pars[4]
        parF = dp_pars[5]
        parG = dp_pars[6]
        parH = dp_pars[7]
        parL = dp_pars[8]

        arg = (1.0 + parA*dp_y + parB*dp_y*dp_y + parC*dp_x + parD*dp_x*dp_x + parE*dp_x*dp_y + parF*dp_y*dp_y*dp_y + parG*dp_x*dp_x*dp_y + parH*dp_x*dp_y*dp_y + parL*dp_x*dp_x*dp_x)
        return arg
#********************************************************

#********************************************************
def get_eff_M(matrix_name,corr_eff,off_diag_f):
    eff_M = np.load(matrix_name)
    n_dims = eff_M.shape[0] 

    #+++++++++++++++++++++++++++
    for a in range(n_dims):
        #+++++++++++++++++++++++++++
        for b in range(n_dims):
            f = corr_eff

            if a!=b:
                f = corr_eff * off_diag_f

            eff_M[a,b] *= f
        #+++++++++++++++++++++++++++
    #+++++++++++++++++++++++++++
    


    return eff_M
#********************************************************

#********************************************************
def gen_DP(dp_pars,n_dp_bins,n_events,smear_matrix_name,core_scale,off_scale,DP_Range=[-1.1,1.1],add_gaussian_noise=0.0,add_flat_noise=0.0):
        efficiency_matrix = get_eff_M(smear_matrix_name,core_scale,off_scale)

        dp_x = np.linspace(DP_Range[0],DP_Range[1],n_dp_bins)
        dp_y = np.linspace(DP_Range[0],DP_Range[1],n_dp_bins)

        N_gen = np.zeros((n_dp_bins*n_dp_bins))
        DN_gen = np.zeros((n_dp_bins*n_dp_bins))

        N_gen_rec = np.zeros((n_dp_bins*n_dp_bins))
        DN_gen_rec = np.zeros((n_dp_bins*n_dp_bins))

        dp_x_out = np.zeros((n_dp_bins*n_dp_bins))
        dp_y_out = np.zeros((n_dp_bins*n_dp_bins))
        global_bin = np.zeros((n_dp_bins*n_dp_bins))
        
        #++++++++++++++++++++++++++++++++++++
        for ev in range(n_events):
           #++++++++++++++++++++++++++++++++++++
           for i in range(n_dp_bins):
              #++++++++++++++++++++++++++++++++++++
              for j in range(n_dp_bins): 

                gbin = i + n_dp_bins*j 
                if ev == 0:
                   dp_x_out[gbin] = dp_x[i]
                   dp_y_out[gbin]  = dp_y[j]
                   global_bin[gbin]  = gbin 
            
                N_gen[gbin] += one_DP_function(dp_pars,dp_x[i],dp_y[j])
              #++++++++++++++++++++++++++++++++++++ 
          #++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++

        N_gen_rec = np.matmul(efficiency_matrix,N_gen)
        N_gen = np.where(N_gen > 0.0,N_gen,0.0)
        N_gen_rec = np.where(N_gen_rec > 0.0,N_gen_rec,0.0)

        if add_gaussian_noise != 0.0:
            gaussian_noise = np.random.normal(1.0,add_gaussian_noise,n_dp_bins*n_dp_bins)
            
            N_gen_rec = np.multiply(N_gen_rec,gaussian_noise)

        if add_flat_noise != 0.0:
            flat_noise = np.random.uniform(1.0-add_flat_noise,1.0+add_flat_noise,n_dp_bins*n_dp_bins)
            N_gen_rec = np.multiply(N_gen_rec,flat_noise)

        out_vec = np.vstack(( 
              dp_x_out,
              dp_y_out,
              N_gen,
              DN_gen,
              N_gen_rec,
              DN_gen_rec,
              global_bin
        ))
        
        return np.transpose(out_vec), efficiency_matrix
#********************************************************



DP_Test_Pars = [-1.104,0.142,0.0,0.073,0.0,0.154,0.0,0.0,0.0]

eff_d = [0.2,0.0005]
eff_off_d = [-0.001,0.001]

dalitz_data, eff_matrix = gen_DP(DP_Test_Pars,11,100000,'raw_semar_matrix.npy',0.2,3.0,add_gaussian_noise=0.0,add_flat_noise=0.0)

fig,ax = plt.subplots(1,3)

ax[0].errorbar(x=dalitz_data[:,6],y=dalitz_data[:,2],yerr=dalitz_data[:,3],fmt='bs')
ax[1].errorbar(x=dalitz_data[:,6],y=dalitz_data[:,4],yerr=dalitz_data[:,5],fmt='ko')

ax[2].matshow(eff_matrix)

plt.show()

np.save('dalitz_data_30Corr.npy',dalitz_data)
np.save('eff_matrix_30Corr.npy',eff_matrix)






