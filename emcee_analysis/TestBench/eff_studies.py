import numpy as np
import math
import emcee
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

raw_dalitz_data = np.load('dalitz_data_v10.npy')
dp_acc = np.load('DP_Acc.npy')
eff_M = np.load('raw_semar_matrix.npy')

def rebin_data(dp_data,acc):
    active_DP_data = []
    active_gbin = 0

    counter = 0
    n_dp_bins = dp_data.shape[0]

    new_eff_M = []
    #+++++++++++++++++++++++++
    for el in dp_data:
        current_acc = acc[counter]
        if el[2]*current_acc > 0.0:
            out_list = [el[0],el[1],el[2]*current_acc,el[3]*current_acc,el[4]*current_acc,el[5]*current_acc,active_gbin]

            active_DP_data.append(out_list)
            active_gbin += 1

            m_list = []
            #+++++++++++++++++++++
            for i in range(n_dp_bins):
               if acc[i] * dp_data[i][2] > 0.0:
                 m_list.append(eff_M[counter,i])
            #+++++++++++++++++++++

            new_eff_M.append(m_list)

        counter += 1
    #+++++++++++++++++++++++++

    return np.array(active_DP_data), np.array(new_eff_M)


dalitz_data, new_S = rebin_data(raw_dalitz_data,dp_acc)

global_bin = dalitz_data[:,6]

N_true = dalitz_data[:,2]
DN_true = dalitz_data[:,3]

N_rec = dalitz_data[:,4]
DN_rec = dalitz_data[:,5]

eff = N_rec / (N_true + 1e-5)

def approx_eff(X,x_scale,f_glob,p):
    dX = (np.reshape(X,(X.shape[0],1)) - X) / x_scale
    
    k = (1.0 + 0.5*dX**2)**p

    return f_glob / k


def calc_eff_corr(x,a,b):
    arg = x / np.max(x)

    corr = 1.0 + a*arg + b*arg*arg
    eff_corr = eff * corr

    return eff_corr

def func(x,a,b):
    eff_corr = calc_eff_corr(x,a,b)

    return N_rec / (eff_corr) 

p0 = [0.0,0.0]
popt, pcov = curve_fit(func,global_bin,N_true)

print(popt)

# test_M = approx_eff(global_bin,popt[0],popt[1])

# print(test_M)

N_fit = func(global_bin,*popt)
eff_fit = calc_eff_corr(global_bin,*popt)




# length_scale = 1.0
# f_scale = 0.1
# p = 10.0*np.pi
# folding = approx_eff(global_bin,length_scale,f_scale)
# N_app = np.sum(N_true * folding,1)

# eff_app = N_app / (N_true + 1e-5)



# new_S *= 0.2

# N_app = np.matmul(new_S,N_true)
# eff_app = N_app / (N_true + 1e-5)

fig,ax = plt.subplots(1,2)

ax[0].plot(global_bin,N_true,'ko')
ax[0].plot(global_bin,N_fit,'rs')

ax[1].plot(global_bin,eff,'ko')
ax[1].plot(global_bin,eff_fit,'rs')

plt.show()

# M = approx_eff(global_bin,1.0,0.3,7.1)

# plt.matshow(M)
# plt.show()