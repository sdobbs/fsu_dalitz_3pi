from shutil import move
import numpy as np
import math
import emcee

from matplotlib import pyplot as plt

raw_dalitz_data = np.load('dalitz_data_15Corr.npy')
dp_acc = np.load('DP_Acc.npy')


def get_efficiency_error(n_rec,n_gen,neg_variance_val):
    nom_1 = (n_rec + 1.0)*(n_rec + 2.0)
    nom_2 = (n_rec + 1.0)*(n_rec + 1.0)

    denom_1 = (n_gen + 2.0)*(n_gen + 3.0)
    denom_2 = (n_gen + 2.0)*(n_gen + 2.0)

    variance = nom_1/denom_1 - nom_2/denom_2

    return np.where(variance>=0.0,variance,neg_variance_val)

def rebin_data(dp_data,acc):
    active_DP_data = []
    active_gbin = 0

    counter = 0
    #+++++++++++++++++++++++++
    for el in dp_data:
        current_acc = acc[counter]
        if el[2]*current_acc > 0.0:
            out_list = [el[0],el[1],el[2]*current_acc,el[3]*current_acc,el[4]*current_acc,el[5]*current_acc,active_gbin]

            active_DP_data.append(out_list)
            active_gbin += 1

        counter += 1
    #+++++++++++++++++++++++++

    return np.array(active_DP_data)


dalitz_data = rebin_data(raw_dalitz_data,dp_acc)



global_bin = dalitz_data[:,6]

N_true = dalitz_data[:,2]
DN_true = dalitz_data[:,3]

N_rec = dalitz_data[:,4]
DN_rec = dalitz_data[:,5]

eff = N_rec / (N_true + 1e-5)
D_eff = get_efficiency_error(N_rec,N_true,1e-8)

N_eta = N_rec / eff

DN_eta = np.sqrt(DN_rec*DN_rec + (N_eta * D_eff)**2) / eff


        


fig,ax = plt.subplots(1,3)

ax[0].errorbar(x=global_bin,y=N_true,yerr=DN_true,fmt='rd')
ax[1].errorbar(x=global_bin,y=N_rec,yerr=DN_rec,fmt='ko')

ax[2].errorbar(x=global_bin,y=eff,yerr=D_eff,fmt='bs')
ax[2].set_ylim(0.0,1.0)

plt.show()


#*********************************************
def oneD_Dalitz_Function(theta,x):
        norm = theta[0]
        parA = theta[1]
        parB = theta[2]
        parC = theta[3]
        parD = theta[4]
        parE = theta[5]
        parF = theta[6]
        parG = theta[7]
        parH = theta[8]
        parL = theta[9]

        DP_X = x[:,0]
        DP_Y = x[:,1]

        return norm*(1.0 + parA*DP_Y + parB*DP_Y*DP_Y + parC*DP_X + parD*DP_X*DP_X + parE*DP_X*DP_Y + parF*DP_Y*DP_Y*DP_Y + parG*DP_X*DP_X*DP_Y + parH*DP_X*DP_Y*DP_Y + parL*DP_X*DP_X*DP_X)

def log_prior(theta,constr_err):
        arg_c = theta[3]/constr_err[0]
        arg_e = theta[5]/constr_err[1]
        arg_h = theta[8]/constr_err[2]
        arg_l = theta[9]/constr_err[3]

        norm = 0.0
        #++++++++++++++++++++++++
        for err in constr_err:
            norm += math.log(2.0*math.pi*err*err)
        #++++++++++++++++++++++++

        log_P = -0.5*(arg_c*arg_c + arg_e*arg_e * arg_h*arg_h + arg_l*arg_l + norm)
        return log_P

def log_likelihood(theta,x,y,yerr):
    y_fit = oneD_Dalitz_Function(theta,x)

    sigma2 = yerr**2
    return -0.5 * np.sum((y - y_fit) ** 2 / sigma2 + np.log(sigma2))

def log_probability(theta, x, y, yerr,constr_err):
    lp = log_prior(theta,constr_err)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)
#*********************************************

def get_start_pars(min_vals,max_vals,nWalkers):
    start_par_list = []
    n_pars = len(min_vals)

    #+++++++++++++++++++++++++
    for i in range(n_pars):
        current_par = np.random.uniform(min_vals[i],max_vals[i],size=(nWalkers,1))
        start_par_list.append(current_par)
    #+++++++++++++++++++++++++

    return np.concatenate(start_par_list,1)

c_err = [0.035]*4

minDPVals = [100000,-1.2,0.05,-0.01,0.05,-0.01,0.1,-0.01,-0.01,-0.01]
maxDPVals = [200000,-1.0,0.2,0.01,0.1,0.01,0.2,0.01,0.01,0.01]

pos = get_start_pars(minDPVals,maxDPVals,250)
nPars = pos.shape[1]


par_idx = [i for i in range(nPars)]
fit_par_idx = [i for i in range(nPars-1)]


nwalkers, ndim = pos.shape

#mover = emcee.moves.GaussianMove(mode='random',cov=[1000,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])

mover = [
        (emcee.moves.DEMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ]

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(dalitz_data[:,[0,1]],N_eta,DN_eta,c_err)
)
sampler.run_mcmc(pos,10000, progress=True)

tau = sampler.get_autocorr_time()

min_tau = np.min(tau)
max_tau = np.max(tau)

n_discard = int(2.0*max_tau)
n_thin = int(0.5*min_tau)

flat_samples = sampler.get_chain(discard=n_discard, thin=n_thin, flat=True)

mean_values = np.mean(flat_samples,0)
err_values = np.std(flat_samples,0)

fit_pars = mean_values
fit_pars_errs = err_values

N_fit = oneD_Dalitz_Function(fit_pars,dalitz_data[:,[0,1]])

true_pars = [1.104,0.142,0.0,0.073,0.0,0.154,0.0,0.0,0.0]
fit_pars[1] = -fit_pars[1]

active_bin = 0
n_eta_list = []
dn_eta_list = []
n_fit_list = []
bin_list = []

#+++++++++++++++++++++
for i in range(N_eta.shape[0]):
    if N_eta[i] > 0.0:
        n_eta_list.append(N_eta[i])
        dn_eta_list.append(DN_eta[i])
        n_fit_list.append(N_fit[i])
        bin_list.append(active_bin)

        active_bin += 1 
#+++++++++++++++++++++

reb_N_eta = np.array(n_eta_list)
reb_DN_eta = np.array(dn_eta_list)
reb_N_fit = np.array(n_fit_list)
reb_global_bin = np.array(bin_list)

plt.rcParams.update({'font.size': 20})

figf,axf = plt.subplots(1,2)


axf[0].errorbar(x=reb_global_bin,y=reb_N_eta,yerr=reb_DN_eta,fmt='ko',markersize=10.0,label='Toy Data')
axf[0].plot(reb_global_bin,reb_N_fit,'r-',linewidth=2.0,label='Fit')
axf[0].legend()
axf[0].grid(True)
axf[0].set_xlabel('Global Bin')
axf[0].set_ylabel(r'$\eta\rightarrow\pi^+\pi^-\pi^0$' + ' Yields')

axf[1].plot(fit_par_idx,true_pars,'bd',linewidth=2.0,label='True',markersize=10.0)
axf[1].errorbar(x=fit_par_idx,y=fit_pars[1:nPars],yerr=fit_pars_errs[1:nPars],fmt='rs',label='Fit',markersize=10.0)
axf[1].set_xticks(fit_par_idx)
axf[1].set_xticklabels(['a','b','c','d','e','f','g','h','l'])
axf[1].set_ylabel('Parameter Value')
axf[1].grid(True)
axf[1].legend()

plt.show()










    
        

