import numpy as np
import math
import emcee
import random

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


do_basic_fit = False
data_name = 'GlueX-2018-01'
n_runs = 15000
n_walkers = 250
nSigma = 10


c_err = [0.035]*4

minDPVals = [10000,-1.2,0.05,-0.01,0.05,-0.01,0.1,-0.01,-0.01,-0.01]
maxDPVals = [30000,-1.0,0.2,0.01,0.1,0.01,0.2,0.01,0.01,0.01,0.01]

min_f_val = -2.0
max_f_val = 2.0
f_step = 0.05

add_name = '_no_acc_cut.npy'
idx_offset = 0

if do_basic_fit:
    add_name = '.npy'
else:
    idx_offset = 1

dp_data = np.load('/Users/daniellersch/Desktop/eta3Pi_DalitzAna/npy_dalitz_data/New_DP_Ana_May22_2018S_nbins11_kfit_cut3_imgg_cut1_zv_cut1' + add_name)


global_bin = dp_data[:,4]
N_eta = dp_data[:,2]
DN_eta = dp_data[:,3]

dp_acc = dp_data[:,7]

#********************************************************
def dalitz_objective_func(x,par_norm,par_a,par_b,par_c,par_d,par_e,par_f,par_g,par_h,par_l):
    DP_X = x[:,0]
    DP_Y = x[:,1]

    return par_norm*(1.0 + par_a*DP_Y + par_b*DP_Y*DP_Y + par_c*DP_X + par_d*DP_X*DP_X + par_e*DP_X*DP_Y + par_f*DP_Y*DP_Y*DP_Y + par_g*DP_X*DP_X*DP_Y + par_h*DP_X*DP_Y*DP_Y + par_l*DP_X*DP_X*DP_X) 


#-----------------------------------

def run_chi2_fit(DP_Data,start_values):
    dp_x_values = DP_Data[:,[0,1]]
    dp_y_values = DP_Data[:,2]
    dp_y_errors = DP_Data[:,3]

    acc_cut = dp_acc > 0.8
    parameter_bounds = ([-np.inf]*len(start_values),[np.inf]*len(start_values))

    dp_values, dp_errors = curve_fit(dalitz_objective_func,dp_x_values[acc_cut],dp_y_values[acc_cut],p0=start_values,sigma=dp_y_errors[acc_cut],absolute_sigma=True,bounds=parameter_bounds)
    
    return dp_values,dp_errors

#-----------------------------------

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

#----------------------------

def get_dp_acceptance(x):
    f = 1.0 / (1.0 + np.exp(-x))

    return np.where(dp_acc>=f,1.0,0.0)

#----------------------------

def basic_log_prior(theta,constr_err):
        arg_c = theta[3]/constr_err[0]
        arg_e = theta[5]/constr_err[1]
        arg_h = theta[8]/constr_err[2]
        arg_l = theta[9]/constr_err[3]

        
        norm = 0.0
        #++++++++++++++++++++++++
        for err in constr_err:
               norm += math.log(2.0*math.pi*err*err)
        #++++++++++++++++++++++++

        return -0.5*(arg_c*arg_c + arg_e*arg_e * arg_h*arg_h + arg_l*arg_l + norm)

#----------------------------

def advanced_log_prior(theta,constr_err):
    f_acc = theta[10]

    if f_acc > -3.0 and f_acc < 3.0:
        return basic_log_prior(theta,constr_err)

    return -np.inf

#----------------------------

def basic_log_likelihood(theta,x,y,yerr):
    y_fit = oneD_Dalitz_Function(theta,x)
    
    sigma2 = np.where(yerr>0.0,yerr**2,1.0)

    arg = (y - y_fit) ** 2 / sigma2 + np.log(sigma2)
    val = np.where(y>0.0,arg,0.0)

    return  -0.5 * np.sum(val)

#----------------------------

def advanced_log_likelihood(theta,x,y,yerr):
    y_fit = oneD_Dalitz_Function(theta,x)
    
    acc = get_dp_acceptance(theta[10])
    yerr = yerr*acc

    sigma2 = np.where(yerr>0.0,yerr**2,1.0)
    y_fit = y_fit*acc
    y = y*acc

    arg = (y - y_fit) ** 2 / sigma2 + np.log(sigma2)
    #val = np.where(y>0.0,arg,0.0)

    return  -0.5 * np.sum(arg)
             
#----------------------------

def basic_log_probability(theta, x, y, yerr,constr_err):
    lp = basic_log_prior(theta,constr_err)
    if not np.isfinite(lp):
        return -np.inf
    return lp + basic_log_likelihood(theta, x, y, yerr)

#----------------------------

def advanced_log_probability(theta, x, y, yerr,constr_err):
    lp = advanced_log_prior(theta,constr_err)
    if not np.isfinite(lp):
        return -np.inf
    return lp + advanced_log_likelihood(theta, x, y, yerr)

#----------------------------

def get_start_pars(min_vals,max_vals):
    start_par_list = []
    n_pars = len(min_vals)

    #+++++++++++++++++++++++++
    for i in range(n_pars):
        current_par = random.uniform(min_vals[i],max_vals[i])
        start_par_list.append(current_par)
    #+++++++++++++++++++++++++

    return start_par_list

#----------------------------

def get_emcee_start_pars(init_fit_values,init_fit_errors,n_sigma,nWalkers):
    left_b = init_fit_values - n_sigma * init_fit_errors
    right_b = init_fit_values + n_sigma * init_fit_errors

    start_pars = np.random.uniform(left_b,right_b,size=(nWalkers,left_b.shape[0]))

    #----------------------------
    if do_basic_fit is False:
        out_pars = np.concatenate([
            start_pars,
            np.random.uniform(min_f_val,max_f_val,(nWalkers,1))
        ],axis=1)

        return out_pars
    #----------------------------

    return start_pars


#Calculate difference between values:
def calc_diff(a,b):
        return a-b

#---------------------------

#Calculate the corresponding error:
def calc_diff_error(err_a,err_b):
        return math.sqrt(err_a*err_a + err_b*err_b)

#---------------------------

#Get a chi2 between the GlueX results and other experiments:
def calc_diff_chi2(diff_values,diff_err_values):
        chisq = 0.0
         
        #+++++++++++++++++++++++++ 
        for d,d_err in zip(diff_values,diff_err_values):
              if d_err == 0.0:
                  chisq += d*d
              else:
                  chisq += d*d / (d_err*d_err)
        #+++++++++++++++++++++++++ 

        return chisq

#---------------------------
   
#Now run the comparison:
def calculate_differences(values,errors,ref_values,ref_errors):
    diffs = [calc_diff(x,y) for x,y in zip(values,ref_values)]
    diff_errors = [calc_diff_error(err_x,err_y) for err_x,err_y in zip(errors,ref_errors)]

    d_chisq = calc_diff_chi2(diffs,diff_errors)

    return diffs, diff_errors, d_chisq
#********************************************************

chi2_start_pars = get_start_pars(minDPVals,maxDPVals)

dp_pars_chi2_fit, dp_errs_chi2_fit = run_chi2_fit(dp_data,chi2_start_pars)

N_chi2_fit = dalitz_objective_func(dp_data[:,[0,1]],*dp_pars_chi2_fit)

fig,ax = plt.subplots()

ax.errorbar(x=global_bin,y=N_eta,yerr=DN_eta,fmt='ko')
ax.plot(global_bin,N_chi2_fit,'r-')

plt.show()

cov_m = np.diagonal(dp_errs_chi2_fit)

pos = get_emcee_start_pars(dp_pars_chi2_fit,cov_m,nSigma,n_walkers)
nPars = pos.shape[1]


if do_basic_fit is False:
    cov_m = np.append(np.diagonal(dp_errs_chi2_fit),f_step)


par_idx = [i for i in range(nPars)]
fit_par_idx = [i for i in range(nPars-1-idx_offset)]
nwalkers, ndim = pos.shape

sampler = None
mover = [
    (emcee.moves.DEMove(), 0.2),
    (emcee.moves.GaussianMove(cov=cov_m, mode='random'), 0.8)
]

#---------------------------------
if do_basic_fit:
    sampler = emcee.EnsembleSampler(
       nwalkers, ndim, basic_log_probability, args=(dp_data[:,[0,1]],N_eta,DN_eta,c_err),moves=mover
    )

else:
    sampler = emcee.EnsembleSampler(
       nwalkers, ndim, advanced_log_probability, args=(dp_data[:,[0,1]],N_eta,DN_eta,c_err),moves=mover
    )
#---------------------------------

sampler.run_mcmc(pos,n_runs, progress=True)

tau = sampler.get_autocorr_time()

min_tau = np.min(tau)
max_tau = np.max(tau)

n_discard = int(2.0*max_tau)
n_thin = int(0.5*min_tau)

flat_samples = sampler.get_chain(discard=n_discard, thin=n_thin, flat=True)

f_val = 0.0
f_val_err = 0.0

mean_values = np.mean(flat_samples,0)
err_values = np.std(flat_samples,0)

fit_pars = mean_values
fit_pars_errs = err_values

if do_basic_fit is False:
    f_val = mean_values[10]
    f_val_err = err_values[10]

    # res_acc = (flat_samples[:,10] >= f_val - f_val_err) & (flat_samples[:,10] <= f_val + f_val_err)

    # fit_pars = np.mean(flat_samples[res_acc],0)
    # fit_pars_errs = np.std(flat_samples[res_acc],0)





N_fit = oneD_Dalitz_Function(fit_pars,dp_data[:,[0,1]])
fit_pars[1] = -fit_pars[1]

kloe_pars = [1.104,0.142,0.0,0.073,0.0,0.154,0.0,0.0,0.0]
kloe_errs = [0.005,0.011,0.0,0.007,0.0,0.011,0.0,0.0,0.0]

wasa_pars = [1.144,0.219,0.0,0.086,0.0,0.115,0.0,0.0,0.0]
wasa_errs = [0.018,0.066,0.0,0.033,0.0,0.037,0.0,0.0,0.0]

gluex_pars = fit_pars[1:nPars-idx_offset]
gluex_errs = fit_pars_errs[1:nPars-idx_offset]

kloe_d, kloe_d_err, kloe_d_chisq = calculate_differences(gluex_pars,gluex_errs,kloe_pars,kloe_errs)
wasa_d, wasa_d_err, wasa_d_chisq = calculate_differences(gluex_pars,gluex_errs,wasa_pars,wasa_errs)


if do_basic_fit is False:
    figt,axt = plt.subplots()

    axt.hist(flat_samples[:,10],100)

    plt.show()

    fit_acc = get_dp_acceptance(f_val)

    print("f-val: " + str(f_val) + " +- " + str(f_val_err))

    N_fit = N_fit * fit_acc 
    N_eta = N_eta * fit_acc 
    DN_eta = DN_eta * fit_acc

figf,axf = plt.subplots(1,3)

axf[0].errorbar(x=global_bin,y=N_eta,yerr=DN_eta,fmt='ko')
axf[0].plot(global_bin,N_fit,'r-',linewidth=2.0)

axf[1].errorbar(x=fit_par_idx,y=kloe_pars,yerr=kloe_errs,fmt='bd',label='KLOE',elinewidth=2.0,markersize=10.0)
axf[1].errorbar(x=fit_par_idx,y=wasa_pars,yerr=wasa_errs,fmt='rs',label='WASA',elinewidth=2.0,markersize=10.0)

axf[1].errorbar(x=fit_par_idx,y=gluex_pars,yerr=gluex_errs,fmt='ko',label=data_name,elinewidth=2.0,markersize=10.0)
axf[1].legend()

axf[2].errorbar(x=fit_par_idx,y=kloe_d,yerr=kloe_d_err,fmt='bd',label='KLOE',elinewidth=2.0,markersize=10.0)
axf[2].errorbar(x=fit_par_idx,y=wasa_d,yerr=wasa_d_err,fmt='rs',label='WASA',elinewidth=2.0,markersize=10.0)
axf[2].plot(fit_par_idx,[0.0]*len(fit_par_idx),'k--',linewidth=2.0)
axf[2].legend()

plt.show()

print(" ")
print("KLOE-chi^2: " + str(kloe_d_chisq))
print("WASA-chi^2: " + str(wasa_d_chisq))


