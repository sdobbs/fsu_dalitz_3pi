import math
import random
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import corner
from matplotlib.ticker import FormatStrFormatter

class Fit_Result_Evaluator(object):
     
    def __init__(self):
          self.data = []

    #Get the reader, i.e. the sampler:
    #*****************************************
    def get_reader(self,filename):
          reader = emcee.backends.HDFBackend(filename + '.h5')
          return reader
    #*****************************************

     #Get the chains and translate them to a DataFrame:
    #*****************************************
    def get_chains(self,yourReader,yourParNames,burn_in_fac=2.0,thin_fac=0.5,show_info=True):
          autocorr_time = yourReader.get_autocorr_time()
          max_autocorr_time = np.max(autocorr_time)
          min_autocorr_time = np.min(autocorr_time)

          burnin = int(burn_in_fac * max_autocorr_time)
          thin = int(thin_fac * min_autocorr_time)

          if show_info:

              print("  ")
              print("------------------------------------------------")
              print("Max. autocorrelation time: " + str(max_autocorr_time))
              print("Min. autocorrelation time: " + str(min_autocorr_time))
              print("Burn-in factor: " + str(burn_in_fac) + " (default: 2.0)")
              print("Thinning factor: " + str(thin_fac) + " (default: 0.5)")
              print("Burn-in length: " + str(burnin))
              print("Thinning: " + str(thin))
              print("------------------------------------------------")
              print("  ")

          raw_sample = None
          #------------------------
          if burn_in_fac == 0.0 or thin_fac == 0.0:
             raw_samples = yourReader.get_chain()
          else:
             raw_samples = yourReader.get_chain(discard=burnin, thin=thin)
          #------------------------

          n_entries = raw_samples.shape[0]
          n_walkers = raw_samples.shape[1]
          n_parameters = raw_samples.shape[2]

          walker_counter = [[w] for w in range(1,n_walkers+1)]
          walkers_in_sample = [walker_counter]*n_entries
          walker_id = np.array(walkers_in_sample)
          flat_walker_id = walker_id.reshape(-1,walker_id.shape[-1])

          ana_samples = yourReader.get_chain(discard=burnin, flat=True, thin=thin)
          log_prob_samples = yourReader.get_log_prob(discard=burnin, flat=True, thin=thin)

          labels_to_show = copy.copy(yourParNames)

          labels_to_show.append("log(P)")
          labels_to_show.append("walker_id")

          all_samples = np.concatenate(
             (ana_samples, log_prob_samples[:, None],flat_walker_id), axis=1
          )

          df = pd.DataFrame(all_samples,columns=labels_to_show)
 
          return df
    #*****************************************

    #Simply show parameters as function
    #of the walk step and show burnin:
    #*****************************************
    def show_chains(self,yourReader,yourParNames,figSaveName=None,fontSize=20,x_axis_range=None,show_burnin=False,burn_in_fac=2.0):
          samples = yourReader.get_chain()
          n_plots = len(yourParNames)

          plt.rcParams.update({'font.size': fontSize})
          fig, ax = plt.subplots(n_plots+1,figsize=(15, 10), sharex=True)
          
          x_axis_min = 0.0
          x_axis_max = len(samples)

          if x_axis_range is not None:
              x_axis_min = x_axis_range[0]
              x_axis_max = x_axis_range[1]

          burnin = 0.0
          if show_burnin:
              autocorr_time = yourReader.get_autocorr_time()
              max_autocorr_time = np.max(autocorr_time)
              burnin = int(burn_in_fac * max_autocorr_time)

          #++++++++++++++++++++++++++++++
          for i in range(n_plots):
            ax[i].plot(samples[:, :, i], "k", alpha=0.3)
            ax[i].set_xlim(x_axis_min, x_axis_max)
            ax[i].set_ylabel(yourParNames[i])
            ax[i].yaxis.set_label_coords(-0.1, 0.5)

            if show_burnin:
                max_par = np.max(samples[:, :, i])
                min_par = np.min(samples[:, :, i])
 
                ax[i].plot([burnin,burnin],[min_par,max_par],'r-',linewidth=3.0)

          #++++++++++++++++++++++++++++++
          
          log_prob = yourReader.get_log_prob()
          ax[n_plots].plot(log_prob,"k",alpha=0.3)
          ax[n_plots].set_xlim(x_axis_min, x_axis_max)
          ax[n_plots].set_ylabel('log(P)')

          if show_burnin:
                max_par = np.max(log_prob)
                min_par = np.min(log_prob)
 
                ax[n_plots].plot([burnin,burnin],[min_par,max_par],'r-',linewidth=3.0)

          ax[-1].set_xlabel("Number of Steps")
         
          #------------------------------------
          if figSaveName is not None:
              fig.savefig(figSaveName + '.png')
          else:
              plt.show()
          #------------------------------------
          plt.close(fig)
    #*****************************************

    #Show the corner plot:
    #*****************************************
    def show_corner_plots(self,yourChains,yourParNames,title_size=12,label_size=20,figSaveName=None):
          if title_size != 0.0 and label_size != 0.0:
             fig = corner.corner(yourChains,labels=yourParNames,show_titles=True, title_kwargs={"fontsize": 12},label_kwargs={"fontsize": 20})
             plt.subplots_adjust(bottom=0.10,top=0.95,left=0.1)
          
             #------------------------------------
             if figSaveName is not None:
                fig.savefig(figSaveName + '.png')
             else:
                plt.show()
             #------------------------------------
             plt.close(fig)
    #*****************************************

    #Get and show the DP parameters:
    #*****************************************
    def get_and_show_DP_pars(self,yourChain,yourParNames,dataName,corr_method='spearman',mode='default',logP_Range=None,percentiles=[16,50,84],draw_parameters=False,fontSize=20,inchSize=(20,10),labelSize=15,figSaveName=None):
          #Simply use the mean and cov-matrix to estimate
          #the stat. uncertainites:
          par_values = None
          par_cov = None
          par_errs = None
          max_logP = yourChain['log(P)'].max() 
          mean_logP = yourChain['log(P)'].mean() 

          #-------------------------------------------------------------------
          if mode == 'default':
             par_values = yourChain[yourParNames].mean().values
             par_cov = yourChain[yourParNames].cov().values
             par_errs = np.sqrt(par_cov.diagonal())
          elif mode == 'max':
             par_values = yourChain[yourChain['log(P)']==max_logP][yourParNames].values[0]
             par_cov = yourChain[yourParNames].cov().values
             par_errs = np.sqrt(par_cov.diagonal())
          elif mode == 'range':
             par_values = yourChain[yourChain['log(P)']== max_logP][yourParNames].values[0]
             par_cov = yourChain[yourChain['log(P)']<= max_logP-0.5][yourParNames].cov().values
             par_errs = np.sqrt(par_cov.diagonal())
          elif mode == 'mean':
             cond = (yourChain['log(P)']<=mean_logP + 1) & (yourChain['log(P)']>=mean_logP - 1)
             par_values = yourChain[cond][yourParNames].values[0]
             par_cov = yourChain[yourParNames].cov().values
             par_errs = np.sqrt(par_cov.diagonal())
          elif mode == 'cut' and logP_Range is not None:
             cond = (yourChain['log(P)']<= logP_Range[0]) & (yourChain['log(P)']>= logP_Range[1])
             par_values = yourChain[cond][yourParNames].values[0]
             par_cov = yourChain[yourParNames].cov().values
             par_errs = np.sqrt(par_cov.diagonal())
          #-------------------------------------------------------------------

          par_corrs = yourChain[yourParNames].corr(method=corr_method)

          #Use the percentile:
          n_parameters = len(yourParNames)
          pars = yourChain[yourParNames].values
          perc_val = np.empty(n_parameters)
          perc_minErr = np.empty(n_parameters)
          perc_maxErr = np.empty(n_parameters)

          #+++++++++++++++++++++++++++++++++++++++
          for p in range(0,n_parameters):
              current_vals = np.percentile(pars[:,p],percentiles)
              current_diffs = np.diff(current_vals)

              perc_val[p] = current_vals[1]
              perc_minErr[p] = current_diffs[0]
              perc_maxErr[p] = current_diffs[1]
          #+++++++++++++++++++++++++++++++++++++++

          #Draw the parameters, if needed:
          if draw_parameters:
             fig,pax = plt.subplots(3,4,sharey=True)
             fig.suptitle('Dalitz Plot Parameters for the ' + dataName + ' Data')
             fig.set_size_inches(inchSize)
             plt.rcParams.update({'font.size': fontSize})
             plt.subplots_adjust(hspace=0.5,wspace=0.3)
             
             yourParNames[0] = 'N'
             par_counter = 0
             fmt = None
             #++++++++++++++++++++++++++++++++
             for a in range(3):
                 #++++++++++++++++++++++++++++++++
                 for b in range(4):
                     if a < 2 or (a==2 and b<2):
                        pax[a][b].hist(pars[:,par_counter],bins=100)
                        pax[a][b].grid(True)
                        pax[a][b].set_xlabel('Parameter ' + yourParNames[par_counter],fontsize=fontSize)
                        pax[a][b].tick_params(axis='x', which='major', labelsize=labelSize)
                        pax[a][b].xaxis.set_major_formatter(FormatStrFormatter('%g'))
                     elif a==2 and b==2:
                        pax[a][b].hist(yourChain['log(P)'],bins=100)
                        pax[a][b].grid(True)
                        pax[a][b].set_xlabel('log(P)',fontsize=fontSize)
                        pax[a][b].tick_params(axis='x', which='major', labelsize=labelSize)
                        pax[a][b].xaxis.set_major_formatter(FormatStrFormatter('%g'))
                     par_counter += 1
                 #++++++++++++++++++++++++++++++++
             #++++++++++++++++++++++++++++++++
             pax[0][0].set_ylabel('Entries [a.u.]',fontsize=fontSize)
             pax[1][0].set_ylabel('Entries [a.u.]',fontsize=fontSize)
             pax[2][0].set_ylabel('Entries [a.u.]',fontsize=fontSize)

             if len(yourParNames) == 11:
                pax[2][3].hist(pars[:,10],bins=100)
                pax[2][3].grid(True)
                pax[2][3].set_xlabel('Raw Acceptance Cut',fontsize=fontSize)
                pax[2][3].tick_params(axis='x', which='major', labelsize=labelSize)
                pax[2][3].xaxis.set_major_formatter(FormatStrFormatter('%g'))

             #------------------------------------
             if figSaveName is not None:
                fig.savefig(figSaveName + '.png')
             else:
                plt.show()
             #------------------------------------
             plt.close(fig)

          return [par_values,par_errs,par_corrs,perc_val,perc_minErr,perc_maxErr,max_logP,mean_logP]
    #*****************************************

    #Monitor the fit quality:
    #*****************************************
    def rebin_data(self,Y_meas,dY,Y_fit,eff,d_eff):
        active_bin = 0
        nPoints = Y_meas.shape[0]

        new_Y_meas = []
        new_dY_meas = []
        new_Y_fit = []
        new_eff = []
        new_d_eff = []
        new_gBin = []

        #++++++++++++++++++++++++++
        for p in range(nPoints):
            if Y_meas[p] > 0.0:
                
                new_Y_meas.append(Y_meas[p])
                new_dY_meas.append(dY[p])
                new_Y_fit.append(Y_fit[p])
                new_gBin.append(active_bin)

                if eff is not None and d_eff is not None:
                    new_eff.append(eff[p])
                    new_d_eff.append(d_eff[p]) 

                active_bin += 1
        #++++++++++++++++++++++++++

        if len(new_d_eff) > 0:
            return np.array(new_Y_fit), np.array(new_Y_meas), np.array(new_dY_meas), np.array(new_gBin), np.array(new_eff), np.array(new_d_eff)
        
        return np.array(new_Y_fit), np.array(new_Y_meas), np.array(new_dY_meas), np.array(new_gBin), None, None
        

    #-----------------------------

    #Calulate fit chi2:
    def get_fit_chiSquare(self,parameters,DP_Data):
        norm = parameters[0]
        parA = parameters[1]
        parB = parameters[2]
        parC = parameters[3]
        parD = parameters[4]
        parE = parameters[5]
        parF = parameters[6]
        parG = parameters[7]
        parH = parameters[8]
        parL = parameters[9]

        acceptance = 1.0
        if parameters.shape[0] > 10:
            acc_cut = 1.0 / (1.0 + np.exp(-parameters[10]))
            acceptance = np.where(DP_Data[:,5]>acc_cut,1.0,0.0)

        DP_X = DP_Data[:,0] * acceptance
        DP_Y = DP_Data[:,1] * acceptance
        N_meas = DP_Data[:,2] * acceptance
        DN_meas = DP_Data[:,3] * acceptance

        eff = None
        d_eff = None

        if DP_Data.shape[1] > 6:
            eff = DP_Data[:,6] *acceptance
            d_eff = DP_Data[:,7] *acceptance
        
        N_fit = norm*(1.0 + parA*DP_Y + parB*DP_Y*DP_Y + parC*DP_X + parD*DP_X*DP_X + parE*DP_X*DP_Y + parF*DP_Y*DP_Y*DP_Y + parG*DP_X*DP_X*DP_Y + parH*DP_X*DP_Y*DP_Y + parL*DP_X*DP_X*DP_X)
        sigma = np.where(DN_meas > 0.0,DN_meas**2,1.0)

        N_fit = N_fit * acceptance
        
        if parameters.shape[0] > 10:
            reb_N_fit, reb_N_meas, reb_dN_meas, reb_gBin, reb_eff, reb_d_eff = self.rebin_data(N_meas,DN_meas,N_fit,eff,d_eff)

            return [np.sum((N_meas - N_fit) ** 2 / sigma),reb_N_fit, reb_N_meas, reb_dN_meas, reb_gBin, reb_eff, reb_d_eff]


        return [np.sum((N_meas - N_fit) ** 2 / sigma), N_fit,N_meas,DN_meas,DP_Data[:,4],eff,d_eff]

    #-----------------------------

    def check_c_symmetry(self,DP_Pars,DP_Errs,dataName,figSaveName):
          c_pars = [
              DP_Pars[3],
              DP_Pars[5],
              DP_Pars[8],
              DP_Pars[9]
          ]

          c_errs = [
              DP_Errs[3],
              DP_Errs[5],
              DP_Errs[8],
              DP_Errs[9]
          ]

          x_values = [0,1,2,3]
          x_labels = ['c','e','h','l']
    
          fig,sx = plt.subplots(figsize=(12,8))
          plt.subplots_adjust(bottom=0.2,top=0.87,left=0.17)

          #fig,sx = plt.subplots()
          plt.rcParams.update({'font.size': 30})
          sx.errorbar(x_values,c_pars,c_errs,fmt='ko',label=dataName,markersize=12,linewidth=3.0)
          sx.plot([0,3],[0.0,0.0],'r--',linewidth=3.0,label='C-Conservation')
          sx.set_xticks(x_values)
          sx.set_xticklabels(tuple(x_labels))
          sx.set_ylim(-0.1,0.1)
          sx.set_ylabel('Parameter Values',fontsize=30)
          sx.legend()
          sx.grid(True)

          #----------------------------- 
          if figSaveName is not None:
              fig.savefig(figSaveName)
          else:
              plt.show()
          #----------------------------- 

          plt.close(fig)

    #-----------------------------

    def show_DP_fit_results(self,DP_Results,DP_Data,parNames,dataName,x_step_size=11.0,plot_label='default',figSaveName=None,fontSize=20,labelFontSize=25,show_eff=False,saveResults=None):
          DP_values = DP_Results[0]
          DP_errors = DP_Results[1]
          DP_corr = DP_Results[2]
          DP_perc_values = DP_Results[3]
          DP_perc_minErr = DP_Results[4]
          DP_perc_maxErr = DP_Results[5]
          n_dp_parameters = len(parNames)

          if saveResults is not None:
              np.save(saveResults + '_DP_values.npy',DP_values)
              np.save(saveResults + '_DP_errors.npy',DP_errors)

          chiSquare = None
          N_fit = None
          eff_fit = None

          chiSquare, N_fit, N_meas, DN_meas, global_bin, eff, d_eff = self.get_fit_chiSquare(DP_values,DP_Data)

          ndf = N_fit[N_fit > 0.0].shape[0] - n_dp_parameters
          chiSquare_per_NDF = round(chiSquare / float(ndf),2)

          plt.rcParams.update({'font.size': fontSize})
          fig, ax = plt.subplots(figsize=(12, 8))
          plt.subplots_adjust(bottom=0.2,top=0.87,left=0.17,right=0.8)

          ax.errorbar(global_bin,N_meas,DN_meas,fmt='ko',label=dataName)
          ax.set_xlabel('Global Bin',fontsize=labelFontSize)
          ax.set_ylabel(r'$d^{2}\Gamma/dXdY$ [a.u.]',fontsize=labelFontSize)

          if plot_label == 'default':
               ax.plot(global_bin,N_fit,'r-',linewidth=2.0,label='MCMC-Fit with: ' + r'$\chi^{2}/NDF = $' + str(chiSquare_per_NDF))
          else:
               ax.plot(global_bin,N_fit,'r-',linewidth=3.0,label=plot_label)

          ax.set_xticks(np.arange(0,N_fit.shape[0]+1,step=x_step_size))
          ax.legend(loc='lower left')
          ax.grid(True)
          
          if eff is not None and d_eff is not None and show_eff:
             ax2 = ax.twinx() 

             color = 'blue'
             ax2.set_ylabel('Efficiency ' + r'$\epsilon$', color=color)
             ax2.set_ylim(0.0,1.0)
             ax2.errorbar(global_bin,eff,d_eff,fmt='bo')
             ax2.tick_params(axis='y', labelcolor=color)
          
          figSaveName_results = None 
          figSaveName_corr = None
          figSaveName_csym = None
          #------------------------------------
          if figSaveName is not None:
              figSaveName_results = figSaveName + '_DP_fit_results.png'
              figSaveName_corr = figSaveName + '_DP_parameter_correlations.png'
              figSaveName_csym = figSaveName + '_DP_C_symmetry'
              
              fig.savefig(figSaveName_results)
          else:
              plt.show()
          #------------------------------------
          plt.close(fig)

          print(" ")
          print("Found DP parameters:")
          print("--------------------")
          print(" ")
          #++++++++++++++++++++++++
          for p in range(n_dp_parameters):
              print(parNames[p] + ": " + str(DP_values[p]) + " +- " + str(DP_errors[p]))
              if DP_perc_values is not None:
                  print(parNames[p] + "(from percentile): " + str(DP_perc_values[p]) + " - " + str(DP_perc_minErr[p]) + " + " + str(DP_perc_maxErr[p]))

              print(" ")
          #++++++++++++++++++++++++
          
          print("With chi2/ndf = " + str(chiSquare_per_NDF))
          print("  ")
          print("Correlation between parameters:")
          
          reduced_correlation_matrix = np.triu(DP_corr,k=0)

          corr_fig,corr_ax = plt.subplots(figsize=(20,10))
          corr_fig.suptitle('Dalitz Plot Parameter Correlations for the ' + dataName + ' Data')
          plt.subplots_adjust(bottom=0.1,top=0.9,left=0.125,right=0.9)

          corr_ax.matshow(reduced_correlation_matrix)
          corr_ax.set_xticks(np.arange(n_dp_parameters))
          corr_ax.set_yticks(np.arange(n_dp_parameters))
          corr_ax.set_xticklabels(parNames)
          corr_ax.set_yticklabels(parNames)
          corr_ax.tick_params(axis='both', which='major', labelsize=25)

          #++++++++++++++++++++++++++++++++++
          for i in range(n_dp_parameters):
             #++++++++++++++++++++++++++++++++++
             for j in range(n_dp_parameters):
               c = round(reduced_correlation_matrix[j,i],3)
               corr_ax.text(i, j, str(c), va='center', ha='center')
             #++++++++++++++++++++++++++++++++++
          #++++++++++++++++++++++++++++++++++
  
          #------------------------------------
          if figSaveName_corr is not None:
              corr_fig.savefig(figSaveName_corr)
          else:
              plt.show()
          #------------------------------------
          plt.close(corr_fig)

          print(" ")
          print("--------------------")
          print(" ")

          self.check_c_symmetry(DP_values,DP_errors,dataName,figSaveName_csym)

    #-----------------------------

    #Show the mean autocorrelation time:
    def show_autocorr_time(self,coreFileName,crit_tau_factor=50,figSaveName=None,fontSize=20,labelFontSize=25):
          n = np.load(coreFileName + '_scanned_tau_iterations.npy')
          y = np.load(coreFileName + '_scanned_tau_mean_values.npy')
          y_max = np.load(coreFileName + '_scanned_tau_max_values.npy')

          acc_data = ~np.isnan(y_max)

          y_max = y_max[acc_data]
          n = n[acc_data]
          y = y[acc_data]


          plt.rcParams.update({'font.size': fontSize})
          fig, ax = plt.subplots(figsize=(8, 6))
          plt.subplots_adjust(bottom=0.2,top=0.95,left=0.2)

          thresh_label = 'Threshold: ' + str(crit_tau_factor) + r'$\tau$'
          ax.plot(n, n / crit_tau_factor, "--k",linewidth=2.0,label=thresh_label)
          
          ax.plot(n, y,linewidth=2.0,label=r'mean $\hat{\tau}$')
          ax.plot(n, y_max,"r-.",linewidth=2.0,label=r'$\tau_{max}$')
          ax.set_xlim(0, n.max())
          ax.set_ylim(0, y_max.max() + 0.1 * (y_max.max() - y_max.min()))
          ax.set_xlabel("Number of Steps",fontsize=labelFontSize)
          ax.set_ylabel('Autocorrelation Time ' + r'$\tau$',fontsize=labelFontSize)
          ax.legend()
          ax.grid(True)

          #------------------------------------
          if figSaveName is not None:
              fig.savefig(figSaveName + '.png')
          else:
              plt.show()
          #------------------------------------
          plt.close(fig)
    #*****************************************

    #Comparison to other experiments:
    #*****************************************
    #Get results from WASA:
    def get_WASA_results(self):
        WASA_values = [1.144,0.219,0.0,0.086,0.0,0.115,0.0,0.0,0.0]
        WASA_errors = [0.018,0.066,0.0,0.033,0.0,0.037,0.0,0.0,0.0]
        WASA_stat_errors = [0.018,0.019,0.018,0.037]

        return WASA_values,WASA_errors,WASA_stat_errors

    #-----------------------------

    #Get the results from KLOE:
    def get_KLOE_results(self):
        KLOE_values = [1.104,0.142,0.0,0.073,0.0,0.154,0.0,0.0,0.0]
        KLOE_errors = [0.005,0.011,0.0,0.007,0.0,0.011,0.0,0.0,0.0]
        KLOE_stat_errors = [0.003,0.003,0.003,0.006]

        return KLOE_values,KLOE_errors,KLOE_stat_errors

    #-----------------------------

    #Calculate difference between values:
    def calc_diff(self,a,b):
          return a-b

    #---------------------------

    #Calculate the corresponding error:
    def calc_diff_error(self,err_a,err_b):
          return math.sqrt(err_a*err_a + err_b*err_b)

    #---------------------------

    #Get a chi2 between the GlueX results and other experiments:
    def calc_diff_chi2(self,diff_values,diff_err_values):
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
    def calculate_differences(self,values,errors,ref_values,ref_errors):
          diffs = [self.calc_diff(x,y) for x,y in zip(values,ref_values)]
          diff_errors = [self.calc_diff_error(err_x,err_y) for err_x,err_y in zip(errors,ref_errors)]

          d_chisq = self.calc_diff_chi2(diffs,diff_errors)

          return diffs, diff_errors, d_chisq

    #---------------------------
 
    #Compare statistical uncertainteis only:
    def compare_stat_errors(self,dp_errors,stat_errors,plot_pars,plot_labels,labelFontSize,figSaveName,exp_label='Expected from GlueX-I'):
          reduced_par_names = ['-a','b','d','f']
          x_values = np.arange(len(reduced_par_names))

          dp_stat_errors = []
          #+++++++++++++++++++++++++
          for i in range(len(dp_errors)):
              if i != 2 and i != 4 and i < 6:
                  dp_stat_errors.append(dp_errors[i])
          #+++++++++++++++++++++++++

          fig,cx = plt.subplots(figsize=(12,8))
          plt.subplots_adjust(bottom=0.2,top=0.87,left=0.17)
          
          cx.plot(x_values,dp_stat_errors,'ko',label=exp_label,markersize=10)
          
          #+++++++++++++++++++++++++++++++
          for stat_err,plot_par,plot_label in zip(stat_errors,plot_pars,plot_labels):
                   cx.plot(x_values,stat_err,plot_par,label=plot_label,markersize=10)
          #+++++++++++++++++++++++++++++++

          cx.set_ylabel('Statistical Uncertainty',fontsize=labelFontSize)
          cx.set_xticks(x_values)
          cx.set_xticklabels(tuple(reduced_par_names))
          cx.legend()
          cx.grid(True)

          if figSaveName is not None:
              fig.savefig(figSaveName)
          else:
              plt.show()

          plt.close(fig)

    #---------------------------

    #Combine statistical and systematic errors:
    def combine_errors(self,stat_errors,sys_errors):
        all_errors = []
        #+++++++++++++++++++++++
        for stat,sys in zip(stat_errors,sys_errors):
            arg = stat*stat + sys*sys
            all_errors.append(math.sqrt(arg))
        #+++++++++++++++++++++++

        return all_errors

    #---------------------------

    #Run comparison:
    def run_comparison(self,dp_values,dp_errors,parNames,dataName,figSaveName,fontSize,labelFontSize,GlueX_sys_errors):
        WASA_values, WASA_errors, WASA_stat_errors  = self.get_WASA_results()
        KLOE_values, KLOE_errors, KLOE_stat_errors  = self.get_KLOE_results()
        
        GlueX_values = copy.copy(dp_values[1:10])
        GlueX_stat_errors = copy.copy(dp_errors[1:10])
        GlueX_errors = None

        #---------------------------------------
        if GlueX_sys_errors is None:
           GlueX_errors = GlueX_stat_errors
        else:
           GlueX_errors = self.combine_errors(GlueX_stat_errors,GlueX_sys_errors)
        #---------------------------------------

        x_labels = copy.copy(parNames[1:])           
          
        GlueX_values[0] = -1.0*GlueX_values[0]
        x_labels[0] = '-a'

        n_parameters = len(GlueX_values)
        x_values = np.arange(n_parameters)

        diff_kloe, diff_err_kloe, chisq_kloe = self.calculate_differences(GlueX_values,GlueX_errors,KLOE_values,KLOE_errors)
        diff_wasa, diff_err_wasa, chisq_wasa = self.calculate_differences(GlueX_values,GlueX_errors,WASA_values,WASA_errors)

        plt.rcParams.update({'font.size': fontSize})
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches((16,8))
        fig.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(bottom=0.2,top=0.87,left=0.17)

        #ax[0].errorbar(x_values,GlueX_values,GlueX_errors,fmt='ko',label=dataName)
        ax[0].errorbar(x_values,KLOE_values,KLOE_errors,fmt='rd',label='KLOE')
        ax[0].errorbar(x_values,WASA_values,WASA_errors,fmt='bs',label='WASA')
        ax[0].set_ylabel('Parameter Values',fontsize=labelFontSize)
        ax[0].set_xticks(x_values)
        ax[0].set_xticklabels(tuple(x_labels))
        ax[0].legend()
        ax[0].grid(True)
 
        ax[1].errorbar(x_values,diff_kloe,diff_err_kloe,fmt='rd',label=dataName + '- KLOE')     
        ax[1].errorbar(x_values,diff_wasa,diff_err_wasa,fmt='bs',label=dataName + '- WASA') 
        ax[1].set_ylabel('Difference',fontsize=labelFontSize)
        ax[1].set_xticks(x_values)
        ax[1].set_xticklabels(tuple(x_labels))
        ax[1].plot([0.0,8.0],[0.0,0.0],'k--',linewidth=2.0)
        ax[1].legend()
        ax[1].set_ylim(-0.3,0.3)
        ax[1].grid(True)

        figSaveName_stats = None
        #------------------------------------
        if figSaveName is not None:
              figSaveName_stats = figSaveName + '_compare_statistics.png'
              fig.savefig(figSaveName + '.png')
        else:
              plt.show()
        #------------------------------------
        plt.close(fig)

        print("  ")
        print("ChiSquare from comparison:")
        print("KLOE: " + str(chisq_kloe))
        print("WASA: " + str(chisq_wasa))
        print("  ")

        self.compare_stat_errors(GlueX_stat_errors,[KLOE_stat_errors,WASA_stat_errors],['rd','bs'],['KLOE','WASA'],labelFontSize,figSaveName_stats)
    #*****************************************