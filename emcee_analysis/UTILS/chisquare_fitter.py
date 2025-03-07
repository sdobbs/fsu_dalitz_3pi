import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares,minimize
import matplotlib.pyplot as plt

class ChiSquare_Fitter(object):

    def __init__(self,kinematic_acceptance=None,kinematic_acceptance_cut=0.8):
          self.data = []
          self.acceptance = 1.0

          if kinematic_acceptance is not None:
            self.acceptance = np.where(kinematic_acceptance>kinematic_acceptance_cut,1.0,0.0)    

    #Define function to minimize:
    #*********************************
    def DP_Fit_Function(self,x,norm,parA,parB,parC,parD,parE,parF,parG,parH,parL):
        DP_X = x[:,0]
        DP_Y = x[:,1]

        N_fit = norm*(1.0 + parA*DP_Y + parB*DP_Y*DP_Y + parC*DP_X + parD*DP_X*DP_X + parE*DP_X*DP_Y + parF*DP_Y*DP_Y*DP_Y + parG*DP_X*DP_X*DP_Y + parH*DP_X*DP_Y*DP_Y + parL*DP_X*DP_X*DP_X)
        
        return N_fit * self.acceptance
    #*********************************

    #Fit the data:
    #*********************************
    def fit_data(self,DP_Data,start_values,use_abs_sigma,par_bounds):
        dp_x_values = DP_Data[:,[0,1]]
        dp_y_values = DP_Data[:,2]
        dp_y_errors = np.where(DP_Data[:,3] > 0.0,DP_Data[:,3],1.0)

        parameter_bounds = ([-np.inf]*len(start_values),[np.inf]*len(start_values))
        #------------------------------
        if par_bounds is not None:
            parameter_bounds = par_bounds
        #------------------------------

        dp_values, dp_errors = curve_fit(self.DP_Fit_Function,dp_x_values,dp_y_values,p0=start_values,sigma=dp_y_errors,absolute_sigma=use_abs_sigma,bounds=parameter_bounds)
        return [dp_values,dp_errors]
    #*********************************

    #Perform fit and show the results:
    #*********************************
    def run_initial_fitter(self,DP_Raw_Data,start_values,parNames,dataSetName,fitName,show_results,figSaveName=None,fontSize=20,labelFontSize=25,use_abs_sigma=True,parBounds=None):
        dp_values = None
        dp_errors = None
        chiSquare = None
        chiSquare_per_NDF = None
        y_fit = None

        DP_Data = np.copy(DP_Raw_Data)

        DP_Data[:,0] *= self.acceptance
        DP_Data[:,1] *= self.acceptance
        DP_Data[:,2] *= self.acceptance
        DP_Data[:,3] *= self.acceptance

        #Get the parameters:
        dp_values, dp_cov = self.fit_data(DP_Data,start_values,use_abs_sigma,parBounds)

        #Compute chi2:
        x_data = DP_Data[:,[0,1]]
        y_data = DP_Data[:,2]
        y_data_error = np.where(DP_Data[:,3] > 0.0,DP_Data[:,3],1.0)
        y_fit = self.DP_Fit_Function(x_data,dp_values[0],dp_values[1],dp_values[2],dp_values[3],dp_values[4],dp_values[5],dp_values[6],dp_values[7],dp_values[8],dp_values[9])

        ndf = x_data.shape[0] - dp_values.shape[0]
        arg = (y_data-y_fit) / y_data_error
        chiSquare = np.sum(arg**2)
        chiSquare_per_NDF = round(chiSquare / ndf,2)
        dp_errors = np.sqrt(np.diag(dp_cov))

      
        if show_results:

           #Show parameters:
           print("  ")
           print("Found DP parameters from initial fit:")
           print("-------------------------------------")
           print("  ")

           #++++++++++++++++++++++++++++
           for p in range(dp_values.shape[0]):
              print(parNames[p] + ": " + str(dp_values[p]) + " +- " + str(dp_errors[p]))
              print("  ")
           #++++++++++++++++++++++++++++

           print("With chi2/NDF = " + str(chiSquare_per_NDF))
           print("-------------------------------------")
           print("  ")

           #Visualize the results:
           plt.rcParams.update({'font.size': fontSize})
           fig, ax = plt.subplots(figsize=(12, 8))
           plt.subplots_adjust(bottom=0.2,top=0.95,left=0.2,right=0.8)

           ax.errorbar(DP_Data[:,4],y_data,y_data_error,fmt='ko',label=dataSetName)
           ax.set_xlabel('Global Bin',fontsize=labelFontSize)
           ax.set_ylabel(r'$N(\eta\rightarrow\pi^{+}\pi^{-}\pi^{0})$ [a.u.]',fontsize=labelFontSize)

           fitName += ' with: ' + r'$\chi^{2}/NDF$' + ' = ' + str(chiSquare_per_NDF) 
           ax.plot(DP_Data[:,4],y_fit,'r-',linewidth=2.0,label=fitName)
           ax.set_xticks(np.arange(0,y_fit.shape[0]+1,step=5.0))
           ax.legend(loc='lower left')

           if DP_Data.shape[1] > 6:
               
               DP_Data[:,6] *= self.acceptance
               DP_Data[:,7] *= self.acceptance

               ax2 = ax.twinx() 
               color = 'blue'
               ax2.set_ylabel('Efficiency ' + r'$\epsilon$', color=color)
               ax2.set_ylim(0.0,1.0)
               ax2.errorbar(DP_Data[:,4],DP_Data[:,6],DP_Data[:,7],fmt='bo')
               ax2.tick_params(axis='y', labelcolor=color)
          
           #------------------------------------
           if figSaveName is not None:
              figSaveName_results = figSaveName + '_DP_chi2Fit_results.png'
              fig.savefig(figSaveName_results)
           else:
              plt.show()
           #------------------------------------
           plt.close(fig)

        return [dp_values,dp_cov,dp_errors,chiSquare_per_NDF]
    #*********************************

    #Generate start values
    #from initial fit:
    #*********************************
    def get_start_values_from_init_fit(self,fit_vals,fit_errs,n_sigma):
        minVals = [v-n_sigma*e for v,e in zip(fit_vals,fit_errs)]
        maxVals = [v+n_sigma*e for v,e in zip(fit_vals,fit_errs)]

        minVals[0] = int(minVals[0])
        maxVals[0] = int(maxVals[0])

        return [minVals,maxVals]
    #*********************************



    #Run the fitter, by using an efficienvy matrix:
    #///////////////////////////////////////////////////////////////////////

    #Function to minimize:
    #*********************************
    def fit_data_effM(self,DP_Data,start_values,effM,d_effM):
        dp_x_values = DP_Data[:,[0,1]]
        dp_y_values = DP_Data[:,7]
        dp_y_errors = DP_Data[:,8]

        def objective_function(x,dp_xy,N,dN,eff_M,d_eff_M):
            N_fit = self.DP_Fit_Function(dp_xy,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

            diff = N - np.matmul(eff_M,N_fit)
           
            sigma = dN**2 + np.matmul(d_eff_M,N_fit**2)
            #sigma = dN**2

            arg = diff**2/sigma

            return np.sum(arg)

        def errFit(hess_inv, resVariance):
            return np.sqrt( np.diag( hess_inv * resVariance))

        parameter_bounds = ([-np.inf]*len(start_values),[np.inf]*len(start_values))


        res = minimize(objective_function,x0=start_values,args=(dp_x_values,dp_y_values,dp_y_errors,effM,d_effM),method='BFGS')
        dp_values = res.x

        h_inv = res.jac
        res_v = res.fun / (dp_y_values.shape[0]-len(start_values))

        #dp_errors = errFit(h_inv,res_v)
   
        dp_errors = np.zeros_like(dp_values)

        return [dp_values,dp_errors]
    #*********************************

    #Testing:
    #*********************************
    def run_likelihood_fitter(self,dp_data,eff_M,d_eff_M,percentage,n_iterations,start_pars,par_bounds):
        
        def log_likelihood(theta,x,y,yerr,effM,d_effM):
            N_fit = self.DP_Fit_Function(x,theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],theta[9])

            diff = y - np.matmul(effM,N_fit)
            sigma2 = yerr**2 + np.matmul(d_effM,N_fit**2)

            return -0.5 * np.sum(diff ** 2 / sigma2 + np.log(sigma2))

        nll = lambda *args: -log_likelihood(*args)

        initial = np.array(start_pars) + 0.1 * np.random.randn(len(start_pars))

        # test_L = log_likelihood(initial,dp_data[:,[0,1]],dp_data[:,7],dp_data[:,8],eff_M)

        # print(test_L)
        
        fit_results = []
        #++++++++++++++++++++++++++++++++++
        for it in range(1,n_iterations+1):
            print("   Running fit iteration: " + str(it) + "/" + str(n_iterations))

            idx = np.random.choice(dp_data.shape[0],int(percentage*dp_data.shape[0]),replace=False)
            fit_data = dp_data[idx]
            eff_M_fit = eff_M[idx,idx]
            d_eff_M_fit = d_eff_M[idx,idx]
            
            soln = minimize(nll,initial,args=(fit_data[:,[0,1]],fit_data[:,7],fit_data[:,8],eff_M_fit,d_eff_M_fit),bounds=par_bounds)

            L = log_likelihood(soln.x,fit_data[:,[0,1]],fit_data[:,7],fit_data[:,8],eff_M_fit,d_eff_M_fit)

            current_results = soln.x
            
            current_results = np.append(current_results,L)

            current_results = np.reshape(current_results,(current_results.shape[0],1)).T

            fit_results.append(current_results)
        #++++++++++++++++++++++++++++++++++

        result_vec = np.concatenate(fit_results,axis=0)

        plt.hist(result_vec[:,10],100)
        

        dp_values = np.zeros(10)
        dp_errors = np.zeros(10)

        #++++++++++++++++++++++++
        for p in range(10):
            dp_values[p] = np.mean(result_vec[:,p])
            dp_errors[p] = np.std(result_vec[:,p])
        #++++++++++++++++++++++++

        return dp_values, dp_errors
    #*********************************

