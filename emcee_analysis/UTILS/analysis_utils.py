import math
import numpy as np
import random
import copy

class DP_Analysis_Utils(object):

    def __init__(self):
          self.data = []

    #Get raw yields (for systematic studies)
    #*************************************************
    def get_raw_yields(self,yields,yield_errs,effs,eff_errs):
        raw_yields = np.multiply(yields,effs)
        raw_yield_errs = np.sqrt(np.multiply(yield_errs,effs) + np.multiply(yields,eff_errs))

        return [raw_yields, raw_yield_errs]
    #*************************************************

    #Remove empty bins from the dalitz plot
    #The input data should be organized as follows:
    #data[0] = X
    #data[1] = Y
    #data[2] = Yields
    #data[3] = Yield Errors
    #data[4] = global bin --> will be replaced in this function
    #if aplicable:
    #data[5] = efficiency
    #data[6] = error efficiency 
    #data[7] = kinematic acceptance 
    #data[8] = Raw Yields
    #data[9] = Error raw yields
    #*************************************************
    def rebin_DP(self,yourDPData,include_raw_yields=False):
        active_DP_data = []
        active_gbin = 0
        #++++++++++++++++++++++++++++++++++++++
        for data in yourDPData:
            if data[2] > 0.0:
                out_list = [data[0],data[1],data[2],data[3],active_gbin,data[5]]
  
                if len(data) > 6: 
                   out_list.append(data[6]) #--> Add efficiency 
                   out_list.append(data[7]) #--> Add efficiency error
                   
                   
                   if include_raw_yields: #--> Add raw yields, in case the efficiency information is stored:
                      out_list.append(data[8])
                      out_list.append(data[9])
                      out_list.append(data[4]) #--> Add the former global binning --> needed for the efficiency matrix
                      
                
                active_DP_data.append(out_list)
                active_gbin += 1
        #++++++++++++++++++++++++++++++++++++++

        return np.array(active_DP_data) 
    #*************************************************

    #Resize the efficiency matrix (if available)
    #*************************************************
    def resize_efficiency_matrix(self,rebinned_DP_Data,eff_M,d_eff_M):
        activ_gbin = rebinned_DP_Data[:,4]
        old_gbin = rebinned_DP_Data[:,9]
        
        new_dim = activ_gbin.shape[0]

        new_eff_M = np.zeros((new_dim,new_dim))
        new_d_eff_M = np.zeros((new_dim,new_dim))

        #++++++++++++++++++++++++++++++++++++++
        for i in range(new_dim):

          a = int(old_gbin[i])
          #++++++++++++++++++++++++++++++++++++++
          for j in range(new_dim):

            b = int(old_gbin[j])

            new_eff_M[i,j] = eff_M[a,b]
            new_d_eff_M[i,j] = d_eff_M[a,b]           

          #++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++

        return new_eff_M, new_d_eff_M
    #*************************************************

    #Add the individual (efficiency corrected) data 
    #*************************************************
    def add_data(self,data_collection,include_raw_yields=False):
        dp_x = copy.copy(data_collection[0][:,0])
        dp_y = copy.copy(data_collection[0][:,1])
        gbin = copy.copy(data_collection[0][:,4])
        acc = copy.copy(data_collection[0][:,5])

        sum_n = np.zeros(shape=data_collection[0][:,2].shape)
        sum_err_n = np.zeros(shape=data_collection[0][:,3].shape)

        sum_n_raw = np.zeros(shape=data_collection[0][:,2].shape)
        sum_err_n_raw = np.zeros(shape=data_collection[0][:,3].shape)

        #+++++++++++++++++++++++++++++++
        for data in data_collection:
            sum_n += data[:,2]
            sum_err_n += data[:,3]**2

            if data_collection[0].shape[1] > 5 and include_raw_yields:
               n_raw, dn_raw = self.get_raw_yields(data[:,2],data[:,3],data[:,5],data[:,6]) 
               sum_n_raw += n_raw
               sum_err_n_raw += dn_raw**2
        #+++++++++++++++++++++++++++++++

        new_results = None
        if include_raw_yields:
          new_results = np.vstack((
            dp_x,
            dp_y,
            sum_n,
            np.sqrt(sum_err_n),
            gbin,
            acc,
            np.zeros(shape=data_collection[0][:,2].shape),#Not nice, but these are place holders, so the indexing is consistent....
            np.zeros(shape=data_collection[0][:,2].shape),
            sum_n_raw,
            np.sqrt(sum_err_n_raw)
          ))
        else:
          new_results = np.vstack((
            dp_x,
            dp_y,
            sum_n,
            np.sqrt(sum_err_n),
            gbin,
            acc
          ))
        
        return np.transpose(new_results)
    #*************************************************

    #Calculate asymmetry:
    #*************************************************
    def calculate_asymmetry_from_data(self,dp_x,yields):
        neg_cond = (dp_x < 0.0)
        pos_cond = (dp_x > 0.0)

        neg_yields = np.where(neg_cond,yields,0.0)
        pos_yields = np.where(pos_cond,yields,0.0)

        N_minus = np.sum(neg_yields)
        N_plus = np.sum(pos_yields)

        A = (N_plus - N_minus) / (N_plus + N_minus)
        return A

    #************************************************* 

    #Generate DP with known parameters for algorithm calibration / debugging:

    #Define function with uncertainty on Dalitz Parameters:
    #*************************************************
    def one_DP_function(self,dp_pars,dp_errs,dp_x,dp_y):
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
    #*************************************************

    #Generate the dalitz plot:
    #*************************************************
    def gen_DP(self,dp_pars,dp_errs,n_dp_bins,n_events,DP_Range=[-1.1,1.1],add_gaussian_noise=0.0,add_flat_noise=0.0):
        dp_x = np.linspace(DP_Range[0],DP_Range[1],n_dp_bins)
        dp_y = np.linspace(DP_Range[0],DP_Range[1],n_dp_bins)

        N_gen = np.zeros((n_dp_bins*n_dp_bins))
        DN_gen = np.zeros((n_dp_bins*n_dp_bins))
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
            
                N_gen[gbin] += self.one_DP_function(dp_pars,dp_errs,dp_x[i],dp_y[j])
              #++++++++++++++++++++++++++++++++++++ 
          #++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++

        counter = 0
        #++++++++++++++++++++++++++++++++++++
        for n in N_gen:
            if n > 0.0:
              DN_gen[counter] = math.sqrt(n)
            else:
              DN_gen[counter] = 0.0

            counter += 1
        #++++++++++++++++++++++++++++++++++++

        if add_gaussian_noise != 0.0:
            gaussian_noise = np.random.normal(1.0,add_gaussian_noise,n_dp_bins*n_dp_bins)
            
            N_gen = np.multiply(N_gen,gaussian_noise)

        if add_flat_noise != 0.0:
            flat_noise = np.random.uniform(1.0-add_flat_noise,1.0+add_flat_noise,n_dp_bins*n_dp_bins)
            N_gen = np.multiply(N_gen,flat_noise)

        out_vec = np.vstack(( 
              dp_x_out,
              dp_y_out,
              N_gen,
              DN_gen,
              global_bin
        ))
        
        return np.transpose(out_vec)
    #*************************************************

    #*************************************************
    def normalize_Dalitz_Data(self,DP_Data):
        norm_val = 0.0

        min_x = np.min(np.abs(DP_Data[:,0]))
        min_y = np.min(np.abs(DP_Data[:,1]))

        #+++++++++++++++++++++++++++++
        for dat in DP_Data:
            
          if math.fabs(dat[0]) == min_x and math.fabs(dat[1]) == min_y:
            norm_val = dat[2]
        #+++++++++++++++++++++++++++++

        DP_Data[:,2] = DP_Data[:,2] / norm_val
        DP_Data[:,3] = DP_Data[:,3] / norm_val

        return DP_Data
    #*************************************************