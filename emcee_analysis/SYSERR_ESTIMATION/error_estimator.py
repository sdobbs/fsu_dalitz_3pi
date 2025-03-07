import numpy as np
import math
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import os
import pandas as pd

#w/halld-scifs17exp/home/chandra/CascadeFSU/kpksxi0/MCSimulation/mergedFall2018/MCSimulation

class Error_Estimator(object):

    def __init__(self):
          self.data = []

    #Subract errors according to Barlow test:
    #***********************************************
    def subtract_errors(self,ref_error,current_error):
        diff = ref_error*ref_error - current_error*current_error
        return np.sqrt(np.fabs(diff))
    #***********************************************

    #Calculate the systematic errors (experimental!):
    #***********************************************
    #Note: This function does not perform a fit, because a fit with a single constant allows for reversing the calculation
    def calc_sys_error_from_constant_fit(self,values,errors,ref_index,var_index):
        reduced_values = np.delete(values[:,var_index],ref_index)
        reduced_errors = np.delete(errors[:,var_index],ref_index)
        ref_value = values[ref_index,var_index] * np.ones_like(reduced_values)

        arg_sigma = 1.0 / (reduced_errors**2)
        arg_m = (np.abs(ref_value) - np.abs(reduced_values)) * arg_sigma

        sigma_t_inv = np.sum(arg_sigma)
        m = np.sum(arg_m)
        
        sigma_t = 1.0 / sigma_t_inv
        q = sigma_t
        p = q*m

        err_1 = -p + math.sqrt(p*p + q)
        err_2 = p - math.sqrt(p*p + q)

        err_min = 0.0
        err_max = 0.0

        if err_1 < err_2:
           err_min = err_1
           err_max = err_2
        else:
           err_min = err_2
           err_max = err_1
    
        return [m,sigma_t,err_min,err_max,reduced_errors,reduced_values,ref_value]
    #***********************************************

    #Calculate pseudo chi2:
    #***********************************************
    def calc_sys_error_from_linear_fit(self,values,errors,ref_index,var_index):
        reduced_values = np.delete(values[:,var_index],ref_index)
        reduced_errors = np.delete(errors[:,var_index],ref_index)
        ref_value = values[ref_index,var_index]
        ref_value_vec = ref_value * np.ones(reduced_values.shape[0])

        def calc_pseudo_chiSquare(x):
            arg_low = (ref_value_vec - x[0]*np.ones_like(ref_value_vec) - reduced_values) / reduced_errors
            arg_high = (ref_value_vec + x[0]*np.ones_like(ref_value_vec) - reduced_values) / reduced_errors
            
            arg = np.where(reduced_values < ref_value_vec,arg_low**2,arg_high**2)
            return np.sum(arg)

        res = minimize(calc_pseudo_chiSquare,[0.1],method='nelder-mead',options={'xatol': 1e-8})
        
        high_val = ref_value + math.fabs(res.x[0])
        low_val = ref_value - math.fabs(res.x[0])

        min_err = low_val - ref_value
        max_err = high_val - ref_value
    
        return [min_err,max_err,ref_value,low_val,high_val]
    #***********************************************

    #Try a simple line fit:
    #***********************************************
    def fit_data(self,variations,values,errors,ref_index,var_index):
        reduced_variations = np.delete(variations,ref_index)
        reduced_values = np.delete(values[:,var_index],ref_index)
        reduced_errors = np.delete(errors[:,var_index],ref_index)
        ref_value = values[ref_index,var_index]

        def linear_func(x,m,b):
            return m*x + b

        par, pcov = curve_fit(linear_func,reduced_variations,reduced_values,p0=[0.0,ref_value],sigma=reduced_errors,absolute_sigma=True)

        perr = np.sqrt(np.diag(pcov))

        fit_values = linear_func(variations,par[0],par[1])
        fit_values_low = linear_func(variations,par[0]-perr[0],par[1]-perr[1])
        fit_values_high = linear_func(variations,par[0]+perr[0],par[1]+perr[1])


        return [fit_values,fit_values_low,fit_values_high]
    #***********************************************

    #Write data out to a data frame:
    #***********************************************
    def load_result_df(self,df_dir,df_name):
        full_file_name = df_dir + '/' + df_name + '.csv'
        result_df = None

        #---------------------------------------------
        if os.path.exists(full_file_name):
            print("  ")
            print("   >>> DataFrame already exists. Going to load existing one. <<<")
            print("  ")

            result_df = pd.read_csv(full_file_name)
        else:
            print("  ")
            print("   >>> DataFrame does not exist. Going to create a fresh new one. <<<")
            print("  ")

            df_columns = ['a','b','c','d','e','f','g','h','l','flag']
            result_df = pd.DataFrame(columns=df_columns)
        #---------------------------------------------

        return result_df
    #***********************************************
        