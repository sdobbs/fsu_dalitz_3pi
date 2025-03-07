import numpy as np

class fitter_utils(object):

    def __init__(self,kinematic_acceptance):
        self.data = []
        self.kinematic_acceptance = kinematic_acceptance

    #List of helpful loss functions, that might
    #be used for minimization
    #All loss functions shown here have been adapted from: 
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    #**********************************
    #The linear loss:
    def linear_loss(self,X):
        return X

    #-------------------------

    #Huber loss:
    def huber_loss(self,X,delta):
        cond = X <= delta
        return (cond)*X + (~cond)*2.0*delta*(np.sqrt(X) - 0.5*delta)

    #-------------------------
    
    #Soft l1 loss:
    def soft_l1_loss(self,X):
        return 2 * (np.sqrt(1 + X) - 1)

    #-------------------------

    #Cauchy loss:
    def cauchy_loss(self,X):
        return np.log(1.0 + X)

    #-------------------------

    #Arctan loss:
    def arctan_loss(self,X):
        return np.arctan(X)

    #-------------------------

    #Get a specified loss function:
    def get_loss_function(self,func_str,func_par):

        #------------------------------
        if func_str == 'linear':
            def loss_function(X):
                return self.linear_loss(X)
            
            return loss_function
        
        elif func_str == 'huber':
            def loss_function(X):
                return self.huber_loss(X,func_par)

            return loss_function

        elif func_str == 'soft_l1':
            def loss_function(X):
                return self.soft_l1_loss(X)
            
            return loss_function

        elif func_str == 'cauchy':
            def loss_function(X):
                return self.cauchy_loss(X)
            
            return loss_function

        elif func_str == 'arctan':
            def loss_function(X):
                return self.arctan_loss(X)
            
            return loss_function

        elif func_str == 'ensemble':
            def loss_function(X):
                arg = self.linear_loss(X)
                arg += self.huber_loss(X,func_par)
                arg += self.soft_l1_loss(X)
                arg += self.cauchy_loss(X)
                arg += self.arctan_loss(X)

                return np.mean(arg)

            return loss_function 
        #------------------------------

        print("  ")       
        print(">>> Warning! This loss function is not specified! <<<")
        print(">>> Please choose one of the following: <<<")
        print(">>> 'linear' <<<")
        print(">>> 'huber' <<<")
        print(">>> 'soft_l1' <<<")
        print(">>> 'cauchy' <<<")
        print(">>> 'arctan' <<<")
        print("  ")  

        return None
    #**********************************

    #Get a generic likelihood function and posterior probability, 
    #utilizing the provided objective function
    #Again, the idea to this has been adapted from:
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    #**********************************
    #Log likelihood
    def log_likelihood(self,theta,x,y,yerr,model,loss_function):
        y_fit = model(theta,x)
        res = (y-y_fit) / yerr

        return -0.5*np.sum(loss_function(res**2))

    #----------------------------

    #Posterior probability
    def log_probability(self,theta,x,y,yerr,model,loss_function,prior_function):
        lp = prior_function(theta)
        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(theta,x,y,yerr,model,loss_function)
    #**********************************

    #Get the objective function:
    #**********************************
    def get_objective_function(self,model,func_str,func_par,prior_func):
        loss_func = self.get_loss_function(func_str,func_par)
        
        if self.kinematic_acceptance is None:
            #----------------------------------------------------------------------
            if prior_func is None:
                def objective_function(theta,x,y,yerr):
                    return self.log_likelihood(theta,x,y,yerr,model,loss_func)

                return objective_function

            else:
                def objective_function(theta,x,y,yerr):
                    return self.log_probability(theta,x,y,yerr,model,loss_func,prior_func)

                return objective_function
            #----------------------------------------------------------------------
        else:

            #----------------------------------------------------------------------
            if prior_func is None:
                def objective_function(theta,x,y,yerr):
                    return self.log_likelihood_kin_acc(theta,x,y,yerr,model,loss_func)

                return objective_function

            else:
                def objective_function(theta,x,y,yerr):
                    return self.log_probability_kin_acc(theta,x,y,yerr,model,loss_func,prior_func)

                return objective_function
            #----------------------------------------------------------------------
    #**********************************

    #Same as above, but this time the kinematic acceptance cut is done by the fitter:
    #**********************************
    def apply_acceptance_cut(self,acc_cut):
        cut = 1.0 / (1.0 + np.exp(-acc_cut))
        acc = np.where(self.kinematic_acceptance>cut,1.0,0.0)

        return acc, cut


    #---------------------------- 

    #Log likelihood
    def log_likelihood_kin_acc(self,theta,x,y,yerr,model,loss_function):
        acc,_ = self.apply_acceptance_cut(theta[10])

        y_fit = model(theta,x)
        y_fit = y_fit * acc
        y = y * acc
        yerr = yerr * acc        

        sigma_squared = np.where(yerr>0.0,yerr**2,1.0)
        res = (y-y_fit) / np.sqrt(sigma_squared)

        return -0.5*np.sum(loss_function(res**2))

    #----------------------------

    #Posterior probability
    def log_probability_kin_acc(self,theta,x,y,yerr,model,loss_function,prior_function):
        lp = prior_function(theta)
        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood_kin_acc(theta,x,y,yerr,model,loss_function)
    #**********************************


