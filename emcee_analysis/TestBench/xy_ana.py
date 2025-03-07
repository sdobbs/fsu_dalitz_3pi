#!/usr/bin/env python3

import sys
import numpy as np
from READ_ROOT_DATA.root_data_reader import ROOT_Data_Reader
from ROOT import TH1D,TH2D,TCanvas,kBlack,kRed,TSVDUnfold,TFile,TMath
from array import array
import matplotlib.pyplot as plt
from UTILS.chisquare_fitter import ChiSquare_Fitter
from UTILS.analysis_utils import DP_Analysis_Utils
from scipy.optimize import minimize
import math
import random

#Basic definitions:
#**********************************************
data_set = int(sys.argv[1])
fileInitName = sys.argv[2]
ana_name = sys.argv[3]

r_matrix_name = None
#-------------------------------
if len(sys.argv) > 4:
      r_matrix_name = sys.argv[4]
#-------------------------------

reader = ROOT_Data_Reader()
init_fitter = ChiSquare_Fitter()
ana_utils = DP_Analysis_Utils()

data_dict = {
   0: ['17','2017','GlueX 2017','GlueX-2017'],
   1: ['18S','2018S','GlueX 2018-01','GlueX-2018-01'],
   2: ['18F','2018F','GlueX 2018-08','GlueX-2018-08']
}

r_m_file_dict = {
      0: "/Volumes/BunchOfStuff/GlueX_Eta_Data/"
}


addName = data_dict[data_set][1] + '_' + ana_name
rootDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/root_dalitz_data/'
npyDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/npy_dalitz_data'


dataFileName = fileInitName + addName
n_DP_bins = reader.get_n_dp_bins_from_name(dataFileName)
accFileName = 'kinematic_acceptance_ratio' + str(n_DP_bins)
outFileName = dataFileName + '.npy'
fullSaveName = npyDir + '/' + outFileName

graphsDict = {
    'data_graph': 'gr_nEvents_Data',
    'mc_rec_graph': 'gr_nEvents_MC_Rec',
    'mc_true_graph': 'gr_nEvents_MC_True',
    'pipig_bkg_graph': 'gr_nEvents_MC_PiPiG',
    'acc_graph': 'DP_kinAccR',
}

filesDict = {
    'DP_Data_File': rootDir + '/' + dataFileName,
    'DP_Acc_File': rootDir + '/' + accFileName,
}
#**********************************************
print("  ")

DP_data_file = reader.get_root_file(filesDict['DP_Data_File'])
data_graphs = reader.get_data_graphs(DP_data_file,graphsDict)

acc_data_file = reader.get_root_file(filesDict['DP_Acc_File'])
acc_graph = reader.get_graph_from_file(acc_data_file,graphsDict['acc_graph'])

alt_ana_name = 'nbins11_kfit_cut0_imgg_cut0_zv_cut1'
smear_m_file = reader.get_root_file(r_m_file_dict[data_set] + r_matrix_name)
smear_m = reader.get_graph_from_file(smear_m_file,'R_Matrix_' + ana_name)

#--------------------------------------------------------------------

def get_smear_and_folding_matrices(dp_data,graph_collection,acc_graph,gbin_smearing_matrix):
    mc_rec_graph = data_graphs[1]
    mc_true_graph = data_graphs[2]

    n_points = mc_true_graph.GetN()
    n_orig_points = int(math.sqrt(n_points))

    X_bin_edges = np.linspace(-1.1,1.1,n_orig_points+1)
    Y_bin_edges = np.linspace(-1.1,1.1,n_orig_points+1)

    X_smearing_matrix = np.zeros((n_orig_points,n_orig_points))
    Y_smearing_matrix = np.zeros((n_orig_points,n_orig_points))

    X_rec = np.zeros(n_orig_points)
    Y_rec = np.zeros(n_orig_points)

    X_true = np.zeros(n_orig_points)
    Y_true = np.zeros(n_orig_points)

    acc = np.zeros(n_points)

    np_X = dp_data[:,0]
    np_Y = dp_data[:,1]


    X_dig = np.digitize(np_X,X_bin_edges)
    Y_dig = np.digitize(np_Y,Y_bin_edges)
    #++++++++++++++++++++++++++++++++++++
    for p in range(n_points):
        x_mc_rec = array('d',[0.0])
        y_mc_rec = array('d',[0.0])

        x_mc_true = array('d',[0.0])
        y_mc_true = array('d',[0.0])

        x_acc = array('d',[0.0])
        y_acc = array('d',[0.0])

        acc_graph.GetPoint(p,x_acc,y_acc)

        if y_acc[0] > 0.8:
            acc[p] = 1.0

        mc_rec_graph.GetPoint(p,x_mc_rec,y_mc_rec)
        mc_true_graph.GetPoint(p,x_mc_true,y_mc_true)

        x_bin_X = X_dig[p] - 1
        x_bin_Y = Y_dig[p] - 1

        X_rec[x_bin_X] += y_mc_rec[0]
        Y_rec[x_bin_Y] += y_mc_rec[0]


        X_true[x_bin_X] += y_mc_true[0]
        Y_true[x_bin_Y] += y_mc_true[0]
        #++++++++++++++++++++++++++++++++++++
        for k in range(n_points):
            y_bin_X = X_dig[k] - 1
            y_bin_Y = Y_dig[k] - 1

            content = gbin_smearing_matrix.GetBinContent(p+1,k+1)

            X_smearing_matrix[x_bin_X,y_bin_X] += content
            Y_smearing_matrix[x_bin_Y,y_bin_Y] += content
        #++++++++++++++++++++++++++++++++++++
    #++++++++++++++++++++++++++++++++++++

    row_sum_X = np.sum(X_smearing_matrix,1)
    X_folding_matrix = np.where(row_sum_X==0.0,0.0,X_smearing_matrix/row_sum_X)

    row_sum_Y = np.sum(Y_smearing_matrix,1)
    Y_folding_matrix = np.where(row_sum_Y==0.0,0.0,Y_smearing_matrix/row_sum_Y)

    #+++++++++++++++++++++++++++
    for k in range(n_orig_points):
        x_rec = X_rec[k]
        x_true = X_true[k]

        x_eff = 0.0
        if x_true > 0.0:
            x_eff = x_rec / x_true
            
        X_folding_matrix[k,:] *= x_eff

        y_rec = Y_rec[k]
        y_true = Y_true[k]

        y_eff = 0.0
        if y_true > 0.0:
            y_eff = y_rec / y_true
            
        Y_folding_matrix[k,:] *= y_eff
    #+++++++++++++++++++++++++++

    return X_smearing_matrix, Y_smearing_matrix, X_folding_matrix, Y_folding_matrix, X_true, X_rec, Y_true, Y_rec, acc

#--------------------------------------------------------------------

def get_X_and_Y_from_Ampl(amplitude,amplitde_err):
    n_bins = amplitude.shape[0]
    n_orig_bins = int(math.sqrt(n_bins))

    gbin = np.linspace(0,n_bins-1,n_bins).astype(int)
    gbin_reshaped = np.reshape(gbin,(n_orig_bins,n_orig_bins))

    N_yield = amplitude[gbin_reshaped]

    X_out = np.sum(N_yield,0)
    Y_out = np.sum(N_yield,1)

    if amplitde_err is not None:
       N_yield_err = amplitde_err[gbin_reshaped]
       X_err_sq = np.sum(N_yield_err**2,0)
       Y_err_sq = np.sum(N_yield_err**2,1)

       return X_out, Y_out, np.sqrt(X_err_sq), np.sqrt(Y_err_sq)

    else:
        return X_out, Y_out

#--------------------------------------------------------------------

def translate_vec_to_th1d(arr_c,arr,h_th1d,reverse_process=False):

    if reverse_process == False:
       #+++++++++++++++++++++++++++++++
       for p in range(arr_c.shape[0]):
           b = h_th1d.GetXaxis().FindFixBin(arr_c[p])
           h_th1d.SetBinContent(b,arr[p]) 
       #+++++++++++++++++++++++++++++++
    else:
       #+++++++++++++++++++++++++++++++
       for p in range(h_th1d.GetNbinsX()):
           content = h_th1d.GetBinContent(p+1)
           arr[p] = content
       #+++++++++++++++++++++++++++++++

#--------------------------------------------------------------------

def translate_matrix_to_th2d(arr_c,arr_c2,mat,h_th2d,reverse_process=False):

    if reverse_process == False:
       #+++++++++++++++++++++++++++++++
       for p in range(arr_c.shape[0]):
           b1 = h_th2d.GetXaxis().FindFixBin(arr_c[p])

           #+++++++++++++++++++++++++++++++
           for k in range(arr_c2.shape[0]):
               b2 = h_th2d.GetYaxis().FindFixBin(arr_c2[k])

               h_th2d.SetBinContent(b1,b2,mat[p][k])
           #+++++++++++++++++++++++++++++++
       #+++++++++++++++++++++++++++++++
    else:
       #+++++++++++++++++++++++++++++++
       for p in range(h_th2d.GetNbinsX()):
           
           #+++++++++++++++++++++++++++++++
           for k in range(h_th2d.GetNbinsY()):
               content = h_th2d.GetBinContent(p+1,k+1)
               mat[p][k] = content
           #+++++++++++++++++++++++++++++++
       #+++++++++++++++++++++++++++++++

#--------------------------------------------------------------------


npy_dp_data = reader.get_DP_data(filesDict,graphsDict,11)
rebinned_data = ana_utils.rebin_DP(npy_dp_data,include_raw_yields=True)

n_active_bins = rebinned_data.shape[0]
n_dp_bins = npy_dp_data.shape[0]

X_smear, Y_smear, X_fold, Y_fold, X_t, X_r, Y_t, Y_r, Acc = get_smear_and_folding_matrices(npy_dp_data,data_graphs,acc_graph,smear_m)

# X_r_test = np.matmul(X_fold,X_t)
# Y_r_test = np.matmul(Y_fold,Y_t)


N_test = init_fitter.DP_Fit_Function(npy_dp_data[:,[0,1]],100000,-1.095,0.145,0.0,0.088,0.0,0.141,-0.044,0.0,0.0)
N_test = np.where(Acc==1.0,N_test,0.0)

X_o, Y_o = get_X_and_Y_from_Ampl(N_test,None)

X_o_r = np.matmul(X_fold,X_o)
Y_o_r = np.matmul(Y_fold,Y_o)

idx = np.array([i for i in range(11)])

X_edges = np.linspace(-1.1,1.1,12)
Y_edges = np.linspace(-1.1,1.1,12)

X_x = (X_edges[:-1] + X_edges[1:]) / 2
Y_x = (Y_edges[:-1] + Y_edges[1:]) / 2

X_true_dist = TH1D("X_true_dist","",11,-1.1,1.1)
Y_true_dist = TH1D("Y_true_dist","",11,-1.1,1.1)

X_rec_dist = TH1D("X_rec_dist","",11,-1.1,1.1)
Y_rec_dist = TH1D("Y_rec_dist","",11,-1.1,1.1)

X_Smear_M = TH2D("X_Smear_M","",11,-1.1,1.1,11,-1.1,1.1)
Y_Smear_M = TH2D("Y_Smear_M","",11,-1.1,1.1,11,-1.1,1.1)

X_toy_dist = TH1D("X_toy_dist","",11,-1.1,1.1)
Y_toy_dist = TH1D("Y_toy_dist","",11,-1.1,1.1)

X_smear_dist = TH1D("X_smear_dist","",11,-1.1,1.1)
Y_smear_dist = TH1D("Y_smear_dist","",11,-1.1,1.1)

translate_vec_to_th1d(X_x[idx],X_t[idx],X_true_dist)
translate_vec_to_th1d(Y_x[idx],Y_t[idx],Y_true_dist)

translate_vec_to_th1d(X_x[idx],X_r[idx],X_rec_dist)
translate_vec_to_th1d(Y_x[idx],Y_r[idx],Y_rec_dist)

translate_vec_to_th1d(X_x[idx],X_o[idx],X_toy_dist)
translate_vec_to_th1d(Y_x[idx],Y_o[idx],Y_toy_dist)

translate_vec_to_th1d(X_x[idx],X_o_r[idx],X_smear_dist)
translate_vec_to_th1d(Y_x[idx],Y_o_r[idx],Y_smear_dist)

translate_matrix_to_th2d(X_x[idx],Y_x[idx],X_smear,X_Smear_M)
translate_matrix_to_th2d(X_x[idx],Y_x[idx],Y_smear,Y_Smear_M)

tsvdunf_X = TSVDUnfold(X_smear_dist,X_rec_dist,X_true_dist,X_Smear_M)
X_unfold = tsvdunf_X.Unfold(2)
X_unfold.SetName("X_unfold")

tsvdunf_Y = TSVDUnfold(Y_smear_dist,Y_rec_dist,Y_true_dist,Y_Smear_M)
Y_unfold = tsvdunf_Y.Unfold(3)
Y_unfold.SetName("Y_unfold")

N_meas = npy_dp_data[:,8]
N_meas = np.where(Acc==1.0,N_meas,0.0)

dN_meas = npy_dp_data[:,9]
dN_meas = np.where(Acc==1.0,dN_meas,0.0)

X_meas, Y_meas, dX_meas, dY_meas = get_X_and_Y_from_Ampl(N_meas,dN_meas)

X_meas_dist = TH1D("X_meas_dist","",11,-1.1,1.1)
Y_meas_dist = TH1D("Y_meas_dist","",11,-1.1,1.1)

dX_meas_dist = TH2D("dX_meas_dist","",11,-1.1,1.1,11,-1.1,1.1)
dY_meas_dist = TH2D("dY_meas_dist","",11,-1.1,1.1,11,-1.1,1.1)

#+++++++++++++++++++++++++++++++
for a in range(11):
   #+++++++++++++++++++++++++++++++
   for b in range(11):
        if a == b:
           dX_meas_dist.SetBinContent(a+1,b+1,dX_meas[a])
           dY_meas_dist.SetBinContent(a+1,b+1,dY_meas[a])
        else:
           dX_meas_dist.SetBinContent(a+1,b+1,0.0)
           dY_meas_dist.SetBinContent(a+1,b+1,0.0)
   #+++++++++++++++++++++++++++++++ 
#+++++++++++++++++++++++++++++++

translate_vec_to_th1d(X_x[idx],X_meas[idx],X_meas_dist)
translate_vec_to_th1d(Y_x[idx],Y_meas[idx],Y_meas_dist)

tsvdunf_X_meas = TSVDUnfold(X_smear_dist,X_rec_dist,X_true_dist,X_Smear_M)
X_unfold_meas = tsvdunf_X_meas.Unfold(11)
X_unfold_meas.SetName("X_meas_unfold")
X_bcov = tsvdunf_X_meas.GetBCov()
X_bcov.SetName("X_bcov")

tsvdunf_Y_meas = TSVDUnfold(Y_smear_dist,Y_rec_dist,Y_true_dist,Y_Smear_M)
Y_unfold_meas = tsvdunf_Y_meas.Unfold(11)
Y_unfold_meas.SetName("Y_meas_unfold")
Y_bcov = tsvdunf_Y_meas.GetBCov()
Y_bcov.SetName("Y_bcov")

# outf = TFile("bla.root","RECREATE")

# X_true_dist.Write()
# X_rec_dist.Write()
# X_toy_dist.Write()
# X_smear_dist.Write()
# X_Smear_M.Write()
# X_unfold.Write()
# X_meas_dist.Write()
# X_unfold_meas.Write()
# dX_meas_dist.Write()
# X_bcov.Write()

# Y_true_dist.Write()
# Y_rec_dist.Write()
# Y_toy_dist.Write()
# Y_smear_dist.Write()
# Y_Smear_M.Write()
# Y_unfold.Write()
# Y_meas_dist.Write()
# Y_unfold_meas.Write()
# dY_meas_dist.Write()
# Y_bcov.Write()

# outf.Write()

X_unf = np.zeros(11)
translate_vec_to_th1d(X_x[idx],X_unf,X_unfold_meas,reverse_process=True)
Y_unf = np.zeros(11)
translate_vec_to_th1d(Y_x[idx],Y_unf,Y_unfold_meas,reverse_process=True)

X_unf_err = np.zeros((11,11))
Y_unf_err = np.zeros((11,11))
translate_matrix_to_th2d(X_x[idx],Y_x[idx],X_unf_err,X_bcov,reverse_process=True)
translate_matrix_to_th2d(X_x[idx],Y_x[idx],Y_unf_err,Y_bcov,reverse_process=True)

X_err = np.sqrt(np.diag(X_unf_err))
Y_err = np.sqrt(np.diag(Y_unf_err))


# N_test = init_fitter.DP_Fit_Function(npy_dp_data[:,[0,1]],10000,-1.095,0.145,0.0,0.088,0.0,0.141,-0.044,0.0,0.0)
# N_test = np.where(Acc==1.0,N_test,0.0)

# X_o, Y_o, dX_o, dY_o = get_X_and_Y_from_Ampl(N_test,np.sqrt(N_test))

# print(X_err)

def run_XY_fitter(X,Y,dX,dY,DP_Data,Acc,start_values):
    n_points = DP_Data.shape[0]
    n_orig_points = int(math.sqrt(n_points))

    idx = np.array([i for i in range(n_orig_points)])

    dX = np.where(dX==0.0,0.00001,dX)
    dY = np.where(dY==0.0,0.00001,dY)

    def objective_function(x,dp_data,x_meas,y_meas,dx_meas,dy_meas,acc,sort_i):
        N_fit = init_fitter.DP_Fit_Function(dp_data[:,[0,1]],x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

        #N_fit = init_fitter.DP_Fit_Function(dp_data[:,[0,1]],x[0],x[1],x[2],0.0,x[4],0.0,x[6],x[7],0.0,0.0)
        N_fit = np.where(acc==1.0,N_fit,0.0)

        x_fit, y_fit = get_X_and_Y_from_Ampl(N_fit,None)

        res_x = (x_meas[sort_i] - x_fit[sort_i]) / dx_meas[sort_i]
        res_y = (y_meas[sort_i] - y_fit[sort_i]) / dy_meas[sort_i]

        return np.sum(res_x**2) + np.sum(res_y**2)

    res = minimize(objective_function,x0=start_values,args=(DP_Data,X,Y,dX,dY,Acc,idx),method='BFGS')
    dp_pars = res.x

    print("Function value: " + str(res.fun))

    N_res = init_fitter.DP_Fit_Function(DP_Data[:,[0,1]],dp_pars[0],dp_pars[1],dp_pars[2],dp_pars[3],dp_pars[4],dp_pars[5],dp_pars[6],dp_pars[7],dp_pars[8],dp_pars[9])
    N_res = np.where(Acc==1.0,N_res,0.0)


    X_res, Y_res = get_X_and_Y_from_Ampl(N_res,None)
    
    return X_res[idx], Y_res[idx], dp_pars



start_pars = [
    random.uniform(10000,30000),
    random.uniform(-1.2,-1.0),
    random.uniform(0.05,0.2),
    random.uniform(-0.01,0.01),
    random.uniform(0.05,0.1),
    random.uniform(-0.01,0.01),
    random.uniform(0.1,0.2),
    random.uniform(-0.01,0.01),
    random.uniform(-0.01,0.01),
    random.uniform(-0.01,0.01)
]

X_fit, Y_fit, DP_Pars = run_XY_fitter(X_unf,Y_unf,X_err,Y_err,npy_dp_data,Acc,start_pars)

for p in DP_Pars:
    print(p)

#X_fit, Y_fit, DP_Pars = run_XY_fitter(X_o,Y_o,dX_o,dY_o,npy_dp_data,Acc,start_pars)

fig,ax = plt.subplots(1,2)

ax[0].errorbar(x=X_x[idx],y=X_unf[idx],yerr=X_err[idx],fmt='ko')
ax[0].plot(X_x[idx],X_fit[idx],'rs-')

ax[1].errorbar(x=Y_x[idx],y=Y_unf[idx],yerr=Y_err[idx],fmt='ko')
ax[1].plot(Y_x[idx],Y_fit[idx],'rs-')


# ax[0].errorbar(x=X_x[idx],y=X_o[idx],yerr=dX_o[idx],fmt='ko')
# ax[0].plot(X_x[idx],X_fit[idx],'rs-')

# ax[1].errorbar(x=Y_x[idx],y=Y_o[idx],yerr=dY_o[idx],fmt='ko')
# ax[1].plot(Y_x[idx],Y_fit[idx],'rs-')


plt.show()











