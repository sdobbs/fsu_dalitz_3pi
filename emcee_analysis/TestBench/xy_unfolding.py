#!/usr/bin/env python3

import sys
import numpy as np
from READ_ROOT_DATA.root_data_reader import ROOT_Data_Reader
from ROOT import TH1D,TH2D,TCanvas,kBlack,kRed,TSVDUnfold,TFile,TMath
from array import array
import matplotlib.pyplot as plt
from UTILS.chisquare_fitter import ChiSquare_Fitter
from UTILS.analysis_utils import DP_Analysis_Utils

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
     # 0: "/Users/daniellersch/Desktop/eta3Pi_DalitzAna/Efficiency_Matrices/"
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

smear_m_file = reader.get_root_file(r_m_file_dict[data_set] + r_matrix_name)
alt_ana_name = 'nbins11_kfit_cut0_imgg_cut0_zv_cut1'
smear_m = reader.get_graph_from_file(smear_m_file,'R_Matrix_' + ana_name)

print(smear_m)

#smear_m = reader.get_graph_from_file(smear_m_file,'h_gbin11_true_vs_gbin11_rec')

npy_dp_data = reader.get_DP_data(filesDict,graphsDict,11)
rebinned_data = ana_utils.rebin_DP(npy_dp_data,include_raw_yields=True)

n_active_bins = rebinned_data.shape[0]
n_dp_bins = npy_dp_data.shape[0]
n_orig_bins = 11

#X-values:
#----------
X_true_dist = TH1D('X_true_dist','',11,-1.1,1.1)
X_rec_dist = TH1D('X_rec_dist','',11,-1.1,1.1)
X_smear_matrix = TH2D('X_smear_matrix','',11,-1.1,1.1,11,-1.1,1.1)

np_X_smear_matrix = np.zeros((11,11))

X_toy_dist = TH1D('X_toy_dist','',11,-1.1,1.1)
np_X_toy = np.zeros(n_orig_bins)
X_smear_dist = TH1D('X_smear_dist','',11,-1.1,1.1)
np_X_smear = np.zeros(n_orig_bins)

#Y-values:
#---------
Y_true_dist = TH1D('Y_true_dist','',11,-1.1,1.1)
Y_rec_dist = TH1D('Y_rec_dist','',11,-1.1,1.1)
Y_smear_matrix = TH2D('Y_smear_matrix','',11,-1.1,1.1,11,-1.1,1.1)

np_Y_smear_matrix = np.zeros((11,11))

Y_toy_dist = TH1D('Y_toy_dist','',11,-1.1,1.1)
np_Y_toy = np.zeros(n_orig_bins)
Y_smear_dist = TH1D('Y_smear_dist','',11,-1.1,1.1)
np_Y_smear = np.zeros(n_orig_bins)


eff_vec = np.zeros(n_dp_bins)

#Get a test DP:
#---------------
DP_Test = init_fitter.DP_Fit_Function(npy_dp_data[:,[0,1]],10000,-1.095,0.145,0.0,0.081,0.0,0.141,-0.044,0.0,0.0)

plt.plot(npy_dp_data[:,4],DP_Test,'ko')
plt.show()

np_X = npy_dp_data[:,0]
np_Y = npy_dp_data[:,1]

mc_rec_graph = data_graphs[1]
mc_true_graph = data_graphs[2]
#+++++++++++++++++++++++++++++
for p in range(n_dp_bins):
    x_mc_rec = array('d',[0.0])
    y_mc_rec = array('d',[0.0])

    x_mc_true = array('d',[0.0])
    y_mc_true = array('d',[0.0])

    x_acc = array('d',[0.0])
    y_acc = array('d',[0.0])

    mc_rec_graph.GetPoint(p,x_mc_rec,y_mc_rec)
    mc_true_graph.GetPoint(p,x_mc_true,y_mc_true)
    acc_graph.GetPoint(p,x_acc,y_acc)

    bin_x_true = X_true_dist.GetXaxis().FindFixBin(np_X[p])
    X_true_dist.AddBinContent(bin_x_true,y_mc_true[0])

    bin_x_rec = X_rec_dist.GetXaxis().FindFixBin(np_X[p])
    X_rec_dist.AddBinContent(bin_x_rec,y_mc_rec[0])

    bin_y_true = Y_true_dist.GetXaxis().FindFixBin(np_Y[p])
    Y_true_dist.AddBinContent(bin_y_true,y_mc_true[0])

    bin_y_rec = Y_rec_dist.GetXaxis().FindFixBin(np_Y[p])
    Y_rec_dist.AddBinContent(bin_y_rec,y_mc_rec[0])

    bin_x_smearX = X_smear_matrix.GetXaxis().FindFixBin(np_X[p])
    bin_x_smearY = Y_smear_matrix.GetXaxis().FindFixBin(np_Y[p])

    #test_val = DP_Test[p]
    test_val = y_mc_true[0]

    if y_mc_rec[0] > 0.0:
        eff_vec[p] = y_mc_rec[0] / y_mc_true[0]

    p_acc = 0.0
    if y_acc[0] > -0.8:
       bin_x_toy = X_toy_dist.GetXaxis().FindFixBin(np_X[p])
       X_toy_dist.AddBinContent(bin_x_toy,test_val)

       bin_y_toy = Y_toy_dist.GetXaxis().FindFixBin(np_Y[p])
       Y_toy_dist.AddBinContent(bin_y_toy,test_val)

       p_acc = 1.0

       

    
    # #+++++++++++++++++++++++++++++
    # for k in range(n_dp_bins):
    #     x_acc = array('d',[0.0])
    #     y_acc = array('d',[0.0])

    #     acc_graph.GetPoint(k,x_acc,y_acc)

    #     k_acc = 0.0
    #     if y_acc[0] > 0.8:
    #         k_acc = 1.0

    #     bin_y_smearX = X_smear_matrix.GetYaxis().FindFixBin(np_X[k])
    #     bin_y_smearY = Y_smear_matrix.GetYaxis().FindFixBin(np_Y[k])



    #     content_X = smear_m.GetBinContent(p+1,k+1)
    #     np_X_smear_matrix[bin_x_smearX-1][bin_y_smearX-1] += content_X

    #     content_Y = smear_m.GetBinContent(p+1,k+1)
    #     np_Y_smear_matrix[bin_x_smearY-1][bin_y_smearY-1] += content_Y
    # #+++++++++++++++++++++++++++++    

#+++++++++++++++++++++++++++++


#+++++++++++++++++++++++++++++
for p in range(n_dp_bins):

    bin_x_smearX = X_smear_matrix.GetXaxis().FindFixBin(np_X[p])
    bin_x_smearY = Y_smear_matrix.GetXaxis().FindFixBin(np_Y[p])

    #+++++++++++++++++++++++++++++
    for k in range(n_dp_bins):

      bin_y_smearX = X_smear_matrix.GetYaxis().FindFixBin(np_X[k])  
      bin_y_smearY = Y_smear_matrix.GetYaxis().FindFixBin(np_Y[k])

      content_X = smear_m.GetBinContent(p+1,k+1)
      #content_X *= eff_vec[k] 

      np_X_smear_matrix[bin_x_smearX-1][bin_y_smearX-1] += content_X

      content_Y = smear_m.GetBinContent(p+1,k+1)
      #content_Y *= eff_vec[k] 

      np_Y_smear_matrix[bin_x_smearY-1][bin_y_smearY-1] += content_Y

    #+++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++






row_sum_X = np.sum(np_X_smear_matrix,1)
folding_X_matrix = np.where(row_sum_X==0.0,0.0,np_X_smear_matrix/row_sum_X)

row_sum_Y = np.sum(np_Y_smear_matrix,1)
folding_Y_matrix = np.where(row_sum_Y==0.0,0.0,np_Y_smear_matrix/row_sum_Y)

# folding_X_matrix = np_X_smear_matrix
# folding_Y_matrix = np_Y_smear_matrix

#+++++++++++++++++++++++++++++++++++++++++++
for k in range(n_orig_bins):
    x_true = X_true_dist.GetBinContent(k+1)
    x_rec = X_rec_dist.GetBinContent(k+1)

    x_eff = 0.0
    if x_true > 0.0:
        x_eff = x_rec / x_true

    folding_X_matrix[k,:] *= x_eff

    y_true = Y_true_dist.GetBinContent(k+1)
    y_rec = Y_rec_dist.GetBinContent(k+1)

    y_eff = 0.0
    if y_true > 0.0:
        y_eff = y_rec / y_true
    
    folding_Y_matrix[k,:] *= y_eff

    np_X_toy[k] = X_toy_dist.GetBinContent(k+1)
    np_Y_toy[k] = Y_toy_dist.GetBinContent(k+1)

    # np_X_toy[k] = x_true
    # np_Y_toy[k] = y_true
#+++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++
for i in range(n_orig_bins):
    #+++++++++++++++++++++++++++++
    for j in range(n_orig_bins):

        #if np_X_smear_matrix[i,j] > 0.0:
        X_smear_matrix.SetBinContent(i+1,j+1,np_X_smear_matrix[i,j])

        #if np_Y_smear_matrix[i,j] > 0.0:
        Y_smear_matrix.SetBinContent(i+1,j+1,np_Y_smear_matrix[i,j])
    #+++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++

np_X_smear = np.matmul(folding_X_matrix,np_X_toy)
np_Y_smear = np.matmul(folding_Y_matrix,np_Y_toy)

dp_x_test = np.concatenate([
    np.reshape(np_X_smear,(np_X_smear.shape[0],1)),
    np.reshape(np_Y_smear,(np_Y_smear.shape[0],1))
],axis=1)

#+++++++++++++++++++++++++++++++++++++++++++
for i in range(n_orig_bins):
    X_smear_dist.SetBinContent(i+1,np_X_smear[i])

    Y_smear_dist.SetBinContent(i+1,np_Y_smear[i])
#+++++++++++++++++++++++++++++++++++++++++++

# toy_dp = TH1F('toy_dp','',n_active_bins,-0.5,n_active_bins+0.5)
# smeared_toy_dp = TH1F('smeared_toy_dp','',n_active_bins,-0.5,n_active_bins+0.5)
# unf_toy_dp = TH1F('unf_toy_dp','',n_active_bins,-0.5,n_active_bins+0.5)




tsvdunf_X = TSVDUnfold(X_smear_dist,X_rec_dist,X_true_dist,X_smear_matrix)
unfresult_X = tsvdunf_X.Unfold( 11 )#2
unfresult_X.SetName("Unfolded_X")
X_bcov = tsvdunf_X.GetBCov()

tsvdunf_Y = TSVDUnfold(Y_smear_dist,Y_rec_dist,Y_true_dist,Y_smear_matrix)
unfresult_Y = tsvdunf_Y.Unfold( 11 )#3
unfresult_Y.SetName("Unfolded_Y")
Y_bcov = tsvdunf_Y.GetBCov()

#++++++++++++++++++++++++++++++++
for i in range(n_orig_bins):
    x_new_err = TMath.Sqrt(X_bcov.GetBinContent(i+1,i+1))
    y_new_err = TMath.Sqrt(Y_bcov.GetBinContent(i+1,i+1))

    unfresult_X.SetBinError(i+1,x_new_err)
    unfresult_Y.SetBinError(i+1,y_new_err)
#++++++++++++++++++++++++++++++++


outf = TFile("bla_xy_flat.root","RECREATE")

X_true_dist.Write()
X_rec_dist.Write()
X_smear_matrix.Write()

X_toy_dist.Write()
X_smear_dist.Write()

unfresult_X.Write()

Y_true_dist.Write()
Y_rec_dist.Write()
Y_smear_matrix.Write()

Y_toy_dist.Write()
Y_smear_dist.Write()

unfresult_Y.Write()

outf.Write()






# mc_true_dist = TH1D('mc_true_dist','',n_active_bins,-0.5,n_active_bins+0.5)
# mc_rec_dist = TH1D('mc_rec_dist','',n_active_bins,-0.5,n_active_bins+0.5)
# smear_matrix = TH2D('smear_matrix','',n_active_bins,-0.5,n_active_bins+0.5,n_active_bins,-0.5,n_active_bins+0.5)

# toy_data = TH1D('toy_data','',n_active_bins,-0.5,n_active_bins+0.5)
# rec_toy_data = TH1D('rec_toy_data','',n_active_bins,-0.5,n_active_bins+0.5)
# err_toy_data = TH2D('err_toy_data','',n_active_bins,-0.5,n_active_bins+0.5,n_active_bins,-0.5,n_active_bins+0.5)

# np_smear_matrix = np.zeros((n_active_bins,n_active_bins))
# np_eff = np.zeros(n_active_bins)
# np_toy_data = np.zeros(n_active_bins)
# np_rec_toy_data = np.zeros(n_active_bins)

# mc_rec_graph = data_graphs[1]
# mc_true_graph = data_graphs[2]
# #+++++++++++++++++++++++++++++
# for p in range(n_active_bins):
#     x_mc_rec = array('d',[0.0])
#     y_mc_rec = array('d',[0.0])

#     x_mc_true = array('d',[0.0])
#     y_mc_true = array('d',[0.0])

#     old_bin = int(rebinned_data[:,9][p])

#     mc_rec_graph.GetPoint(old_bin,x_mc_rec,y_mc_rec)
#     mc_true_graph.GetPoint(old_bin,x_mc_true,y_mc_true)

#     err_mc_true = mc_true_graph.GetErrorY(old_bin)
#     err_mc_rec = mc_rec_graph.GetErrorY(old_bin)

#     mc_true_dist.SetBinContent(p+1,y_mc_true[0])
#     mc_true_dist.SetBinError(p+1,err_mc_true)

#     mc_rec_dist.SetBinContent(p+1,y_mc_rec[0])
#     mc_rec_dist.SetBinError(p+1,err_mc_rec)

#     #+++++++++++++++++++++++++++++
#     for k in range(n_active_bins):
#         old_bin_k = int(rebinned_data[:,9][k])

#         current_content = smear_m.GetBinContent(old_bin+1,old_bin_k+1)

#         if current_content > 0.0:
#            smear_matrix.SetBinContent(p+1,k+1,current_content) 
#         else:
#            smear_matrix.SetBinContent(p+1,k+1,10.0)

#         np_smear_matrix[p][k] = current_content
#     #+++++++++++++++++++++++++++++

#     if y_mc_true[0] > 0.0:
#         np_eff[p] = y_mc_rec[0] / y_mc_true[0]

#     np_toy_data[p] = y_mc_true[0]
# #+++++++++++++++++++++++++++++

# row_sum = np.sum(np_smear_matrix,1)

# folding_matrix = np.where(row_sum==0.0,0.0,np_smear_matrix/row_sum)

# err_folding_matrix = np.zeros_like(folding_matrix)

# #+++++++++++++++++++++++++++++++++++++++++++
# for k in range(folding_matrix.shape[0]):
#     folding_matrix[k,:] *= np_eff[k]

#     err_folding_matrix[k,:] = folding_matrix[k,:]*(1 - folding_matrix[k,:]) / np_toy_data[p]
# #+++++++++++++++++++++++++++++++++++++++++++


# mc_true_dist.SetLineWidth(2)
# mc_true_dist.SetLineColor(6)
# mc_true_dist.SetMarkerColor(6)

# mc_rec_dist.SetLineWidth(2)
# mc_rec_dist.SetLineColor(4)
# mc_rec_dist.SetMarkerColor(4)

# toy_data.SetLineWidth(2)
# toy_data.SetLineColor(kBlack)
# toy_data.SetMarkerColor(kBlack)

# rec_toy_data.SetLineWidth(2)
# rec_toy_data.SetLineColor(kRed)
# rec_toy_data.SetMarkerColor(kRed)

# tsvdunf = TSVDUnfold(rec_toy_data,err_toy_data,mc_rec_dist,mc_true_dist,smear_matrix)
# #tsvdunf.SetNormalize(False)

# unfresult = tsvdunf.Unfold( 2 )

# unfresult.SetLineColor(8)

# # ddist = tsvdunf.GetD()



# outf = TFile("bla.root","RECREATE")

# mc_true_dist.Write()
# mc_rec_dist.Write()
# toy_data.Write()
# rec_toy_data.Write()
# unfresult.Write()
# #ddist.Write()
# outf.Write()










