#!/usr/bin/env python3

import sys
import numpy as np
from READ_ROOT_DATA.root_data_reader import ROOT_Data_Reader
from ROOT import TH1D,TH2D,TCanvas,kBlack,kRed,TSVDUnfold,TFile,TMath
from array import array
import matplotlib.pyplot as plt
from UTILS.chisquare_fitter import ChiSquare_Fitter
from UTILS.analysis_utils import DP_Analysis_Utils
from scipy import ndimage
import math

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

smear_m_file = reader.get_root_file(r_m_file_dict[data_set] + r_matrix_name)
smear_m = reader.get_graph_from_file(smear_m_file,'R_Matrix_' + ana_name)

npy_dp_data = reader.get_DP_data(filesDict,graphsDict,11)
rebinned_data = ana_utils.rebin_DP(npy_dp_data,include_raw_yields=True)

n_active_bins = rebinned_data.shape[0]
n_dp_bins = npy_dp_data.shape[0]

mc_true_dist = TH1D('mc_true_dist','',n_active_bins,-0.5,n_active_bins+0.5)
mc_rec_dist = TH1D('mc_rec_dist','',n_active_bins,-0.5,n_active_bins+0.5)
smear_matrix = TH2D('smear_matrix','',n_active_bins,-0.5,n_active_bins+0.5,n_active_bins,-0.5,n_active_bins+0.5)

toy_data = TH1D('toy_data','',n_active_bins,-0.5,n_active_bins+0.5)
smeared_toy_data = TH1D('smeared_toy_data','',n_active_bins,-0.5,n_active_bins+0.5)

#np_toy_data = np.zeros(n_active_bins)

np_toy_data = init_fitter.DP_Fit_Function(rebinned_data[:,[0,1]],100000,-1.095,0.145,0.0,0.081,0.0,0.141,-0.044,0.0,0.0)

np_smear_matrix = np.zeros((n_active_bins,n_active_bins))

mc_rec_graph = data_graphs[1]
mc_true_graph = data_graphs[2]
#+++++++++++++++++++++++++++++
for p in range(n_active_bins):
    x_mc_rec = array('d',[0.0])
    y_mc_rec = array('d',[0.0])

    x_mc_true = array('d',[0.0])
    y_mc_true = array('d',[0.0])

    x_acc = array('d',[0.0])
    y_acc = array('d',[0.0])

    old_bin = int(rebinned_data[:,9][p])

    mc_rec_graph.GetPoint(old_bin,x_mc_rec,y_mc_rec)
    mc_true_graph.GetPoint(old_bin,x_mc_true,y_mc_true)
    acc_graph.GetPoint(old_bin,x_acc,y_acc)

    mc_true_dist.SetBinContent(p+1,y_mc_true[0])
    mc_rec_dist.SetBinContent(p+1,y_mc_rec[0])


    #+++++++++++++++++++++++++++++
    for k in range(n_active_bins):
        old_bin_k = int(rebinned_data[:,9][k])

        current_content = smear_m.GetBinContent(old_bin+1,old_bin_k+1)
        smear_matrix.SetBinContent(p+1,k+1,current_content) 
        np_smear_matrix[p][k] = current_content
    #+++++++++++++++++++++++++++++
    
    toy_data.SetBinContent(p+1,np_toy_data[p])
    #np_toy_data[p] = y_mc_true[0]
#+++++++++++++++++++++++++++++

row_sum = np.sum(np_smear_matrix,1)
folding_matrix = np.where(row_sum==0.0,0.0,np_smear_matrix/row_sum)

#++++++++++++++++++++++++++
for k in range(np_smear_matrix.shape[0]):
    y_true = mc_true_dist.GetBinContent(k+1)
    y_rec = mc_rec_dist.GetBinContent(k+1)

    eff = 0.0
    if y_true > 0.0:
        eff = y_rec / y_true
        folding_matrix[k,:] *= eff
#++++++++++++++++++++++++++


np_smeared_toy_data = np.matmul(folding_matrix,np_toy_data)
#++++++++++++++++++++++++++
for k in range(np_smear_matrix.shape[0]):
   smeared_toy_data.SetBinContent(k+1,np_smeared_toy_data[k])
#++++++++++++++++++++++++++

tsvdunf = TSVDUnfold(smeared_toy_data,mc_rec_dist,mc_true_dist,smear_matrix)
unfolded_toy_data = tsvdunf.Unfold( 3 )
unfolded_toy_data.SetName('unfolded_toy_data')

# filtered_unf_toy_data = TH1D('filtered_unf_toy_data','',n_active_bins,-0.5,n_active_bins+0.5)

# def get_gbin(x,y,nBins,min_x,max_x):
#     dp_delta = (max_x - min_x) / float(nBins)

#     gbin = math.floor((x + 1.1) / dp_delta) + nBins * math.floor((y + 1.1) / dp_delta)
#     return gbin

# bin_collection = np.zeros(n_dp_bins)    

# x_values = np.zeros(n_dp_bins) 

# #++++++++++++++++++++++++
# for p in range(n_active_bins):
#     counts = unfolded_toy_data.GetBinContent(p+1)

#     x = rebinned_data[:,0][p]
#     y = rebinned_data[:,1][p]

#     g_bin = get_gbin(x,y,11,-1.1,1.1)

#     #old_bin = int(rebinned_data[:,9][p])
#     bin_collection[g_bin] = counts
#     x_values[g_bin] = g_bin
# #++++++++++++++++++++++++

# plt.plot(x_values,bin_collection,'ko')
# plt.show()






outf = TFile("bla.root","RECREATE")

mc_true_dist.Write()
mc_rec_dist.Write()
toy_data.Write()
smeared_toy_data.Write()
unfolded_toy_data.Write()

outf.Write()


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










