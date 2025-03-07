#!/usr/bin/env python3

import sys
import numpy as np
from READ_ROOT_DATA.root_data_reader import ROOT_Data_Reader
from UTILS.analysis_utils import DP_Analysis_Utils

#Basic definitions:
#**********************************************
data_set = int(sys.argv[1])
fileInitName = sys.argv[2]
ana_name = sys.argv[3]
dp_acc_cut = 0.8
outfile_add_name = '.npy'

#-------------------------------
if len(sys.argv) > 4:
      dp_acc_cut = float(sys.argv[4])
      outfile_add_name = '_no_acc_cut_norm.npy'
# #-------------------------------

# r_matrix_name = None
# #-------------------------------
# if len(sys.argv) > 4:
#       r_matrix_name = sys.argv[4]
# #-------------------------------

reader = ROOT_Data_Reader()
ana_utils = DP_Analysis_Utils()

data_dict = {
   0: ['17','2017','GlueX 2017','GlueX-2017'],
   1: ['18S','2018S','GlueX 2018-01','GlueX-2018-01'],
   2: ['18F','2018F','GlueX 2018-08','GlueX-2018-08']
}

r_m_file_dict = {
      0: "/Users/daniellersch/Desktop/eta3Pi_DalitzAna/raw_dp_root_data/mc_hists/"
}


addName = data_dict[data_set][1] + '_' + ana_name
rootDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/root_dalitz_data'
npyDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/npy_dalitz_data'

dataFileName = fileInitName + addName
n_DP_bins = reader.get_n_dp_bins_from_name(dataFileName)
accFileName = 'kinematic_acceptance_ratio' + str(n_DP_bins)
outFileName = dataFileName + outfile_add_name
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

#Load the ROOT data
#**********************************************
print("Load ROOT data and translate it to a numpy array...")

npy_dp_data = reader.get_DP_data(filesDict,graphsDict,n_DP_bins,kin_acc_cut=dp_acc_cut)

npy_dp_data = ana_utils.normalize_Dalitz_Data(npy_dp_data)

print("...done!")
print("  ")
#**********************************************

#**********************************************
print("Write numpy array to: " + fullSaveName + "...")

np.save(fullSaveName,npy_dp_data)

print("...done! Have a great day!")
print("  ")
#**********************************************

#if r_matrix_name is not None:
#
#    m_file_name = r_m_file_dict[data_set] + r_matrix_name
#    #**********************************************
#    print("Calculate efficiency matrices...")
#
#    eff_M, d_eff_M = reader.calc_eff_matrix(filesDict,graphsDict,m_file_name,ana_name)
#
#    print("...done!")
#    print(" ")
#    #**********************************************
#
#    #**********************************************
#    print("Write effiency matrix to file...")
#
#    eff_out_name = npyDir + '/' + dataFileName + '_effMatrix_r.npy'
#    d_eff_out_name = npyDir + '/' + dataFileName + '_deffMatrix_r.npy'
#
#    print("   ")
#    print("   >>> " + eff_out_name + " <<<")
#    np.save(eff_out_name,eff_M)
#    print("   ")
#    print("   >>> " + d_eff_out_name + " <<<")
#    np.save(d_eff_out_name,d_eff_M)
#
#
#    print("...done!")
#    print(" ")
#    #**********************************************

