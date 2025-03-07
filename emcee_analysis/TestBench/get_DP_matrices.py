import numpy as np
from ROOT import TFile,TH2D,TH2F
import ROOT
from array import array

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

data_dir = '/Volumes/BunchOfStuff/GlueX_Eta_Data/New_MC2017_BinMig_Smear_Matrix.root'
file = ROOT.TFile.Open(data_dir)

acc_file = ROOT.TFile.Open('/Users/daniellersch/Desktop/eta3Pi_DalitzAna/root_dalitz_data/kinematic_acceptance_ratio11.root')
acc_graph = acc_file.Get('DP_kinAccR')


alt_ana_name = 'nbins11_kfit_cut0_imgg_cut1_zv_cut1'
m_name = 'R_Matrix_' + alt_ana_name


smear_m_h = file.Get(m_name)

smear_matrix = np.zeros((121,121))
dp_acc = np.zeros(121)


for k in range(121):
    x_acc = array('d',[0.0])
    y_acc = array('d',[0.0])
    
    acc_graph.GetPoint(k,x_acc,y_acc)
    dp_acc[k] = y_acc[0]

    for l in range(121):

        content = smear_m_h.GetBinContent(k+1,l+1)
        smear_matrix[k,l] = content


smear_matrix = smear_matrix / np.max(smear_matrix)
dp_fin_acc = np.where(dp_acc>0.8,dp_acc,0.0)


fig,ax = plt.subplots(1,2)

ax[0].imshow(smear_matrix)

ax[1].plot(dp_acc,'ko')
ax[1].plot(dp_fin_acc,'bs')

plt.show()

np.save('raw_semar_matrix.npy',smear_matrix)
np.save('DP_Acc.npy',dp_fin_acc)
np.save('raw_DP_Acc.npy',dp_acc)
