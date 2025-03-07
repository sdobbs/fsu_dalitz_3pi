import math
import numpy as np
import random
import ROOT
from ROOT import TFile, TGraphErrors,TH2F
from array import array
import copy

class ROOT_Data_Reader(object):

    def __init__(self):
          self.data = []

    #Get the root file:  
    #*************************************************
    def get_root_file(self,fullFilePath):
        fullName = fullFilePath + '.root'
        rootFile = ROOT.TFile.Open(fullName)

        return rootFile
    #*************************************************

    #Load single graph from file:
    #*************************************************
    def get_graph_from_file(self,yourROOTFile,yourGraphName):
        if yourROOTFile is None or yourGraphName is None or yourGraphName == "" or yourGraphName == " ":
            return None
        else:
           graph = yourROOTFile.Get(yourGraphName)
           return graph
    #*************************************************

    #Get the graphs:
    #*************************************************
    def get_data_graphs(self,yourROOTFile,yourGraphNames):
        data_graph = self.get_graph_from_file(yourROOTFile,yourGraphNames['data_graph'])
        mc_rec_graph = self.get_graph_from_file(yourROOTFile,yourGraphNames['mc_rec_graph'])
        mc_true_graph = self.get_graph_from_file(yourROOTFile,yourGraphNames['mc_true_graph'])
        pipig_bkg_graph = self.get_graph_from_file(yourROOTFile,yourGraphNames['pipig_bkg_graph'])

        out_data = [data_graph,mc_rec_graph,mc_true_graph]
        if pipig_bkg_graph is not None:
            out_data.append(pipig_bkg_graph)

        return out_data
    #*************************************************

    #Get the number of DP bins, based on the file name:
    #*************************************************
    def get_n_dp_bins_from_name(self,file_name,dp_bins=[9,10,11,12,13]):
        n_bins = 0
        #++++++++++++++++++++++ 
        for b in dp_bins:
            name="nbins" + str(b)
            if name in file_name:
                n_bins = b
        #++++++++++++++++++++++ 

        return n_bins
    #*************************************************

    #Calculate the efficiency error accroding to:
    #"Treatment of Errors in Efficiency Caclulations" by T. Ullrich and Z. Xu
    #arXiv:physics/70701199v2
    #*************************************************
    def get_efficiency_error(self,n_rec,n_gen,neg_variance_val):
        nom_1 = (n_rec + 1.0)*(n_rec + 2.0)
        nom_2 = (n_rec + 1.0)*(n_rec + 1.0)

        denom_1 = (n_gen + 2.0)*(n_gen + 3.0)
        denom_2 = (n_gen + 2.0)*(n_gen + 2.0)

        variance = nom_1/denom_1 - nom_2/denom_2
        if variance >= 0.0:
          return math.sqrt(variance)
        else:
          return neg_variance_val
    #*************************************************

    #Calculate the efficiency corrected eta-yields
    #*************************************************
    #Get eta-yields from existing graphs:
    def get_eta_yields(self,yourAnaGraphs,yourAccGraph,divide_by_eff,kin_acc_cut,pipig_corr,neg_variance_val):
        data_graph = yourAnaGraphs[0]
        mc_rec_graph = yourAnaGraphs[1]
        mc_true_graph = yourAnaGraphs[2]
        pipig_bkg_graph = None

        if pipig_corr != 0.0 and len(yourAnaGraphs) == 4:
            pipig_bkg_graph = yourAnaGraphs[3]

        n_dp_points = data_graph.GetN()
        eta_yields = []
        error_eta_yields = []
        rec_efficiency = []
        error_rec_efficiency = []
        eta_acceptance = []
        eta_yields_raw = []
        error_eta_yields_raw = []
        #++++++++++++++++++++++++++++++++++++
        for p in range(n_dp_points):
            x_data = array('d',[0.0])
            y_data = array('d',[0.0])
            
            x_mc_rec = array('d',[0.0])
            y_mc_rec = array('d',[0.0])

            x_mc_true = array('d',[0.0])
            y_mc_true = array('d',[0.0])

            x_pipig_bkg = array('d',[0.0])
            y_pipig_bkg = array('d',[0.0])

            x_acc = array('d',[0.0])
            y_acc = array('d',[0.0])

            data_graph.GetPoint(p,x_data,y_data)
            mc_rec_graph.GetPoint(p,x_mc_rec,y_mc_rec)
            mc_true_graph.GetPoint(p,x_mc_true,y_mc_true)
            yourAccGraph.GetPoint(p,x_acc,y_acc)
            dy_pipig_bkg = 0.0

            if pipig_bkg_graph is not None:
                pipig_bkg_graph.GetPoint(p,x_pipig_bkg,y_pipig_bkg)
                dy_pipig_bkg = pipig_bkg_graph.GetErrorY(p)

            N_eta = 0.0
            DN_eta = 0.0
            N_eta_raw = 0.0
            DN_eta_raw = 0.0
           
            efficiency = 0.0
            err_efficiency = 0.0
            #------------------------------------

            if y_mc_true[0] > 0.0:
                efficiency = y_mc_rec[0] / y_mc_true[0]
                err_efficiency = self.get_efficiency_error(y_mc_rec[0],y_mc_true[0],neg_variance_val)
                err_arg1 = data_graph.GetErrorY(p)

                #------------------------------------
                if efficiency > 0.0:
                    if divide_by_eff:
                       N_eta = (y_data[0] - pipig_corr*y_pipig_bkg[0]) / efficiency
                       err_arg2 = err_efficiency*N_eta
                
                       DN_eta_s = (err_arg1*err_arg1 + pipig_corr*pipig_corr*dy_pipig_bkg*dy_pipig_bkg + err_arg2*err_arg2) / (efficiency**2)
                       DN_eta = math.sqrt(DN_eta_s)
                    else:
                       N_eta = y_data[0]
                       DN_eta = err_arg1
                #------------------------------------
 
                N_eta_raw = (y_data[0] - pipig_corr*y_pipig_bkg[0])
                DN_eta_raw = math.sqrt((err_arg1*err_arg1 + pipig_corr*pipig_corr*dy_pipig_bkg*dy_pipig_bkg))
            #------------------------------------

            if y_acc[0] > kin_acc_cut:
              eta_yields.append(N_eta)
              error_eta_yields.append(DN_eta)      

              eta_yields_raw.append(N_eta_raw)
              error_eta_yields_raw.append(DN_eta_raw)
            else:
              eta_yields.append(0.0)
              error_eta_yields.append(0.0)

              eta_yields_raw.append(0.0)
              error_eta_yields_raw.append(0.0)

            rec_efficiency.append(efficiency)
            error_rec_efficiency.append(err_efficiency) 

            eta_acceptance.append(y_acc[0])
        #++++++++++++++++++++++++++++++++++++

        return [eta_yields,error_eta_yields,rec_efficiency,error_rec_efficiency,eta_acceptance,eta_yields_raw,error_eta_yields_raw]

    #----------------------------------------

    #Combine everything:
    def get_DP_data(self,yourFileDict,yourGraphDict,yourNDPBins,DP_Range=[-1.1,1.1],divide_by_eff=True,kin_acc_cut=0.8,pipig_corr=0.0,neg_variance_val=0.0000001):
        DP_data_file = self.get_root_file(yourFileDict['DP_Data_File'])
        data_graphs = self.get_data_graphs(DP_data_file,yourGraphDict)

        acc_data_file = self.get_root_file(yourFileDict['DP_Acc_File'])
        acc_graph = self.get_graph_from_file(acc_data_file,yourGraphDict['acc_graph'])

        N_eta, DN_eta, efficiency, err_efficiency, eta_acc, N_eta_raw, DN_eta_raw = self.get_eta_yields(data_graphs,acc_graph,divide_by_eff,kin_acc_cut,pipig_corr,neg_variance_val)

        #Handle binning properly:
        #----------------------------------------------------
        DP_width = DP_Range[1] - DP_Range[0]

        bin_width = DP_width / float(yourNDPBins)
        DP_X_binned = np.arange(DP_Range[0],DP_Range[1],bin_width) 
        DP_X_binned = np.append(DP_X_binned,DP_Range[1])
        DP_Y_binned = np.arange(DP_Range[0],DP_Range[1],bin_width)
        DP_Y_binned = np.append(DP_Y_binned,DP_Range[1])

        DP_X = (DP_X_binned[:-1] + DP_X_binned[1:]) / float(2)
        DP_Y = (DP_Y_binned[:-1] + DP_Y_binned[1:]) / float(2)
        #----------------------------------------------------

        DP_DATA = []
        bins = []
        #+++++++++++++++++++++++++++++++++++
        for i in range(yourNDPBins):
            #+++++++++++++++++++++++++++++++++++
            for j in range(yourNDPBins):
                gbin = i + yourNDPBins*j
                bins.append(gbin)

                data = [DP_X[i],DP_Y[j],N_eta[gbin],DN_eta[gbin],gbin,eta_acc[gbin],efficiency[gbin],err_efficiency[gbin],N_eta_raw[gbin],DN_eta_raw[gbin]] 
                DP_DATA.append(data)
            #+++++++++++++++++++++++++++++++++++
        #+++++++++++++++++++++++++++++++++++

        #Sort the data for convenience
        DP_DATA = [x for _,x in sorted(zip(bins,DP_DATA)) ]

        return np.array(DP_DATA)
    #*************************************************

    #Calculate the efficiency matrix: 
    #*************************************************
    def calc_eff_matrix(self,yourFileDict,yourGraphDict,eff_m_fileName,ana_name):
        DP_data_file = self.get_root_file(yourFileDict['DP_Data_File'])
        data_graphs = self.get_data_graphs(DP_data_file,yourGraphDict)

        r_m_hist_name = "R_Matrix_" + ana_name
        dr_m_hist_name = "dR_Matrix_" + ana_name

        R_Data_File = self.get_root_file(eff_m_fileName)
        R_Matrix_Hist = self.get_graph_from_file(R_Data_File,r_m_hist_name)
        dR_Matrix_Hist = self.get_graph_from_file(R_Data_File ,dr_m_hist_name)
        
        mc_rec_graph = data_graphs[1]
        mc_true_graph = data_graphs[2]

        n_dp_points = mc_rec_graph.GetN()

        eff_M = np.zeros((n_dp_points,n_dp_points))
        d_eff_M = np.zeros((n_dp_points,n_dp_points))
        #++++++++++++++++++++++++++++++++++++
        for i in range(n_dp_points):
            x_mc_rec = array('d',[0.0])
            y_mc_rec = array('d',[0.0])
            mc_rec_graph.GetPoint(i,x_mc_rec,y_mc_rec)
            
            #++++++++++++++++++++++++++++++++++++
            for j in range(n_dp_points):
                x_mc_true = array('d',[0.0])
                y_mc_true = array('d',[0.0])

                mc_true_graph.GetPoint(j,x_mc_true,y_mc_true)
                
                if y_mc_true[0] > 0.0:
                    epsilon_ij_tilde = y_mc_rec[0] / y_mc_true[0]
                    d_epsilon_ij_tilde = self.get_efficiency_error(y_mc_rec[0],y_mc_true[0],0.000000001)
               
                    r_ij = R_Matrix_Hist.GetBinContent(i+1,j+1)
                    dr_ij = dR_Matrix_Hist.GetBinContent(i+1,j+1)

                    # eff_M[i][j] = epsilon_ij_tilde * r_ij
                    # d_eff_M[i][j] = math.sqrt(r_ij*r_ij*d_epsilon_ij_tilde*d_epsilon_ij_tilde + epsilon_ij_tilde*epsilon_ij_tilde*dr_ij*dr_ij)
                    eff_M[i][j] = r_ij
                    d_eff_M[i][j] = math.sqrt(dr_ij*dr_ij)
            #++++++++++++++++++++++++++++++++++++

        #++++++++++++++++++++++++++++++++++++

        return eff_M, d_eff_M
    #*************************************************