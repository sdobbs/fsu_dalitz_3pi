#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TFile.h"

//All output-graphs should be defined here:
//*********************************************************************************
TGraphErrors *gr_nEvents_Data;
TGraphErrors *gr_nEvents_MC_Rec;
TGraphErrors *gr_nEvents_MC_True;
TGraphErrors *gr_misc; //Fill whatever you want to monitor here....

//Monitoring the fit quality / data - MC agreement:
TGraph *gr_chiSquare_fit;
TGraph *gr_chiSquare_match;
TGraph *gr_mean_fit;
TGraph *gr_sigma_fit;
TGraph *gr_mean_mc;
TGraph *gr_sigma_mc;
//*********************************************************************************

//Initialize the graphs:
//*********************************************************************************
void init_graphs(Int_t n_points){
    gr_nEvents_Data = new TGraphErrors(n_points);
    gr_nEvents_MC_Rec = new TGraphErrors(n_points);
    gr_nEvents_MC_True = new TGraphErrors(n_points);
    gr_misc = new TGraphErrors(n_points);
    
    gr_chiSquare_fit = new TGraph(n_points);
    gr_chiSquare_match = new TGraph(n_points);
    gr_mean_fit = new TGraph(n_points);
    gr_sigma_fit = new TGraph(n_points);
    gr_mean_mc = new TGraph(n_points);
    gr_sigma_mc = new TGraph(n_points);
}
//*********************************************************************************

//Write everything to an output-file:
//*********************************************************************************
void write_out_results(TString outFileName,TString misc_gr_name){
    TFile *outfile = new TFile(outFileName,"RECREATE");
    
    gr_nEvents_Data->Write("gr_nEvents_Data");
    gr_nEvents_MC_Rec->Write("gr_nEvents_MC_Rec");
    gr_nEvents_MC_True->Write("gr_nEvents_MC_True");
    gr_misc->Write(misc_gr_name);
    
    gr_chiSquare_fit->Write("gr_chiSquare_fit");
    gr_chiSquare_match->Write("gr_chiSquare_match");
    gr_mean_fit->Write("gr_mean_fit");
    gr_sigma_fit->Write("gr_sigma_fit");
    gr_mean_mc->Write("gr_mean_mc");
    gr_sigma_mc->Write("gr_sigma_mc");
    
    outfile->Write();
    outfile->Close();
}
//*********************************************************************************

//Show the results:
//*********************************************************************************
void show_results(){
    TCanvas *c_out = new TCanvas("c_out","",1600,800);
    c_out->Divide(3,2);
    c_out->cd(1);
    gr_nEvents_Data->Draw("AP");
    c_out->cd(2);
    gr_nEvents_MC_Rec->Draw("AP");
    c_out->cd(3);
    gr_nEvents_MC_True->Draw("AP");
    c_out->cd(4);
    gr_misc->Draw("AP");
    c_out->cd(5);
    gr_chiSquare_fit->Draw("AP");
    c_out->cd(6);
    gr_chiSquare_match->Draw("AP");
    c_out->cd();
}
//*********************************************************************************
