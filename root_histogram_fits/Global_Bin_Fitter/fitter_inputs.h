#include "TFile.h"
#include "TH2.h"
#include "TH1.h"

TH2F *input_Data_noAcc;//no acceptance cut applied
TH2F *input_Data_Acc;//acceptance cut applied
TH2F *input_MC_Rec;
TH2F *input_MC_True;
TH2F *input_misc; //Whatever you want ot monitor during you scan...

//Load a file:
//*********************************************************************************
TFile *getFile(TString fileName){
    TFile *out = 0;
    out = TFile::Open(fileName);
    return out;
}
//*********************************************************************************

//Get histograms:
//*********************************************************************************
//Get 2D histograms:
TH2F *getTwoDHistogram(TFile *inFile, TString histname){
    TH2F *out = (TH2F*)inFile->Get(histname);
    return out;
}

//-----------------------------------------------------------------

//Get 1D histograms:
TH1F *getOneDHistogram(TFile *inFile, TString histname){
    TH1F *out = (TH1F*)inFile->Get(histname);
    return out;
}
//*********************************************************************************

//Set the input histograms:
//*********************************************************************************
void set_input_histograms(TFile **input_files,TString *input_names){
    input_Data_noAcc = getTwoDHistogram(input_files[0],input_names[0]);
    input_Data_Acc = getTwoDHistogram(input_files[1],input_names[1]);
    input_MC_Rec = getTwoDHistogram(input_files[2],input_names[2]);
    input_MC_True = getTwoDHistogram(input_files[3],input_names[3]);
    input_misc = getTwoDHistogram(input_files[4],input_names[4]);
}
//*********************************************************************************
