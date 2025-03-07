#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TF1.h"
#include "TMath.h"
#include "TAxis.h"
#include "TGraphErrors.h"

//Handle the axis setting properly:
//*********************************************************************************
void setAxis(TAxis *yourAxis, TString yourTitle, Double_t yourOffset, Double_t yourSize,Double_t yourLabelSize){
    yourAxis->SetTitle(yourTitle);
    yourAxis->SetTitleOffset(yourOffset);
    yourAxis->SetTitleSize(yourSize);
    yourAxis->SetLabelSize(yourLabelSize);
}
//*********************************************************************************

//Get 1D projection:
//*********************************************************************************
TH1F *getOneDProjection(TH2F *inHist,TString proName,Double_t left_val,Double_t right_val,Int_t axis){
    Int_t left_bin,right_bin;
    
    if(axis == 0){
        left_bin = inHist->GetXaxis()->FindFixBin(left_val);
        right_bin = inHist->GetXaxis()->FindFixBin(right_val);
        TH1F *out = (TH1F*)inHist->ProjectionY(proName,left_bin,right_bin,"");
        
        return out;
    }else{
        left_bin = inHist->GetYaxis()->FindFixBin(left_val);
        right_bin = inHist->GetYaxis()->FindFixBin(right_val);
        TH1F *out = (TH1F*)inHist->ProjectionX(proName,left_bin,right_bin,"");
        
        return out;
    }
}
//*********************************************************************************

//Determine chi2 to compare two distributions (e.g. measured vs. fitted histogram)
//This function is actually implemented in root...
//*********************************************************************************
Double_t get_matchChiSquare(TH1F *histData, TH1F *histFit,Int_t left_bin,Int_t right_bin,Bool_t use_per_ndf){
    Double_t chiSquare = 0.0;
    Double_t counts_data,counts_fit;
    Double_t error = 0.0;
    Double_t arg = 0.0;
    Int_t n_fitted_points = 0;
    
    //++++++++++++++++++++++++++++++++++++++
    for(Int_t p=left_bin;p<=right_bin;p++){
        error = histData->GetBinError(p);
        
        if(error > 0.0){
            counts_data = histData->GetBinContent(p);
            counts_fit = histFit->GetBinContent(p);
            arg = (counts_data - counts_fit) / error;
            chiSquare += arg*arg;
            
            n_fitted_points ++;
        }
    }
    //++++++++++++++++++++++++++++++++++++++
    
    if(use_per_ndf){
        return chiSquare*TMath::Power(n_fitted_points,-1);
    }else return chiSquare;
}
//*********************************************************************************

//Get integral boarders from histogram:
//*********************************************************************************
Int_t *get_integration_bins(TH1F *histData,Double_t center_value,Double_t percentage){
    Int_t *integration_bins = 0;
    integration_bins = new Int_t[2];
    
    Int_t center_bin = histData->GetXaxis()->FindFixBin(center_value);
    Double_t height = histData->GetBinContent(center_bin);
    Double_t left_height = height;
    Double_t right_height = height;
    
    Int_t left_bin,right_bin;
    left_bin = right_bin = center_bin;
    
    while(left_height > 0.5*height){
        left_bin += -1;
        left_height = histData->GetBinContent(left_bin);
    }
    
    while(right_height > 0.5*height){
        right_bin += 1;
        right_height = histData->GetBinContent(right_bin);
    }
    
    Double_t fwhm = histData->GetXaxis()->GetBinCenter(right_bin) - histData->GetXaxis()->GetBinCenter(left_bin);
    Double_t sigma = fwhm / 2.355;
    
    integration_bins[0] = histData->GetXaxis()->FindFixBin(center_value - sigma*percentage);
    integration_bins[1] = histData->GetXaxis()->FindFixBin(center_value + sigma*percentage);

    return integration_bins;
}
//*********************************************************************************

//Get the peak integral (but also applies background correction to your data histogram!!!)
//*********************************************************************************
Double_t *get_integral(TH1F *histData,TH1F *histBkg,Int_t left_int,Int_t right_int,Bool_t is_fitted_peak,Bool_t is_mc_dist){
    Double_t *results = 0;
    results = new Double_t [2];
    
    //----------------------------------------
    if(is_mc_dist){
        results[0] = histData->Integral(left_int,right_int);
        results[1] = TMath::Sqrt(results[0]);
    }else{
        Double_t int_error = histData->Integral(left_int,right_int) + histBkg->Integral(left_int,right_int);
        Double_t int_value = 0.0;
        
        if(is_fitted_peak){
            int_value = histData->Integral(left_int,right_int);
        }else{
            histData->Add(histBkg,-1);
            int_value = histData->Integral(left_int,right_int);
        }
        
        results[0] = int_value;
        results[1] = TMath::Sqrt(int_error);
    }
    //----------------------------------------
    
    return results;
}
//*********************************************************************************

//Scale two histograms (assuming the have the same binning!):
//*********************************************************************************
TH1F *get_scaled_histogram(TH1F *hist_to_scale,TH1F *ref_hist,Double_t center_value){
    Int_t center_bin = hist_to_scale->GetXaxis()->FindFixBin(center_value);
    
    Double_t current_height = hist_to_scale->GetBinContent(center_bin);
    Double_t new_height = ref_hist->GetBinContent(center_bin);
    
    TH1F *scaled_hist = (TH1F*)hist_to_scale->Clone();
    scaled_hist->Scale(new_height*TMath::Power(current_height,-1));
    
    return scaled_hist;
}
//*********************************************************************************

//Handle accidental subtraction:
//***************************************************************************************************
TH2F *get_h2f_accs(TH2F* hist_noCut,TH2F *hist_Cut,Int_t n_bunches,TString hist_name){
    Double_t scale = 1.0 / (2.0*n_bunches);
    
    hist_noCut->Add(hist_Cut,-1);
    hist_noCut->Scale(scale);
    hist_Cut->Add(hist_noCut,-1);
    
    Double_t x_min = hist_Cut->GetXaxis()->GetXmin();
    Double_t x_max = hist_Cut->GetXaxis()->GetXmax();
    Int_t nbins_x = hist_Cut->GetNbinsX();
    
    Double_t y_min = hist_Cut->GetYaxis()->GetXmin();
    Double_t y_max = hist_Cut->GetYaxis()->GetXmax();
    Int_t nbins_y = hist_Cut->GetNbinsY();
    
    TH2F *out_hist = new TH2F(hist_name,"",nbins_x,x_min,x_max,nbins_y,y_min,y_max);
    Double_t content = 0.0;
    //+++++++++++++++++++++++++++++++++
    for(Int_t i=1;i<=nbins_x;i++){
        //+++++++++++++++++++++++++++++++++
        for(Int_t j=1;j<=nbins_y;j++){
            content = hist_Cut->GetBinContent(i,j);
            if(content > 0.0)out_hist->SetBinContent(i,j,content);
        }
        //+++++++++++++++++++++++++++++++++
    }
    //+++++++++++++++++++++++++++++++++
    
    return out_hist;
}
//***************************************************************************************************

//Get integral from a 2D histogram:
//***************************************************************************************************
Double_t getInt2dHist(TH2F *inHist){
    Int_t nBinsX = inHist->GetNbinsX();
    Int_t nBinsY = inHist->GetNbinsY();
    
    Double_t totalContent = 0.0;
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(Int_t i=0;i<nBinsX;i++){
        for(Int_t j=0;j<nBinsY;j++){
            totalContent += inHist->GetBinContent(i+1,j+1);
        }
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    return totalContent;
}
//***************************************************************************************************

//Apply a simple threshold cut on a given quantity (preferably the true histogram...)
//***************************************************************************************************
bool accept_current_event(Double_t value, Double_t threshold){
    if(value > threshold){
        return true;
    }else return false;
}
//***************************************************************************************************

//Normalize yields (within a graph by the initial number of events):
//***************************************************************************************************
void normalize_yields(TGraphErrors *yield_gr,TH2F *input_histogram){
    Double_t x,y,dy;
    Int_t n_points = yield_gr->GetN();
    Double_t integral = input_histogram->Integral();
    Double_t norm = TMath::Power(integral,-1);
    
    //+++++++++++++++++++++++++++++++++++++
    for(Int_t p=0;p<n_points;p++){
        yield_gr->GetPoint(p,x,y);
        dy = yield_gr->GetErrorY(p);
        
        yield_gr->SetPoint(p,x,y*norm);
        yield_gr->SetPointError(p,0.0,dy*norm);
    }
    //+++++++++++++++++++++++++++++++++++++
}
//***************************************************************************************************

