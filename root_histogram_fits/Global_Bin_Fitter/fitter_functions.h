#include "TMath.h"
#include "TF1.h"
#include "TH1.h"

//Function parameters that shall NOT be touched!
//*********************************************************************************
Double_t left_reject_point = 0.0;
Double_t right_reject_point = 0.0;

const int bkg_pol_order = 3;
const int n_pars_signal = 5;
const int n_pars_background = 1 + bkg_pol_order;
//*********************************************************************************


//Fit the background with/without rejecting the peak region:
//Use exp(pol(N)) function for the eta-background, because background is mainly exponential...
//*********************************************************************************
//Core function: exp(pol(N))
Double_t Bkg_coreFunction(Double_t *x, Double_t *par){
    Double_t fpol = 0.0;
    //+++++++++++++++++++++++++++++++++++
    for(Int_t i=0;i<n_pars_background;i++){
        fpol += par[i]*pow(x[0],i);
    }
    //+++++++++++++++++++++++++++++++++++
    Double_t arg = TMath::Exp(fpol);
   
    //Want to ensure convergence + your signal should not have negative entries....
    if(arg >= 0.0){
        return arg;
    }else return 0.0;
}

//-----------------------------------------------------------------

//Reject the signal region:
Double_t Bkg_coreFunction_reject(Double_t *x, Double_t *par){
    Double_t response = Bkg_coreFunction(x,par);
    
    if(x[0] > left_reject_point && x[0] < right_reject_point){
        TF1::RejectPoint();
        return 0;
    }else return response;
}
//*********************************************************************************


//Fit the signal: Gauss + exp. tails:
//*********************************************************************************
Double_t Sig_coreFunction(Double_t *x, Double_t *par){
    Double_t val = 0.0;
    Double_t scale = par[0];
    Double_t mean = par[1];
    Double_t sigma = par[2];
    Double_t kL = par[3];
    Double_t kR = par[4];
    
    if(sigma > 0.0){
        Double_t arg = (x[0] - mean)/sigma;
        
            //----------------------------------------
            if(arg < -kL){
                val = TMath::Exp(0.5*kL*kL + kL*arg);
            }else if(arg >= -kL && arg < kR){
                val = TMath::Exp(-0.5*arg*arg);
            }else if(arg >= kR){
                val = TMath::Exp(0.5*kR*kR - kR*arg);
            }
            //----------------------------------------
        return scale*val;
    }else return 0.0;
}

//-----------------------------------------------------------------

//Just add an offset to the signal, so you can perform a fit to the peak region...
Double_t Sig_coreFunction_offset(Double_t *x, Double_t *par){
    Double_t response = Sig_coreFunction(x,par);
    Double_t offset = par[5];
    
    return response + offset;
}

//-----------------------------------------------------------------

Double_t full_fit(Double_t *x, Double_t *par){
    Double_t out = 0;
    out = Bkg_coreFunction(x,par) + Sig_coreFunction(x,&par[n_pars_background]);
    return out;
}
//*********************************************************************************

//Fit parameters and distributions:
//*********************************************************************************
Double_t bkg_pars[n_pars_background];
Double_t bkg_pars_errs[n_pars_background];
Double_t sig_pars[n_pars_signal];
Double_t sig_pars_errs[n_pars_signal];

TH1F *signal_distribution;
TH1F *background_distribution;
TH1F *full_distribution;

//SETTERS:
void set_bkg_par(Double_t value,Int_t index){
    bkg_pars[index] = value;
}

void set_bkg_err(Double_t value,Int_t index){
    bkg_pars_errs[index] = value;
}

void set_sig_par(Double_t value,Int_t index){
    sig_pars[index] = value;
}

void set_sig_err(Double_t value,Int_t index){
    sig_pars_errs[index] = value;
}

//GETTERS:
Double_t *get_bkg_pars(){
    return bkg_pars;
}

Double_t *get_bkg_pars_errs(){
    return bkg_pars_errs;
}

Double_t *get_sig_pars(){
    return sig_pars;
}

Double_t *get_sig_pars_errs(){
    return sig_pars_errs;
}
//*********************************************************************************


//Now run the fit itself:
//*********************************************************************************
Double_t fit_histogram(TH1F *histData,Double_t *init_reject_boarders,Double_t *fit_boarders,Double_t *signal_init_pars,Double_t scan_step_size,Int_t n_fit_iterations,Bool_t fix_bkg_pars,Bool_t fix_sig_pars,Bool_t fit_bkg_only,TString *fit_options,Bool_t show_ndf){
    Double_t currentChi2 = 0.0;
    Double_t chi2Limit = 10000000.0;
    Int_t ndf;
    
    //++++++++++++++++++++++++++++++++++++++++++++++++
    for(Int_t a=0;a<n_fit_iterations;a++){
        left_reject_point = init_reject_boarders[0] + a*scan_step_size;
        
        //++++++++++++++++++++++++++++++++++++++++++++++++
        for(Int_t b=0;b<n_fit_iterations;b++){
            right_reject_point = init_reject_boarders[1] + b*scan_step_size;
            
            //Fit the background first:
            //=========================================
            TF1 *backgroundFit = new TF1("backgroundFit",Bkg_coreFunction_reject,fit_boarders[0],fit_boarders[1],n_pars_background);
            
            //++++++++++++++++++++++++++++++++++++++++
            for(Int_t h=0;h<n_pars_background;h++){
                backgroundFit->SetParameter(h,1.0);
            }
            //++++++++++++++++++++++++++++++++++++++++
            
            histData->Fit("backgroundFit",fit_options[0]);
            //=========================================
            
            //---------------------------------------------
            if(!fit_bkg_only){
                //Now fit the signal:
                //=========================================
                TF1 *signalFit = new TF1("signalFit",Sig_coreFunction_offset,left_reject_point,right_reject_point,n_pars_signal+1);
                //++++++++++++++++++++++++++++++++++++++++
                for(Int_t i=0;i<n_pars_signal+1;i++){
                    signalFit->SetParameter(i,signal_init_pars[i]);
                }
                //++++++++++++++++++++++++++++++++++++++++
                
                histData->Fit("signalFit",fit_options[1]);
                //=========================================
                
                //Now run a combined fit:
                //=========================================
                TF1 *fitAll = new TF1("fitAll",full_fit,fit_boarders[0],fit_boarders[1],n_pars_signal+n_pars_background);
                
                //++++++++++++++++++++++++++++++++++++++++
                for(Int_t h=0;h<n_pars_signal + n_pars_background;h++){
                    //---------------------------
                    if(h < n_pars_background){
                        if(fix_bkg_pars){
                            fitAll->FixParameter(h,backgroundFit->GetParameter(h));
                        }else fitAll->SetParameter(h,backgroundFit->GetParameter(h));
                    }else{
                        if(fix_sig_pars){
                            fitAll->FixParameter(h,signalFit->GetParameter(h-n_pars_background));
                        }else fitAll->SetParameter(h,signalFit->GetParameter(h-n_pars_background));
                    }
                    //---------------------------
                }
                //++++++++++++++++++++++++++++++++++++++++
                
                histData->Fit("fitAll",fit_options[2]);
                //=========================================
                
                //Now choose 'best' fit:
                currentChi2 = fitAll->GetChisquare();
                ndf = fitAll->GetNDF();
                
                //-----------------------------------
                if(currentChi2 < chi2Limit){
                    chi2Limit = currentChi2;
                    
                    //Retreive the fit paramaters:
                    //++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    for(Int_t z=0;z<n_pars_signal + n_pars_background;z++){
                        //---------------------------
                        if(z < n_pars_background){
                            set_bkg_par(fitAll->GetParameter(z),z);
                            set_bkg_err(fitAll->GetParError(z),z);
                        }else{
                            set_sig_par(fitAll->GetParameter(z),z-n_pars_background);
                            set_sig_err(fitAll->GetParError(z),z-n_pars_background);
                        }
                        //---------------------------
                    }
                    //++++++++++++++++++++++++++++++++++++++++++++++++++++++
                }
                //-----------------------------------
                
                signalFit->Clear();
                backgroundFit->Clear();
                fitAll->Clear();
            }else{
                //Now choose 'best' fit:
                currentChi2 = backgroundFit->GetChisquare();
                ndf = backgroundFit->GetNDF();
                
                //-----------------------------------
                if(currentChi2 < chi2Limit){
                    chi2Limit = currentChi2;
                    
                    //Retreive the fit paramaters:
                    //++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    for(Int_t z=0;z<n_pars_background;z++){
                        set_bkg_par(backgroundFit->GetParameter(z),z);
                        set_bkg_err(backgroundFit->GetParError(z),z);
                    }
                    //++++++++++++++++++++++++++++++++++++++++++++++++++++++
                }
                //-----------------------------------
                
                backgroundFit->Clear();
            }
            //---------------------------------------------
        }
        //++++++++++++++++++++++++++++++++++++++++++++++++
    }
    //++++++++++++++++++++++++++++++++++++++++++++++++
    
    //retreive the 'best' chisquare for judging the fit quality
    if(show_ndf){
        return chi2Limit*TMath::Power(ndf,-1);
    }else return chi2Limit;
}
//*********************************************************************************

//Now get the fitted spectra!
//*********************************************************************************
void get_fit_hists(TH1F *histData,Bool_t fit_bkg_only){
    Int_t n_bins = histData->GetNbinsX();
    Double_t min_x = histData->GetXaxis()->GetXmin();
    Double_t max_x = histData->GetXaxis()->GetXmax();

    //Get the background distribution
    TF1 *bkg = new TF1("bkg",Bkg_coreFunction,min_x,max_x,n_pars_background);
    Double_t *p_bkg = get_bkg_pars();
    //+++++++++++++++++++++++++++++++++++++++++
    for(Int_t h=0;h<n_pars_background;h++){
        bkg->FixParameter(h,p_bkg[h]);
    }
    //+++++++++++++++++++++++++++++++++++++++++
    
    bkg->SetNpx(n_bins);
    background_distribution = (TH1F*)bkg->GetHistogram();
    background_distribution->SetName("background_distribution");
    
    //--------------------------------
    if(!fit_bkg_only){
        //Signal distribution
        TF1 *sig = new TF1("sig",Sig_coreFunction,min_x,max_x,n_pars_signal);
        Double_t *p_sig = get_sig_pars();
        //+++++++++++++++++++++++++++++++++++++++++
        for(Int_t h=0;h<n_pars_signal;h++){
            sig->FixParameter(h,p_sig[h]);
        }
        //+++++++++++++++++++++++++++++++++++++++++
        
        sig->SetNpx(n_bins);
        signal_distribution = (TH1F*)sig->GetHistogram();
        signal_distribution->SetName("signal_distribution");
        
        //Full distribution:
        full_distribution = (TH1F*)signal_distribution->Clone();
        full_distribution->Add(background_distribution,1);
        full_distribution->SetName("full_distribution");
        
        sig->Clear();
    }
    //--------------------------------
    
    bkg->Clear();
}
//*********************************************************************************


//Clear the histograms:
//*********************************************************************************
void clear_histograms(){
    signal_distribution->Clear();
    background_distribution->Clear();
    full_distribution->Clear();
}
//*********************************************************************************
