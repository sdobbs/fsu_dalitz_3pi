#include "TMath.h"
#include "TCanvas.h"
#include "fitter_inputs.h"
#include "fitter_outputs.h"
#include "fitter_functions.h"
#include "fitter_utils.h"
#include "fitter_settings.h"
#include "TGraphErrors.h"
#include "TLine.h"
#include "TSystem.h"
#include "TStyle.h"
#include <iostream>

using namespace std;

//Do some cosmetics on the y-axis:
//x-axis should be handled by the input histograms
Double_t y_axis_offset = 0.85;
Double_t y_axis_size = 0.075;
Double_t y_axis_label_size = 0.05;

//Run the fit procedures:
//*********************************************************************************
Int_t fit_histograms(Int_t nBins,Int_t axis,TH2F *histData,TH2F *histMC_Rec,TH2F *histMC_True,TH2F *hist_misc,TGraphErrors *gr_data,TGraphErrors *gr_mc_rec,TGraphErrors *gr_mc_true,TGraphErrors *gr_misc,TGraph *gr_chiSqaure_fit,TGraph *gr_chiSqaure_match,TGraph *gr_mean_fit,TGraph *gr_sigma_fit,TGraph *gr_mean_mc,TGraph *gr_sigma_mc,Bool_t show_ndf,TString saveName,Bool_t is_auto_mode){
    set_limits(nBins*nBins);
    
    Double_t *minX = get_min_X();
    Double_t *maxX = get_max_X();
    
    Int_t fitter_mode = -1;
    if(is_auto_mode) fitter_mode = 2;
    
    Double_t inspection_value = minX[0];
    
    const int n_bins_to_scan = nBins*nBins;

    //----------------------------------------
    if(fitter_mode == -1){
    
        cout <<"   "<< endl;
        cout <<"Allright! Nearly everything is set up. In which mode do you whish to procede?"<< endl;
        cout <<"0 - Regular analysis mode"<< endl;
        cout <<"1 - Inspection mode, i.e. look at a particular value / bin."<< endl;
        cout <<"2 - Auto mode, i.e. perform all fits without asking for approval (saves time, but might cause a mess)."<< endl;
        cin >> fitter_mode;
    
        if(fitter_mode == 1){
           cout <<"Please choose DP bin you whish to look at"<< endl;
           cout <<"Between "<<minX[0]<<" and "<<maxX[n_bins_to_scan-1]<< endl;
           cin >> inspection_value;
           cout <<"     "<< endl;
        }
    
        if(fitter_mode < 0 || fitter_mode > 2 || inspection_value < minX[0] || inspection_value > maxX[n_bins_to_scan-1]){
           cout <<"Sorry. Wrong input. Program will stop. Have a nice day!"<< endl;
           cout <<"   "<< endl;
           return 0;
        }
    }else{
        cout <<"   "<< endl;
        cout <<"You are running in auto mode. So be aware that you might miss a few nasty details... Good luck!"<< endl;
        cout <<"   "<< endl;
    }
    //----------------------------------------
    
    //Yields:
    Double_t nEvents_Data[n_bins_to_scan];
    Double_t nEvents_MC_Rec[n_bins_to_scan];
    Double_t nEvents_MC_True[n_bins_to_scan];
    Double_t nEvents_misc[n_bins_to_scan];
    
    //And their errors:
    Double_t dnEvents_Data[n_bins_to_scan];
    Double_t dnEvents_MC_Rec[n_bins_to_scan];
    Double_t dnEvents_MC_True[n_bins_to_scan];
    Double_t dnEvents_misc[n_bins_to_scan];
    
    //Histograms:
    TH1F *pro_Data[n_bins_to_scan];
    TH1F *pro_MC_Rec[n_bins_to_scan];
    TH1F *pro_MC_True[n_bins_to_scan];
    TH1F *pro_misc[n_bins_to_scan];
    
    TCanvas *c_mon = new TCanvas("c_mon","",1600,500);
    c_mon->Divide(3);
    
    Double_t fit_boarders[2] = {0.0,0.0};
    Double_t init_reject_points[2] = {0.0,0.0};
    Double_t signal_fit_params[6] = {
        get_fit_scale(),
        get_fit_mass(),
        get_fit_sigma(),
        get_fit_KL(),
        get_fit_KR(),
        get_fit_offset()
    };
    TString fitter_options[3] = {
        get_pre_fit_bkg_options(),
        get_pre_fit_sig_options(),
        get_full_fit_options()
    };
    Double_t fit_chiSquare = 0.0;
    Bool_t fit_bkg_only = get_report_bkg_fit_only();
    Bool_t use_fit_sig_yields = get_use_yields_from_fitted_signal();
    Double_t match_chiSquare = 0.0;
    Double_t mean_fit = 0.0;
    Double_t sigma_fit = 0.0;
    Double_t mean_mc = 0.0;
    Double_t sigma_mc = 0.0;
    
    cout<<"  "<< endl;
    cout<<"READ SETTINGS FOR A TEST"<< endl;
    cout <<"Integration percentage: "<<get_integration_percentage()<< endl;
    cout <<"N fit routines: "<<get_n_fit_routines()<< endl;
    cout <<"Use only bkg fit: "<<get_report_bkg_fit_only() << endl;
    cout <<"Use yields from fit: "<<get_use_yields_from_fitted_signal()<< endl;
    cout<<"  "<< endl;
    
    cout<<"  "<< endl;
    
    Double_t zoom_left = get_zoom_left_signal();
    Double_t zoom_right = get_zoom_right_signal();
    Double_t current_x_value = 0.0;
    Int_t accept_hist_fit = -1;
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for(Int_t i=0;i<n_bins_to_scan;i++){
        nEvents_Data[i] = nEvents_MC_Rec[i] = nEvents_MC_True[i] = nEvents_misc[i] = 0.0;
        dnEvents_Data[i] = dnEvents_MC_Rec[i] = dnEvents_MC_True[i] = dnEvents_misc[i] = 0.0;
        
        current_x_value = 0.5*(minX[i] + maxX[i]);
        
        if((fitter_mode == 1 && current_x_value != inspection_value))continue;
        
        TString addName = Form("_scan%i",i);
        
        //cout <<"Debug: "<< endl;
        pro_Data[i] = getOneDProjection(histData,"data"+addName,minX[i],maxX[i],axis);
        //cout <<"Data: "<<pro_Data[i]->Integral()<< endl;
        pro_MC_Rec[i] = getOneDProjection(histMC_Rec,"mc_rec"+addName,minX[i],maxX[i],axis);
        //cout <<"MC Rec: "<<pro_MC_Rec[i]->Integral()<< endl;
        pro_MC_True[i] = getOneDProjection(histMC_True,"mc_true"+addName,minX[i],maxX[i],axis);
        //cout <<"MC True: "<<pro_MC_True[i]->Integral()<< endl;
        pro_misc[i] = getOneDProjection(hist_misc,"misc"+addName,minX[i],maxX[i],axis);
        
        setAxis(pro_Data[i]->GetYaxis(),"Entries [a.u.]",y_axis_offset,y_axis_size,y_axis_label_size);
        
        nEvents_MC_True[i] = pro_MC_True[i]->Integral();
        dnEvents_MC_True[i] = TMath::Sqrt(nEvents_MC_True[i]);
        //-----------------------------------------------
        bool accept_scan = accept_current_event(nEvents_MC_True[i],n_true_threshold);
        if(accept_scan){
            cout <<"    "<< endl;
            cout <<"Start fitting procedure for bin: "<<i<<" with min: "<<minX[i]<<" and max: "<<maxX[i]<< endl;
            
            fit_boarders[0] = get_left_fit_boarder();
            fit_boarders[1] = get_right_fit_boarder();
            
            init_reject_points[0] = get_left_reject_point_start();
            init_reject_points[1] = get_right_reject_point_start();
            
            TLine *left_line,*right_line;
            //------------------------------------------
            if(axis == 0){
                left_line = new TLine(minX[i],0.0,minX[i],histData->GetYaxis()->GetXmax());
                left_line->SetLineWidth(2.0);
                left_line->SetLineColor(kBlack);
                
                right_line = new TLine(maxX[i],0.0,maxX[i],histData->GetYaxis()->GetXmax());
                right_line->SetLineWidth(2.0);
                right_line->SetLineColor(kBlack);
            }else{
                left_line = new TLine(0.0,minX[i],histData->GetXaxis()->GetXmax(),minX[i]);
                left_line->SetLineWidth(2.0);
                left_line->SetLineColor(kBlack);
                
                right_line = new TLine(0.0,maxX[i],histData->GetXaxis()->GetXmax(),maxX[i]);
                right_line->SetLineWidth(2.0);
                right_line->SetLineColor(kBlack);
            }
            //------------------------------------------
            
            //Draw the 2D hist from which we draw the projections to fit:
            c_mon->cd(1);
            histData->Draw("COLZ");
            left_line->Draw("same");
            right_line->Draw("same");
            
            //Run the actual fitting procedure here:
            //------------------------------
            while(accept_hist_fit != 2){
                //Fit:
                fit_chiSquare = fit_histogram(pro_Data[i],init_reject_points,fit_boarders,signal_fit_params,get_fit_step_size(),get_n_fit_routines(),get_fix_bkg(),get_fix_sig(),fit_bkg_only,fitter_options,show_ndf);
                //Retreive hists:
                get_fit_hists(pro_Data[i],fit_bkg_only);
                pro_Data[i]->GetXaxis()->SetRangeUser(fit_boarders[0],fit_boarders[1]);
                
                //Plot the fitted distributions:
                TLegend *fit_leg = new TLegend(0.6,0.6,0.9,0.9);
                fit_leg->SetFillColor(0);
                fit_leg->AddEntry(pro_Data[i],"Data");
                fit_leg->AddEntry(background_distribution,"Fit: exp[pol(3)]");
                
                background_distribution->SetLineWidth(3.0);
                background_distribution->SetLineColor(kRed);
                
                c_mon->cd(2);
                if(fit_bkg_only){
                    pro_Data[i]->Draw("E");
                    background_distribution->Draw("same");
                }else{
                    full_distribution->SetLineColor(kYellow);
                    full_distribution->SetLineWidth(3.0);
                    full_distribution->SetFillColor(kYellow);
                    full_distribution->SetFillStyle(3001);
                    
                    signal_distribution->SetLineWidth(3.0);
                    signal_distribution->SetLineColor(8);
                    
                    fit_leg->AddEntry(signal_distribution,"Fit: Gaus + exp. Tail");
                    fit_leg->AddEntry(full_distribution,"Total Fit");
                    
                    full_distribution->GetXaxis()->SetRangeUser(fit_boarders[0],fit_boarders[1]);
                    setAxis(full_distribution->GetXaxis(),pro_Data[i]->GetXaxis()->GetTitle(),y_axis_offset,y_axis_size,y_axis_label_size);
                    setAxis(full_distribution->GetYaxis(),pro_Data[i]->GetYaxis()->GetTitle(),y_axis_offset,y_axis_size,y_axis_label_size);
                    
                    full_distribution->Draw("");
                    pro_Data[i]->Draw("sameE");
                    background_distribution->Draw("same");
                    signal_distribution->Draw("same");
                    
                }
                fit_leg->Draw("same");
                
                //Plot the background corrected spectra:
                //Get data integration boarders
                Int_t *integral_bins_data = get_integration_bins(pro_Data[i],signal_fit_params[1],get_integration_percentage());
                //Get MC integration boarders
                Int_t *integral_bins_mc = get_integration_bins(pro_MC_Rec[i],signal_fit_params[1],get_integration_percentage());
                
                //Get yields:
                TH1F *pure_signal_distribution = (TH1F*)pro_Data[i]->Clone();
                Double_t *data_yields = 0;
                //--------------------------
                if(use_fit_sig_yields && !fit_bkg_only){
                    data_yields = get_integral(signal_distribution,background_distribution,integral_bins_data[0],integral_bins_data[1],true,false);
                }else data_yields = get_integral(pure_signal_distribution,background_distribution,integral_bins_data[0],integral_bins_data[1],false,false);
                //--------------------------
                
                Double_t *mc_rec_yields = get_integral(pro_MC_Rec[i],background_distribution,integral_bins_data[0],integral_bins_data[1],false,true);
                Double_t *misc_yields = get_integral(pro_misc[i],background_distribution,integral_bins_data[0],integral_bins_data[1],false,true);
                
                //Get normalized MC distribution:
                TH1F *mc_reference_hist = get_scaled_histogram(pro_MC_Rec[i],pure_signal_distribution,signal_fit_params[1]);
                mc_reference_hist->SetLineWidth(3.0);
                mc_reference_hist->SetLineColor(kRed);
                
                TLine *left_int_data,*right_int_data;
                TLine *left_int_mc,*right_int_mc;
                
                left_int_data = new TLine(pure_signal_distribution->GetXaxis()->GetBinCenter(integral_bins_data[0]),0.0,pure_signal_distribution->GetXaxis()->GetBinCenter(integral_bins_data[0]),pure_signal_distribution->GetMaximum()*0.95);
                left_int_data->SetLineWidth(3.0);
                left_int_data->SetLineColor(kBlack);
                
                right_int_data = new TLine(pure_signal_distribution->GetXaxis()->GetBinCenter(integral_bins_data[1]),0.0,pure_signal_distribution->GetXaxis()->GetBinCenter(integral_bins_data[1]),pure_signal_distribution->GetMaximum()*0.95);
                right_int_data->SetLineWidth(3.0);
                right_int_data->SetLineColor(kBlack);
                
                left_int_mc = new TLine(mc_reference_hist->GetXaxis()->GetBinCenter(integral_bins_mc[0]),0.0,mc_reference_hist->GetXaxis()->GetBinCenter(integral_bins_mc[0]),mc_reference_hist->GetMaximum()*0.95);
                right_int_mc = new TLine(mc_reference_hist->GetXaxis()->GetBinCenter(integral_bins_mc[1]),0.0,mc_reference_hist->GetXaxis()->GetBinCenter(integral_bins_mc[1]),mc_reference_hist->GetMaximum()*0.95);
                
                left_int_mc->SetLineWidth(3.0);
                left_int_mc->SetLineColor(kBlack);
                left_int_mc->SetLineStyle(2);
                
                right_int_mc->SetLineWidth(3.0);
                right_int_mc->SetLineColor(kBlack);
                right_int_mc->SetLineStyle(2);
                
                pure_signal_distribution->GetXaxis()->SetRangeUser(zoom_left,zoom_right);
                pure_signal_distribution->SetLineColor(kBlack);
                pure_signal_distribution->SetMarkerColor(kBlack);
                pure_signal_distribution->SetFillColor(kYellow);
                pure_signal_distribution->SetFillStyle(3001);
                
                TLegend *leg_fin = new TLegend(0.6,0.6,0.9,0.9);
                leg_fin->SetFillColor(0);
                leg_fin->AddEntry(pure_signal_distribution,"Data");
                if(!fit_bkg_only)leg_fin->AddEntry(signal_distribution,"Fit");
                leg_fin->AddEntry(mc_reference_hist,"MC");
                
                //Show the final distributions:
                c_mon->cd(3);
                pure_signal_distribution->Draw();
                mc_reference_hist->Draw("same");
                if(!fit_bkg_only)signal_distribution->Draw("same");
                leg_fin->Draw("same");
                left_int_data->Draw("same");
                right_int_data->Draw("same");
                left_int_mc->Draw("same");
                right_int_mc->Draw("same");
                
                c_mon->Update();
                gSystem->ProcessEvents();
                
                if(saveName != "" && saveName != " "){
                    c_mon->SaveAs(saveName + addName + ".png");
                }
                
                match_chiSquare = get_matchChiSquare(pure_signal_distribution,mc_reference_hist,pure_signal_distribution->GetXaxis()->FindFixBin(zoom_left),pure_signal_distribution->GetXaxis()->FindFixBin(zoom_right),show_ndf);
                
                if(!fit_bkg_only){
                    mean_fit = signal_distribution->GetMean(1);
                    sigma_fit = signal_distribution->GetStdDev(1);
                }
                
                mean_mc = mc_reference_hist->GetMean(1);
                sigma_mc = mc_reference_hist->GetStdDev(1);
                
                //--------------------------------
                if(fitter_mode != 2){
                    cout <<"            "<< endl;
                    cout <<"Does the fit look ok? "<< endl;
                    cout <<"0 - Redo fit with new fit boarders"<< endl;
                    cout <<"1 - Redo fit with new initial peak rejection boarders"<< endl;
                    cout <<"2 - Accept the fit and take the results"<< endl;
                    cout <<"3 - Set everything to 0"<< endl;
                    cin >> accept_hist_fit;
                    
                    if(accept_hist_fit == 0){
                        cout <<"   "<< endl;
                        cout <<"Please type in left fit boarder"<< endl;
                        cin >> fit_boarders[0];
                        cout <<"Please type in right fit boarder"<< endl;
                        cin >> fit_boarders[1];
                    }
                    
                    if(accept_hist_fit == 1){
                        cout <<"   "<< endl;
                        cout <<"Please type in left reject boarder"<< endl;
                        cin >> init_reject_points[0];
                        cout <<"Please type in right reject boarder"<< endl;
                        cin >> init_reject_points[1];
                    }
                    
                    if(accept_hist_fit == 3){
                        nEvents_Data[i] = 0.0;
                        dnEvents_Data[i] = 0.0;
                        
                        nEvents_MC_Rec[i] = 0.0;
                        dnEvents_MC_Rec[i] = 0.0;
                        
                        nEvents_misc[i] = 0.0;
                        dnEvents_misc[i] = 0.0;
                    }
                }else{
                    accept_hist_fit = 2;
                }
                //--------------------------------
                
                if(accept_hist_fit == 2){
                    nEvents_Data[i] = data_yields[0];
                    dnEvents_Data[i] = data_yields[1];
                    
                    nEvents_MC_Rec[i] = mc_rec_yields[0];
                    dnEvents_MC_Rec[i] = mc_rec_yields[1];
                    
                    nEvents_misc[i] = misc_yields[0];
                    dnEvents_misc[i] = misc_yields[1];
                }
                
                left_int_data->Clear();
                right_int_data->Clear();
                left_int_mc->Clear();
                right_int_mc->Clear();
                
                pure_signal_distribution->Clear();
                mc_reference_hist->Clear();
                
                fit_leg->Clear();
                leg_fin->Clear();
            }
            //------------------------------
            
            if(fitter_mode != 1){
               gr_data->SetPoint(i,current_x_value,nEvents_Data[i]);
               gr_data->SetPointError(i,0.0,dnEvents_Data[i]);
            
               gr_mc_rec->SetPoint(i,current_x_value,nEvents_MC_Rec[i]);
               gr_mc_rec->SetPointError(i,0.0,dnEvents_MC_Rec[i]);
            
               gr_mc_true->SetPoint(i,current_x_value,nEvents_MC_True[i]);
               gr_mc_true->SetPointError(i,0.0,dnEvents_MC_True[i]);
            
               gr_misc->SetPoint(i,current_x_value,nEvents_misc[i]);
               gr_misc->SetPointError(i,0.0,dnEvents_misc[i]);
            
               gr_chiSqaure_fit->SetPoint(i,current_x_value,fit_chiSquare);
               gr_chiSqaure_match->SetPoint(i,current_x_value,match_chiSquare);
                
               gr_mean_fit->SetPoint(i,current_x_value,mean_fit);
               gr_sigma_fit->SetPoint(i,current_x_value,sigma_fit);
                
               gr_mean_mc->SetPoint(i,current_x_value,mean_mc);
               gr_sigma_mc->SetPoint(i,current_x_value,sigma_mc);
            }
                
            cout <<"...finished fitting procedure for bin: "<<i<< endl;
            accept_hist_fit = -1;
            
            left_line->Clear();
            right_line->Clear();
        }
        //-----------------------------------------------
        
        //Clear stuff:
        pro_Data[i]->Clear();
        pro_MC_Rec[i]->Clear();
        pro_MC_True[i]->Clear();
        pro_misc[i]->Clear();
    }
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    return fitter_mode;
}
//*********************************************************************************


//Now run the full analysis:
//*********************************************************************************
void run_analysis(Int_t data_set,Int_t nBins,Int_t kfit_cut,Int_t imgg_cut,Int_t zv_cut,Int_t ebeam,Int_t cal,TString outName,Bool_t is_auto_mode){
    gStyle->SetOptFit(0);
    gStyle->SetOptStat(0);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadLeftMargin(0.15);
    Double_t title_offset = 0.85;
    Double_t title_size = 0.075;
    Double_t label_size = 0.05;
    
    cout <<"     "<< endl;
    cout <<"*************************************"<< endl;
    cout <<"*                                   *"<< endl;
    cout <<"* Welcome to the Dalitz Plot Fitter *"<< endl;
    cout <<"*                                   *"<< endl;
    cout <<"*************************************"<< endl;
    cout <<"     "<< endl;
    
    //1.) Handle the output file name:
    //====================================================
    cout <<"Define output file name..."<< endl;
    
    TString data_names[3] = {"_2017","_2018S","_2018F"};
    
    TString anaName = Form("_nbins%i_kfit_cut%i_imgg_cut%i_zv_cut%i",nBins,kfit_cut,imgg_cut,zv_cut);
    if(ebeam >= 0){
        anaName += Form("_ebeam%i",ebeam);
    }
    
    if(cal == 0){
        anaName += "_fcal";
    }else if(cal == 1){
        anaName += "_bcal";
    }
    
    TString saveFileName = outName + data_names[data_set] + anaName + ".root";
    
    cout <<"...done!"<< endl;
    cout <<"   "<< endl;
    //====================================================
    
    //2.) Load the files with histograms:
    //====================================================
    cout <<"Load data and get histograms..."<< endl;
    
    TString f_data = data_files[data_set];
    TString f_mc_rec = mc_rec_files[data_set];
    TString f_mc_true = mc_true_files[data_set];
    TString f_misc = misc_files[data_set];
    
    TFile *in_files[5];
    in_files[0] = getFile(f_data);
    in_files[1] = in_files[0];//include acceptance correction
    in_files[2] = getFile(f_mc_rec);
    in_files[3] = getFile(f_mc_true);
    in_files[4] = getFile(f_misc);
    
    TString histogram_names[5] = {
        get_histname_rec(nBins,kfit_cut,0,imgg_cut,zv_cut,ebeam,cal),
        get_histname_rec(nBins,kfit_cut,1,imgg_cut,zv_cut,ebeam,cal),
        get_histname_rec(nBins,kfit_cut,1,imgg_cut,zv_cut,ebeam,cal),
        get_histname_true(nBins,ebeam,cal),
        get_histname_misc(nBins,kfit_cut,1,imgg_cut,zv_cut,ebeam,cal)
    };
    
    set_input_histograms(in_files,histogram_names);
    
    cout <<"...done!"<< endl;
    cout <<"   "<< endl;
    //====================================================
    
    //3.) Handle accidentals:
    //====================================================
    cout <<"Take care of accidental subtraction..."<< endl;
    
    TH2F *input_Data = get_h2f_accs(input_Data_noAcc,input_Data_Acc,n_beam_bunches,"input_Data");
    setAxis(input_Data->GetXaxis(),"Global Bin",0.85,0.075,0.05);
    setAxis(input_Data->GetYaxis(),"M(#pi^{+},#pi^{-},#gamma_{1},#gamma_{2})-M(#gamma_{1},#gamma_{2})",1.0,0.075,0.05);
    
    cout <<"...done!"<< endl;
    cout <<"   "<< endl;
    //====================================================
    
    //4.) Set up the graphs:
    //====================================================
    cout <<"Set up graphs for collecting the results..."<< endl;
    
    init_graphs(nBins*nBins);
    
    setAxis(gr_nEvents_Data->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_nEvents_Data->GetYaxis(),"N_{Data}(#eta#rightarrow#pi^{+}#pi^{-}#pi^{0})",title_offset,title_size,label_size);
    
    setAxis(gr_nEvents_MC_Rec->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_nEvents_MC_Rec->GetYaxis(),"N_{MC,Rec}(#eta#rightarrow#pi^{+}#pi^{-}#pi^{0})",title_offset,title_size,label_size);
    
    setAxis(gr_nEvents_MC_True->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_nEvents_MC_True->GetYaxis(),"N_{MC,True}(#eta#rightarrow#pi^{+}#pi^{-}#pi^{0})",title_offset,title_size,label_size);
    
    setAxis(gr_misc->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_misc->GetYaxis(),"Rel. N_{MC}(#eta#rightarrow#pi^{+}#pi^{-}#gamma) [%]",title_offset,title_size,label_size);
    
    setAxis(gr_chiSquare_fit->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_chiSquare_match->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    
    if(show_per_ndf){
        setAxis(gr_chiSquare_fit->GetYaxis(),"#chi^{2}_{Fit}/NDF",title_offset,title_size,label_size);
        setAxis(gr_chiSquare_match->GetYaxis(),"#chi^{2}_{Match}/NDF",title_offset,title_size,label_size);
    }else{
        setAxis(gr_chiSquare_fit->GetYaxis(),"#chi^{2}_{Fit}",title_offset,title_size,label_size);
        setAxis(gr_chiSquare_match->GetYaxis(),"#chi^{2}_{Match}",title_offset,title_size,label_size);
    }
    
    setAxis(gr_mean_fit->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_mean_fit->GetYaxis(),"Mean from Fit",title_offset,title_size,label_size);
    
    setAxis(gr_sigma_fit->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_sigma_fit->GetYaxis(),"Std. Dev. from Fit",title_offset,title_size,label_size);
    
    setAxis(gr_mean_mc->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_mean_mc->GetYaxis(),"Mean from MC",title_offset,title_size,label_size);
    
    setAxis(gr_sigma_mc->GetXaxis(),"Global Bin",title_offset,title_size,label_size);
    setAxis(gr_sigma_mc->GetYaxis(),"Std. Dev. from MC",title_offset,title_size,label_size);
    
    cout <<"...done!"<< endl;
    cout <<"   "<< endl;
    //====================================================
    
    //5.) Do the fits:
    //====================================================
    Int_t mode = fit_histograms(nBins,0,input_Data,input_MC_Rec,input_MC_True,input_misc,gr_nEvents_Data,gr_nEvents_MC_Rec,gr_nEvents_MC_True,gr_misc,gr_chiSquare_fit,gr_chiSquare_match,gr_mean_fit,gr_sigma_fit,gr_mean_mc,gr_sigma_mc,show_per_ndf,fit_hist_names,is_auto_mode);
    //====================================================
    
    if(mode == 1)return;
    
    //6.) Show the results:
    //====================================================
    cout <<"Show results..."<< endl;
    
    //Normalize contributions from eta->pi+pi-pi0:
    normalize_yields(gr_misc,input_misc);
    show_results();
    
    cout <<"...done!"<< endl;
    cout <<"   "<< endl;
    //====================================================
    
    //7.) Write the results to a file:
    //====================================================
    cout <<"Write results to: "<< saveFileName <<"..."<< endl;
    
    write_out_results(saveFileName,"gr_nEvents_MC_PiPiG");
    
    cout <<"...done!"<< endl;
    cout <<"   "<< endl;
    //====================================================
}
//*********************************************************************************
