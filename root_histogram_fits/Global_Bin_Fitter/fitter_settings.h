//Basic settings, used for the fitter:


//SETTINGS FOR LOADING THE DATA:
//===================================================================================================
//ROOT-files containing the histograms
TString dataDir = "/Users/daniellersch/Desktop/eta3Pi_DalitzAna/raw_dp_root_data";
TString thrownDir = "/Users/daniellersch/Desktop/eta3Pi_DalitzAna/raw_dp_root_data/gen_hists/";
//Data:
TString data_files[3] = {
    
    dataDir + "/data_hists/tcut_2017_data.root",
    dataDir + "/data_hists/tcut_2018S_data.root",
    dataDir + "/data_hists/tcut_2018F_data.root"
    
    /*
    dataDir + "all_eta3pi_2017_data_noW_zv.root",
    dataDir + "all_eta3pi_2018S_data_noW_zv.root",
    dataDir + "all_eta3pi_2018F_data_noW_zv.root"
     */
};
//MC Rec:
TString mc_rec_files[3] = {
    dataDir + "/mc_hists/tcut_2017_MC.root",
    dataDir + "/mc_hists/tcut_2018S_MC.root",
    dataDir + "/mc_hists/tcut_2018F_MC.root"
};
//MC True:
TString mc_true_files[3] = {
    thrownDir + "new_ana_tcut_2017_Thrown.root",
    thrownDir + "new_ana_tcut_2018S_Thrown.root",
    thrownDir + "new_ana_tcut_2018F_Thrown.root"
};
//Misc. histograms:
/*
TString misc_files[3] = {
    "/Volumes/BunchOfStuff/GlueX_Eta_Data/MC2017_Geant4/Eta3Pi_Hists_2017_Geant4_PiPiG.root",
    "/Volumes/BunchOfStuff/GlueX_Eta_Data/MC2018S_Geant4/Eta3Pi_Hists_2018S_Geant4_PiPiG.root",
    "/Volumes/BunchOfStuff/GlueX_Eta_Data/MC2018F_Geant4/Eta3Pi_Hists_2018F_Geant4_PiPiG.root"
};
*/
TString misc_files[3] = {
    dataDir + "/mc_hists/tcut_2017_MC.root",
    dataDir + "/mc_hists/tcut_2018S_MC.root",
    dataDir + "/mc_hists/tcut_2018F_MC.root"
};

//Define the histogram names:
//This function is custom-made and might be different for someone else:
//rec histograms:
TString get_histname_rec(Int_t gbin,Int_t kfit_cut,Int_t dbrt_cut,Int_t imgg_cut,Int_t zv_cut,Int_t ebeam,Int_t cal){
    TString rec_name = Form("probCut%i/h_dM_vs_GBin%i_probCut%i_dbrtCut%i_imggCut%i_zvCut%i",kfit_cut,gbin,kfit_cut,dbrt_cut,imgg_cut,zv_cut);
    
    if(ebeam >= 0){
        rec_name += Form("_ebeam%i",ebeam);
    }
    
    if(cal == 0){
        rec_name += "_fcal";
    }else if(cal == 1){
        rec_name += "_bcal";
    }
    
    return rec_name;
}

//true histograms:
TString get_histname_true(Int_t gbin,Int_t ebeam,Int_t cal){
    TString true_name = Form("MM_vs_GBin_Gen_%i",gbin);
    
    if(ebeam >= 0){
        true_name = Form("MM_vs_GBin_Gen_egWindow%i",ebeam);
    }
    
    if(cal == 0){
        true_name = "MM_vs_GBin_FCAL";
    }else if(cal == 1){
        true_name = "MM_vs_GBin_BCAL";
    }
    
    return true_name;
}

//misc histograms:
TString get_histname_misc(Int_t gbin,Int_t kfit_cut,Int_t dbrt_cut,Int_t imgg_cut,Int_t zv_cut,Int_t ebeam,Int_t cal){
    TString rec_name = Form("probCut%i/h_dM_vs_GBin%i_probCut%i_dbrtCut%i_imggCut%i_zvCut%i",kfit_cut,gbin,kfit_cut,dbrt_cut,imgg_cut,zv_cut);
    
    if(ebeam > 0){
        rec_name += Form("_ebeam%i",ebeam);
    }
    
    if(cal == 0){
        rec_name += "_fcal";
    }else if(cal == 1){
        rec_name += "_bcal";
    }
    
    return rec_name;
}
//===================================================================================================

//These scan-limits need to be set individually:
//===================================================================================================
Double_t min_X[169];
Double_t max_X[169];

void set_limits(Int_t nbins){ //This function can be changed, depending on your binning / axis settings, but the name should remain the same...
    //+++++++++++++++++++++++++++++++++++
    for(Int_t h=0;h<nbins;h++){
        min_X[h] = h;
        max_X[h] = h;
    }
    //+++++++++++++++++++++++++++++++++++
}

Double_t *get_min_X(){
    return min_X;
}

Double_t *get_max_X(){
    return max_X;
}
//===================================================================================================

//SETTINGS FOT THE HISTOGRAM FITS:
//===================================================================================================
Double_t left_reject_point_start = 0.34;
Double_t right_reject_point_start = 0.44;
Double_t left_fit_boarder = 0.33;
Double_t right_fit_boarder = 0.6;
Double_t zoom_left_signal = 0.35;
Double_t zoom_right_signal = 0.55;
Double_t integration_percentage = 3.0;
Int_t n_fit_routines = 5;
Double_t fit_step_size = 0.01;
Double_t fit_mass = 0.5478 - 0.135;
Double_t fit_sigma = 0.02;
Double_t fit_scale = 30;
Double_t fit_offset = 10;
Double_t fit_KL = 1.0;
Double_t fit_KR = 1.0;

TString pre_fit_sig_options = "LRQN";
TString pre_fit_bkg_options = "BRQN";
TString full_fit_options = "LRQN";

Bool_t fix_bkg = true;
Bool_t fix_sig = false;
Bool_t report_bkg_fit_only = false;
Bool_t use_yields_from_fitted_signal = false;
//===================================================================================================


//Misc. settings:
//===================================================================================================
Int_t n_beam_bunches = 4;
TString fit_hist_names = "";
Bool_t show_per_ndf = true;
Double_t n_true_threshold = 0.0;
//===================================================================================================



//NOTHING TO BE DONE BELOW THIS LINE!!!!!!!!
//###########################################################################################################


//SETTERS:
//===================================================================================================
void set_left_reject_point_start(Double_t val){
    left_reject_point_start = val;
}

//--------------------------------------

void set_right_reject_point_start(Double_t val){
    right_reject_point_start = val;
}

//--------------------------------------

void set_left_fit_boarder(Double_t val){
    left_fit_boarder = val;
}

//--------------------------------------

void set_right_fit_boarder(Double_t val){
    right_fit_boarder = val;
}

//--------------------------------------

void set_zoom_left_signal(Double_t val){
    zoom_left_signal = val;
}

//--------------------------------------

void set_zoom_right_signal(Double_t val){
    zoom_right_signal = val;
}

//--------------------------------------

void set_integration_percentage(Double_t val){
    integration_percentage = val;
}

//--------------------------------------

void set_n_fit_routines(Int_t val){
    n_fit_routines = val;
}

//--------------------------------------

void set_fit_step_size(Double_t val){
    fit_step_size = val;
}


//--------------------------------------

void set_fit_mass(Double_t val){
    fit_mass = val;
}

//--------------------------------------

void set_fit_sigma(Double_t val){
    fit_sigma = val;
}

//--------------------------------------

void set_fit_scale(Double_t val){
    fit_scale = val;
}

//--------------------------------------

void set_fit_offset(Double_t val){
    fit_offset = val;
}

//--------------------------------------

void set_fit_KL(Double_t val){
    fit_KL = val;
}

//--------------------------------------

void set_fit_KR(Double_t val){
    fit_KR = val;
}

//--------------------------------------

void set_pre_fit_sig_options(TString val){
    pre_fit_sig_options = val;
}

//--------------------------------------

void set_pre_fit_bkg_options(TString val){
    pre_fit_bkg_options = val;
}

//--------------------------------------

void set_full_fit_options(TString val){
    full_fit_options = val;
}

//--------------------------------------

void set_fix_bkg(Bool_t val){
    fix_bkg = val;
}

//--------------------------------------

void set_fix_sig(Bool_t val){
    fix_sig = val;
}

//--------------------------------------

void set_report_bkg_fit_only(Bool_t val){
    report_bkg_fit_only = val;
}

//--------------------------------------

void set_use_yields_from_fitted_signal(Bool_t val){
    use_yields_from_fitted_signal = val;
}
//===================================================================================================

//GETTERS:
//===================================================================================================
Double_t get_left_reject_point_start(){
    return left_reject_point_start;
}

//--------------------------------------

Double_t get_right_reject_point_start(){
    return right_reject_point_start;
}

//--------------------------------------

Double_t get_left_fit_boarder(){
    return left_fit_boarder;
}

//--------------------------------------

Double_t get_right_fit_boarder(){
    return right_fit_boarder;
}

//--------------------------------------

Double_t get_zoom_left_signal(){
    return zoom_left_signal;
}

//--------------------------------------

Double_t get_zoom_right_signal(){
    return zoom_right_signal;
}

//--------------------------------------

Double_t get_integration_percentage(){
    return integration_percentage;
}

//--------------------------------------

Int_t get_n_fit_routines(){
    return n_fit_routines;
}

//--------------------------------------

Double_t get_fit_step_size(){
    return fit_step_size;
}

//--------------------------------------

Double_t get_fit_mass(){
    return fit_mass;
}

//--------------------------------------

Double_t get_fit_sigma(){
    return fit_sigma;
}

//--------------------------------------

Double_t get_fit_scale(){
    return fit_scale;
}

//--------------------------------------

Double_t get_fit_offset(){
    return fit_offset;
}

//--------------------------------------

Double_t get_fit_KL(){
    return fit_KL;
}

//--------------------------------------

Double_t get_fit_KR(){
    return fit_KR;
}

//--------------------------------------

TString get_pre_fit_sig_options(){
    return pre_fit_sig_options;
}

//--------------------------------------

TString get_pre_fit_bkg_options(){
    return pre_fit_bkg_options;
}

//--------------------------------------

TString get_full_fit_options(){
    return full_fit_options;
}

//--------------------------------------

Bool_t get_fix_bkg(){
    return fix_bkg;
}

//--------------------------------------

Bool_t get_fix_sig(){
    return fix_sig;
}

//--------------------------------------

Bool_t get_report_bkg_fit_only(){
    return report_bkg_fit_only;
}

//--------------------------------------

Bool_t get_use_yields_from_fitted_signal(){
    return use_yields_from_fitted_signal;
}
//===================================================================================================
