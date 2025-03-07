#include "fitter_routines.h"

void run_fitter(Int_t data_set,Int_t nBins,Int_t kfit_cut,Int_t imgg_cut,Int_t zv_cut,Int_t ebeam,Int_t cal,TString outName){
    
    run_analysis(data_set,nBins,kfit_cut,imgg_cut,zv_cut,ebeam,cal,outName,false);
    
    
}
