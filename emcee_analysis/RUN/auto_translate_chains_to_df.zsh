#!/bin/zsh

data_set=${1}
sys_var=${2}
start=${3}
end=${4}
out_name=${5}

kfit_ref=3
imgg_ref=1

addName="_no_acc_cut"

sys_yield_array=(
    "BkgPol2_"
    "BkgPol4_" 
    "IntR2_"
    "IntR4_"
    "IntR5_"
    "NFit3_"
    "NFit7_"
    "FitBkg_"
    "FitYield_"
    "nbins9_"
    "nbins10_"
    "nbins12_"
    "nbins13_"
)

sys_fit_array=(
    #"nonDiagCov_"
    "stretchMover_"
    "walkMover_"
    #"kdeMover_"
    #"deSnookerMover_"
    "100Walkers_"
    "300Walkers_"
    "linearLoss_"
    "softL1Loss_"
    "cauchyLoss_"
)

for i in `seq $start $end`;
  do
       save_name=$out_name 
       if [ $sys_var = "kfit" ]; then
         ana_name="nbins11_kfit_cut"$i"_imgg_cut"$imgg_ref"_zv_cut1"$addName
       elif [ $sys_var = "imgg" ]; then
         ana_name="nbins11_kfit_cut"$kfit_ref"_imgg_cut"$i"_zv_cut1"$addName
       elif [ $sys_var = "ebeam" ]; then
         ana_name="nbins11_kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1_ebeam"$i$addName
       elif [ $sys_var = "yield_sys" ]; then
            if [ $i -lt 10 ]; then
               ana_name="nbins11_kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1"$addName
               save_name=${sys_yield_array[$i]}$out_name
            else
               ana_name=${sys_yield_array[$i]}"kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1"$addName  
         fi    
       elif [ $sys_var = "fit_sys" ]; then
         ana_name="nbins11_kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1"$addName
         save_name=${sys_fit_array[$i]}$out_name    
       fi 
    
       ./translate_chain_to_df.py $data_set $ana_name $save_name
  done
