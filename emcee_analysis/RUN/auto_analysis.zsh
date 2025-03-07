#!/bin/zsh

coreName=${1}
data_set=${2}
sys_var=${3}
start=${4}
end=${5}
out_name=${6}

kfit_ref=3
imgg_ref=1
nbins_ref=11

addName="_no_acc_cut_norm"

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

for i in `seq $start $end`;
  do
     save_name=$out_name
     if [ $sys_var = "kfit" ]; then
         pre_name=$coreName"_"
         ana_name="nbins"$nbins_ref"_kfit_cut"$i"_imgg_cut"$imgg_ref"_zv_cut1"$addName
       elif [ $sys_var = "imgg" ]; then
         pre_name=$coreName"_"
         ana_name="nbins"$nbins_ref"_kfit_cut"$kfit_ref"_imgg_cut"$i"_zv_cut1"$addName
       elif [ $sys_var = "ebeam" ]; then
         pre_name=$coreName"_"
         ana_name="nbins"$nbins_ref"_kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1_ebeam"$i$addName
       elif [ $sys_var = "yield_sys" ]; then
         if [ $i -lt 10 ]; then
             save_name=${sys_yield_array[$i]}$out_name
             pre_name=$coreName"_"${sys_yield_array[$i]}
             ana_name="nbins"$nbins_ref"_kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1"$addName    
         else
             pre_name=$coreName"_"
             ana_name=${sys_yield_array[$i]}"kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1"$addName 
         fi    
       fi 
      
    ./run_random_walk_analysis.py $data_set $pre_name $ana_name $save_name
  done
      