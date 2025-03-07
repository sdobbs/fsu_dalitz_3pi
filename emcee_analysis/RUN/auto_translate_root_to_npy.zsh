#!/bin/zsh

coreName=${1}
sys_var=${2}
start=${3}
end=${4}

kfit_ref=3
imgg_ref=1
nbins_ref=11



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

for d in `seq 0 2`;
 do
   for i in `seq $start $end`;
     do
       if [ $sys_var = "kfit" ]; then
         pre_name=$coreName"_"
         ana_name="nbins"$nbins_ref"_kfit_cut"$i"_imgg_cut"$imgg_ref"_zv_cut1"
       elif [ $sys_var = "imgg" ]; then
         pre_name=$coreName"_"
         ana_name="nbins"$nbins_ref"_kfit_cut"$kfit_ref"_imgg_cut"$i"_zv_cut1"
       elif [ $sys_var = "ebeam" ]; then
         pre_name=$coreName"_"
         ana_name="nbins"$nbins_ref"_kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1_ebeam"$i
       elif [ $sys_var = "yield_sys" ]; then
         if [ $i -lt 10 ]; then
             pre_name=$coreName"_"${sys_yield_array[$i]}
             ana_name="nbins"$nbins_ref"_kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1"
         else
             pre_name=$coreName"_"
             ana_name=${sys_yield_array[$i]}"kfit_cut"$kfit_ref"_imgg_cut"$imgg_ref"_zv_cut1"
         fi    
       fi 
         #echo $pre_name
         #echo $ana_name
       ./readin_root_data.py $d $pre_name $ana_name -1.0
     done
 done
