#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from UTILS.result_evaluator import Fit_Result_Evaluator
import sys

evaluator = Fit_Result_Evaluator()

data_set = int(sys.argv[1])
ana_name = sys.argv[2]
out_name = sys.argv[3]

data_dict = {
   0: ['17','2017','GlueX 2017','GlueX-2017'],
   1: ['18S','2018S','GlueX 2018-01','GlueX-2018-01'],
   2: ['18F','2018F','GlueX 2018-08','GlueX-2018-08'],
   3: ['All','All','GlueX-I','GlueX-I']
}

coreAddName = data_dict[data_set][1] + '_' + ana_name

print(" ")
#************************************************************
print("Load data from random walk analysis...")

dataDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/mcmc_dalitz_data'
fullLoadName = dataDir + '/' + out_name + '_' + coreAddName
dataSetName = data_dict[data_set][3]

storeDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/pandas_mcmc_dalitz_data'
fullSaveName = storeDir + '/' + out_name + '_' + coreAddName + '_df.csv'

print("...done!")
print(" ")
#************************************************************

#************************************************************
print("Load MCMC reader...")

mcmc_reader = evaluator.get_reader(fullLoadName)
parNames = ['norm','a','b','c','d','e','f','g','h','l','acc_cut']
#parNames = ['norm','a','b','c','d','e','f','g','h','l']

print("...done!")
print(" ")
#***********************************************************

#Select samples:
#************************************************************
print("Determine autocorrelation time and select samples from chains...")

dp_df = evaluator.get_chains(mcmc_reader,parNames)

print("...done!")
print(" ")
#************************************************************

#Write everything to a csv:
#************************************************************
print("Write data to csv: " + fullSaveName + "...")

dp_df.to_csv(fullSaveName)

print("...done! Have a great day!")
print(" ")
#************************************************************