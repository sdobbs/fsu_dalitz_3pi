#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from UTILS.result_evaluator import Fit_Result_Evaluator
import sys

evaluator = Fit_Result_Evaluator()

in_name = sys.argv[1]

print(" ")
#************************************************************
print("Load data from random walk analysis...")

dataDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/mcmc_dalitz_data'
fullLoadName = dataDir + '/' + in_name

storeDir = '/Users/daniellersch/Desktop/eta3Pi_DalitzAna/pandas_mcmc_dalitz_data'
fullSaveName = storeDir + '/' + in_name + '_df.csv'

print("...done!")
print(" ")
#************************************************************

#************************************************************
print("Load MCMC reader...")

mcmc_reader = evaluator.get_reader(fullLoadName)
parNames = ['norm','a','b','c','d','e','f','g','h','l']

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