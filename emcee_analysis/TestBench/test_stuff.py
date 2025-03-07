import numpy as np
from ROOT import TFile,TH2D,TH2F
import ROOT
from array import array

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

m = [1.0,0.7,1.2]
s = [0.03,0.05,0.01]

var = np.random.normal(m,s,size=(10,3))

print(var)