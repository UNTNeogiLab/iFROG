# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:38:26 2020

@author: nota
"""

import numpy as np


date = "200123"
wl = ["840", "850", "865"]
ss = ["BBO", "MOS2"]
settings = "100mW_100mn_1slit_432grating_-11__27t0" 

datapath = {}

for sub in ss:
    for wave in wl:
            datapath[sub+"_"+wave] = np.loadtxt(date + "_" + sub + "_" + str(wave) + "nm_" + settings, delimiter="\t")
            datapath[sub+"_"+wave] = np.delete(datapath[sub+"_"+wave], datapath[sub+"_"+wave].shape[1]-1, axis=1)
            
            datapath[sub+"_"+wave+"_"+"pos"] = np.loadtxt(date + "_" + sub + "_" + str(wave) + "nm_" + settings, delimiter="\t")
            datapath[sub+"_"+wave+"_"+"pos"] = np.delete(datapath[sub+"_"+wave+"_"+"pos"], datapath[sub+"_"+wave+"_"+"pos"].shape[1]-1, axis=1)
