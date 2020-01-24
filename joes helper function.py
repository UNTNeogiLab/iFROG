# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:38:26 2020

@author: nota
"""

import numpy as np

#%% Global Vars

# Input File Definitions
date = "200123"
wl = ["840", "850", "865"]
ss = ["BBO", "MOS2"]
settings = "100mW_100mn_1slit_432grating_-11__27t0" 

#Declare Empty datapath Dict
datapath = {}

um2fs = 2*(10**-6)*(10**15)/(3*10**8)
#%% Function Definitions

#%%Imort Datapath

def importDatapath(date,wl,ss,settings):
    for sub in ss:
        for wave in wl:
                datapath[sub+"_"+wave] = np.loadtxt(date + "_" + sub + "_" + str(wave) + "nm_" + settings, delimiter="\t")
                datapath[sub+"_"+wave] = np.delete(datapath[sub+"_"+wave], datapath[sub+"_"+wave].shape[1]-1, axis=1)
                
                datapath[sub+"_"+wave+"_"+"pos"] = np.loadtxt(date + "_" + sub + "_" + str(wave) + "nm_" + settings + "_pos", delimiter="\t")

#%% Find T0
                
def findT0(date,wl,ss,settings):
    for sub in ss:
        for wave in wl:
                datapath[sub+"_"+wave+"_"+"t0ind"] = int(np.mean(np.argpartition(datapath[sub+"_"+wave].sum(axis=0),-50)[-50:]))
                #converting to int is probably a bad assumption may need revision
                
                datapath[sub+"_"+wave+"_"+"t0"] = datapath[sub+"_"+wave+"_"+"pos"][datapath[sub+"_"+wave+"_"+"t0ind"]]
                print(datapath[sub+"_"+wave+"_"+"t0ind"])
                print(datapath[sub+"_"+wave+"_"+"pos"][datapath[sub+"_"+wave+"_"+"t0ind"]])
                
#%% Mesh Grid

def meshgrid(date,wl,ss,settings):
    for sub in ss:
        for wave in wl:
                datapath[sub+"_"+wave+"_"+"x"] = (datapath[sub+"_"+wave+"_"+"pos"] - datapath[sub+"_"+wave+"_"+"t0"])*um2fs
                datapath[sub+"_"+wave+"_"+"y"] = np.linspace(1,datapath[sub+"_"+wave].shape[0],num=datapath[sub+"_"+wave].shape[0])
                datapath[sub+"_"+wave+"_"+"xg"],datapath[sub+"_"+wave+"_"+"yg"]=np.meshgrid(datapath[sub+"_"+wave+"_"+"x"],datapath[sub+"_"+wave+"_"+"y"])

#%% Data Inport
                
importDatapath(date,wl,ss,settings)

#%% Grunt Work

#Find T0
findT0(date,wl,ss,settings)

meshgrid(date,wl,ss,settings)

#%% Plot Me


                
            
#%%
            