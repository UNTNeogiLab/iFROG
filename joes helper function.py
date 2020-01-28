# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:38:26 2020

@author: nota
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy import signal
from matplotlib import ticker, cm
from skimage.feature import peak_local_max

#%% Global Vars

# Input File Definitions
date = "200123"
wl = ["840", "850", "865"]
ss = ["BBO", "MOS2"]
settings = "100mW_100mn_1slit_432grating_-11__27t0" 

#Declare Empty datapath Dict
datapath = {}

um2fs = 2*(10**-6)*(10**15)/(3*10**8)
timestep=0.66713  
#%% Function Definitions

#%%Imort Datapath

def importDatapath(date,wl,ss,settings):
    for sub in ss:
        for wave in wl:
                pre = sub+"_"+wave+"_"
                datapath[sub+"_"+wave] = np.loadtxt(date + "_" + sub + "_" + str(wave) + "nm_" + settings, delimiter="\t")
                datapath[sub+"_"+wave] = np.delete(datapath[sub+"_"+wave], datapath[sub+"_"+wave].shape[1]-1, axis=1)
                
                datapath[pre+"pos"] = np.loadtxt(date + "_" + sub + "_" + str(wave) + "nm_" + settings + "_pos", delimiter="\t")

#%% Find T0
                
def findT0(date,wl,ss,settings):
    for sub in ss:
        for wave in wl:
                pre = sub+"_"+wave+"_"
                datapath[pre+"t0ind"] = int(np.mean(np.argpartition(datapath[sub+"_"+wave].sum(axis=0),-50)[-50:]))
                #converting to int is probably a bad assumption may need revision
                
                datapath[pre+"t0"] = datapath[pre+"pos"][datapath[pre+"t0ind"]]
                print(datapath[pre+"t0ind"])
                print(datapath[pre+"pos"][datapath[pre+"t0ind"]])
                
#%% Mesh Grid

def meshgrid(date,wl,ss,settings):
    for sub in ss:
        for wave in wl:
                pre = sub+"_"+wave+"_"
                datapath[pre+"x"] = (datapath[pre+"pos"] - datapath[pre+"t0"])*um2fs
                datapath[pre+"y"] = np.linspace(1,datapath[sub+"_"+wave].shape[0],num=datapath[sub+"_"+wave].shape[0])
                datapath[pre+"xg"],datapath[pre+"yg"]=np.meshgrid(datapath[pre+"x"],datapath[pre+"y"])
                

#%% fft

def fft(date,wl,ss,settings):
    for sub in ss:
        for wave in wl:
            pre = sub+"_"+wave+"_"
            
            datapath[pre+"fft"] = np.fft.fftshift(np.fft.fft(datapath[sub+"_"+wave]),axes=1)
            datapath[pre+"W"] = np.fft.fftshift(np.fft.fftfreq(datapath[pre+"pos"].size,timestep))
            
            xfft = datapath[pre+"W"]
            yfft = np.linspace(1,datapath[pre+"fft"].shape[0],num=datapath[pre+"fft"].shape[0])
            
            datapath[pre+"xgfft"], datapath[pre+"ygfft"] = np.meshgrid(xfft,yfft)
            
#%% Test Plot

def plot(date,wl,ss,settings):
    for i, sub in enumerate(ss):
        for j, wave in enumerate(wl):
                pre = sub+"_"+wave+"_"
                data = datapath[sub+"_"+wave]  
                datapath[sub+"_"+wave+"_plt"] = plt.contourf(datapath[pre+"xg"],          
                datapath[pre+"yg"],data,np.linspace(data.min(), data.max(), 1000), cmap='jet')
#%% Data Inport
                
importDatapath(date,wl,ss,settings)

#%% Grunt Work

#Find T0
findT0(date,wl,ss,settings)

meshgrid(date,wl,ss,settings)

fft(date,wl,ss,settings)

#%% Plotting

plot(date,wl,ss,settings)

            
##%% Plotting to Take over the World
#
#
#
#def fftplot(date,wl,ss,settings):
#    for sub in ss:
#        for wave in wl:
#            
#            