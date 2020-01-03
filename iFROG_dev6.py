

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:11:57 2019

@author: briansquires
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from matplotlib import ticker, cm

#%% Data Import

databbo = np.loadtxt("191231_BBO_840_20mW_10MHz_iRp_432GR_0000AE_10msE_8DIV_fine", delimiter="\t")
databbo = np.delete(databbo, databbo.shape[1]-1, axis=1)
datamos2 = np.loadtxt("191230_MOS2_840_20mW_10MHz_iRp_432GR_0966AE_1000msE_8DIV_fine", delimiter="\t")
datamos2 = np.delete(datamos2, datamos2.shape[1]-1, axis=1)

posb=np.loadtxt("191231_BBO_840_20mW_10MHz_iRp_432GR_0000AE_10msE_8DIV_fine_pos", delimiter="\t")
#posb=np.delete(posb,5000,axis=0)
posm=np.loadtxt("191230_MOS2_840_20mW_10MHz_iRp_432GR_0966AE_1000msE_8div_fine_pos", delimiter="\t")
#posm=np.delete(posm,5000,axis=0)
#%%


#truncation (data specific)
datamos2=datamos2[::,:6080]
posm=np.resize(posm,[6080,])

tzerob = -11704.60461
tzerom = -11707.53711

um2fs = 2*(10**-6)*(10**15)/(3*10**8)

xb=(posb-tzerob)*um2fs
yb=np.linspace(1,databbo.shape[0],num=databbo.shape[0])
xbg,ybg=np.meshgrid(xb,yb)

xm=(posm-tzerom)*um2fs
ym=np.linspace(10,datamos2.shape[0]+10,num=datamos2.shape[0])
xmg,ymg=np.meshgrid(xm,ym)


#%%Normalization

databbo = databbo/np.amax(databbo)
datamos2 = datamos2/np.amax(datamos2)



#%%
div=datamos2/(databbo)



#%% Plot Data

f , (ax1,ax2,ax3) =plt.subplots(1, 3, sharey=True)

f.suptitle('840nm')

bboplt = ax1.contourf(xbg,ybg,databbo,np.linspace(databbo.min(), databbo.max(), 10), cmap='jet')
ax1.set_title('BBO')
ax1.set_xlim(-500,500)
ax1.set_ylim(600,800)

cbar = f.colorbar(bboplt)


mos2plt = ax2.contourf(xmg,ymg,datamos2,np.linspace(datamos2.min(), datamos2.max(), 10),cmap='jet')
ax2.set_title('MoS2')
ax2.set_xlim(-500,500)
ax2.set_ylim(600,800)

divplt= ax3.contour(xbg,ybg,div,np.linspace(div.min(), div.max(), 10),cmap='jet')
ax3.set_title('MoS2/BBO')
ax3.set_xlim(-500,500)
ax3.set_ylim(600,800)


#%% Fourier Transform
timestep=0.66713  ##this is the time for one 100nm step.  This returns the FFT
                    # x-axis in THz

bfft = np.fft.fft(databbo)
Wb = np.fft.fftfreq(posb.size,timestep)
           
mfft = np.fft.fft(datamos2)
Wm = np.fft.fftfreq(posm.size,timestep)         

divfft=np.fft.fft(div)
#divfft=np.fft.fftfreq(div.size,timestep) #######
W=np.fft.fftfreq(div[1].size,timestep)
#W=np.fft.fftshift(W)

xfft=W
yfft=np.linspace(1,divfft.shape[0],num=divfft.shape[0])
xgfft, ygfft =np.meshgrid(xfft,yfft)


#%% Plot FFT
g, (gx1,gx2) = plt.subplots(1,2, sharey=True)
xlim1 = .32  ### Sets x min
xlim2 = .38  ### and max in THz
# =============================================================================
# xlim1 = W.min() 
# xlim2 = W.max()
# =============================================================================

ylim1=650        ###Sets ymin
ylim2 = 720      ###and max in pixel number ---- need to convert to wavelength
RealFFT = gx1.contourf(xgfft,ygfft,(divfft.real),100,
                       cmap='jet')
gx1.set_title('Re{FFT}')

gx1.set_xlim(xlim1,xlim2)
gx1.set_ylim(ylim1,ylim2)

ImgFFT = gx2.contourf(xgfft,ygfft,(divfft.imag),100,
                      cmap='jet')
gx2.set_title('Im{FFT}')

gx2.set_xlim(xlim1,xlim2)
gx2.set_ylim(ylim1,ylim2)

#%%   FFT Filtering

divfftfilt = divfft[ylim1:ylim2,1200:1650]  ###magic numbers for 840
divfftfiltRe = divfftfilt.real
Wfilt = W[1200:1650]
divfilt = np.fft.ifft(divfftfilt)
xfilt=np.linspace(0,divfilt.shape[1],divfilt.shape[1])
yfilt=np.linspace(0,divfilt.shape[0],divfilt.shape[0])

xgfilt, ygfilt = np.meshgrid(xfilt, yfilt)



filtplt, (filtaxRe,filtaxIm) = plt.subplots(1,2, sharey=True)

CSre = filtaxRe.contourf(xfilt,yfilt,divfilt.real,500,cmap='jet',vmin=-10,vmax=40)
#filtaxRe.set_xlim(16,45)

CSim = filtaxIm.contourf(xfilt,yfilt,divfilt.imag,500,cmap='jet',vmin=5,vmax=10)
#filtaxIm.set_xlim(16,45)


#%% interpolate

xgrid, ygrid = np.meshgrid(256,1024)

grid_z0 = griddata((xfilt,yfilt), divfilt.real, (xgrid, ygrid), method='nearest')

plt.contourf(xgrid, ygrid, grid_z0)













