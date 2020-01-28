

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:11:57 2019

@author: briansquires, joetoney
"""

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from numba import jit
from scipy import signal
from matplotlib import ticker, cm
from skimage.feature import peak_local_max

#%% Data Import

databbo = np.loadtxt("/Users/tracebivens/Documents/GitHub/iFROG/200123_BBO_865nm_100mW_100mn_1slit_432grating_-11__27t0", delimiter="\t")
databbo = np.delete(databbo, databbo.shape[1]-1, axis=1)
datamos2 = np.loadtxt("/Users/tracebivens/Documents/GitHub/iFROG/200123_MoS2_865nm_100mW_100mn_1slit_432grating_-11__27t0", delimiter="\t")
datamos2 = np.delete(datamos2, datamos2.shape[1]-1, axis=1)

posb=np.loadtxt("/Users/tracebivens/Documents/GitHub/iFROG/200123_BBO_865nm_100mW_100mn_1slit_432grating_-11__27t0_pos", delimiter="\t")
#posb=np.delete(posb,5000,axis=0)
posm=np.loadtxt("/Users/tracebivens/Documents/GitHub/iFROG/200123_MoS2_865nm_100mW_100mn_1slit_432grating_-11__27t0_pos", delimiter="\t")
#posm=np.delete(posm,5000,axis=0)
#%%


#truncation (data specific)
# =============================================================================
# datamos2=datamos2[::,:6080]
# posm=np.resize(posm,[6080,])
# =============================================================================

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

def CheckR_M(data):
    summed = np.sum(data, axis=0)
    summed = summed - summed.min()
    r = summed[3900] / summed.max()
    m = (np.abs(np.log2(r)) + 1) / 2
    return r, m, summed
#%%
div=datamos2/(databbo)


#%%Joe Plot
datamos2=datamos2
plt.matshow(datamos2, aspect='auto')
plt.show()

# =============================================================================
# #%%Variance
# var = [] 
# posold = 0
# for i , pos in enumerate(posm):
#     var.append(posold - pos)
#     posold = pos
# var = var[1::]
# s=(np.var(var))**-2
# m=np.var(var)
# mean=np.mean(var)
# print(var)
# print(s)
# print(m)
# print(mean)
# =============================================================================

#%% Plot Data

f , (ax1,ax2,ax3) =plt.subplots(1, 3, sharey=True)

f.suptitle('840nm')

bboplt = ax1.contourf(xbg,ybg,databbo,np.linspace(databbo.min(), databbo.max(), 1000), cmap='jet')
ax1.set_title('BBO')
ax1.set_xlim(-500,500)
#ax1.set_ylim(600,800)

cbar = f.colorbar(bboplt)


mos2plt = ax2.contourf(xmg,ymg,datamos2,np.linspace(datamos2.min(), datamos2.max(), 1000),cmap='jet')
ax2.set_title('MoS2')
ax2.set_xlim(-500,500)
#ax2.set_ylim(600,800)

divplt= ax3.contour(xbg,ybg,div,np.linspace(div.min(), div.max(), 1000),cmap='jet')
ax3.set_title('MoS2/BBO')
ax3.set_xlim(-500,500)
#ax3.set_ylim(600,800)


#%%Matplot Data
f , (ax1,ax2,ax3) =plt.subplots(1, 3, sharex=True, sharey=True)

f.suptitle('840nm')


bboplt = ax1.matshow(databbo, aspect='auto')
ax1.set_title('BBO')
#ax1.set_xlim(-500,500)
ax1.set_ylim(500,700)

cbar = f.colorbar(bboplt)


mos2plt = ax2.matshow(datamos2, aspect='auto')
ax2.set_title('MoS2')
#ax2.set_xlim(-500,500)
ax2.set_ylim(500,700)

divplt= ax3.matshow(div, aspect='auto')
ax3.set_title('MoS2/BBO')
#ax3.set_xlim(-500,500)
ax3.set_ylim(500,700)






#%% Fourier Transform
timestep=0.66713  ##this is the time for one 100nm step.  This returns the FFT
                    # x-axis in THz


###   Try np.fft.rfft and np.fft.hfft
###   Check to make sure the resulting plots are symmetric in the frequency domain

bfft = np.fft.fft(databbo)
bfft = np.fft.fftshift(bfft, axes=1)
Wb = np.fft.fftfreq(posb.size,timestep)
Wb = np.fft.fftshift(Wb)

xfft=Wb
yfft=np.linspace(1,bfft.shape[0],num=bfft.shape[0])
xgbfft, ygbfft =np.meshgrid(xfft,yfft)

mfft = np.fft.fft(datamos2)
mfft = np.fft.fftshift(mfft, axes=1)
Wm = np.fft.fftfreq(posm.size,timestep)   
Wm = np.fft.fftshift(Wm)      

xfft=Wm
yfft=np.linspace(1,mfft.shape[0],num=mfft.shape[0])
xgmfft, ygmfft =np.meshgrid(xfft,yfft)

# =============================================================================
# divfft=np.fft.fft(div)
# divfft = np.fft.fftshift(divfft, axes=1)
# W=np.fft.fftfreq(div[1].size,timestep)
# W = np.fft.fftshift(W)
# 
# xfft=W
# yfft=np.linspace(1,divfft.shape[0],num=divfft.shape[0])
# xgdfft, ygdfft =np.meshgrid(xfft,yfft)
# 
# =============================================================================

#%% Plot FFT
#g, (gx1,gx2) = plt.subplots(1,2, sharey=True)
#xlim1 = .32  ### Sets x min
#xlim2 = .38  ### and max in THz
## =============================================================================
##xlim1 = W.min() 
##xlim2 = W.max()
## =============================================================================
#
#ylim1=650        ###Sets ymin
#ylim2 = 720      ###and max in pixel number ---- need to convert to wavelength
#RealFFT = gx1.contourf(xgdfft,ygdfft,(divfft.real),100,
#                       cmap='jet')
#gx1.set_title('Re{FFT}')
#
#gx1.set_xlim(xlim1,xlim2)
#gx1.set_ylim(ylim1,ylim2)
#
#ImgFFT = gx2.contourf(xgdfft,ygdfft,(divfft.imag),100,
#                      cmap='jet')
#gx2.set_title('Im{FFT}')
#
#gx2.set_xlim(xlim1,xlim2)
#gx2.set_ylim(ylim1,ylim2)

#%% Plot FFT NEW
g, ((gx1,gx2),(gx3,gx4),(gx5,gx6)) = plt.subplots(3,2, sharex=True, sharey=True)

# =============================================================================
# xlim1 = .32  ### Sets x min
# xlim2 = .38  ### and max in THz
# =============================================================================
xlim1 = W.min() 
xlim2 = W.max()

ylim1=500        ###Sets ymin
ylim2 = 700      ###and max in pixel number ---- need to convert to wavelength

# =============================================================================
# divfft = np.absolute(divfft)
# mfft = np.absolute(mfft)
# bfft = np.absolute(bfft)
# =============================================================================


RealFFT = gx1.contourf(xgdfft,ygdfft,(divfft.real),100,cmap='jet')
gx1.set_title('Re{FFT}')

gx1.set_xlim(xlim1,xlim2)
gx1.set_ylim(ylim1,ylim2)

ImgFFT = gx2.contourf(xgdfft,ygdfft,(divfft.imag),100,cmap='jet')
gx2.set_title('Im{FFT}')

gx2.set_xlim(xlim1,xlim2)
gx2.set_ylim(ylim1,ylim2)



RealFFT = gx3.contourf(xgmfft,ygmfft,(mfft.real),100,cmap='jet')
gx3.set_title('M Re{FFT}')

gx3.set_xlim(xlim1,xlim2)
gx3.set_ylim(ylim1,ylim2)

ImgFFT = gx4.contourf(xgmfft,ygmfft,(mfft.imag),100,cmap='jet')
gx4.set_title('M Im{FFT}')

gx4.set_xlim(xlim1,xlim2)
gx4.set_ylim(ylim1,ylim2)



RealFFT = gx5.contourf(xgbfft,ygbfft,(bfft.real),100,cmap='jet')
gx5.set_title('B Re{FFT}')

gx5.set_xlim(xlim1,xlim2)
gx5.set_ylim(ylim1,ylim2)

ImgFFT = gx6.contourf(xgbfft,ygbfft,(bfft.imag),100,cmap='jet')
gx6.set_title('B Im{FFT}')

gx6.set_xlim(xlim1,xlim2)
gx6.set_ylim(ylim1,ylim2)


#%% Window FFT Filter
from scipy import signal
window = signal.general_gaussian(bfft[1].size, p=2, sig=300)
plt.plot(window)

#%%   FFT Filtering
xlim1 = 2912
xlim2 = 3168
ylim1 = 480
ylim2 = 736

mfftfilt = mfft[ylim1:ylim2,xlim1:xlim2]
bfftfilt = bfft[ylim1:ylim2,xlim1:xlim2]
divfftfilt = divfft[ylim1:ylim2,xlim1:xlim2]  

mfilt = np.fft.ifft(mfftfilt)
Tmfilt = np.fft.fftfreq(Wm.size,Wm[1]-Wm[0])
bfilt = np.fft.ifft(bfftfilt)
Tbfilt = np.fft.fftfreq(Wb.size,Wb[1]-Wb[0])
divfilt = np.fft.ifft(divfftfilt)
Tdfilt = np.fft.fftfreq(W.size,W[1]-W[0])
xfilt=np.linspace(0,divfilt.shape[1],divfilt.shape[1])
yfilt=np.linspace(0,divfilt.shape[0],divfilt.shape[0])

xgfilt, ygfilt = np.meshgrid(xfilt, yfilt)



filtplt, (filtaxM,filtaxB, filtaxD) = plt.subplots(1,3, sharex=True, sharey=True)

#CSM = filtaxM.contourf(xfilt,yfilt,np.abs(mfilt),500,cmap='jet')#,vmin=-10,vmax=10)
#filtaxRe.set_xlim(16,45)

CSB = filtaxB.contourf(xfilt,yfilt,np.abs(bfilt),500,cmap='jet')#,vmin=-10,vmax=10)
#filtaxIm.set_xlim(16,45)

#CSD = filtaxD.contourf(xfilt,yfilt,np.abs(divfilt),500,cmap='jet')#,vmin=-10,vmax=10)
#%% Find t0
def FindTZero(data_array):
    """
    Finds t0 in the raw iFROG trace.
    NOT WORKING
    Parameters
    ----------
    data_array : ndarray
        2d numpy array containing raw iFROG data.

    Returns
    -------
    t0 : int
        The index value of the absolute maximum of the input array.

    """
    t0 = peak_local_max(data_array, num_peaks=1)
    return t0
#%% get FROG traces
def FrogMath(fft_array, wav, bandwidth=128, norm=False):
    """
    Workflow: do bbo_wavelength = FrogMath(bfft, wav)
    or mos2_wavelength = FrogMath(mfft, wav)
    wav can be found at the bottom of the script

    Parameters
    ----------
    fft_array : TYPE
        DESCRIPTION.
    wav : TYPE
        DESCRIPTION.
    bandwidth : TYPE, optional
        DESCRIPTION. The default is 128.
    norm : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    peaks, band_indices, match = FindBandIndices(fft_array)
    center = peaks[2,0], peaks[2,1]
    center_peak = fft_array[center]
    if norm == True:
        fft_array = fft_array / center_peak
    else:
        pass
    dc = fft_array[wav-bandwidth:wav+bandwidth,band_indices[0]-bandwidth:band_indices[0]+bandwidth]
    fm = fft_array[wav-bandwidth:wav+bandwidth,band_indices[1]-bandwidth:band_indices[1]+bandwidth]
    shg = fft_array[wav-bandwidth:wav+bandwidth,band_indices[2]-bandwidth:band_indices[2]+bandwidth]
    window = signal.windows.general_gaussian(dc[1].size, p=2, sig=bandwidth)
    dc = dc * window
    fm = fm * window
    shg = shg * window
    dc_ifft = np.fft.ifft(dc)
    fm_ifft = np.fft.ifft(fm)
    shg_ifft = np.fft.ifft(shg)
    dc_mag = np.abs(dc)
    dc_ifft_mag = np.abs(dc_ifft)
    fm_mag = np.abs(fm)
    fm_ifft_mag = np.abs(fm_ifft)
    shg_mag = np.abs(shg)
    shg_ifft_mag = np.abs(shg_ifft)
    
    return {'dc':dc,'dc_mag':dc_mag, 'dc_ifft':dc_ifft,
            'dc_ifft_mag':dc_ifft_mag, 'fm':fm, 'fm_mag':fm_mag, 
            'fm_ifft':fm_ifft, 'fm_ifft_mag':fm_ifft_mag, 'shg':shg,
            'shg_mag':shg_mag, 'shg_ifft':shg_ifft,
            'shg_ifft_mag':shg_ifft_mag, 'band_indices':band_indices,
            'match':match}
#%%
def FrogLoop(fft_array, pump_wavelength, optical_bandwidth,
             delay_bandwidth=250):
    """
    IGNORE, NOT WORKING YET

    Parameters
    ----------
    fft_array : TYPE
        DESCRIPTION.
    pump_wavelength : TYPE
        DESCRIPTION.
    optical_bandwidth : TYPE
        DESCRIPTION.
    delay_bandwidth : TYPE, optional
        DESCRIPTION. The default is 250.

    Returns
    -------
    fft_mag : TYPE
        DESCRIPTION.
    ifft_mag : TYPE
        DESCRIPTION.

    """
    shg_wavelength = pump_wavelength // 2
    wavelength_high = shg_wavelength + (optical_bandwidth // 2)
    wavelength_low = shg_wavelength - (optical_bandwidth // 2)
    band_indices = FindBandIndices(fft_array.real)
    clipped = [fft_array[wavelength_low : wavelength_high,
                         i - delay_bandwidth : i + delay_bandwidth]
               for i in band_indices]
    window = signal.windows.general_gaussian(delay_bandwidth*2, p=2, 
                                             sig=delay_bandwidth)
    smoothed = [i * window for i in clipped]
    del(window)
    ifft = [np.fft.ifft(i) for i in smoothed]
    ifft_mag = [np.abs(i) for i in ifft]
    fft_mag = [np.abs(i) for i in smoothed]
    
    return fft_mag, ifft_mag
#%% Find local maxima in FFT arrays
def FindBandIndices(fft_array, with_neg = False):
    """
    Finds the 5 expected band maxima in a fourier-transformed iFROG trace
    and returns the corresponding indices. Requires real values.

    Parameters
    ----------
    fft_array : 2d complex array
        iFROG data after FFT has been performed.
    with_neg : bool
        Set to True to retrieve all frequency bands, otherwise returns only
        dc and positive frequencies.

    Returns
    -------
    shg_neg : int
        The index value of the maximum in the negative frequency component
        of the shg band.
    fm_neg : int
        The index value of the maximum in the negative frequency component
        of the fm band.
    dc : int
        The index value of the maximum in the DC base band.
    fm_pos : int
        The index value of the maximum in the positive frequency component
        of the fm band.
    shg_pos: int
        The index value of the maximum in the positive frequency component 
        of the shg band.

    """
    peaks = peak_local_max(fft_array.real, min_distance = 100, num_peaks=5)
    band_indices = peaks[:,1]
    band_indices.sort()
    shg_neg = band_indices[0]
    fm_neg = band_indices[1]
    dc = band_indices[2]
    fm_pos = band_indices[3]
    shg_pos = band_indices[4]
    match = CheckIndices(band_indices, fft_array)
    if with_neg == False:
        return peaks, [dc, fm_pos, shg_pos], match
    else:
        return peaks, band_indices, match
    
#%% Check peak distances
def CheckIndices(band_indices, fft_array):
    """
    Helper function. Verifies that the band indices are equally spaced.

    Parameters
    ----------
    band_indices : int list
        Array of indices.
    fft_array : 2d array
        Fourier transformed iFROG trace.

    Returns
    -------
    None.

    """
    if band_indices[2] == fft_array.shape[1] / 2:
        print('DC band ok')
        dc = True
    else:
        print('DC peak location mismatch. Check peak finding params.')
        dc = False
    if band_indices[2]-band_indices[1] == band_indices[3]-band_indices[2]:
        print('FM bands ok')
        fm = True
    else:
        print('FM band mismatch. Check peak finding params.')
        fm = False
    if band_indices[2]-band_indices[0] == band_indices[4]-band_indices[2]:
        print('SHG bands ok')
        shg = True
    else:
        print('SHG band mismatch. Check fitting params.')
        shg = False
    match = [dc, fm, shg]
    return match
        
#%% select bands
def SelectBands(dc_band, fm_band, shg_band, fft_array, bandwidth=200):
    """
    Ignore until further notice

    Parameters
    ----------
    dc_band : TYPE
        DESCRIPTION.
    fm_band : TYPE
        DESCRIPTION.
    shg_band : TYPE
        DESCRIPTION.
    fft_array : TYPE
        DESCRIPTION.
    bandwidth : TYPE, optional
        DESCRIPTION. The default is 200.

    Returns
    -------
    window : TYPE
        DESCRIPTION.
    window_neg : TYPE
        DESCRIPTION.
    window_pos : TYPE
        DESCRIPTION.

    """
    window = signal.windows.general_gaussian(fft_array[1].size, p=2,
                                             sig=bandwidth)
    window_neg = np.append(window[neg_band:window.size], np.zeros(neg_band))
    window_pos = np.append(np.zeros(window.size-pos_band), window[0:pos_band])
    window = window_neg + window_pos
    return window, window_neg, window_pos
    
#%%thing
def Norm(array):
    norm = array - array.min()
    div = array.max() - array.min()
    norm = norm / div
    return norm
#%%
def MagPlot(dictionary, vmin=0, vmax=50000):
    plotlist = ['dc_mag', 'dc_ifft_mag', 'fm_mag', 'fm_ifft_mag',
                'shg_mag', 'shg_ifft_mag']
    for i in plotlist:
        plt.matshow(dictionary[i], vmin=vmin, vmax=vmax)
#%%
def StretchPlot(dictionary):
    plotlist = ['dc_mag', 'dc_ifft_mag', 'fm_mag', 'fm_ifft_mag',
                'shg_mag', 'shg_ifft_mag']
    for i in plotlist:
        plt.matshow(dictionary[i])
#%%
def FrogSave(dictionary, name):
    if all(dictionary['match']) == True:
        fm_ifft_mag = dictionary['fm_ifft_mag']
        np.savetxt(name, fm_ifft_mag, delimiter='\t')
    else:
        fm_ifft_mag = dictionary['shg_ifft_mag']
        np.savetxt(name, fm_ifft_mag, delimiter='\t')
# =============================================================================
# #%% interpolate
# 
# xgrid, ygrid = np.meshgrid(256,1024)
# 
# grid_z0 = griddata((xfilt,yfilt), divfilt.real, (xgrid, ygrid), method='nearest')
# 
# plt.contourf(xgrid, ygrid, grid_z0)
# =============================================================================

"""
fundamental window sigma: 200
fundamental slice width: 400
fm window sigma: 300
fm slice width: 600
shg window sigma: 100
shg slice width: 200

840 center: 670
850 center: 600
865 center: 485
870 center: 450
"""