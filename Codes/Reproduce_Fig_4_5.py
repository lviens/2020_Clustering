#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:40:23 2020

@author: Loic
"""

import  sys
import numpy as np
from matplotlib import gridspec
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
import geopy.distance
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#%% Load functions
# Function to band-pass filter the waveforms with a Butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


init_fold ='./' 
sys.path.append(init_fold)
from Function_clustering_DFs import Clustering_PCA_GMM
#%% Initial variables

np.random.seed(1) # To ensure data reproducibility

# Folders

# Output directory for Figures
dir_out = '../Figures'
data_folder = '../Data'

# Number of PCs
n_components_PCA = 10 # Number of PCs to be kept

# Virtual source and receiver names and components
virt = 'M.KME18'
rec = 'N.ABNH'
cmp  = 'ZZ'
timesave = 300 # Time to select for the anti-causal and causal parts

# Period range of the band-pass filter
period1 = 4 # in s
period2 = 10 # in s
filtlow = 1/period2 # Frequency (in Hz)
filthigh = 1/period1 # Frequency (in Hz)


# Coordinates of the stations
coords_1 = (33.3860, 136.3828) # Coordinates of the virtual source
coords_2 = (34.6325, 137.2313) # Coordinates of the receiver


#%%

dist = (geopy.distance.distance(coords_1, coords_2).km) # Compute distance between the two stations

# Read and load data
name_virtZ =  data_folder + '/' + virt + '_' + rec + '_All_2015_2016_Z_Z_365_0.5hr.mat' 
data_mat = sio.loadmat(name_virtZ) 
ZZfin2 = data_mat['ZZf'] # get DFs data
ZZmt=np.squeeze(data_mat['ZZmt'])
ZZdy=np.squeeze(data_mat['ZZdy'])
delta = 1/ data_mat['dt']
ZZdatetot=(data_mat['ZZtot'])
dt = data_mat['dt']

# Filter and taper the data
ZZfin =[]
hann2 = signal.hann(21) 
halflen = int((len(hann2)-1)/2+1 )
for i in range(len(ZZfin2)): 
    filtdata = []
    filtdata =  butter_bandpass_filter(ZZfin2[i], lowcut= filtlow, highcut = filthigh, fs = 1/dt, order=4) 
    filtdata[int(len(filtdata)/2):int(len(filtdata)/2)+halflen] = filtdata[int(len(filtdata)/2):int(len(filtdata)/2)+halflen] *hann2[:halflen]
    filtdata[int(len(filtdata)/2)-(halflen-1):int(len(filtdata)/2)] = filtdata[int(len(filtdata)/2)-(halflen-1):int(len(filtdata)/2)] *hann2[halflen:]
    ZZfin.append(filtdata[int(len(filtdata)/2 -timesave*1/dt +1): int( len(filtdata)/2+ timesave *1/dt+1  )])


#%% Plot figure 4
# Create right-hand side vertical axis
dtmt=[]
textval=[]
for ti in np.arange(4,13):
    dtmt.append(np.where(ZZmt==ti)[-1][0])
    textval.append('2015/' + str(int(ti)).zfill(2) )

for ti in np.arange(1,4):
    dtmt.append(np.where(ZZmt==ti)[-1][0])
    textval.append('2016/' + str(int(ti)).zfill(2) )

t = np.arange( -len(ZZfin[0])/2/delta , len(ZZfin[0])/2/delta, dt) # time vector

# Plot figure
plt.rcParams.update({'font.size': 14})
fig =plt.figure(figsize= (10,10))
# Plot all the 30-min DFs
gs = gridspec.GridSpec(2,1, height_ratios=[1,3]) 
ax0 = plt.subplot(gs[1])
im =plt.imshow( ZZfin/np.max(np.abs(ZZfin)), aspect = 'auto', clim= (-.015 , .015), extent=[t[0],timesave, len(ZZfin), 0 ] , cmap = 'coolwarm')
plt.text(-375, -10, '(b)')
plt.grid(linewidth=.2)
plt.xlabel('Time (s)', fontweight = 'bold')
plt.ylabel('Waveform #', fontweight = 'bold')

#make colorbar
cbbox = inset_axes(ax0, '10%', '20%', loc = 7)
cbbox.tick_params(axis='both' ,labelleft='False', labeltop='False', labelright='False', labelbottom='False') 
cbbox.set_facecolor([1,1,1,0.9])
cbaxes = inset_axes(cbbox, '30%', '90%', loc = 6)

cb=fig.colorbar(im,cax=cbaxes)
cb.set_ticks([-.01 , 0 , 0.01])
cb.set_ticklabels(['  - ', ' 0 ', ' + '])
cbbox.xaxis.set_visible(False)
cbbox.yaxis.set_visible(False)

ax2 = ax0.twinx()
ax2.set_ylim(0, len(ZZfin))
plt.yticks(dtmt, textval )
ax2.invert_yaxis()

# Plot stack over the year
ax1 = plt.subplot(gs[0])
avetot = np.mean(ZZfin,axis = 0 )
plt.plot(t, avetot/ np.max(np.abs(avetot)) , linewidth = 3)
plt.ylabel('Normalized amp.', fontweight = 'bold')
plt.text(-375, 1.4, '(a)')
plt.ylim(-1.1 , 1.1)
plt.xlim(t[0], t[-1])
plt.title('Raw stack: '  + virt[2:] + '-' + rec[2:] + ', ' +cmp[0] +  '-' + cmp[1]  +  ' (' + str(np.round(dist)) + ' km), Filter: '+ str(period1) + '-' + str(period2) + ' s' )
plt.grid()
plt.show() 
fig.savefig(dir_out+ '/Fig_4.pdf', dpi=200)


#%% Scale the data
min_max_scaler = preprocessing.RobustScaler()
X_train_minmax = min_max_scaler.fit_transform(ZZfin)

ZZfin3 = X_train_minmax
x = np.array(ZZfin3)
x_or = np.array(ZZfin)

#%% Perform clustering
initval = 2
range_GMM = np.arange(initval, 16) # To check the best number of cluster. The code will determine the best number of clusters in the given range
pca_output, var, models, n_clusters, gmixfinPCA, probs, BICF = Clustering_PCA_GMM(X_train_minmax, n_components_PCA,range_GMM)

#%% Assign colors to each cluster
c_dict =  {0:'r',1:'b',2:'g',3:'orange',4:'m',5:'r',6:'b',7:'g',8:'y',9:'k',10:'r',11:'b',12:'g',13:'y',14:'k'}
colors = np.array([c_dict[i] for i in gmixfinPCA])

#%% Compute the variance of the data on the first 2 PCs and select the cluster with the lowest variance
savestdpca = 10000
savescoreind = []
for i in np.arange(n_clusters):
    stdpca = []
    a0test = [j for j, e in enumerate(gmixfinPCA) if e == i]
    stdpca = np.var(pca_output[a0test,0:2])
    if stdpca < savestdpca and len(pca_output[a0test,0])>100:
        savescoreind = i
        savestdpca = stdpca

#%% Plot Figure 5
fnt = 11
plt.rcParams.update({'font.size': fnt})
a0test = [j for j, e in enumerate(gmixfinPCA) if e == savescoreind]

figa, axs = plt.subplots(n_clusters, 2,figsize=(10,8) )
figa.set_rasterized(True)

axs[0,0].set_aspect('equal')
axs[0,0].set_rasterized(True)
axs[0,0].scatter(pca_output[:,0], pca_output[:, 1], c=colors, alpha=0.25, s = 10 ) #
axs[0,0].scatter(pca_output[a0test,0], pca_output[a0test, 1], c=colors[a0test], alpha=0.25, s = 10 ) 
axs[0,0].set_xlabel('Principal component 1', fontsize=fnt)
axs[0,0].set_ylabel('Principal component 2', fontsize=fnt)
axs[0,0].text(-90, 110, '(a)', fontsize=14)
axs[0,0].set_title(virt[2:] +' - '+ rec[2:] + ' stations \n' + cmp[0] +  '-' + cmp[1]  + ' components \n PCA + GMM clustering', fontsize=fnt)
axs[0,0].grid(linewidth = .3)

axs[1,0].plot(range_GMM, BICF, linewidth =3)
axs[1,0].plot(n_clusters, BICF[n_clusters-initval], 'or', markersize =10)
axs[1,0].set_xlabel('Number of Clusters', fontsize=fnt)
axs[1,0].set_xticks(np.linspace(2,16, 8))
axs[1,0].set_xlim(2, 15)
axs[1,0].set_ylabel('BIC score', fontsize=fnt)
axs[1,0].text(-.5, BICF[0]+ BICF[0]*.001, '(b)', fontsize=14)
axs[1,0].grid(linewidth = .3)
axs[1,0].set_rasterized(True)


ampmulplt = 1000
axtest = []
for i in np.arange(n_clusters):
    scoreamp = []
    a0test = [j for j, e in enumerate(gmixfinPCA) if e == i]
    meanb1 = np.mean(x_or[a0test] ,axis = 0 )
    
    axs[i,1].plot(t,meanb1*ampmulplt, label = 'normal', color =  c_dict[i], linewidth =1.5)
    if i == savescoreind:
        axs[i,1].set_title('Cluster ' +str(i+1) + ': '+ str(len(a0test))+ ' waveforms (selected)'  , fontsize=fnt)
    else: 
        axs[i,1].set_title('Cluster ' +str(i+1) + ': '+ str(len(a0test))+ ' waveforms'  , fontsize=fnt)
    axs[i,1].set_ylabel('Amplitude\n(x ' + str(ampmulplt) + ')', fontsize=fnt)
    if i ==0:
        saveamp0 = np.max(np.abs(meanb1*ampmulplt))
        
    axs[i,1].set_ylim(-np.max(np.abs(meanb1*ampmulplt))- .05*saveamp0, np.max(np.abs(meanb1*ampmulplt))+ .05*saveamp0)
    axs[i,1].set_xlim(-timesave, timesave)
    axs[i,1].grid()
    axs[i,1].set_rasterized(True)
    
axs[i,1].set_xlabel('Time (s)', fontsize=fnt)
axs[i,1].grid(linewidth = .3)
                
axs[0,0].set_position([0.065, .5, .4, .4])
axs[1,0].set_position([0.125, .09, .3, .3])
axs[0,1].text(-450, saveamp0 + .15*saveamp0, '(c)', fontsize=fnt+2)
 
axs[0,1].set_position([0.55, .79, .42, .16])
axs[1,1].set_position([0.55, .55, .42, .16])
axs[2,1].set_position([0.55, .31, .42, .16])
axs[3,1].set_position([0.55, .075, .42, .16])
axs[2,0].axis('off')
axs[3,0].axis('off')

plt.show() 
figa.savefig(dir_out + '/Fig_5.pdf', dpi=300)

