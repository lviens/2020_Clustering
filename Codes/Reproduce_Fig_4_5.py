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
import matplotlib.pyplot as plt
import geopy.distance
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import scale


#%% Load functions
# Function to band-pass filter the waveforms with a Butterworth filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


init_fold = './' 
sys.path.append(init_fold)
from Function_clustering_DFs import Clustering_PCA_GMM
#%% Initial variables

np.random.seed(1) # To ensure data reproducibility

# Folders
dir_out = '../Figures' # Output directory for Figures
data_folder = '../Data' # Input data directory

# Number of PCs
n_components_PCA = 20 # Number of PCs to be kept

# Virtual source and receiver names and components
virt = 'M.KME18'
rec = 'N.ARIH'
cmp  = 'ZZ'
timesave = 300 # Time to select for the anti-causal and causal parts

# Period range of the band-pass filter
period1 = 4 # in s
period2 = 10 # in s
filtlow = 1/period2 # Frequency (in Hz)
filthigh = 1/period1 # Frequency (in Hz)

# Coordinates of the stations
coords_1 = (33.3860, 136.3828) # Coordinates of the virtual source
coords_2 = (34.6912, 137.5604 ) # Coordinates of the receiver
dist = (geopy.distance.distance(coords_1, coords_2).km) # Compute distance between the two stations


#%%
# Read and load data
name_virtZ =  data_folder + '/' + virt + '_' + rec + '_All_2016_Z_Z_365_0.5hr.mat' 
print(name_virtZ)
data_mat = sio.loadmat(name_virtZ) 
ZZfin2 = data_mat['ZZf'] # get DFs data
ZZmt=np.squeeze(data_mat['ZZmt'])
ZZdy=np.squeeze(data_mat['ZZdy'])
delta = 1/ data_mat['dt']
ZZdatetot=(data_mat['ZZtot'])
dt = data_mat['dt']

# Filter the data
ZZfin =[]
for i in range(len(ZZfin2)): 
    filtdata = []
    filtdata =  butter_bandpass_filter(ZZfin2[i], lowcut= filtlow, highcut = filthigh, fs = 1/dt, order=4) 
    ZZfin.append(filtdata[int(len(filtdata)/2 -timesave*1/dt ): int( len(filtdata)/2+ timesave *1/dt+1  )])


#%% Energy ratio stack method

tsign = int((dist/3.)*delta)
E_SNR =[]
                   
scoreamptotef = []
corrcoefinitot = []
for io in np.arange(len(ZZfin)):
    avetotef =ZZfin[io]
    E_spur = sum(avetotef[int(len(avetotef)/2)-tsign :int(len(avetotef)/2)+tsign ]**2)
    E_sign = sum(avetotef[int(len(avetotef)/2)+tsign :int(len(avetotef)/2)+tsign*3 ]**2)
    E_SNR.append(E_sign/E_spur)

perc =20
nb_stackso = int(perc*len(ZZfin)/100)
sort = np.argsort(E_SNR)[::-1]
ZZfinstc = []
for i in np.arange(nb_stackso):
    ZZfinstc.append(ZZfin[sort[i]])
finwavff20plt = np.mean(ZZfinstc,axis=0)

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
    
dtmt.append(len(ZZmt))
textval.append('2016/04' )

t = np.arange( -len(ZZfin[0])/2/delta , len(ZZfin[0])/2/delta, dt) # time vector

# Plot figure
plt.rcParams.update({'font.size': 14})
fig =plt.figure(figsize= (10,12))
gs = gridspec.GridSpec(3,1, height_ratios=[  4, 1,1]) 

ax0 = plt.subplot(gs[0])# Plot all the 30-min DFs
im =plt.imshow( ZZfin/np.max(np.abs(ZZfin)), aspect = 'auto', clim= (-.015 , .015), extent=[t[0],timesave, len(ZZfin), 0 ] , cmap = 'coolwarm')
plt.title( virt[2:] + '-' + rec[2:] + ' stations (' +cmp[0] +  '-' + cmp[1]  +  ' component, ' +str(len(ZZfin)) + ' DFs)\n Distance: ' + str(np.round(dist)) + ' km, BP filter: '+ str(period1) + '-' + str(period2) + ' s')
plt.text(-375, -10, '(a)')
plt.grid(linewidth=.3)
plt.grid(linewidth =.1, which = 'minor')
plt.ylabel('Waveform #', fontweight = 'bold')
cbbox = inset_axes(ax0, '10%', '20%', loc = 7)
cbbox.set_facecolor([1,1,1,0.9])
cbaxes = inset_axes(cbbox, '30%', '90%', loc = 6)
 
# Colorbar
cb=fig.colorbar(im,cax=cbaxes) #make colorbar
cb.set_ticks([-.01 , 0 , 0.01])
cb.set_ticklabels(['  - ', ' 0 ', ' + '])
cbbox.xaxis.set_visible(False)
cbbox.yaxis.set_visible(False)
ax2 = ax0.twinx()
ax2.set_ylim(0 , len(ZZfin) )
plt.yticks(dtmt, textval )
ax2.invert_yaxis()

# Plot stack over the year
ax1 = plt.subplot(gs[1])
avetot = np.mean(ZZfin,axis = 0 )
plt.plot(t, avetot/ np.max(np.abs(avetot)) , linewidth = 3,color ='k')
plt.ylabel('Norm. amp.', fontweight = 'bold')
plt.text(-375, 1.7, '(b)')
plt.ylim(-1.4 , 1.4)
plt.xlim(np.min(t), timesave)
plt.title('Raw stack'   )
plt.grid(linewidth =.3)
plt.grid(linewidth =.1, which = 'minor')

# Plot the Energy stack ratio DF
ax2 = plt.subplot(gs[2])

plt.plot(t, finwavff20plt/ np.max(np.abs(finwavff20plt)) , color = 'k', linewidth = 3)    
plt.ylabel('Norm. amp.', fontweight = 'bold')
plt.text(-375, 1.7, '(c)')
plt.ylim(-1.4 , 1.4)
plt.xlim(np.min(t), timesave)
plt.title('Energy ratio stack (' + str(perc)+ '%)'  )
plt.xlabel('Time (s)', fontweight = 'bold')
plt.fill_between(np.arange(-tsign/delta, tsign/delta, 1) , -2 , 2, color='grey', alpha = .2)
plt.fill_between(np.arange(tsign/delta, tsign/delta + tsign/delta*2, 1) , -2 , 2, color='grey', alpha = .4)
plt.text(-20, 1, 'Noise')
plt.text(101, 1, 'Signal')
plt.grid(linewidth =.3)
plt.grid(linewidth =.1, which = 'minor')

# Move the subplot positions
pos1 = ax0.get_position()
pos2 = [pos1.x0 , pos1.y0-.03 ,  pos1.width, pos1.height+.1 ]
ax0.set_position(pos2)
ax0.set_xticks(np.arange(-300,350, 50), minor = True)
ax0.tick_params(bottom=True, top=True, left=True, right=False)
ax0.tick_params(bottom=True, top=True, left=True, right=False, which = 'minor')

pos1 = ax1.get_position()
pos2 = [pos1.x0 , pos1.y0-.05 ,  pos1.width, pos1.height+.01 ]
ax1.set_position(pos2)
ax1.tick_params(bottom=True, top=True, left=True, right=True)
ax1.tick_params(bottom=True, top=True, left=True, right=True, which = 'minor')
ax1.set_xticks(np.arange(-300,350, 50), minor = True)
ax1.set_yticks(np.arange(-1,1.5, .5 ), minor = True)

pos1 = ax2.get_position()
pos2 = [pos1.x0 , pos1.y0-.07 ,  pos1.width, pos1.height+.01 ]
ax2.set_position(pos2)
ax2.tick_params(bottom=True, top=True, left=True, right=True)
ax2.set_xticks(np.arange(-300,350, 50), minor = True)
ax2.set_yticks(np.arange(-1,1.5, .5 ), minor = True)
ax2.tick_params(bottom=True, top=True, left=True, right=True, which = 'minor')

plt.show() 
fig.savefig(dir_out+ '/Fig_4.png', dpi=100)


#%% Standardize the data (scale function from sklearn)
X_train_minmax = scale(ZZfin)
x_or = np.array(ZZfin)

#%% Perform clustering
initval = 2
range_GMM = np.arange(initval, 16) # To check the best number of cluster. The code gets the best number of clusters in the given range
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

fnt =11
plt.rcParams.update({'font.size': fnt})
figa, axs = plt.subplots(n_clusters, 2,figsize=(9.5,8) )
figa.set_rasterized(True)

# Plot Fig. 5a
axs[0,0].set_aspect('equal')
axs[0,0].set_rasterized(True)
axs[0,0].scatter(pca_output[:,0], pca_output[:, 1], c=colors, alpha=0.25, s = 10 ) #
a0test = [j for j, e in enumerate(gmixfinPCA) if e == savescoreind]
axs[0,0].scatter(pca_output[a0test,0], pca_output[a0test, 1], c=colors[a0test], alpha=0.25, s = 10 ) #
axs[0,0].set_xlabel('Principal component 1', fontsize=fnt)
axs[0,0].set_ylabel('Principal component 2', fontsize=fnt)
axs[0,0].set_title(virt[2:] +' - '+ rec[2:] + ' stations \n' + cmp[0] +  '-' + cmp[1]  + ' component \n PCA + GMM clustering', fontsize=fnt)
axs[0,0].grid(linewidth = .3)
axs[0,0].text(-135, 120, '(a)', fontsize=14)
axs[0,0].set_xlim(-100, 100)
axs[0,0].set_ylim(-95, 95)

# Plot Fig. 5b
axs[1,0].plot(range_GMM, BICF, linewidth =3)
axs[1,0].plot(n_clusters, BICF[n_clusters-initval], 'or', markersize =10)
axs[1,0].set_xlabel('Number of Clusters', fontsize=fnt)
axs[1,0].set_xlim(2, 15)
axs[1,0].set_ylabel('BIC score', fontsize=fnt)
axs[1,0].text(-1, BICF[0]+ BICF[0]*.001, '(b)', fontsize=14)
axs[1,0].grid(linewidth = .3)
axs[1,0].set_rasterized(True)
axs[1,0].set_xticks(np.arange(2,16,2))
 
# Plot Fig. 5c
ampmulplt = 1000
for i in np.arange(n_clusters): # Loop on the clusters to stack the data
    a0test = [j for j, e in enumerate(gmixfinPCA) if e == i]
    a0 =[]
    for tt in a0test:
        if probs[tt,i]>0.0: # As the GMM is a soft clustering algorithm, selecting only the DFs with high (>.95) probability of belonging to the cluster could improve the results.
            a0.append(tt)          
    meanb1 = []
    meanb1 = np.mean(x_or[a0] ,axis = 0 )

    axs[i,1].plot(t,meanb1*ampmulplt, label = 'normal', color =  c_dict[i], linewidth =1.5)
    if i == savescoreind:
        axs[i,1].set_title('Cluster ' +str(i+1) + ': '+ str(len(a0))+ ' waveforms (selected)'  , fontsize=fnt)
    else: 
        axs[i,1].set_title('Cluster ' +str(i+1) + ': '+ str(len(a0))+ ' waveforms'  , fontsize=fnt)
    axs[i,1].set_ylabel('Amplitude\n(x ' + str(ampmulplt) + ')', fontsize=fnt)
    if i ==0:
        saveamp0 = np.max(np.abs(meanb1*ampmulplt))     
    axs[i,1].set_ylim(-np.max(np.abs(meanb1*ampmulplt))- .05*saveamp0, np.max(np.abs(meanb1*ampmulplt))+ .05*saveamp0)
    axs[i,1].set_xlim(-timesave, timesave)
    axs[i,1].grid()
    axs[i,1].set_rasterized(True)
       
axs[i,1].set_xlabel('Time (s)', fontsize=fnt)
axs[0,1].text(-450, saveamp0 + .125*saveamp0, '(c)', fontsize=fnt+2)
axs[i,1].grid(linewidth = .3)
   
# Set the position of each subplot
axs[0,0].set_position([0.085, .55, .35, .35])
axs[1,0].set_position([0.115, .12, .3, .3])
if n_clusters == 4:
    axs[0,1].set_position([0.55, .79, .42, .16])
    axs[1,1].set_position([0.55, .55, .42, .16])
    axs[2,1].set_position([0.55, .31, .42, .16])
    axs[3,1].set_position([0.55, .075, .42, .16])
    axs[2,0].axis('off')
    axs[3,0].axis('off')
elif n_clusters == 5:
    ayy = 0.19
    iniy = 0.065
    axs[0,1].set_position([0.55, iniy+ayy*4, .42, .12])
    axs[1,1].set_position([0.55, iniy+ayy*3, .42, .12])
    axs[2,1].set_position([0.55, iniy+ayy*2, .42, .12])
    axs[3,1].set_position([0.55, iniy+ayy , .42, .12])
    axs[4,1].set_position([0.55, iniy, .42, .12])
    axs[2,0].axis('off')
    axs[3,0].axis('off')
    axs[4,0].axis('off')
figa.savefig(dir_out + '/Fig_5.png', dpi=300)
