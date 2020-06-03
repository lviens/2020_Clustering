#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:31:47 2020

@author: Loic
"""

from __future__ import division
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import chirp
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from Function_clustering_DFs import Clustering_PCA_GMM

#%%
np.random.seed(1)
dir_ini = './'
sys.path.append(dir_ini)

# Output directory for Figures
dir_out = '../Figures'

#%% Set up the Chirp functions
maxtraing_len = int(10000) # Total number of waveforms
delta = 2 # sampling rate
t_len = 200 # Total duration of the signal in s from -len(t_len)/2 to  len(t_len)/2
t = np.linspace(0,t_len,int(t_len*delta))-100 # Time vector of each waveform


hann2 = signal.hann(21) # Hannin window to taper the signals
half_len =round(len(hann2)/2-.001)+1 # Half lenght of the taper

noise_level = 1. # Level of noise (1 -> max value is the max amplitude of the waveform without noise)
nbosc = .114  # Frequency of the spurious wave 

 
nbinitt = [2000, 4000 ,6000]
ZZfin = []
cpt = 0
cpt2 = 0
cpt3 = 0
cpt4 = 0
cpt5 = 0
xrawsav = []
indclus = []
indo =11
for i in range(0,maxtraing_len):
    x1 = np.zeros(int(t_len*delta))
    lend = round((( (len(t)- len(t)/indo )) -(len(t)/2 + len(t)/indo)) /2)
    
    # Create chirp waveform and taper.
    chirpp =0.5*chirp(np.arange(0,lend, 1/delta), f0 = .05,t1 =lend, f1=.25,method='linear' )
    chirpp[:half_len]=chirpp[:half_len]*hann2[:half_len]
    chirpp[-half_len:]=chirpp[-half_len:]*hann2[-half_len:]
    
    # Create spurious waveform and taper
    spurr=.75*(-np.cos(nbosc*2*np.pi*t[int(len(t)/2 - len(t)/indo): int(len(t)/2+ len(t)/indo )]) )
    spurr[:half_len]=spurr[:half_len]*hann2[:half_len]
    spurr[-half_len:]=spurr[-half_len:]*hann2[-half_len:]
    
    if  (i>=0 and i<nbinitt[0]): # Causal and acausal
        x1[int(len(t)/2 + len(t)/indo): int(len(t)- len(t)/indo+1 )] = chirpp
        x1[int(len(t)/indo): int(len(t)/2 - len(t)/indo+1 )] = chirpp[::-1]
        if i ==0:
            xrawsav.append(x1)
        indclus.append(3)
        cpt3 +=1
        
    elif (i>= nbinitt[0] and i<nbinitt[1]):#Causal, acausal, and spurious
        x1[int(len(t)/2 - len(t)/indo):int(len(t)/2 + len(t)/indo)] =  spurr
        x1[int(len(t)/indo): int(len(t)/2 - len(t)/indo+1 )] = chirpp[::-1]
        x1[int(len(t)/2 + len(t)/indo): int(len(t)- len(t)/indo )+1] = chirpp
        indclus.append(1)
        if i==nbinitt[0]:
            xrawsav.append(x1)
        cpt +=1
        
    elif  (i>= nbinitt[1] and i<nbinitt[2]): # acausal and spurious
        x1[int(len(t)/indo): int(len(t)/2 - len(t)/indo+1 )] = chirpp[::-1]
        x1[int(len(t)/2 - len(t)/indo):int(len(t)/2 + len(t)/indo)] = spurr
        if i ==nbinitt[1]:
            xrawsav.append(x1)
        indclus.append(2)
        cpt2 +=1
        
    elif  (i>= nbinitt[2] and i<maxtraing_len): # Noise
        if i ==nbinitt[2]:
            xrawsav.append(x1)
        indclus.append(4)
        cpt4 +=1
    noisef = np.random.randn(int(t_len*delta)) # create noise
    ZZfin.append(x1 + noise_level*noisef/np.max(np.abs(noisef))) # add noise to the data
    

#%% Suffle and normalize the waveforms
X_trainf, ind = shuffle(ZZfin,  indclus) 
ZZfin = ZZfin 
x_or = np.array(X_trainf)

#%% Plot figure 1 (plot waveforms)
climy =np.max(np.abs(ZZfin))
fig1 = plt.figure(figsize=(12.5,7))
plt.rcParams.update({'font.size': 12})
ax1 = fig1.add_subplot(121)
ax1.tick_params( direction = 'out', right = True, top = True)

ax1.arrow(30, -300, 20, 0, head_width=200, head_length=5, fc='k', ec='k',clip_on=False, lw= 3)
plt.text(20, -200, 'A', fontweight='bold')
plt.text(59, -200, 'B', fontweight='bold')
ax1.arrow(-30, -300, -20, 0, head_width=200, head_length=5, fc='k', ec='k',clip_on=False, lw= 3)
plt.text(-26, -200, 'B', fontweight='bold')
plt.text(-65, -200, 'A', fontweight='bold')
im4 = plt.imshow(ZZfin, extent = (t[0], t[-1], len(X_trainf), 1), aspect = 'auto', clim= (-climy,climy))
plt.plot(t, ZZfin[1000]*400*-1 +1400, 'k')
plt.plot(t, xrawsav[0]*400*-1 +600, 'silver')

plt.plot(t, ZZfin[3000]*400*-1 +3400, 'k')
plt.plot(t, xrawsav[1]*400*-1 +2600, 'silver')

plt.plot(t, ZZfin[5000]*400*-1 +5400, 'k')
plt.plot(t, xrawsav[2]*400*-1 +4600, 'silver')

plt.plot(t, ZZfin[8000]*400*-1 +8400, 'k')
plt.plot(t, xrawsav[3]*400*-1 +7600, 'silver')

valx = 30.5
valy = 500
plt.plot([55+valx,55+valx],[ valy-400 ,valy], 'k', linewidth=2)
plt.plot([53+valx,57.+valx],[ valy-400 ,valy-400], 'k', linewidth=2)
plt.plot([53+valx,57.+valx],[ valy ,valy], 'k', linewidth=2)
plt.text(88,425, '1', fontsize = 14)

plt.text(0.5, 1.065, 'Synthetic waveforms',
         horizontalalignment='center',
         fontsize=14,
         transform = ax1.transAxes)

plt.xlabel('Time (s)')
plt.ylabel('Waveform #')
plt.text(-110, -500 , '(a)', fontsize = 14)
plt.grid(linewidth = .2)

ax = fig1.add_subplot(122)
ax.yaxis.tick_right()
ax.tick_params( direction = 'out', left = True, right= True, top = True)
plt.imshow(X_trainf, extent = (t[0], t[-1], len(X_trainf), 1), aspect = 'auto', clim= (-climy,climy))
plt.plot(t, np.mean(ZZfin,axis =0)*400*2.5*-1 +700, 'silver')
plt.grid(linewidth = .2)

valx = 9.5
valy = 600
plt.plot([75+valx,75+valx],[ valy-400 ,valy], 'k', linewidth=2)
plt.plot([73+valx,77.+valx],[ valy-400 ,valy-400], 'k', linewidth=2)
plt.plot([73+valx,77.+valx],[ valy ,valy], 'k', linewidth=2)
plt.text(87,525, '0.4', fontsize = 14)

plt.text(0.5, 1.065, 'Randomly shuffled waveforms',
         horizontalalignment='center',
         fontsize=14,
         transform = ax.transAxes)

ax.arrow(30, -300, 20, 0, head_width=200, head_length=5, fc='k', ec='k',clip_on=False, lw= 3)
plt.text(20, -200, 'A', fontweight='bold')
plt.text(59, -200, 'B', fontweight='bold')

ax.arrow(-30, -300, -20, 0, head_width=200, head_length=5, fc='k', ec='k',clip_on=False, lw= 3)
plt.text(-25, -200, 'B', fontweight='bold')
plt.text(-65, -200, 'A', fontweight='bold')

plt.xlabel('Time (s)')

plt.text(-110, -500 , '(b)', fontsize = 14)

colorbar_ax = fig1.add_axes([0.48, 0.45, 0.025, 0.15])
cb2 = fig1.colorbar(im4, cax=colorbar_ax,orientation='vertical', ticks=[-2,-1, 0 , 1,2])
cb2.ax.set_title('      Amplitude', fontsize = 13)

ax.set_position([0.55, .075, .39, .83])
ax1.set_position([0.07, .075, .39, .83])

plt.show()
fig1.savefig(dir_out + '/Fig_1.pdf', dpi=300)


#%% Do the clustering

n_components_PCA = 2 # number of principal components on which the clustering is performed
range_GMM = np.arange(2, 16) # To check the best number of cluster. The code will determine the best number of clusters in the given range
pca_output, var, models, n_clusters, gmixfinPCA, probs, BICF = Clustering_PCA_GMM(X_trainf, n_components_PCA,range_GMM)


#%% Set up colors for each cluster
size = 50 * probs.max(1) ** 2 
c_dict =  {0:'r',1:'b',2:'g',3:'y',4:'m',5:'r',6:'b',7:'g',8:'y',9:'k',10:'r',11:'b',12:'g',13:'y',14:'k'}
colors = np.array([c_dict[i] for i in gmixfinPCA])


#%% Compute accuracy of the classifier using sklearn.metrics.accuracy_score
fin_mat = np.zeros(int(len(ind)))
for i in np.arange(n_clusters):
    a0test = [j for j, e in enumerate(gmixfinPCA) if e == i]
    clutt = []
    mean_val = []
    for ii in a0test:
        clutt.append(ind[ii])
    mean_val = round(np.mean(clutt))
    fin_mat[a0test] = mean_val
    
score = accuracy_score(ind,fin_mat)*100 


#%% Plot Figure 2 (clustering results)
fnt = 14
nb = 2
figa, axs = plt.subplots(n_clusters, 2,figsize=(10,8) )
figa.set_rasterized(True)

# First two PCs plot
axs[0,0].set_aspect('equal')
axs[0,0].set_rasterized(True)
axs[0,0].scatter(pca_output[:,0], pca_output[:, 1], c=colors, alpha=0.25, s = 10 ) 
axs[0,0].set_xlabel('Principal component 1', fontsize=fnt)
axs[0,0].set_ylabel('Principal component 2', fontsize=fnt)
axs[0,0].text(-6.5, np.max(pca_output[:, 1])+ np.max(pca_output[:, 1])*.2, '(a)', fontsize=14)
axs[0,0].set_title('PCA + GMM clustering', fontsize=12)
axs[0,0].grid(linewidth = .3)

# BIC score plot
axs[1,0].plot(range_GMM, BICF, linewidth =3)
axs[1,0].plot(n_clusters, BICF[n_clusters-2], 'or', markersize =10)
axs[1,0].set_xlabel('Number of Clusters', fontsize=fnt)
axs[1,0].set_ylabel('BIC score', fontsize=fnt)
axs[1,0].text(-2.5, np.max(BICF)+np.max(BICF)*.001, '(b)', fontsize=14)
axs[1,0].grid(linewidth = .3)
axs[1,0].set_xticks(np.linspace(2,16, 8))
axs[1,0].set_xlim(2, 15)
#axs[1,0].set_ylim(35, 40)
axs[1,0].set_rasterized(True)

# Loop over the clusters to stack the waveforms and plot them
for i in np.arange(n_clusters):
    a0test = [j for j, e in enumerate(gmixfinPCA) if e == i]
    a0 =[]       
    for tt in a0test:
        if probs[tt,i]>.0:
            a0.append(tt)
    meanb1 = []
    meanb1 = np.mean(x_or[a0] ,axis = 0) # waveform stack for each cluster
    if nb ==2 :
        axs[i,1].text(-140, 1.15, '(c)', fontsize=14)
    nb = nb+2
    
    axs[i,1].plot(t,meanb1, label = 'normal', color =  c_dict[i], linewidth =2)
    if i==0:
        axs[i,1].set_title('Clustering total accuracy: ' + str(np.round(score,5) ) + '%\n Cluster ' +str(i+1) + ': '+ str(len(a0))+' waveforms' , fontsize=12)
    else:
        axs[i,1].set_title('Cluster ' +str(i+1) + ': '+ str(len(a0))+' waveforms' , fontsize=12)
    axs[i,1].set_ylabel('Amplitude', fontsize=fnt)
    axs[i,1].set_xlim(t[0], t[-1])
    axs[i,1].set_ylim(-.85, .85)
    axs[i,1].grid()
    axs[i,1].set_rasterized(True)
    
    
axs[i,1].set_xlabel('Time (s)', fontsize=fnt)
axs[i,1].grid(linewidth = .3)
axs[0,0].set_position([0.065, .5, .4, .4])
axs[1,0].set_position([0.125, .09, .3, .3])

if n_clusters == 4: # Make the plot cleaner
    axs[0,1].set_position([0.55, .79, .42, .16])
    axs[1,1].set_position([0.55, .55, .42, .16])
    axs[2,1].set_position([0.55, .31, .42, .16])
    axs[3,1].set_position([0.55, .075, .42, .16])
    axs[2,0].axis('off')
    axs[3,0].axis('off')
    
plt.show() 
figa.savefig(dir_out + '/Fig_2.pdf', dpi=300)

