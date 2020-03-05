#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:57:54 2020

@author: Loic
"""

import numpy as np
from sklearn.decomposition import PCA 
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

def Clustering_PCA_GMM(mat, PC_nb, range_GMM):
    """
    Inputs:
        mat: Matrix of correlation functions
        PC_nb: number of principal components to keep (typically 5 - 15)
        range_GMM: range of number of clusters to determine the best number of clusters using the knee (or elbow) method (typically 2 - 15)
    Outputs:
        pca_output: Output of the PCA
        var: Cumulative explained variance for each of from 1 to "PC_nb"
        models: GMM for different number of clusters
        n_clusters: Best number of clusters determined by the knee method
        gmixfinPCA: Clustering of the data with "n_clusters"
        probs: Probability that the data belong to the cluster they were assigned to.
        BICF: BIC score for the different number of clusters

    """
    # Perform PCA with the number of principal components given in input
    pca = PCA(n_components=PC_nb)
    pca_output = pca.fit_transform(mat)
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) # Cumulative variance

    print('The first ' + str(PC_nb) + ' PCs explain ' + str(var[-1]) + ' % of the cumulative variance')
    
    # Compute the GMM for different number of clusters set by "range_GMM"
    models = [GaussianMixture(n, covariance_type='full', random_state=0,max_iter=500).fit(pca_output) for n in range_GMM]
    # Compute the Bayesian information criterion (BIC) for each model
    BICF =[m.bic(pca_output)/1000 for m in models]
     # Determine the best number of clusters using the knee (or elbow) method from the BIC scores
    kn = KneeLocator(range_GMM, BICF, S=1, curve='convex', direction='decreasing')
    n_clusters = kn.knee
    
    # Perform clustering for the best number of clusters
    gmix = GaussianMixture(n_components=n_clusters, covariance_type='full',max_iter=500)
    gmix.fit(pca_output)
    gmixfinPCA = gmix.predict(pca_output)
    probs = gmix.predict_proba(pca_output)
    
    return pca_output, var, models, n_clusters, gmixfinPCA, probs, BICF
    
