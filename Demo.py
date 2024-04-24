# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:45:19 2024

@author: Ali Şenol
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import time

from sklearn import metrics
from ANDClust import ANDClust
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
get_ipython().magic('clear all -sf')

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


datasets = {1,2,3,4,5,6}
for selected_dataset in datasets:
    if selected_dataset == 1:
        data = np.loadtxt("Datasets/3_Spirals.txt", delimiter=',', dtype=float)
        X = data[:, 0:2]
        labels_true = data[:, 2]
        dataset_name = "1_ThreeSpirals_"
        dataset_name2 = "Three Spirals"
        N = 16
        eps = 0.67
        k = 3
        b_width = 2
        krnl = 'tophat'

    elif selected_dataset == 2:
        X = np.loadtxt("Datasets/varydensity.txt", delimiter='\t', dtype=float)
        labels_true = np.loadtxt(
            "Datasets/varydensity-gt.txt", delimiter='\t', dtype=float)
        dataset_name = "2_varyingdensity_"
        dataset_name2 = "Varying Density"
        N =47
        eps = 0.95
        k = 4
        b_width = 0.05
        krnl = 'linear'
        
    elif selected_dataset == 3:
        data = np.loadtxt("Datasets/cure-t1-2000n-2D.txt",
                          delimiter=',', dtype=float)
        X = data[:, 0:2]
        labels_true = data[:, 2]
        dataset_name = "3_cure-t1-2000n-2D_"
        dataset_name2 = "Cure-t1-2000n-2D_"
        N = 2 # ->30 dene
        eps = 3.18
        k = 2
        b_width = 2.24
        krnl = 'cosine'

    elif selected_dataset == 4:
        data = np.loadtxt("Datasets/Aggregation.txt",
                          delimiter=',', dtype=float)
        X = data[:, 0:2]
        labels_true = data[:, 2]
        labels_true = np.ravel(labels_true)
        dataset_name = "4_aggregation_"
        dataset_name2 = "Aggregation"
        N =30
        eps = 0.0875
        k = 29
        b_width = 0.5
        krnl = 'linear'
        
    elif selected_dataset == 5:
        dataset = np.loadtxt("Datasets/DS2.txt", delimiter='\t', dtype=float)
        X = dataset[:, :2]
        labels_true = dataset[:, 2]
        dataset_name = "5_DS1_"
        dataset_name2 = "DS1"
        N =9
        eps = 0.155
        k = 20
        b_width = 4
        krnl = 'gaussian'
        
    elif selected_dataset == 6:
        dataset = np.loadtxt("Datasets/DS3.txt", delimiter='\t', dtype=float)
        X = dataset[:, :2]
        labels_true = dataset[:, 2]
        dataset_name = "6_DS3_"
        dataset_name2 = "DS3"
        N =34
        eps = 5.21
        k = 9
        b_width = 8.2
        krnl = 'tophat'

    # ####MinMaxNormalization#######################################################
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X = scaler.transform(X)
    d = X.shape[1]
 

    start=time.time()
    #Clustering with ANDClust ######################
    andClust=ANDClust(X,N,k,eps,krnl,b_width)
    labels=andClust.labels_ 
    ################################################
    end=time.time()
    print("Elapsed time= ",end-start) 
    
    ARI=adjusted_rand_score(labels,labels_true)
    NMI = normalized_mutual_info_score(labels, labels_true)
    Purity = purity_score(labels, labels_true)  

    print("####### Results: ##########")
    print("ARI=", ARI)
    andClust.plotGraph("ARI",ARI,dataset_name)
    
    print("Purity=", Purity)
    andClust.plotGraph("Purity",Purity,dataset_name) 
   
    print("NMI=", NMI)
    andClust.plotGraph("NMI",NMI,dataset_name)
        







