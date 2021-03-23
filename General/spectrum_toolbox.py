#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 16:47:52 2021

@author: mathieudagreou
"""
import itertools
import scipy.sparse as sp
import numpy as np

#%%
"""Some functions used to build kernels on sequences"""


def preindexation(k,alphabet = 'ATCG'):
    """
    Compute a preindexation of all sequences of length k built with letters from alphabet
    
    Parameters
    ----------
    k : int
        length
    alphabet : string, optional
        DESCRIPTION. The default is 'ATCG'.

    Returns
    -------
    A dictionnary whose keys are subsequences and values are integers

    """
    subsequences = dict()    
    voc = [''.join(x) for x in itertools.product(alphabet, repeat=k)]
    i = 0
    for w in voc:
        subsequences[w] = i
        i+=1
    return(subsequences) 

def Spectrum_embedding(sequences,k,preindex = None):
    """
    Compute the spectrum embedding of a list of sequences

    Parameters
    ----------
    sequences : string
    k : integer
    preindex : None or dictionnary, optional
        If None, preindex is computed in the function. The default is None.

    Returns
    -------
    Sparse matrix

    """
    if preindex == None:
        preindex = preindexation(k)
    n_sequences = sequences.shape[0]
    
    embedding = sp.lil_matrix((n_sequences,4**k))

    for i in range(n_sequences):   # cvery sequence
        for w in range(len(sequences[i])-k+1): # every pattern w 
            j = int(preindex[sequences[i][w:w+k]])
            embedding[i,j] += 1
        
    embedding = embedding.tocsr()
    
    return(embedding)


def Mismatch_embedding(sequences,k,m,preindex = None):
    """
    Compute the spectrum embedding of a list of sequences

    Parameters
    ----------
    sequences : string
    k : integer
    preindex : None or dictionnary, optional
        If None, preindex is computed in the function. The default is None.

    Returns
    -------
    Sparse matrix

    """
    if preindex == None:
        preindex = preindexation(k)
    n_sequences = sequences.shape[0]
    
    embedding = sp.lil_matrix((n_sequences,4**k))

    for i in range(n_sequences):   # cvery sequence
        if i%500==0:
            print(i)
        for w in range(len(sequences[i])-k+1): # every pattern w 
            for j, b in enumerate(list(preindex.keys())):  # for all element of the
                embedding[i,j] += sum(1 for a, c in zip(sequences[i][w:w+k],b) if a != c)<=m
            
    embedding = embedding.tocsr()
    
    return(embedding)
###################################################
#import sys
# import pandas as pd
# import numpy as np
# import os
# from tqdm import tqdm
# os.chdir(r"C:\Users\pierr\Desktop\MVA\Kernel\kaggle\mva-KMML-challenge\data")

# X = pd.read_csv('Xtr0.csv')
# y = pd.read_csv('ytr0.csv')
# X.set_index('Id',inplace = True)

# X.head()
# ######################################
# params={}
# params["k"]=4
# preindex = preindexation(params['k'])
# X_emb = Spectrum_embedding(X['seq'].to_numpy(),params['k'],preindex = preindex)
# print(X_emb[0])

# #####################""
# params={}
# params["k"]=4
# params["m"]=1

# preindex = preindexation(params['k'])
# X_emb = Mismatch_embedding(X['seq'].to_numpy(),params['k'],params["m"],preindex = preindex)
# print(X_emb[0])












