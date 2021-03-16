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

    for i in range(n_sequences):
        for w in range(len(sequences[i])-k+1):
            j = int(preindex[sequences[i][w:w+k]])
            embedding[i,j] += 1
        
    embedding = embedding.tocsr()
    
    return(embedding)
