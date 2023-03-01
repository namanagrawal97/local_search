# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:28:09 2023

@author: Yapicilab
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as stats
import matplotlib.ticker as ticker

def binning(list_to_bin,kernel_size):
    i=0
    binned_list=[]
    while i<len(list_to_bin):
        binned_list.append(np.sum(list_to_bin[i:i+kernel_size])/kernel_size)
        i=i+kernel_size
    return binned_list

def find_index(l,t): #This function helps us find indexes
    for j in l:
        if j>t:
            return l.index(j)
            break
        else:
            continue

def dict_to_df(input_dict):
    df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in input_dict.items() ]))    
    sorted_index_df = df.mean().sort_values().index
    df=df[sorted_index_df]
    return(df)

def df_to_csv(input_df):
    input_df.to_csv("results\\{}.csv".format(input_df),index=False)

def sort_df_mean(input_df):
    sorted_index_df = input_df.mean().sort_values().index
    input_df=input_df[sorted_index_df]
    return(input_df)    