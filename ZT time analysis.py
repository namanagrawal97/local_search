# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:48:05 2022

@author: Yapicilab
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math

os.chdir('G:\\My Drive\\local_search\\ZT')
some_df={}
timelist=sorted(os.listdir())
timelist.remove('desktop.ini')
for timing in timelist:
    fnames = sorted(glob.glob(timing+'/'+'*.csv'))
    inst_vel_all=[]
    for u in fnames: #goes thru files in the folder.
        df=pd.read_csv(u, header=None)
        df=df.dropna(axis=1, how='all')
        if(df.shape[1]==10):
            data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
        elif(df.shape[1]==8):
            data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
        elif(df.shape[1]==6):
             data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2']
        elif(df.shape[1]==4):
             data_header = ['Time', 'Latency', 'Fx1', 'Fy1']
        df.columns=data_header
        latency=list(df['Latency'])
        latency[0]=0
        df['Time']=df['Time']-120
        for i in range(2,len(data_header),2):
            x_coord=list(df[data_header[i]])
            y_coord=list(df[data_header[i+1]])
            time=list(df['Time'])
            jumps=[]
            inst_vel=np.zeros_like(x_coord)
            for j in range(0,len(x_coord)-1,1):
                inst_vel[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2)/latency[j+1]
            
            # median_inst_vel=np.median(inst_vel)
            inst_vel_all.extend(inst_vel)
            newlist = [x for x in inst_vel_all if np.isnan(x) == False]
    some_df[timing]=newlist

# some_df=some_df.dropna(axis=0)
labels, data2 = [*zip(*some_df.items())]  # 'transpose' items to parallel key, value lists
fig, ax = plt.subplots()

ax.boxplot(data2, showfliers=False)
# ax=sns.stripplot(data2, size=2)
ax.set_xticklabels(labels)
plt.show()
