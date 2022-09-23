# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:56:57 2022

@author: Yapicilab
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
import sklearn
import numpy.linalg as LA
from sklearn.model_selection import train_test_split

#import plotly.express as px
# import rhinoscriptsyntax as rs

os.chdir('C:\\Users\\Yapicilab\\Dropbox\\screening')

starvationlist=["0","8","16","24"]
foodlist=["ss15725","ss30373","ss30377"]
starvation="24"


rad_thres_list=[75,80,85,90]
time_thres_list=[0.5,1,1.5,2]
efficienty_matrix=pd.DataFrame(columns=rad_thres_list)
k=0
for rad_thres in rad_thres_list:
    k=k+1
    l=0
    time_thres_array=np.zeros_like(time_thres_list)
    for time_thres in time_thres_list:
        
        radial_distance_all=[]
        food_index_all=[]
        inst_vel_all=[]
        heading_vector_all=[]
        est_first_food=[]
        flyindex=[]
        
        l=l+1
        print(rad_thres,time_thres)
        
        def find_split_points(some_list):
            splitlist=[]
            for i in range(0, len(some_list)-1,1):
                if some_list[i+1]-some_list[i]>1:
                    splitlist.append(i)
                else:
                    continue
            return splitlist
        def rad_dist_classifier(rad_dist_list,rad_dist_index,rad_threshold):
            rad_dist_list=list(rad_dist_list)
            for i in rad_dist_list:
                if i<rad_threshold:
                    rad_dist_index[rad_dist_list.index(i)]=1
                else:
                    continue
        
        
        real_food=pd.read_csv('food_timing_multi_lines.csv')
        
        
        for food in foodlist:
            fnames = sorted(glob.glob(food+'/'+starvation+'/'+'*.csv'))
            for u in fnames: #goes thru files in the folder.
                # print(u)
                """
                Loading the data into a dataframe
                """
                
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
                    rad_dist=np.zeros_like(x_coord)
                    rad_dist_index=np.zeros_like(x_coord)
                    food_index=np.zeros_like(x_coord)
                    for j in range(0,len(x_coord)-1,1):
                        rad_dist[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2)
                    rad_dist_classifier(rad_dist, rad_dist_index, rad_thres)
                    FRAresidence=list(np.where(rad_dist_index==1)[0])
                    splitlist=find_split_points(FRAresidence)
                    split_list = [FRAresidence[q: w] for q, w in zip([0] + splitlist, splitlist + [None])]
                    # print(i/2, splitlist[0]/24, time[FRAresidence[0]])
                    for p in range(0, len(split_list),1):
                        # print(p)
                        if ((len(split_list[p])/24) > time_thres):
                            timestamp=split_list[p][0]
                            est_first_food.append(time[timestamp])
                            flyindex.append(i/2)
                            break
                        else:
                            continue
                        # radial_distance_all.extend(rad_dist)
                    # food_index_all.extend(food_index)
        real_food['Estimated_first_food']=list(est_first_food)                        
        """
        Here we make a loop to compare estimated first bout timing and real food bout timing
        
        """
        hits=0
        misses=0
        for i,j in zip(list(real_food['Estimated_first_food']),real_food['final_first_bout']):
            if j<i+1.5 and j>i-1.5:
                hits=hits+1
            else:
                misses=misses+1
        print("Hit percentage is",hits/real_food.shape[0])
        print("Miss percentage is",misses/real_food.shape[0])
        print("Hits/Miss ratio is",hits/misses)
        # time_thres_array[l]=hits/real_food.shape[0]
        # efficienty_matrix.iat[k,l]=hits/real_food.shape[0]