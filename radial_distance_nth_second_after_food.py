# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:55:56 2022

@author: Yapicilab
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:27:56 2022

@author: Yapicilab
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
import scipy.stats as stats
import dabest
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import mannwhitneyu

os.chdir('C:\\Users\\Yapicilab\\Dropbox\\screening')
real_food_all=pd.read_csv('food_timing_multi_lines.csv')


# starvationlist=["0","8","16","24"]


foodlist=os.listdir()
foodlist.remove('food_timing_multi_lines.csv')
foodlist.remove('results')
foodlist.remove('ss46202')
# foodlist.remove('ss45950')
# foodlist.remove('w1118')
# foodlist=foodlist[0:39]
genotypelist=foodlist
# genotypelist=["ss32463"]
starvation="24"
time_list=[5,10,30,60]

time_thres=60

real_food_all=pd.read_csv('food_timing_multi_lines.csv')

def find_index(l,t):
    for j in l:
        if j>t:
            return l.index(j)
            break
        else:
            continue

rad_dist_dict={}

for genotype in genotypelist:
    
    radial_distances=[]
    print(genotype)
    fnames = sorted(glob.glob(genotype+'/'+starvation+'/'+'*.csv'))
    index=0
    real_food_df=real_food_all[real_food_all['genotype']==genotype]
    real_food_bout_list=list(real_food_df['final_first_bout'])
    
    for u in fnames: #goes thru files in the folder.
        
        # print(index)
        # print(u)
        """
        Loading the data into a dataframe
        """
        
        df=pd.read_csv(u, header=None)
        print("before",df.shape[1])
        df=df.dropna(axis=1,thresh=20000)
        print("after",df.shape[1])
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
            index=index+1
            first_bout_time=real_food_bout_list[index-1] #Extracting the first bout time from manual annotation and storing it in a variable
            if (first_bout_time==2000):
                continue
            else:
                pass
    
                time=list(df['Time'])
                split_point_index=find_index(time, first_bout_time)
                print(index,first_bout_time,split_point_index)
                
                x_coord=list(df[data_header[i]])
                y_coord=list(df[data_header[i+1]])
                del x_coord[0:split_point_index]
                del y_coord[0:split_point_index]
                
                rad_dist=list(np.zeros_like(x_coord))
                
                for j in range(0,len(x_coord)-1,1):
                    rad_dist[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2)
                    
                if len(rad_dist)<time_thres*24:
                    pass
                else:
                    del rad_dist[time_thres*24:-1]
                    del rad_dist[0:(time_thres-1)*24-1]
                    rad_dist.pop()
                    radial_distances.append(np.mean(rad_dist))

    # rad_dist_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in radial_distances.items() ]))
    # rad_dist_df['mean']=rad_dist_df.mean(axis=1)
    rad_dist_dict[genotype]=radial_distances

rad_dist_med_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rad_dist_dict.items() ])) #convert dictionary to dataframe

sorted_index_rad = rad_dist_med_all_df.median().sort_values().index #Use the median values to sort the index

rad_dist_med_all_df=rad_dist_med_all_df[sorted_index_rad] #Using the sorted Index to sort the Dataframes

rad_dist_med_all_df.to_csv("results\\rad_dist_mean_{}thSecond_df.csv".format(time_thres),index=False)

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=rad_dist_med_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=rad_dist_med_all_df,size=2,color=".25", orient="h")
ax.set_xlabel('Radial Distance')
ax.set_yticklabels(rad_dist_med_all_df.columns, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, labelsize=5)
ax.set_title('Radial Distance {}th second after first contact with Food'.format(time_thres), fontsize=13)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.xaxis.grid(True)
fig.savefig("results\\Radial Distance {}th second after first contact with Food.png".format(time_thres),format='png', dpi=600, bbox_inches = 'tight')

plt.show()