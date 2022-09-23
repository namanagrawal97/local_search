# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:12:51 2022

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
# genotypelist=["w1118"]
starvation="24"
time_list=[5,10,30,60]

real_food_all=pd.read_csv('food_timing_multi_lines.csv')

def find_index(l,t):
    for j in l:
        if j>t:
            return l.index(j)
            break
        else:
            continue


for cutoff in time_list: 
    
    inst_vel_dict={}
    rad_dist_dict={}
    tot_dist_dict={}
    
    rad_dist_swt_dict={}
    inst_vel_swt_dict={}
    
    rad_dist_med_dict={}
    inst_vel_med_dict={}
    tot_dist_med_dict={}
    
    rad_dist_med_all_dict={}
    inst_vel_med_all_dict={}
    
    rad_dist_mean_all_dict={}
    inst_vel_mean_all_dict={}
    
    rad_dist_mwu_dict={}
    inst_vel_mwu_dict={}
    tot_dist_mwu_dict={}
    
    number_of_flies={}
    
    
    for genotype in genotypelist:
        print(genotype)
        fnames = sorted(glob.glob(genotype+'/'+starvation+'/'+'*.csv'))
        index=0
        real_food_df=real_food_all[real_food_all['genotype']==genotype]
        real_food_bout_list=list(real_food_df['final_first_bout'])
        inst_vel_all=[]
        rad_dist_all=[]
        tot_dist_all=[]
        
        inst_vel_median_all=[]
        rad_dist_median_all=[]
        tot_dist_median_all=[]
        
        inst_vel_mean_all=[]
        rad_dist_mean_all=[]
        tot_dist_mean_all=[]
        
        for u in fnames: #goes thru files in the folder.
            
            # print(index)
            # print(u)
            """
            Loading the data into a dataframe
            """
            
            df=pd.read_csv(u, header=None)
            print("before",df.shape[1])
            df=df.dropna(axis=1, how='all',thresh=40000)
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
                # print(index,first_bout_time)
    
                    time=list(df['Time'])
                    split_point_index=find_index(time, first_bout_time)
                    print(index,first_bout_time,split_point_index)
                    
                    x_coord=list(df[data_header[i]])
                    y_coord=list(df[data_header[i+1]])
                    del x_coord[0:split_point_index]
                    del y_coord[0:split_point_index]
                    
                    inst_vel=list(np.zeros_like(x_coord))
                    rad_dist=list(np.zeros_like(x_coord))
                    distance_travelled=list(np.zeros_like(x_coord))
                    
                    for j in range(0,len(x_coord)-1,1):
                        rad_dist[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2)
                        inst_vel[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2)/latency[j+1]
                        distance_travelled[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2)
                    
                    wall_touching_point=find_index(rad_dist, 520)
                    if wall_touching_point is None:    
                        wall_touching_point=len(rad_dist)-1
                    else:
                        wall_touching_point=wall_touching_point
                    print(wall_touching_point)
                    del distance_travelled[wall_touching_point:-1]
                    del rad_dist[cutoff*24:-1] #Delete everything after declared time
                    del inst_vel[cutoff*24:-1]
                    rad_dist.pop()#Remove the last element which is somehow always 0
                    inst_vel.pop()
                    total_distance_travelled=sum(distance_travelled)
                    
                    rad_dist_median_all.append(np.median(rad_dist))
                    inst_vel_median_all.append(np.median(inst_vel))
    
                    rad_dist_mean_all.append(np.mean(rad_dist))
                    inst_vel_mean_all.append(np.mean(inst_vel))                    
                    
                    rad_dist_all.extend(rad_dist)
                    inst_vel_all.extend(inst_vel)
                    tot_dist_all.append(total_distance_travelled)
                    
                    inst_vel_all = [x for x in inst_vel_all if np.isnan(x) == False]
                    rad_dist_all = [x for x in rad_dist_all if np.isnan(x) == False]
                    tot_dist_all = [x for x in tot_dist_all if np.isnan(x) == False]                
            
        inst_vel_med_dict[genotype]=np.median(inst_vel_all)
        rad_dist_med_dict[genotype]=np.median(rad_dist_all)
        tot_dist_med_dict[genotype]=np.median(tot_dist_all)
    
        inst_vel_dict[genotype]=inst_vel_all
        rad_dist_dict[genotype]=rad_dist_all
        tot_dist_dict[genotype]=tot_dist_all
    
        rad_dist_med_all_dict[genotype]=rad_dist_median_all
        inst_vel_med_all_dict[genotype]=inst_vel_median_all
        
        rad_dist_mean_all_dict[genotype]=rad_dist_mean_all
        inst_vel_mean_all_dict[genotype]=inst_vel_mean_all
    
    "All radial distance and inst velocity raw data"
    rad_dist_med_dict = dict(sorted(rad_dist_med_dict.items(), key=lambda x: x[1], reverse=True))
    inst_vel_med_dict = dict(sorted(inst_vel_med_dict.items(), key=lambda x: x[1], reverse=True))
    tot_dist_med_dict = dict(sorted(tot_dist_med_dict.items(), key=lambda x: x[1], reverse=True))
    
    inst_vel_med_dict_labels, inst_vel_med_dict_data = [*zip(*inst_vel_med_dict.items())]  # 'transpose' items to parallel key, value lists
    rad_dist_med_dict_labels, rad_dist_med_dict_data = [*zip(*rad_dist_med_dict.items())]  # 'transpose' items to parallel key, value lists
    tot_dist_med_dict_labels, tot_dist_med_dict_data = [*zip(*tot_dist_med_dict.items())]  # 'transpose' items to parallel key, value lists
    
    inst_vel_dict_labels, inst_vel_dict_data = [*zip(*inst_vel_dict.items())]  # 'transpose' items to parallel key, value lists
    rad_dist_dict_labels, rad_dist_dict_data = [*zip(*rad_dist_dict.items())]  # 'transpose' items to parallel key, value lists
    tot_dist_dict_labels, tot_dist_dict_data = [*zip(*tot_dist_dict.items())]  # 'transpose' items to parallel key, value lists
    
    inst_vel_dict_final = {k: inst_vel_dict[k] for k in inst_vel_med_dict_labels}
    rad_dist_dict_final = {k: rad_dist_dict[k] for k in rad_dist_med_dict_labels}
    tot_dist_dict_final = {k: tot_dist_dict[k] for k in tot_dist_med_dict_labels}
    
    inst_vel_dict_final_labels, inst_vel_dict_final_data = [*zip(*inst_vel_dict_final.items())]  # 'transpose' items to parallel key, value lists
    rad_dist_dict_final_labels, rad_dist_dict_final_data = [*zip(*rad_dist_dict_final.items())]  # 'transpose' items to parallel key, value lists
    tot_dist_dict_final_labels, tot_dist_dict_final_data = [*zip(*tot_dist_dict_final.items())]  # 'transpose' items to parallel key, value lists
    
    rad_dist_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rad_dist_dict_final.items() ]))
    inst_vel_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_dict_final.items() ]))
    tot_dist_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tot_dist_dict_final.items() ]))
    
    """
    Median of Radial Distance and Instantaneous Velocity Dataframe
    
    """
    rad_dist_med_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rad_dist_med_all_dict.items() ])) #convert dictionary to dataframe
    inst_vel_med_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_med_all_dict.items() ]))
    
    sorted_index_rad = rad_dist_med_all_df.median().sort_values().index #Use the median values to sort the index
    sorted_index_inst = inst_vel_med_all_df.median().sort_values().index
    
    rad_dist_med_all_df=rad_dist_med_all_df[sorted_index_rad] #Using the sorted Index to sort the Dataframes
    inst_vel_med_all_df=inst_vel_med_all_df[sorted_index_inst]
    
    rad_dist_med_all_df.to_csv("results\\rad_dist_med_all_{}_df.csv".format(cutoff),index=False)
    inst_vel_med_all_df.to_csv("results\\inst_vel_med_all_{}_df.csv".format(cutoff),index=False)
    
    """
    Mean of Radial Distance and Instantaneous Velocity Dataframe
    
    """
    rad_dist_mean_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rad_dist_mean_all_dict.items() ])) #convert dictionary to dataframe
    inst_vel_mean_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_mean_all_dict.items() ]))
    
    sorted_index_rad = rad_dist_mean_all_df.median().sort_values().index #Use the median values to sort the index
    sorted_index_inst = inst_vel_mean_all_df.median().sort_values().index
    
    rad_dist_mean_all_df=rad_dist_mean_all_df[sorted_index_rad] #Using the sorted Index to sort the Dataframes
    inst_vel_mean_all_df=inst_vel_mean_all_df[sorted_index_inst]
    
    rad_dist_mean_all_df.to_csv("results\\rad_dist_mean_all{}_df.csv".format(cutoff),index=False)
    inst_vel_mean_all_df.to_csv("results\\inst_vel_mean_all{}_df.csv".format(cutoff),index=False)

"""
PLOTTING
"""

"""
Here we generate comparative boxplots for genotypes
"""


fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=rad_dist_dict_final_data, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
# ax = sns.violinplot(data=rad_dist_dict_final_data, split=True)
ax.set_xlabel('Radial Distance')
ax.set_yticklabels(rad_dist_dict_final_labels, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Radial Distance 10 seconds after first contact with Food', fontsize=13)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()
# fig.savefig('radial_distance_10seconds.png',format='png', dpi=600, bbox_inches = 'tight')

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=inst_vel_dict_final_data, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
# ax = sns.stripplot(data=data2, color=".25",size=3,orient="h")
ax.set_xlabel('Instantenous Velocity')
ax.set_yticklabels(inst_vel_dict_final_labels, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Instantenous Velocity 10 seconds after first contact with Food', fontsize=13)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()
# fig.savefig('inst_vel_10seconds.png',format='png', dpi=600, bbox_inches = 'tight')

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=tot_dist_dict_final_data, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=tot_dist_dict_final_data,color=".25",size=2, orient='h')
# ax = sns.violinplot(data=rad_dist_dict_final_data, split=True)
ax.set_xlabel('Distance')
ax.set_yticklabels(tot_dist_dict_final_labels, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Distance travelled before touching wall', fontsize=13)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()
fig.savefig('results\\total_distance_before_wall_touching.png',format='png', dpi=600, bbox_inches = 'tight')


"""
Here we generate comparative boxplots for genotypes, but median only
"""


fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=rad_dist_med_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=rad_dist_med_all_df,size=2,color=".25", orient="h")
ax.set_xlabel('Radial Distance')
ax.set_yticklabels(rad_dist_med_all_df.columns, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, labelsize=5)
ax.set_title('Radial Distance 10 seconds after first contact with Food', fontsize=13)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()
fig.savefig('results\\radial_distance_10seconds_medians.png',format='png', dpi=600, bbox_inches = 'tight')

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=inst_vel_med_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=inst_vel_med_all_df, color=".25",size=2, orient="h")
ax.set_xlabel('Instantaneous Velocity')
ax.set_yticklabels(inst_vel_med_all_df.columns, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Instantaneous Velocity 10 seconds after first contact with Food', fontsize=13)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()
fig.savefig('results\\inst_vel_10seconds_medians.png',format='png', dpi=600, bbox_inches = 'tight')



"""
Here we generate Histograms of our raw data to see what the raw data looks like
"""


fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Individual genotype Radial Distance (10s after food) distribution", fontsize=18, y=0.95)

# loop through tickers and axes
for ticker, ax in zip(list(rad_dist_dict_labels), axs.ravel()):
    sns.histplot(x=rad_dist_df[ticker],ax=ax, bins=10, kde=True, alpha=0.5)
    # ax.tick_params(axis='y', labelrotation = 0, size=2)
    ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
    ax.set_xlim(0,540)
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set(ylabel=None)
    # ax.text(450, 1000, "n={}".format(number_of_flies[ticker]))
# fig.savefig('results\\Ind_gen_distb_rad_dist.png',format='png', dpi=600, bbox_inches = 'tight')
plt.show()

fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Individual genotype Inst. Velocity (10s after food) distribution", fontsize=18, y=0.95)
for ticker, ax in zip(list(inst_vel_dict_labels), axs.ravel()):
    sns.histplot(x=inst_vel_df[ticker],ax=ax,bins=10, kde=True, alpha=0.5)
    # ax.tick_params(axis='y', labelrotation = 0, size=2)
    ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
    ax.set_xlabel("")
    
    ax.set_yticks([])
    ax.set(ylabel=None)
# fig.savefig('results\\Ind_gen_distb_inst_vel.png',format='png', dpi=600, bbox_inches = 'tight')
plt.show()



"""
Here we generate QQ plots to check for normality of our data
"""


fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("QQ PLOT Individual genotype Radial Distance (10s after food) distribution", fontsize=18, y=0.95)

# loop through tickers and axes
for ticker, ax in zip(list(rad_dist_dict_final_labels), axs.ravel()):
    qqplot(data=rad_dist_df[ticker],ax=ax, line='s', marker='o', markersize=1, alpha=0.5)
    # ax.tick_params(axis='y', labelrotation = 0, size=2)
    ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
    # ax.set_xlim(0,540)
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set(ylabel=None)
fig.savefig('results\\Ind_gen_qq_rad_dist.png',format='png', dpi=600, bbox_inches = 'tight')
plt.show()

fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("QQ PLOT Individual genotype Inst. Vel. (10s after food) distribution", fontsize=18, y=0.95)

# loop through tickers and axes
for ticker, ax in zip(list(inst_vel_dict_final_labels), axs.ravel()):
    qqplot(data=inst_vel_df[ticker],ax=ax, line='s', marker='o', markersize=1, alpha=0.5)
    # ax.tick_params(axis='y', labelrotation = 0, size=2)
    ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
    # ax.set_xlim(0,540)
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set(ylabel=None)
fig.savefig('results\\Ind_gen_qq_inst_vel.png',format='png', dpi=600, bbox_inches = 'tight')
plt.show()

