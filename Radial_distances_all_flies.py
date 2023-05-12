# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:44:14 2022

@author: Yapicilab
"""

"""
Readme:
    
This code takes 2 inputs :
    1. "food_timing_multi_lines.csv", which is the csv file containing time labels at which each fly of each genotype first ate food.
    2. XY coordinates of the fly. The code first enters the folder of the genotype ("foodlist" contains the list of all genotypes in the data folder), and then enter the "24" folder.
        Inside the "24" folder, there are 2 csv files, containing data for 8 flies. The code parses through each one by one.

Within the "for u in fnames:" loop, the code then takes the XY coordinates of each fly, and deletes the data prior to when the fly first ate food
It then calculates some parameters, stores them in lists. It then deletes the data in those lists after a specified "time_thres".

It saves the parameters in CSV files.

The code generates plots of radial distance from 0 to "time_thres" for individual fly in each genotype. 
"""




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import copy
import scipy.stats as stats
import matplotlib.ticker as ticker
from conversion_func import *
from plot_func import *

os.chdir('E:\\Dropbox\\yeast_starvation') #SET THIS TO YOUR FOLDER WHERE YOU HAVE KEPT THE DATA FILES
real_food_all=pd.read_csv('food_timing_yeast_starvation.csv')


starvationlist=["0","8","16","24","32"]
# starvationlist=["0"]


foodlist=os.listdir()
foodlist.remove('food_timing_yeast_starvation.csv')
foodlist.remove('food_timing_sugar.csv')

foodlist.remove('food_timing_yeast_starvation.xlsx')
foodlist.remove('results')
foodlist.remove('bad_data')
# foodlist.remove('water')

# foodlist.remove('w1118')
# foodlist.remove('ss39947') #Removes this genotype because there is some problem with this data
genotypelist=foodlist


genotypelist=["10mM","100mM","500mM","1M"]
# "ss35281",
# "ss38164",
# "ss38535",
# "ss39883",
# "ss39916",
# "ss40945",
# "ss43355",
# "ss46373",
# "ss49430"
# ] #USE THIS LINE OF CODE TO SET YOUR GENOTYPE.
genotypelist=["10mM"]
starvationlist=["0"]

starvation="24"
# time_list=[5,10,30,60]

time_thres=60
# real_food_all=pd.read_csv('food_timing_multi_lines.csv') #Reading the Manual Labelling of food eating 

def find_index(l,t):
    for j in l:
        if j>t:
            return l.index(j)
            break
        else:
            continue
    
rad_dist_dict={} #Creating empty dictionary to store data

for genotype in genotypelist:
    for starvation in starvationlist:        
        try:
            radial_distances={}
            radial_distances_full={}
            food_indexes={}
            print(genotype)
            # fnames = sorted(glob.glob(genotype+'/'+'*.csv'))
            fnames = sorted(glob.glob(genotype+'/'+starvation+'/'+'*.csv'))
            print(fnames)
            index=0
            real_food_df=real_food_all[real_food_all['genotype']==genotype]
            real_food_df=real_food_df[real_food_df['starvation']==int(starvation)]
            real_food_bout_list=list(real_food_df['final_first_bout'])
            
            for u in fnames: #goes thru files in the folder.
                print(u)
                """
                Loading the data into a dataframe
                """
                
                df=pd.read_csv(u, header=None)
                print("before",df.shape[1])
                # df=df.dropna(axis=1,thresh=20000)
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
                df['Time']=df['Time']-round(df['Time'][0])
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
                        food_indexes[index]=split_point_index
                        x_coord=list(df[data_header[i]])
                        y_coord=list(df[data_header[i+1]])
                        
                        x_coord_full=copy.deepcopy(x_coord)
                        y_coord_full=copy.deepcopy(y_coord)
                        
                        del x_coord[0:split_point_index]
                        del y_coord[0:split_point_index]
                        
                        rad_dist=list(np.zeros_like(x_coord))
                        rad_dist_full=list(np.zeros_like(x_coord_full))
                        for j in range(0,len(x_coord)-1,1):
                            rad_dist[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2)
                        
                        for j in range(0,len(x_coord_full)-1,1):
                            rad_dist_full[j]=np.sqrt((x_coord_full[j]-540)**2+(y_coord_full[j]-540)**2)

                        rad_dist_full.pop()
                        del rad_dist[time_thres*24:-1]
                        rad_dist.pop()
                        radial_distances[index]=rad_dist
                        radial_distances_full[index]=rad_dist_full
            
            rad_dist_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in radial_distances.items() ]))
            
            rad_dist_df[rad_dist_df>540]=np.nan #Remove problem in tracking
            rad_dist_df=rad_dist_df.mul(conversion)
            rad_dist_full_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in radial_distances_full.items() ]))
            rad_dist_full_df[rad_dist_full_df>540]=np.nan #Remove problem in tracking
            rad_dist_full_df=rad_dist_full_df.mul(conversion)
            food_indexes_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in food_indexes.items() ]))

            """
            Here we plot Radial Distances for 60 seconds only
            """
            
            fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
            plt.subplots_adjust(hspace=0.5)
            fig.suptitle(" {} {}hr Radial Distance ({}s after food) distribution".format(genotype, starvation, time_thres), fontsize=18, y=0.95)
            fig.supxlabel('time (seconds)')
            fig.supylabel('radial distance (mm)')
            # loop through tickers and axes
            for fly_numbers, ax in zip(list(rad_dist_df.columns), axs.ravel()):
                ax.plot(np.arange(0,time_thres*24,1), rad_dist_df[fly_numbers])
                ax.axhline(y=60*conversion, color='orange')
                ax.set_title("{}".format(fly_numbers),fontsize=9)
                ax.set_ylim(0,540*conversion)
                ax.set_xlim(0,time_thres*24)
                ax.set_xlabel("")
                ax.set_xticks(ticks=np.arange(0,time_thres*24+time_thres,time_thres*24/6),labels=np.arange(0,70,10))
                
            # handles, labels = ax.get_legend_handles_labels()
            # fig.legend(handles, ["Fly", "Rad_Dist=60"], loc='right')
            fig.savefig('results\\rad_dist_all_flies\\{}_{}_rad_dist_{}_Seconds.png'.format(genotype,starvation,time_thres),format='png', dpi=600, bbox_inches = 'tight')
            plt.show()
        
            """
            Here we plot the radial distance during the entire experiment
            """
            
            fig, axs = plt.subplots(nrows=8, ncols=1, figsize=(18, 12))
            plt.subplots_adjust(hspace=0.5)
            fig.suptitle(" {} {}hr Radial Distance".format(genotype, starvation), fontsize=18, y=0.95)
            fig.supxlabel('time (min)')
            fig.supylabel('radial distance (mm)')

            # loop through tickers and axes
            for fly_numbers, ax in zip(list(rad_dist_full_df.columns), axs.ravel()):
                ax.plot(np.arange(0,len(rad_dist_full_df[fly_numbers]),1), rad_dist_full_df[fly_numbers])
                ax.axhline(y=60*conversion, color='orange')
                ax.axvline(x=int(food_indexes_df[fly_numbers]),color='r')
                ax.set_title("{}".format(fly_numbers),fontsize=9)
                ax.set_ylim(0,540*conversion)
                ax.set_xlim(0,len(time))
                ax.set_xlabel("")
        
                ax.set_xticks(ticks=np.arange(0,len(time),len(time)/14),labels=np.arange(0,15,1))
            # fig.legend(handles, ["Fly", "Rad_Dist=60"], loc='right')
            fig.savefig('results\\rad_dist_all_flies\\full\\{}_{}_rad_dist_Seconds.png'.format(genotype,starvation),format='png', dpi=600, bbox_inches = 'tight')
            plt.show()
        
        
        
        except:
            pass
        