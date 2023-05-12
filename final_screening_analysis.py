# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:12:51 2022

@author: Yapicilab
"""
"""
Readme:
    
This code takes 2 inputs :
    1. "food_timing_multi_lines.csv", which is the csv file containing time labels at which each fly of each genotype first ate food.
    2. XY coordinates of the fly. The code first enters the folder of the genotype ("foodlist" contains the list of all genotypes in the data folder), and then enter the "24" folder.
        Inside the "24" folder, there are 2 csv files, containing data for 8 flies. The code parses through each one by one.

Within the "for u in fnames:" loop, the code then takes the XY coordinates of each fly, and deletes the data prior to when the fly first ate food
It then calculates some parameters, stores them in lists. It then deletes the data in those lists after a specified "time_thres", which loops through [5,10,30,60].

It saves the parameters in CSV files.

The code also generates boxplots to compare genotypes for average radial distance and median radial distance within the "time_thres". 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as stats
import matplotlib.ticker as ticker
from mpl_toolkits import mplot3d



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
# genotypelist=["1yeast","5yeast","10yeast"]


 #Removes this genotype because there is some problem with this data
# genotypelist=foodlist
diameter_of_arena=50.8 #in mm
num_of_pixels=1080

conversion=diameter_of_arena/num_of_pixels

# starvation="24"
time_list=[60]

# time_thres=60

def find_index(l,t): #This function helps us find indexes
    for j in l:
        if j>t:
            return l.index(j)
            break
        else:
            continue
def count_jumps(sequence):
    count = 0
    for i in range(len(sequence)-1):
        if sequence[i+1] - sequence[i] >= 12:
            count += 1
    return count
def find_distance_below_threshold(distances, threshold):
    below_threshold_indices = [i for i, distance in enumerate(distances) if distance <= threshold]
    return below_threshold_indices

i=0#initializing counter
for time_thres in time_list: 
    
    inst_vel_dict={}
    rad_dist_dict={} #Creating empty dictionaries for saving some data
    tot_dist_dict={}
     
    rad_dist_med_dict={}
    inst_vel_med_dict={}
    tot_dist_med_dict={}
    
    rad_dist_med_all_dict={}
    inst_vel_med_all_dict={}
    
    rad_dist_mean_all_dict={}
    inst_vel_mean_all_dict={}
   
    number_of_flies={}
    tort_coeff_dict={}
    
    compiled_df=pd.DataFrame(columns=['genotype','starvation','rad_dist','inst_vel','tort_coeff','tot_dist','foodvisits_60','foodvisits_full'])
    compiled_mean_df=pd.DataFrame(columns=['genotype','starvation','rad_dist','inst_vel','tot_dist'])

    
    for genotype in genotypelist:
        for starvation in starvationlist:                    
            i=i+1
            flies=0
            print(genotype,starvation)
            fnames = sorted(glob.glob(genotype+'/'+'*.csv'))
            fnames = sorted(glob.glob(genotype+'/' + starvation +'/'+'*.csv')) #Enter the genotype folder and "24" folder
            index=0 #Counter for counting the number of iterations
            
            real_food_df=real_food_all[real_food_all['genotype']==genotype] #Extracts data for this genotype from Manual labels of first food eating labels
            real_food_df=real_food_df[real_food_df['starvation']==int(starvation)]
            real_food_bout_list=list(real_food_df['final_first_bout'])
            
            inst_vel_all=[]
            rad_dist_all=[] #Lists to store all the raw data
            tot_dist_all=[]
            tort_coeff_all=[]
            
            inst_vel_median_all=[]
            rad_dist_median_all=[] #Lists to store the medians of data
            tot_dist_median_all=[]
            
            inst_vel_mean_all=[]
            rad_dist_mean_all=[] #Lists to store the means of data
            tot_dist_mean_all=[]
            
            for u in fnames: #goes through files in the folder.
                
                """
                Loading the data into a dataframe
                """
                
                df=pd.read_csv(u, header=None)
                # print("before",df.shape[1])
                # df=df.dropna(axis=1,thresh=20000)
                # print("after",df.shape[1])
                if(df.shape[1]==10):
                    data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
                elif(df.shape[1]==8):
                    data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
                elif(df.shape[1]==6):
                     data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2']
                elif(df.shape[1]==4):
                     data_header = ['Time', 'Latency', 'Fx1', 'Fy1']
                df.columns=data_header
                
                latency=list(df['Latency']) #Just making a list for Latency, which is the second column in our data csv files. We dont use this anywhere.
                latency[0]=0 #Usually the latency of the first observation is really high, so we manually set it to zero.
                
                df['Time']=df['Time']-round(df['Time'][0]) #Since our experiment begins after 2 minutes, we subtract this time from the data
                
                for i in range(2,len(data_header),2):
                    index=index+1
                    first_bout_time=real_food_bout_list[index-1] #Extracting the first bout time from manual annotation and storing it in a variable
                    if (first_bout_time==2000):
                        continue
                        flies=flies+0
                    else:
                        pass
                        flies=flies+1
                        time=list(df['Time'])
                        split_point_index=find_index(time, first_bout_time)
                        print(index,first_bout_time,split_point_index)
                        
                        x_coord=list(df[data_header[i]])
                        y_coord=list(df[data_header[i+1]])
                        del x_coord[0:split_point_index] #Delete data from time=0 to time= First eating bout
                        del y_coord[0:split_point_index]
                        
                        inst_vel_full=list(np.zeros_like(x_coord)) #create an empty list
                        rad_dist_full=list(np.zeros_like(x_coord)) #create an empty list
                        distance_travelled_full=list(np.zeros_like(x_coord)) #create an empty list
                        
                        for j in range(0,len(x_coord)-1,1):
                            rad_dist_full[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2) #Calculate Radial Distance
                            inst_vel_full[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2)/latency[j+1] #Calculate Instantenous Velocity
                            distance_travelled_full[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2) #Calculate Total Distance Travelled
                        
                        wall_touching_point=find_index(rad_dist_full, 520)  #Here I find the time point at which the fly first touched the wall after eating food
                        if wall_touching_point is None:    
                            wall_touching_point=len(rad_dist_full)-1
                        else:
                            wall_touching_point=wall_touching_point
                        print(wall_touching_point)
                        tort_distance=distance_travelled_full
                        displacement=np.sqrt((x_coord[-1]-x_coord[0])**2+(y_coord[-1]-y_coord[0])**2)
                        
                        rad_dist=rad_dist_full[0:time_thres*24]
                        distance_travelled=distance_travelled_full[0:time_thres*24]
                        tort_distance=tort_distance[0:time_thres*24]
                        inst_vel=inst_vel_full[0:time_thres*24]
                        
                        # del distance_travelled[wall_touching_point:-1] #Delete data from the Total_Distance list after fly has touched wall
                        # del tort_distance[time_thres*24:-1]
                        # del rad_dist[time_thres*24:-1] #Delete everything after declared time
                        # del inst_vel[time_thres*24:-1]
                        # rad_dist.pop() #Remove the last element which is somehow always 0
                        # inst_vel.pop() #Remove the last element which is somehow always 0
                        total_distance_travelled=np.nansum(distance_travelled) 
                        
                        tort=np.sum(tort_distance)/displacement
                        
                        rad_dist_median_all.append(np.median(rad_dist)) #Calculate the median radial distance and append to a list
                        inst_vel_median_all.append(np.median(inst_vel)) #Calculate the median Instantenous Velocity and append to a list
        
                        rad_dist_mean_all.append(np.nanmean(rad_dist))  #Calculate the mean radial distance and append to a list
                        inst_vel_mean_all.append(np.nanmean(inst_vel))  #Calculate the mean Instantenous Velocity and append to a list                 
                        
                        compiled_df=compiled_df.append({'genotype':genotype,'starvation':starvation,'fly_num':index,'rad_dist':np.nanmean(rad_dist),'inst_vel':np.nanmean(inst_vel),'tort_coeff':tort,'tot_dist':total_distance_travelled, 'foodvisits_60':count_jumps(find_distance_below_threshold(rad_dist,60)),'foodvisits_full':count_jumps(find_distance_below_threshold(rad_dist_full,60))},ignore_index=True)

                        
                        rad_dist_all.extend(rad_dist) #Append the Radial Distance list to one huge raw data list
                        inst_vel_all.extend(inst_vel) #Append the Instantaneous Velocity list to one huge raw data list
                        tot_dist_all.append(total_distance_travelled) #Append the Total Distance Travelled list to one huge raw data list
                        tort_coeff_all.append(tort)
                        
                        inst_vel_all = [x for x in inst_vel_all if np.isnan(x) == False]
                        rad_dist_all = [x for x in rad_dist_all if np.isnan(x) == False] #Clean NaNs from the huge list
                        tot_dist_all = [x for x in tot_dist_all if np.isnan(x) == False]                
        
            number_of_flies[genotype]=index
    
            inst_vel_dict[genotype]=inst_vel_all #Append the huge raw data list to Dictionary with the Corresponding genotype+starvation  
            rad_dist_dict[genotype]=rad_dist_all
            tot_dist_dict[genotype]=tot_dist_all
            tort_coeff_dict[genotype]=tort_coeff_all
            
            inst_vel_med_dict[genotype]=np.median(inst_vel_all) #Just taking the median of the huge raw data list and appending it to another dictionary
            rad_dist_med_dict[genotype]=np.median(rad_dist_all) 
            tot_dist_med_dict[genotype]=np.median(tot_dist_all)    
        
            rad_dist_med_all_dict[genotype]=rad_dist_median_all #Append the Median Radial Distance LIST to Dictionary with the Corresponding Genotype
            inst_vel_med_all_dict[genotype]=inst_vel_median_all #Append the Median Instantaneous Velocity LIST to Dictionary with the Corresponding Genotype
            
            rad_dist_mean_all_dict[genotype]=rad_dist_mean_all #Append the Mean Radial Distance LIST to Dictionary with the Corresponding Genotype
            inst_vel_mean_all_dict[genotype]=inst_vel_mean_all #Append the Mean Instantaneous Velocity LIST to Dictionary with the Corresponding Genotype
            
            compiled_mean_df=compiled_mean_df.append({'genotype':genotype,'starvation':starvation,'rad_dist':np.nanmean(rad_dist_mean_all),'rad_dist_var':np.nanvar(rad_dist_mean_all),'inst_vel':np.nanmean(inst_vel_mean_all),'tot_dist':np.nanmean(tot_dist_all),'tot_flies':flies},ignore_index=True)
            
    "All radial distance and inst velocity raw data"
    
    rad_dist_med_dict = dict(sorted(rad_dist_med_dict.items(), key=lambda x: x[1], reverse=True)) #Sorting the dictionary according to medians
    inst_vel_med_dict = dict(sorted(inst_vel_med_dict.items(), key=lambda x: x[1], reverse=True))
    tot_dist_med_dict = dict(sorted(tot_dist_med_dict.items(), key=lambda x: x[1], reverse=True))
    
    inst_vel_med_dict_labels, inst_vel_med_dict_data = [*zip(*inst_vel_med_dict.items())]  #Taking the sorted dictionary's labels
    rad_dist_med_dict_labels, rad_dist_med_dict_data = [*zip(*rad_dist_med_dict.items())]  
    tot_dist_med_dict_labels, tot_dist_med_dict_data = [*zip(*tot_dist_med_dict.items())]  
    
    inst_vel_dict_labels, inst_vel_dict_data = [*zip(*inst_vel_dict.items())]  
    rad_dist_dict_labels, rad_dist_dict_data = [*zip(*rad_dist_dict.items())]  #Extracting labels of complete unsorted raw data for plotting later
    tot_dist_dict_labels, tot_dist_dict_data = [*zip(*tot_dist_dict.items())]  
    
    inst_vel_dict_final = {k: inst_vel_dict[k] for k in inst_vel_med_dict_labels}
    rad_dist_dict_final = {k: rad_dist_dict[k] for k in rad_dist_med_dict_labels} #Using the sorted labels to sort complete raw data
    tot_dist_dict_final = {k: tot_dist_dict[k] for k in tot_dist_med_dict_labels}
    
    inst_vel_dict_final_labels, inst_vel_dict_final_data = [*zip(*inst_vel_dict_final.items())]  
    rad_dist_dict_final_labels, rad_dist_dict_final_data = [*zip(*rad_dist_dict_final.items())]  #Extracting labels of sorted raw data for plotting later
    tot_dist_dict_final_labels, tot_dist_dict_final_data = [*zip(*tot_dist_dict_final.items())]  
    
    rad_dist_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rad_dist_dict_final.items() ]))
    inst_vel_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_dict_final.items() ]))   #Converting the dictionary of sorted rawdata to a Dataframe
    tot_dist_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tot_dist_dict_final.items() ]))
    
    
    """
    Median of Radial Distance and Instantaneous Velocity Dataframe
    
    """
    rad_dist_med_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rad_dist_med_all_dict.items() ])) #convert dictionary to dataframe
    inst_vel_med_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_med_all_dict.items() ]))
    
    sorted_index_rad = rad_dist_med_all_df.median().sort_values().index  #Use the median values to sort the index
    sorted_index_inst = inst_vel_med_all_df.median().sort_values().index
    
    rad_dist_med_all_df=rad_dist_med_all_df[sorted_index_rad]  #Using the sorted Index to sort the Dataframes
    inst_vel_med_all_df=inst_vel_med_all_df[sorted_index_inst]
    
    # rad_dist_med_all_df.to_csv("results\\rad_dist_med_all_{}_df.csv".format(time_thres),index=False) #writing the df to a csv file
    # inst_vel_med_all_df.to_csv("results\\inst_vel_med_all_{}_df.csv".format(time_thres),index=False)
    
    """
    Mean of Radial Distance and Instantaneous Velocity Dataframe
    
    """
    rad_dist_mean_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in rad_dist_mean_all_dict.items() ])) #convert dictionary to dataframe
    inst_vel_mean_all_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_mean_all_dict.items() ]))
    tort_coeff_all_df= pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in tort_coeff_dict.items() ]))
    
    sorted_index_rad = rad_dist_mean_all_df.median().sort_values().index #Use the median values to sort the index
    sorted_index_inst = inst_vel_mean_all_df.median().sort_values().index
    sorted_index_tort=tort_coeff_all_df.median().sort_values().index
    
    rad_dist_mean_all_df=rad_dist_mean_all_df[sorted_index_rad] #Using the sorted Index to sort the Dataframes
    inst_vel_mean_all_df=inst_vel_mean_all_df[sorted_index_inst]
    tort_coeff_all_df=tort_coeff_all_df[sorted_index_tort]
    
    
    rad_dist_mean_all_df.to_csv("results\\rad_dist_mean_all{}_df.csv".format(time_thres),index=False) #writing the df to a csv file
    inst_vel_mean_all_df.to_csv("results\\inst_vel_mean_all{}_df.csv".format(time_thres),index=False)
    
    # tot_dist_df.to_csv("results\\tot_dist_df_{}_df.csv".format(time_thres),index=False)
    compiled_df.to_csv("results\\compiled_{}_df.csv".format(time_thres),index=False)


    # """
    # PLOTTING
    # """
    
    # """
    # Here we generate comparative boxplots for genotypes
    # """
    
    
    # fig, ax= plt.subplots()
    # sns.set_style("white")
    # ax = sns.boxplot(data=rad_dist_dict_final_data, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    # # ax = sns.violinplot(data=rad_dist_dict_final_data, split=True)
    # ax.set_xlabel('Radial Distance')
    # ax.set_yticklabels(rad_dist_dict_final_labels, fontsize=5)
    # ax.tick_params(axis='x', labelrotation = 0, size=2)
    # ax.set_title('Radial Distance 10 seconds after first contact with Food', fontsize=13)
    # ax.set_ylabel('Genotypes')
    # # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    # plt.show()
    # # fig.savefig('radial_distance_10seconds.png',format='png', dpi=600, bbox_inches = 'tight')
    
    # fig, ax= plt.subplots()
    # sns.set_style("white")
    # ax = sns.boxplot(data=inst_vel_dict_final_data, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    # # ax = sns.stripplot(data=data2, color=".25",size=3,orient="h")
    # ax.set_xlabel('Instantenous Velocity')
    # ax.set_yticklabels(inst_vel_dict_final_labels, fontsize=5)
    # ax.tick_params(axis='x', labelrotation = 0, size=2)
    # ax.set_title('Instantenous Velocity 10 seconds after first contact with Food', fontsize=13)
    # ax.set_ylabel('Genotypes')
    # # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    # plt.show()
    # # fig.savefig('inst_vel_10seconds.png',format='png', dpi=600, bbox_inches = 'tight')
    
    # fig, ax= plt.subplots()
    # sns.set_style("white")
    # ax = sns.boxplot(data=tot_dist_dict_final_data, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    # ax = sns.stripplot(data=tot_dist_dict_final_data,color=".25",size=2, orient='h')
    # # ax = sns.violinplot(data=rad_dist_dict_final_data, split=True)
    # ax.set_xlabel('Distance')
    # ax.set_yticklabels(tot_dist_dict_final_labels, fontsize=5)
    # ax.tick_params(axis='x', labelrotation = 0, size=2)
    # ax.set_title('Distance travelled before touching wall', fontsize=13)
    # ax.set_ylabel('Genotypes')
    # # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    # plt.show()
    # # fig.savefig('results\\total_distance_before_wall_touching.png',format='png', dpi=600, bbox_inches = 'tight')
    
    
    # """
    # Here we generate comparative boxplots for genotypes, but median only
    # """
    
    
    # fig, ax= plt.subplots()
    # sns.set_style("white")
    # ax = sns.boxplot(data=rad_dist_med_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    # ax = sns.stripplot(data=rad_dist_med_all_df,size=2,color=".25", orient="h")
    # ax.set_xlabel('Radial Distance')
    # ax.set_yticklabels(rad_dist_med_all_df.columns, fontsize=5)
    # ax.tick_params(axis='x', labelrotation = 0, labelsize=5)
    # ax.set_title('Radial Distance 10 seconds after first contact with Food', fontsize=13)
    # ax.set_ylabel('Genotypes')
    # # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    # plt.show()
    # # fig.savefig('results\\radial_distance_10seconds_medians.png',format='png', dpi=600, bbox_inches = 'tight')
    
    # fig, ax= plt.subplots()
    # sns.set_style("white")
    # ax = sns.boxplot(data=inst_vel_med_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    # ax = sns.stripplot(data=inst_vel_med_all_df, color=".25",size=2, orient="h")
    # ax.set_xlabel('Instantaneous Velocity')
    # ax.set_yticklabels(inst_vel_med_all_df.columns, fontsize=5)
    # ax.tick_params(axis='x', labelrotation = 0, size=2)
    # ax.set_title('Instantaneous Velocity 10 seconds after first contact with Food', fontsize=13)
    # ax.set_ylabel('Genotypes')
    # # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    # plt.show()
    # # fig.savefig('results\\inst_vel_10seconds_medians.png',format='png', dpi=600, bbox_inches = 'tight')
    
    """
    Here we generate comparative boxplots for genotypes, but mean only
    """
    
    
    fig, ax= plt.subplots()
    sns.set_style("white")
    ax = sns.boxplot(data=rad_dist_mean_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    ax = sns.stripplot(data=rad_dist_mean_all_df,size=2,color=".25", orient="h")
    ax.set_xlabel('Radial Distance')
    ax.set_yticklabels(rad_dist_mean_all_df.columns, fontsize=5)
    ax.tick_params(axis='x', labelrotation = 0, labelsize=5)
    ax.set_title('Average Radial Distance {} seconds after first contact with Food'.format(time_thres), fontsize=11)
    ax.set_ylabel('Genotypes')
    # ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.show()
    fig.savefig('results\\radial_distance_{}seconds_means.png'.format(time_thres),format='png', dpi=600, bbox_inches = 'tight')
    
    fig, ax= plt.subplots()
    sns.set_style("white")
    ax = sns.boxplot(data=inst_vel_mean_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    ax = sns.stripplot(data=inst_vel_mean_all_df, color=".25",size=2, orient="h")
    ax.set_xlabel('Instantaneous Velocity')
    ax.set_yticklabels(inst_vel_mean_all_df.columns, fontsize=5)
    ax.tick_params(axis='x', labelrotation = 0, size=2)
    ax.set_title('Average Instantaneous Velocity {} seconds after first contact with Food'.format(time_thres), fontsize=11)
    ax.set_ylabel('Genotypes')
    # ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.show()
    fig.savefig('results\\inst_vel_{}seconds_means.png'.format(time_thres),format='png', dpi=600, bbox_inches = 'tight')
    
    fig, ax= plt.subplots()
    sns.set_style("white")
    ax = sns.boxplot(data=tort_coeff_all_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    ax = sns.stripplot(data=tort_coeff_all_df, color=".25",size=2, orient="h")
    ax.set_xlabel('Tortional Coefficient')
    ax.set_yticklabels(tort_coeff_all_df.columns, fontsize=5)
    ax.tick_params(axis='x', labelrotation = 0, size=2)
    ax.set_title('Tortional Coefficient {} seconds after first contact with Food'.format(time_thres), fontsize=13)
    ax.set_ylabel('Genotypes')
    # ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.show()
    # fig.savefig('results\\tort_coeff_{}seconds.png'.format(time_thres),format='png', dpi=600, bbox_inches = 'tight')
    

"""
Trying to do 3D plotting with Radial Distance, Instantaneous Velocity and Total Distance Travelled
"""

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(rad_dist_mean_all_df['ss31362'].dropna(), inst_vel_mean_all_df['ss31362'].dropna(), tot_dist_df['ss31362'].dropna())
# ax.scatter3D(rad_dist_mean_all_df['ss45929'].dropna(), inst_vel_mean_all_df['ss45929'].dropna(), tot_dist_df['ss45929'].dropna())
# ax.scatter3D(rad_dist_mean_all_df['ss39993'].dropna(), inst_vel_mean_all_df['ss39993'].dropna(), tot_dist_df['ss39993'].dropna())
# ax.scatter3D(rad_dist_mean_all_df['ss45693'].dropna(), inst_vel_mean_all_df['ss45693'].dropna(), tot_dist_df['ss45693'].dropna())
# ax.scatter3D(rad_dist_mean_all_df['ss45926'].dropna(), inst_vel_mean_all_df['ss45926'].dropna(), tot_dist_df['ss45926'].dropna())

# # ax.scatter3D(rad_dist_mean_all_df['w1118'].dropna(), inst_vel_mean_all_df['w1118'].dropna(), tot_dist_df['w1118'].dropna())

# ax.set_xlabel('Rad')
# ax.set_ylabel('Vel')
# ax.set_zlabel('Tot')



# """
# Here we generate Histograms of our raw data to see what the raw data looks like
# """


# fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
# plt.subplots_adjust(hspace=0.5)
# fig.suptitle("Individual genotype Radial Distance (10s after food) distribution", fontsize=18, y=0.95)

# # loop through tickers and axes
# for ticker, ax in zip(list(rad_dist_dict_labels), axs.ravel()):
#     sns.histplot(x=rad_dist_df[ticker],ax=ax, bins=10, kde=True, alpha=0.5)
#     # ax.tick_params(axis='y', labelrotation = 0, size=2)
#     ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
#     ax.set_xlim(0,540)
#     ax.set_xlabel("")
#     ax.set_yticks([])
#     ax.set(ylabel=None)
#     # ax.text(450, 1000, "n={}".format(number_of_flies[ticker]))
# # fig.savefig('results\\Ind_gen_distb_rad_dist.png',format='png', dpi=600, bbox_inches = 'tight')
# plt.show()

# fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
# plt.subplots_adjust(hspace=0.5)
# fig.suptitle("Individual genotype Inst. Velocity (10s after food) distribution", fontsize=18, y=0.95)
# for ticker, ax in zip(list(inst_vel_dict_labels), axs.ravel()):
#     sns.histplot(x=inst_vel_df[ticker],ax=ax,bins=10, kde=True, alpha=0.5)
#     # ax.tick_params(axis='y', labelrotation = 0, size=2)
#     ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
#     ax.set_xlabel("")
    
#     ax.set_yticks([])
#     ax.set(ylabel=None)
# # fig.savefig('results\\Ind_gen_distb_inst_vel.png',format='png', dpi=600, bbox_inches = 'tight')
# plt.show()

"""
Trying to plot heatmap
"""



rad_dist_heat = pd.pivot_table(compiled_mean_df, values='rad_dist', index=['starvation'], columns=['genotype'])
rad_dist_heat = rad_dist_heat[genotypelist]
rad_dist_heat = rad_dist_heat.reindex(starvationlist)
rad_dist_heat=rad_dist_heat.mul(conversion)

rad_dist_var_heat = pd.pivot_table(compiled_mean_df, values='rad_dist_var', index=['starvation'], columns=['genotype'])
rad_dist_var_heat = rad_dist_var_heat[genotypelist]
rad_dist_var_heat = rad_dist_var_heat.reindex(starvationlist)
rad_dist_var_heat=rad_dist_var_heat.mul(conversion**2)

def add_var(x):
    return "var=" + str(round(x, 2))

annot_var_heat = rad_dist_var_heat.round(2).applymap(add_var)

tot_flies_heat = pd.pivot_table(compiled_mean_df, values='tot_flies', index=['starvation'], columns=['genotype'])
tot_flies_heat = tot_flies_heat[genotypelist]
tot_flies_heat = tot_flies_heat.reindex(starvationlist)

def add_n(x):
    return "n=" + str(int(x))

annot_tot_flies_heat = tot_flies_heat.applymap(add_n)

fig,ax=plt.subplots()
ax=sns.heatmap(rad_dist_heat,annot=False, cmap='YlOrRd')
ax=sns.heatmap(rad_dist_heat,annot=annot_var_heat, annot_kws={'va':'bottom'},fmt="", cbar=False, cmap='YlOrRd')
ax=sns.heatmap(rad_dist_heat,annot=annot_tot_flies_heat, annot_kws={'va':'top'},fmt="", cbar=False, cmap='YlOrRd')


# Add axis labels and title
ax.set_xlabel('Food Provided')
ax.set_ylabel('Starvation Level')
ax.set_title('Average Radial Distance (in mm) 60 seconds after eating food')
fig.savefig('results\\sugar_heatmap_rad_dist.png',format='png', dpi=600, bbox_inches = 'tight')
plt.show()


# inst_vel_heat = pd.pivot_table(compiled_mean_df, values='inst_vel', index=['starvation'], columns=['genotype'])
# inst_vel_heat = inst_vel_heat[['1yeast', '5yeast', '10yeast']]
# inst_vel_heat = inst_vel_heat.reindex(starvationlist)
# inst_vel_heat=inst_vel_heat.mul(conversion)

# fig,ax=plt.subplots()
# ax=sns.heatmap(inst_vel_heat, cmap='YlOrRd')

# # Add axis labels and title
# ax.set_xlabel('Food Provided')
# ax.set_ylabel('Starvation Level')
# ax.set_title('Average Speed (in mm/s) 60 seconds after eating food')
# # fig.savefig('results\\SUGAR_heatmap_inst_vel.png',format='png', dpi=600, bbox_inches = 'tight')
# # plt.show()
# # Show the plot
# plt.show()






# """
# Here we generate QQ plots to check for normality of our data
# """


# fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
# plt.subplots_adjust(hspace=0.5)
# fig.suptitle("QQ PLOT Individual genotype Radial Distance (10s after food) distribution", fontsize=18, y=0.95)

# # loop through tickers and axes
# for ticker, ax in zip(list(rad_dist_dict_final_labels), axs.ravel()):
#     qqplot(data=rad_dist_df[ticker],ax=ax, line='s', marker='o', markersize=1, alpha=0.5)
#     # ax.tick_params(axis='y', labelrotation = 0, size=2)
#     ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
#     # ax.set_xlim(0,540)
#     ax.set_xlabel("")
#     ax.set_yticks([])
#     ax.set(ylabel=None)
# fig.savefig('results\\Ind_gen_qq_rad_dist.png',format='png', dpi=600, bbox_inches = 'tight')
# plt.show()

# fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(25, 12))
# plt.subplots_adjust(hspace=0.5)
# fig.suptitle("QQ PLOT Individual genotype Inst. Vel. (10s after food) distribution", fontsize=18, y=0.95)

# # loop through tickers and axes
# for ticker, ax in zip(list(inst_vel_dict_final_labels), axs.ravel()):
#     qqplot(data=inst_vel_df[ticker],ax=ax, line='s', marker='o', markersize=1, alpha=0.5)
#     # ax.tick_params(axis='y', labelrotation = 0, size=2)
#     ax.set_title("{} (n={})".format(ticker,number_of_flies[ticker]),fontsize=9)
#     # ax.set_xlim(0,540)
#     ax.set_xlabel("")
#     ax.set_yticks([])
#     ax.set(ylabel=None)
# fig.savefig('results\\Ind_gen_qq_inst_vel.png',format='png', dpi=600, bbox_inches = 'tight')
# plt.show()

