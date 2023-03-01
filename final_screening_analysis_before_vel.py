# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:03:22 2023

@author: Yapicilab
"""

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

from plot_func import *
from conversion_func import *

os.chdir('C:\\Users\\Yapicilab\\Dropbox\\screening') #SET THIS TO YOUR FOLDER WHERE YOU HAVE KEPT THE DATA FILES
real_food_all=pd.read_csv('food_timing_screening.csv')


# starvationlist=["16","24"]


foodlist=os.listdir()
# foodlist.remove('food_timing_screening.xlsx')
foodlist.remove('food_timing_screening.csv')

foodlist.remove('results')
foodlist.remove('bad_data') #Removes this genotype because there is some problem with this data
genotypelist=foodlist
genotypelist=['ss34089']
#USE THIS LINE OF CODE TO SET YOUR GENOTYPE.
starvation="24"
time_thres=60

# time_thres=60



inst_vel_behav_all_dict={}
inst_vel_behav_binned_all_dict={}

inst_vel_after_mean_all_dict={}
inst_vel_before_mean_all_dict={}
inst_vel_diff_dict={}
inst_vel_ratio_dict={}

inst_vel_onesec_diff_dict={}
inst_vel_onesec_ratio_dict={}

inst_vel_after_binned_mean_all_dict={}
inst_vel_before_binned_mean_all_dict={}
inst_vel_diff_binned_dict={}
inst_vel_ratio_binned_dict={}
number_of_flies={}    

max_vel=0
for genotype in genotypelist:
                
    print(genotype,starvation)
    fnames = sorted(glob.glob(genotype+'/'+starvation+'/'+'*.csv')) #Enter the genotype folder and "24" folder
    index=0 #Counter for counting the number of iterations
    
    real_food_df=real_food_all[real_food_all['genotype']==genotype] #Extracts data for this genotype from Manual labels of first food eating labels
    # real_food_df=real_food_df[real_food_df['starvation']==int(starvation)]
    real_food_bout_list=list(real_food_df['final_first_bout'])
    
    inst_vel_all=[]
    inst_vel_behav=[]
    inst_vel_behav_binned=[]
    inst_vel_behav_dict={}
    inst_vel_behav_binned_dict={}
    
    inst_vel_after_mean_all=[]
    inst_vel_before_mean_all=[]
    inst_vel_diff_all=[]
    inst_vel_ratio_all=[]
    inst_vel_onesec_diff_all=[]
    inst_vel_onesec_ratio_all=[]         

    inst_vel_after_binned_mean_all=[]
    inst_vel_before_binned_mean_all=[]
    inst_vel_diff_binned_all=[]        
    inst_vel_ratio_binned_all=[]

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
        
        df['Time']=df['Time']-round(df['Time'][0]) #Since our experiment begins after 2 minutes, we subtract this time from the data
        
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
                latency=list(df['Latency'])
                latency[0]=np.mean(latency)
                
                x_coord=list(df[data_header[i]])
                y_coord=list(df[data_header[i+1]])
                inst_vel_all=np.zeros_like(x_coord)
                
                for j in range(0,len(x_coord)-1,1):
                    inst_vel_all[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2)/latency[j+1] #Calculate Instantenous Velocity                    
                
                
                inst_vel_before=inst_vel_all[0:split_point_index]
                inst_vel_after=inst_vel_all[split_point_index:len(inst_vel_all)]
                
                inst_vel_before_onesec=inst_vel_before[-24:] #taking just one second of data before food eating
                inst_vel_after_onesec=inst_vel_after[0:24] #taking just one second of data after food eating
                
                if(split_point_index>time_thres*24 and split_point_index<(len(x_coord)-time_thres*24)):
                    inst_vel_before=inst_vel_before[len(inst_vel_before)-time_thres*24:]
                    inst_vel_after=inst_vel_after[0:time_thres*24]
                else:
                    if (len(inst_vel_before)>len(inst_vel_after)):
                        inst_vel_before=inst_vel_before[-len(inst_vel_after):]
                    elif (len(inst_vel_before)<len(inst_vel_after)):
                        inst_vel_after=inst_vel_after[0:len(inst_vel_before)]
                    else:
                        pass
                    
                inst_vel_behav=np.append(inst_vel_before,inst_vel_after)
                
                inst_vel_after_binned=binning(inst_vel_after,10)
                inst_vel_before_binned=binning(inst_vel_before,10)
                
                inst_vel_behav_binned=np.append(inst_vel_before_binned,inst_vel_after_binned)
                
                inst_vel_after_mean_all.append(np.nanmean(inst_vel_after))  #Calculate the mean Instantenous Velocity and append to a list                 
                inst_vel_before_mean_all.append(np.nanmean(inst_vel_before))
                inst_vel_diff_all.append(np.nanmean(inst_vel_after)-np.nanmean(inst_vel_before))
                inst_vel_ratio_all.append(np.nanmean(inst_vel_after)/np.nanmean(inst_vel_before))
                
                inst_vel_onesec_diff_all.append(np.nanmean(inst_vel_after_onesec)-np.nanmean(inst_vel_before_onesec))
                inst_vel_onesec_ratio_all.append(np.nanmean(inst_vel_after_onesec)/np.nanmean(inst_vel_before_onesec))
                
                inst_vel_after_binned_mean_all.append(np.nanmean(inst_vel_after_binned))  #Calculate the mean Instantenous Velocity and append to a list                 
                inst_vel_before_binned_mean_all.append(np.nanmean(inst_vel_before_binned))
                inst_vel_diff_binned_all.append(np.nanmean(inst_vel_after_binned)-np.nanmean(inst_vel_before_binned))
                inst_vel_ratio_binned_all.append(np.nanmean(inst_vel_after_binned)/np.nanmean(inst_vel_before_binned))
                
                inst_vel_behav_dict[index]=inst_vel_behav
                inst_vel_behav_binned_dict[index]=inst_vel_behav_binned
                inst_vel_behav_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_behav_dict.items() ]))
                inst_vel_behav_binned_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in inst_vel_behav_binned_dict.items() ]))

    
    inst_vel_behav_all_dict[genotype]=inst_vel_behav_df            
    inst_vel_behav_binned_all_dict[genotype]=inst_vel_behav_binned_df
    number_of_flies[genotype]=index
    
    
    
    inst_vel_after_mean_all_dict[genotype]=inst_vel_after_mean_all #Append the Mean Instantaneous Velocity LIST to Dictionary with the Corresponding Genotype
    inst_vel_before_mean_all_dict[genotype]=inst_vel_before_mean_all
    inst_vel_diff_dict[genotype]=inst_vel_diff_all
    inst_vel_ratio_dict[genotype]=inst_vel_ratio_all
    
    inst_vel_onesec_diff_dict[genotype]=inst_vel_onesec_diff_all
    inst_vel_onesec_ratio_dict[genotype]=inst_vel_onesec_ratio_all
    
    
    inst_vel_after_binned_mean_all_dict[genotype]=inst_vel_after_binned_mean_all #Append the Mean Instantaneous Velocity LIST to Dictionary with the Corresponding Genotype
    inst_vel_before_binned_mean_all_dict[genotype]=inst_vel_before_binned_mean_all
    inst_vel_diff_binned_dict[genotype]=inst_vel_diff_binned_all
    inst_vel_ratio_binned_dict[genotype]=inst_vel_ratio_binned_all
    
"""
Converting Dictionaries to Dataframe using functions

"""
inst_vel_after_mean_all_df = dict_to_df(inst_vel_after_mean_all_dict)
inst_vel_after_binned_mean_all_df= dict_to_df(inst_vel_after_binned_mean_all_dict)
inst_vel_before_mean_all_df = dict_to_df(inst_vel_before_mean_all_dict)
inst_vel_before_binned_mean_all_df = dict_to_df(inst_vel_before_binned_mean_all_dict)
inst_vel_diff_all_df=dict_to_df(inst_vel_diff_dict)
inst_vel_diff_all_binned_df=dict_to_df(inst_vel_diff_binned_dict)
inst_vel_ratio_df=dict_to_df(inst_vel_ratio_dict)
inst_vel_ratio_binned_df=dict_to_df(inst_vel_ratio_binned_dict)

inst_vel_onesec_diff_df=dict_to_df(inst_vel_onesec_diff_dict)
inst_vel_onesec_ratio_df=dict_to_df(inst_vel_onesec_ratio_dict)    


"""
Converting dataframes to CSV files
"""
# inst_vel_onesec_diff_df.to_csv("results\\inst_vel_onesec_diff_df.csv",index=False)
# inst_vel_onesec_ratio_df.to_csv("results\\inst_vel_onesec_ratio_df.csv",index=False)




"""
Generating individual velocity graphs
"""

# individual_plot_dict(inst_vel_behav_binned_all_dict)




"""
Here we generate comparative boxplots for genotypes
"""
box_plot(inst_vel_onesec_ratio_df)
box_plot(inst_vel_onesec_diff_df)
# ax.set_title("Ratio of Avg Speed one second after and before eating the food", fontsize=11)
# plt.show()

"""
This section is for plotting Raw Instantaneous velocity and deciding on the Kernel Size
"""

# kernel_size = 10
# kernel = np.ones(kernel_size) / kernel_size
# data_convolved_10 = np.convolve(inst_vel_after, kernel, mode='valid')

# kernel_size = 12
# kernel = np.ones(kernel_size) / kernel_size
# data_convolved_12 = np.convolve(inst_vel_after, kernel, mode='valid')
 
# kernel_size = 20
# kernel = np.ones(kernel_size) / kernel_size
# data_convolved_20 = np.convolve(inst_vel_after, kernel, mode='valid')

 
# plt.plot(inst_vel_after[0:240])
# plt.plot(data_convolved_10[0:240])
# plt.plot(data_convolved_12[0:240])
# plt.plot(data_convolved_20[0:240])
# plt.xlabel("Frames")
# plt.ylabel("Velocity (pixel/seconds)")
# plt.legend(("Raw Instantaneous Velocity","Kernel Size 10","Kernel Size 12", "Kernel Size 20"), fontsize=5)
# plt.savefig("results\\Instantaneous Velocity Kernels", dpi=1200,bbox_inches='tight')

# """
# Here we wanna do averaging by Binning
# """
# inst_vel_after_binned=[]
# i=0
# while i<len(inst_vel_after):
#     inst_vel_after_binned.append(np.sum(inst_vel_after[i:i+5])/5)
#     i=i+5


# fig=plt.figure()
# ax=fig.add_subplot(111, label="1")
# ax2=fig.add_subplot(111, label="2", frame_on=False)
# ax3=fig.add_subplot(111, label="3", frame_on=False)

# ax.plot(np.arange(0,240,1),inst_vel_after[0:240], color="C0")
# ticks = ax.get_xticks()/25
# ax.set_xticklabels(ticks)
# # ax.set_xlabel("x label 1", color="C0")
# # ax.set_ylabel("y label 1", color="C0")
# # ax.tick_params(axis='x', colors="C0")
# # ax.tick_params(axis='y', colors="C0")

# ax2.plot(np.arange(0,240,1),data_convolved_10[0:240], color="C1")
# # ax2.xaxis.tick_top()
# # ax2.yaxis.tick_right()
# # ax2.set_xlabel('x label 2', color="C1") 
# # ax2.set_ylabel('y label 2', color="C1")       
# # ax2.xaxis.set_label_position('top') 
# # ax2.yaxis.set_label_position('right') 
# # ax2.tick_params(axis='x', colors="C1")
# # ax2.tick_params(axis='y', colors="C1")

# ax3.plot(np.arange(0,48,1),inst_vel_after_binned[0:48],  color="C3")

# # ax3.set_xticks([])
# # ax3.set_yticks([])

# plt.show()






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

"""
Trying manual input popup
"""

# from tkinter import *
 
# root = Tk()
# root.geometry("200x150")
 
# frame = Frame(root)
# frame.pack()
 
# my_entry = Entry(frame, width = 20)
# my_entry.insert(0,'Username')
# my_entry.pack(padx = 5, pady = 5)
 
# my_entry2 = Entry(frame, width = 15)
# my_entry2.insert(0,'password')
# my_entry2.pack(padx = 5, pady = 5)
 
# root.mainloop()

# from tkinter import *
 
# def retrieve():
#     return(my_entry.get())
#     return(my_entry2.get())
 
# root = Tk()
# root.geometry("200x150")
 
# frame = Frame(root)
# frame.pack()
 
# my_entry = Entry(frame, width = 20)
# my_entry.insert(0,'Username')
# my_entry.pack(padx = 5, pady = 5)
 
# my_entry2 = Entry(frame, width = 15)
# my_entry2.insert(0,'password')
# my_entry2.pack(padx = 5, pady = 5)
 
# Button = Button(frame, text = "Submit", command = retrieve)
# Button.pack(padx = 5, pady = 5)
 
# root.mainloop()