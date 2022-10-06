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
import scipy.stats as stats
import matplotlib.ticker as ticker

os.chdir('C:\\Users\\Yapicilab\\Dropbox\\screening') #SET THIS TO YOUR FOLDER WHERE YOU HAVE KEPT THE DATA FILES
real_food_all=pd.read_csv('food_timing_multi_lines.csv')


# starvationlist=["0","8","16","24"]


foodlist=os.listdir()
foodlist.remove('food_timing_multi_lines.csv')
foodlist.remove('results')
foodlist.remove('ss46202') #Removes this genotype because there is some problem with this data
genotypelist=foodlist


genotypelist=["ss49422"] #USE THIS LINE OF CODE TO SET YOUR GENOTYPE.
starvation="24"
# time_list=[5,10,30,60]

time_thres=10
real_food_all=pd.read_csv('food_timing_multi_lines.csv') #Reading the Manual Labelling of food eating 

def find_index(l,t):
    for j in l:
        if j>t:
            return l.index(j)
            break
        else:
            continue
    
rad_dist_dict={} #Creating empty dictionary to store data

for genotype in genotypelist:
    
    radial_distances={}
    print(genotype)
    fnames = sorted(glob.glob(genotype+'/'+starvation+'/'+'*.csv'))
    index=0
    real_food_df=real_food_all[real_food_all['genotype']==genotype]
    real_food_bout_list=list(real_food_df['final_first_bout'])
    
    for u in fnames: #goes thru files in the folder.
        
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
                    
                
                del rad_dist[time_thres*24:-1]
                rad_dist.pop()
                radial_distances[index]=rad_dist

    rad_dist_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in radial_distances.items() ]))
    
    rad_dist_df[rad_dist_df>540]=np.nan #Remove problem in tracking
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(" {} Radial Distance ({}s after food) distribution".format(genotype, time_thres), fontsize=18, y=0.95)

    # loop through tickers and axes
    for genotype, ax in zip(list(rad_dist_df.columns), axs.ravel()):
        ax.plot(np.arange(0,time_thres*24,1), rad_dist_df[genotype])
        ax.plot(np.arange(0,time_thres*24,1), [60]*time_thres*24) 
        ax.set_title("{}".format(genotype),fontsize=9)
        ax.set_ylim(0,540)
        ax.set_xlim(0,time_thres*24)
        ax.set_xlabel("")

        ticks = ax.get_xticks()/25
        ax.set_xticklabels(ticks)
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, ["Fly", "Rad_Dist=60"], loc='right')
    # fig.savefig('results\\rad_dist_all_flies\\{}_rad_dist10Seconds.png'.format(genotype),format='png', dpi=600, bbox_inches = 'tight')
    plt.show()
    
    