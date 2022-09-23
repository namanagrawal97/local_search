# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:56:04 2022

@author: naman
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
from statsmodels.graphics.tsaplots import plot_acf
os.chdir('G:\\My Drive\\local_search\\local_search_well\\')
#os.chdir('V:\\local_search_new_new\\')
#os.cwd()
#Set Parameters here. 

foodlist=os.listdir()
foodlist.remove('desktop.ini')
starvationlist=["0","8","16","24"]
t=5
trajtime=120     #Insert time in seconds
norm_revisits_df=pd.DataFrame()
raddistdf3=pd.DataFrame()
raddistdf4=pd.DataFrame(index=foodlist, columns=starvationlist)
radial_distance_mean=pd.DataFrame()
total_FRAvisitsdf={}
alllist=[]

# food='1M'
# starvation='24'
# fnames = glob.glob('local_search_well/'+food+'/'+starvation+'/'+'*.csv')
#print(fnames)
# data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']#,'Fx4','Fy4']
# data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']#,'Fx4','Fy4']

k=0
for food in foodlist:
    for starvation in starvationlist:
        print(food+starvation)
        total_FRAvisits=[]
        fnames = sorted(glob.glob(food+'/'+starvation+'/'+'*.csv'))
        FRAvisits=pd.DataFrame()#Loads the dataframe
        afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
        beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
        flynum=0
        
        for u in fnames: #goes thru files in the folder.
            # print(u)
            df=pd.read_csv(u, header=None)
            if(df.shape[1]==10):
                data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
                data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
            elif(df.shape[1]==8):
                data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
                data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
            df.columns=data_header#sets the column header
            df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
            for i in range(0,len(data_header2),2):
                flynum=flynum+1
                empty=pd.DataFrame()
                # empty=df[(df[data_header2[i]]>440) & (df[data_header2[i]] < 640) & (df[data_header2[i+1]] < 640) & (df[data_header2[i+1]] > 440)]
                empty=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 <= 70**2]
                #here we find when the fly is within 440 and 640 pixels.
                #print(data_header2[i],i)
                #print(data_header2[i+1], 'summer')
                timestamp=empty['Time']
                timestamp=timestamp.astype(int)
                timestamp=timestamp.drop_duplicates()
                timestamp=timestamp.tolist()
                jumps=[]
                #print(len(timestamp), "what")
                #In this for loop, we apply the condition that the fly is in food zone for more than t seconds, we count it as one feeding bout.
                for m in range(0,len(timestamp)-1,1):
                    #print(m)
                    if(timestamp[m+1]-timestamp[m]>t):
                        #print(timestamp[m+1])
                        jumps.append(timestamp[m+1])
                    else:
                        pass
                    #m=m+1
                k=k+1
                try:
                    if(timestamp[2]-timestamp[0]<t):
                        jumps.insert(0,timestamp[0])
                    else:
                        pass
                except:
                    pass
                try:
                    norm_revisits=len(jumps)/(1800-(jumps[0]))
                    total_FRAvisits.append(norm_revisits)
                    print(norm_revisits)
                    norm_revisits_dict={'Food': food, 'Starvation': starvation, "Flynum" :flynum, 'revisits': norm_revisits}
                    norm_revisits_df=norm_revisits_df.append(norm_revisits_dict,ignore_index=True)
                    # FRAvisits[k]=pd.Series(jumps)
                except:
                    pass

norm_revisits_df["state"]=norm_revisits_df["Food"]+"_"+norm_revisits_df["Starvation"]+"h"
idx = pd.DataFrame(norm_revisits_df.groupby('state').median().sort_values('revisits').index)

empty = []
for s in idx['state'].values:
    ugh=norm_revisits_df[norm_revisits_df['state']==s]
    empty.append(ugh)
sorted_df = pd.concat(empty)

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(x="revisits", y="state", 
                  data=sorted_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3)
ax = sns.stripplot(x="revisits", y="state", data=sorted_df, color=".25",size=3)
ax.set_xlabel('Probability')
ax.tick_params(axis='x', labelrotation = 0, size=10)
# ax.set_yticks("state")
ax.set_title('Probability of revisit to the food', fontsize=13)
ax.set_ylabel('state')
ax.xaxis.grid(True)
#sns.despine(trim=True, left=True)
fig.savefig('Probability of revisit to the food.png',format='png', dpi=600, bbox_inches = 'tight')