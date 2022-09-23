# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:22:38 2022

@author: na488
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
os.chdir('G:\\My Drive\\local_search\\local_search_well\\')
#os.chdir('V:\\local_search_new_new\\')
#os.cwd()
#Set Parameters here. 

foodlist=os.listdir()
foodlist.remove('desktop.ini')
starvationlist=["0","8","16","24"]
t=5
trajtime=120     #Insert time in seconds
raddistdf=pd.DataFrame()
raddistdf2=pd.DataFrame()
raddistdf3=pd.DataFrame()
raddistdf4=pd.DataFrame(index=foodlist, columns=starvationlist)
alllist=[]
for food in foodlist:
    df4=[]
    for starvation in starvationlist:
        
        try:
            fnames = glob.glob(food+'/'+starvation+'/'+'*.csv')
            #print(fnames)
            # data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
            # data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
            FRAvisits=pd.DataFrame()#Loads the dataframe
            afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
            beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
            k=0
            
            for u in fnames: #goes thru files in the folder.
                #print(u)
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
                    empty=pd.DataFrame()
                    # empty=df[(df[data_header2[i]]>440) & (df[data_header2[i]] < 640) & (df[data_header2[i+1]] < 640) & (df[data_header2[i+1]] > 440)]
                    empty=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 <= 60**2]
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
                        FRAvisits[k]=pd.Series(jumps)
                        indexing=np.where(df['Time']>jumps[0])#Finds the first feeding bout
                        index=indexing[0][0]
                        aftertrunc=df.truncate(before=index)
                        beforetrunc=df.truncate(after=index)
                        firstvisit=pd.DataFrame()# generates a dataframe of fly positions AFTER the first jump
                        firstvisit['Fx']=aftertrunc[data_header2[i]]
                        firstvisit['Fy']=aftertrunc[data_header2[i+1]]
                        firstvisit['Flynum']=k
                        firstvisit['Time']=aftertrunc['Time']
                        
                        bffirstvisit=pd.DataFrame()# generates a dataframe of fly positions BEFORE the first jump
                        bffirstvisit['Fx']=beforetrunc[data_header2[i]]
                        bffirstvisit['Fy']=beforetrunc[data_header2[i+1]]
                        bffirstvisit['Flynum']=k
                        bffirstvisit['Time']=beforetrunc['Time']
                        
                        
                        afterfirstvisit=pd.concat([afterfirstvisit, firstvisit])
                        beforefirstvisit=pd.concat([beforefirstvisit, bffirstvisit])
                    except:
                        pass
            afterfirstvisit=afterfirstvisit[np.isfinite(afterfirstvisit['Fx'])]
            afterfirstvisit=afterfirstvisit[np.isfinite(afterfirstvisit['Fy'])]
            
            beforefirstvisit=beforefirstvisit[np.isfinite(beforefirstvisit['Fx'])]
            beforefirstvisit=beforefirstvisit[np.isfinite(beforefirstvisit['Fy'])]
            
            flynum=list(afterfirstvisit['Flynum'])
            flynum = list(set(flynum))
            trajectoryafter=pd.DataFrame()
            trajectorybefore=pd.DataFrame()
            realtrajtime=trajtime*24 #Multiplying by the FPS rate
            for f in flynum:
                 #print(f)
                 khali1=pd.DataFrame()
                 khali2=pd.DataFrame()
                 khali1=afterfirstvisit[afterfirstvisit['Flynum']==f]
                 khali2=beforefirstvisit[beforefirstvisit['Flynum']==f]
                 trajtrunc=khali1.head(realtrajtime)
                 trajtrunc['Radial Distance']=np.sqrt(((trajtrunc['Fx']-540)**2)+((trajtrunc['Fy']-540)**2))
                 meanraddistance=np.mean(trajtrunc['Radial Distance'])
                 df3={'Food': food, 'Starvation': starvation, "Flynum" :f, 'Radial Distance': meanraddistance}
                 raddistdf2=raddistdf2.append(df3, ignore_index=True)
                 trajtrunc2=khali2.tail(realtrajtime)
                 trajectoryafter=pd.concat([trajectoryafter,trajtrunc])
                 trajectorybefore=pd.concat([trajectorybefore,trajtrunc2])
                 #df4.append(meanraddistance)
                 #print(df4)
            #raddistdf3[food]=df4
            # print(food, starvation)
            # print(raddistdf3.head())
            trajectoryafter['Radial Distance']=np.sqrt(((trajectoryafter['Fx']-540)**2)+((trajectoryafter['Fy']-540)**2))     
            meanraddist=np.mean(trajectoryafter['Radial Distance'])
            df2 = {'Food': food, 'Starvation': starvation, 'Radial Distance': meanraddist}
            print(df2)
            df4.append(meanraddist)
            raddistdf=raddistdf.append(df2, ignore_index=True)
            #raddistdf4=pd.DataFrame(foodlist.apply(x:series_col))
        except:
            pass
    #df4.append(meanraddist)
    alllist.append(df4)    
raddistdf["state"]=raddistdf["Food"]+"_"+raddistdf["Starvation"]+"h"
grouped=raddistdf.sort_values(by='Radial Distance').index
raddistdf2["state"]=raddistdf2["Food"]+"_"+raddistdf2["Starvation"]+"h"
idx = pd.DataFrame(raddistdf2.groupby('state').median().sort_values('Radial Distance').index)

empty = []
for s in idx['state'].values:
    ugh=raddistdf2[raddistdf2['state']==s]
    empty.append(ugh)
    #print(ugh)
    #print(ugh['Radial Distance'])
    #print(s)
    raddistdf3=pd.concat([raddistdf3,ugh])
sorted_df = pd.concat(empty)

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(x="Radial Distance", y="state", 
                 data=sorted_df, palette="Set3", showfliers = False, showmeans=True, linewidth=1)
ax = sns.stripplot(x="Radial Distance", y="state", data=sorted_df, color=".25", size=3)
ax.set_xlabel('Radial Distance (pixels)')
ax.tick_params(axis='x', labelrotation = 0, size=10)
ax.set_title('Median Radial Distance after first encounter with food', fontsize=13)
ax.set_ylabel('state')
ax.xaxis.grid(True)
#sns.despine(trim=True, left=True)
fig.savefig('Radial Distance median yeast_screening.png',format='png', dpi=600, bbox_inches = 'tight')


