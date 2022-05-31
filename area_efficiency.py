# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:59:54 2022

@author: na488
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
os.chdir('G:\\My Drive\\local_search\\')
#os.chdir('V:\\local_search_new_new\\')
#Set Parameters here. 

food="1M"
starvation="16" 
t=3
trajtime=40    
fnames = sorted(glob.glob('local_search_final/'+food+'/'+starvation+'/'+'*.csv'))
print(fnames)
areas=[x for x in range(1,40,1)]
# approximation=pd.DataFrame(index=areas, columns=["hits", "misses"])
aloha=pd.DataFrame()
for area in areas:
    holder=[]
    data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
    data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
    FRAvisits=pd.DataFrame()#Loads the dataframe
    afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
    beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
    k=0
    
    for u in fnames:#goes thru files in the folder.
        # print(u)
        df=pd.read_csv(u, header=None)
        df.columns=data_header#sets the column header
        df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
        df['Time']=df['Time']-60
        for i in range(0,len(data_header2),2):
            empty=pd.DataFrame()
            # empty=df[(df[data_header2[i]]>604-area) & (df[data_header2[i]] < 604+area) & (df[data_header2[i+1]] < 540+area) & (df[data_header2[i+1]] > 540-area)]
            empty=df[(df[data_header2[i]]-604)**2 + (df[data_header2[i+1]]-540)**2 <= 60**2]
            
            # print(data_header2[i],i)
            # print(data_header2[i+1], 'summer')
            timestamp=empty['Time']
            timestamp=timestamp.astype(int)
            timestamp=timestamp.drop_duplicates()
            timestamp=timestamp.tolist()
            jumps=[]
            # print(len(timestamp), "what")
            #In this for loop, we apply the condition that the fly is in food zone for more than t seconds, we count it as one feeding bout.
            for m in range(0,len(timestamp)-1,1):
                #print(m)
                if(timestamp[m+1]-timestamp[m]>area):
                    # print(timestamp[m+1])
                    jumps.append(timestamp[m+1])
                else:
                    pass
    
            k=k+1
            try:
                if(timestamp[2]-timestamp[0]<area):
                    jumps.insert(0,timestamp[0])
                else:
                    pass
            except:
                pass
            try:
                jumps=pd.Series(jumps)
                FRAvisits=pd.concat((FRAvisits,jumps.rename(k)), axis=1)
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
    actual_time_df=pd.DataFrame()
    actual_time=pd.read_csv('local_search_final/food_finding_time.csv')
    for i in range(1,7,1):
        #khali3=pd.DataFrame()
        khali3=actual_time[actual_time['Index']==i]
        # print(i)
        times=[]
        times=list(khali3['bout_start_total_time'])
        times=pd.Series(times)
        # print(times)
        col_num = str(i)
        # actual_time_df[col_num] = times
        actual_time_df=pd.concat([actual_time_df,times.rename(i)], axis=1)
        
    
    
    hit=0
    miss=0
    for j in range(1, FRAvisits.shape[1]+1,1):
        calc_time=FRAvisits[j].dropna()
        real_time=actual_time_df[j].dropna()
        for i in range(len(FRAvisits[j].dropna())):
            monet=int(calc_time[i])
            lst1=list(range(monet-3,monet+4))
            lst2=list(real_time)
            mozart=list(set(lst1).intersection(lst2))
            if len(mozart)==0:
                miss=miss+1
            else:
                hit=hit+1
    print(area)
    print("hits", hit)
    print("misses", miss)
    # approximation[area,1]=hit
    # approximation[area,2]=miss
    holder={'areas': area, 'hits': hit, 'misses': miss}
    holder=pd.Series(holder)
    aloha=pd.concat([aloha,holder], axis=1)
    # aloha=aloha.transpose()
aloha=aloha.transpose()
aloha['Hits/Misses']=aloha['hits']/aloha['misses']
aloha['Total']=aloha['hits']+aloha['misses']
aloha['HitsRate']=aloha['hits']/aloha['Total']
aloha['MissRate']=aloha['misses']/aloha['Total']
aloha['Hits/Misses_norm'] = (aloha['Hits/Misses'] - aloha['Hits/Misses'].min()) / (aloha['Hits/Misses'].max() - aloha['Hits/Misses'].min()) 
# fig, ax= plt.subplots()
# ax.bar(x=aloha['areas'], height=aloha['Hits/Misses'], label='Hits/Misses')
# ax.bar(x=aloha['areas'], height=aloha['hits'], label='hits')
# ax.bar(x=aloha['areas'], height=aloha['misses'], label='misses')
# ax.set_title('Hits/Misses for circular food eating zone as compared to manual annotation', fontsize=13)
plot=aloha.plot(x='areas', y=['HitsRate','MissRate','Hits/Misses_norm'], kind="bar")
fig = plot.get_figure()
# fig.savefig("output.png")
# ax.set_xlabel('Radius of food residence area')
# ax.set_ylabel('Hits/Misses')
# ax.legend()
plt.show()
# fig.savefig('Hits_misses_circular.png',dpi=600,format='png',bbox_inches='tight')