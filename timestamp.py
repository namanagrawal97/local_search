import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
os.chdir('G:\\My Drive\\local_search\\screening\\')
#os.chdir('V:\\local_search_new_new\\')
#Set Parameters here. 

food="ss34089"
starvation="24" 
t=1
trajtime=40    
fnames = sorted(glob.glob(food+'/'+starvation+'/'+'*.csv'))
print(fnames)
data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3', 'Fx4', 'Fy4']
data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3', 'Fx4', 'Fy4']
FRAvisits=pd.DataFrame()#Loads the dataframe
WALLvisits=pd.DataFrame()#Loads the dataframe

afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
k=0

for u in fnames:#goes thru files in the folder.
    print(u)
    df=pd.read_csv(u, header=None)
    df.columns=data_header#sets the column header
    df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
    df['Time']=df['Time']-120 
    for i in range(0,len(data_header2),2):
        empty=pd.DataFrame()
        empty2=pd.DataFrame()
        
        empty2=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 >= 510**2]
        empty=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 <= 70**2]
        
        print(data_header2[i],i)
        print(data_header2[i+1], 'summer')
        timestamp=empty['Time']
        timestamp=timestamp.astype(int)
        timestamp=timestamp.drop_duplicates()
        timestamp=timestamp.tolist()
        jumps=[]
        
        timestamp_wall=empty2['Time']
        timestamp_wall=timestamp_wall.astype(int)
        timestamp_wall=timestamp_wall.drop_duplicates()
        timestamp_wall=timestamp_wall.tolist()
        jumps_wall=[]
        
        # print(len(timestamp), "what")
        
        #In this for loop, we apply the condition that the fly is in food zone for more than t seconds, we count it as one feeding bout.
        for m in range(0,len(timestamp)-1,1):
            #print(m)
            if(timestamp[m+1]-timestamp[m]>10):
                # print(timestamp[m+1])
                jumps.append(timestamp[m+1])
            else:
                pass

        k=k+1
        try:
            if(timestamp[2]-timestamp[0]<10):
                jumps.insert(0,timestamp[0])
            else:
                pass
        except:
            pass
        #Here we are making the jumps_wall list for when the fly touches the wall
        for n in range(0,len(timestamp_wall)-1,1):
            #print(m)
            if(timestamp_wall[n+1]-timestamp_wall[n]>10):
                # print(timestamp[m+1])
                jumps_wall.append(timestamp_wall[n+1])
            else:
                pass

        k=k+1
        try:
            if(timestamp_wall[2]-timestamp_wall[0]<10):
                jumps_wall.insert(0,timestamp_wall[0])
            else:
                pass
        except:
            pass
        try:
            jumps=pd.Series(jumps)
            jumps_wall=pd.Series(jumps_wall)
            FRAvisits=pd.concat((FRAvisits,jumps.rename(k)), axis=1)
            WALLvisits=pd.concat((WALLvisits,jumps_wall.rename(k)), axis=1)
        
        except:
            pass
FRAvisits.to_csv("Timestamps_FRAvisits_{}_{}h.csv".format(food, starvation), sep=',')
