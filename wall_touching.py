import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
os.chdir('G:\\My Drive\\local_search\\local_search_well\\')
#os.chdir('V:\\local_search_new_new\\')
#Set Parameters here. 

# food="ss34089"
# starvation="24"

foodlist=os.listdir()
foodlist.remove('desktop.ini')
starvationlist=["0","8","16","24"]
 
t=5
trajtime=40    
k=0
distance_df=pd.DataFrame()

for food in foodlist:
    for starvation in starvationlist:
        fnames = sorted(glob.glob(food+'/'+starvation+'/'+'*.csv'))
        print(fnames)
        data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3', 'Fx4', 'Fy4']
        data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3', 'Fx4', 'Fy4']
        FRAvisits=pd.DataFrame()#Loads the dataframe
        WALLvisits=pd.DataFrame()#Loads the dataframe
        afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
        beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe        
        try:
            for u in fnames:#goes thru files in the folder.
                print(u)
                df=pd.read_csv(u, header=None)
                if(df.shape[1]==10):
                    data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
                    data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
                elif(df.shape[1]==8):
                    data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
                    data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
                df.columns=data_header#sets the column header
                df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
                # df['Time']=df['Time']-120 
                flynum=0
                for i in range(0,len(data_header2),2):
                    distance=0
                    flynum=flynum+1
                    empty=pd.DataFrame()
                    empty2=pd.DataFrame()
                    
                    # empty=df[(df[data_header2[i]]>604-area) & (df[data_header2[i]] < 604+area) & (df[data_header2[i+1]] < 540+area) & (df[data_header2[i+1]] > 540-area)]
                    empty2=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 >= 510**2]
                    empty=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 <= 60**2]
                    
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
                        if(timestamp[m+1]-timestamp[m]>t):
                            # print(timestamp[m+1])
                            jumps.append(timestamp[m+1])
                        else:
                            pass
            
                    k=k+1
                    try:
                        if(timestamp[2]-timestamp[0]<t):
                            jumps.insert(0,timestamp[0])
                        else:
                            pass
                    except:
                        pass
                    #Here we are making the jumps_wall list for when the fly touches the wall
                    for n in range(0,len(timestamp_wall)-1,1):
                        #print(m)
                        if(timestamp_wall[n+1]-timestamp_wall[n]>t):
                            # print(timestamp[m+1])
                            jumps_wall.append(timestamp_wall[n+1])
                        else:
                            pass
            
                    k=k+1
                    try:
                        if(timestamp_wall[2]-timestamp_wall[0]<t):
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
                        
                        for y in jumps_wall:
                            if y>=jumps[0]:
                                break
                            else:
                                pass
                        print(y)
                        indexing=np.where(df['Time']>jumps[0])#Finds the first feeding bout
                        index=indexing[0][0]
                        indexing_wall=np.where(df['Time']>y)#Finds the first feeding bout
                        index_wall=indexing_wall[0][0]
                        
                        aftertrunc=df.truncate(before=index)
                        final_trunc=aftertrunc.truncate(after=index_wall)
                        
                        Fx=list(final_trunc[data_header2[i]])
                        Fy=list(final_trunc[data_header2[i+1]])
                        
                        for i in range(0,len(Fx)-1,1):
                            distance= distance+np.sqrt((Fx[i+1]-Fx[i])**2+(Fy[i+1]-Fy[i])**2)
                            # print(distance)
                        distance_dict={'Food': food, 'Starvation': starvation, "Flynum" :flynum, 'Distance': distance}
                        distance_df=distance_df.append(distance_dict,ignore_index=True)
                    except:
                        pass
        except:
                pass            
distance_df["state"]=distance_df["Food"]+"_"+distance_df["Starvation"]+"h"
idx = pd.DataFrame(distance_df.groupby('state').median().sort_values('Distance').index)

empty = []
for s in idx['state'].values:
    ugh=distance_df[distance_df['state']==s]
    empty.append(ugh)
sorted_df = pd.concat(empty)

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(x="Distance", y="state", 
                  data=sorted_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1)
ax = sns.stripplot(x="Distance", y="state", data=sorted_df, color=".25", size=3)
ax.set_xlabel('Distance')
ax.tick_params(axis='both', labelrotation = 0, size=5, direction='in')
# ax.tick_params(axis='y', labelrotation = 0, size=5)

ax.set_title('Distance travelled after eating food before touching wall', fontsize=13)
ax.set_ylabel('state')
ax.xaxis.grid(True)
#sns.despine(trim=True, left=True)
fig.savefig('distance yeast_screening.png',format='png', dpi=600, bbox_inches = 'tight')