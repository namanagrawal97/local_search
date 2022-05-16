# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:59:34 2022

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
fnames = glob.glob('local_search_final/'+food+'/'+starvation+'/'+'*.csv')
print(fnames)


data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
FRAvisits=pd.DataFrame()#Loads the dataframe
afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
k=0

for u in fnames:#goes thru files in the folder.
    print(u)
    df=pd.read_csv(u, header=None)
    df.columns=data_header#sets the column header
    df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
    df['Time']=df['Time']-60
    for i in range(0,len(data_header2),2):
        empty=pd.DataFrame()
        empty=df[(df[data_header2[i]]>554) & (df[data_header2[i]] < 654) & (df[data_header2[i+1]] < 590) & (df[data_header2[i+1]] > 490)]
        #here we find when the fly is within 440 and 640 pixels.
        print(data_header2[i],i)
        print(data_header2[i+1], 'summer')
        timestamp=empty['Time']
        #firstvisit=empty.iloc[:,[i+2,i+3]]
        #firstvisit=firstvisit.dropna()
        #firstvisit=firstvisit.astype(int)
        #firstvisit.columns=['Fx','Fy']
        #firstvisit['Flynum']=k
        #afterfirstvisit=pd.concat([afterfirstvisit, firstvisit])
        #timestamp.to_frame()
        timestamp=timestamp.astype(int)
        timestamp=timestamp.drop_duplicates()
        timestamp=timestamp.tolist()
        jumps=[]
        print(len(timestamp), "what")
        #In this for loop, we apply the condition that the fly is in food zone for more than t seconds, we count it as one feeding bout.
        for m in range(0,len(timestamp)-1,1):
            #print(m)
            if(timestamp[m+1]-timestamp[m]>t):
                print(timestamp[m+1])
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
            # pd.concat([FRAvisits,jumps], ignore_index=True, axis=1)
            # FRAvisits[k]=pd.Series(jumps)
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
afterfirstvisit=afterfirstvisit[np.isfinite(afterfirstvisit['Fx'])]
afterfirstvisit=afterfirstvisit[np.isfinite(afterfirstvisit['Fy'])]

beforefirstvisit=beforefirstvisit[np.isfinite(beforefirstvisit['Fx'])]
beforefirstvisit=beforefirstvisit[np.isfinite(beforefirstvisit['Fy'])]

fig6, axx = plt.subplots(nrows=2, ncols=2,
                        figsize=(17, 15),
                        gridspec_kw={'wspace': 0.25} # ensure proper width-wise spacing.
                       ,sharey=False)
ax1=axx.flat[0]
ax2=axx.flat[1]
ax3=axx.flat[2]
ax4=axx.flat[3]

                       
h=ax2.hist2d(x=afterfirstvisit['Fx'], y=afterfirstvisit['Fy'], bins=(50, 50), cmap=plt.cm.plasma, density=True)
h2=ax1.hist2d(x=beforefirstvisit['Fx'], y=beforefirstvisit['Fy'], bins=(50, 50), cmap=plt.cm.plasma, density=True)
ax2.set_title("cumulative residence after first eating bout", fontsize=13)
#plt.colorbar(h, ax=ax2)
#plt.colorbar(h2, ax=ax1)
#h2.colorbar()
ax1.set_title("cumulative residence before first eating bout", fontsize=13)
#plt.show()
#fig6.suptitle('residence_probability{}{}'.format(food, starvation))
#fig6.savefig('residence_probability{}{}.png'.format(food, starvation),format='png', dpi=300)

flynum=list(afterfirstvisit['Flynum'])
flynum = list(set(flynum))
trajectoryafter=pd.DataFrame()
trajectorybefore=pd.DataFrame()
realtrajtime=trajtime*30
for f in flynum:
     print(f)
     khali1=pd.DataFrame()
     khali2=pd.DataFrame()
     khali1=afterfirstvisit[afterfirstvisit['Flynum']==f]
     khali2=beforefirstvisit[beforefirstvisit['Flynum']==f]
     trajtrunc=khali1.head(realtrajtime)
     trajtrunc2=khali2.tail(realtrajtime)
     trajectoryafter=pd.concat([trajectoryafter,trajtrunc])
     trajectorybefore=pd.concat([trajectorybefore,trajtrunc2])
                
sns.set_style("white")
#fig7,(ax3,ax4) = plt.subplots(nrows=1,ncols=2,sharey=False, figsize=(15,7))
scatter=ax4.scatter(data=trajectoryafter, x='Fx', y='Fy', s=1, c='Flynum', cmap='hsv')
legend1 = ax4.legend(*scatter.legend_elements(), loc="upper right", title="flies")
ax4.add_artist(legend1)
ax4.set_xlim(0,1080)
ax4.set_ylim(0,1080)
ax4.axvline(x=540.0, color="grey")
ax4.axhline(y=604.0, color="grey")
ax4.set_title('trajectory {} seconds after first encounter with food'.format(trajtime), fontsize=13)

scatter2=ax3.scatter(data=trajectorybefore, x='Fx', y='Fy', s=1, c='Flynum', cmap='hsv')
legend2 = ax3.legend(*scatter2.legend_elements(), loc="upper right", title="flies")
ax3.add_artist(legend2)
ax3.set_xlim(0,1080)
ax3.set_ylim(0,1080)
ax3.axvline(x=540.0, color="grey")
ax3.axhline(y=604.0, color="grey")
ax3.set_title('trajectory {} seconds before first encounter with food'.format(trajtime), fontsize=13)

plt.show()
fig6.suptitle('{} {}h'.format(food, starvation), fontsize=20, y=0.5)
fig6.savefig('Trajectory{}{}.png'.format(food, starvation),format='png', dpi=300, bbox_inches = 'tight')


cmap_list=['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
fig,ax=plt.subplots()
empty=pd.DataFrame()
for i in flynum:
    empty=trajectoryafter[trajectoryafter['Flynum']==i]
    ax.scatter(data=empty, x='Fy', y='Fx', s=1, c='Time', cmap=cmap_list[i], label=i)
#ax.legend(*cmap_list.legend_elements(), loc="upper right", title="flies")
plt.legend()
ax = plt.gca()
legend = ax.get_legend()
# legend.legendHandles[0].set_color(cmap_list[1](0.8))
# legend.legendHandles[1].set_color(cmap_list[2](0.8))
# legend.legendHandles[2].set_color(cmap_list[3](0.8))
fig.savefig('Trajectory2{}{}.png'.format(food, starvation),format='png', dpi=300, bbox_inches = 'tight')
plt.show()
    