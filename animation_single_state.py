# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:38:49 2022
@author: Naman
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation, writers
from IPython.display import HTML
from celluloid import Camera

plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'

# Animate a single trajectory

# Set up the graph using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
os.chdir('G:\\My Drive\\local_search\\local_search_well')
#Set Parameters here. 

######################################################################################

food="100mM"
starvation="24"
t=10
trajtime=4800    
fnames = sorted(glob.glob(food+'/'+starvation+'/'+'*.csv'))
print(fnames)
areas=[30,40,50,60,70]

# for f in range(1,9,1):
try:
    pick_traj = 2      # Select a trajectory to simulate
    
    ######################################################################################
    
    data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']#,'Fx4','Fy4']
    data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']#,'Fx4','Fy4']
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
            # empty=df[(df[data_header2[i]]>554) & (df[data_header2[i]] < 654) & (df[data_header2[i+1]] < 590) & (df[data_header2[i+1]] > 490)]
            empty=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 <= 60**2]
            timestamp=empty['Time']
            timestamp=timestamp.astype(int)
            timestamp=timestamp.drop_duplicates()
            timestamp=timestamp.tolist()
            jumps=[]
            # print(len(timestamp), "what")
    
            #In this for loop, we apply the condition that the fly is in food zone for more than t seconds, we count it as one feeding bout.
            for m in range(0,len(timestamp)-1,1):
                if(timestamp[m+1]-timestamp[m]>t):
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
    
    afterfirstvisit=afterfirstvisit[np.isfinite(afterfirstvisit['Fx'])]
    afterfirstvisit=afterfirstvisit[np.isfinite(afterfirstvisit['Fy'])]
    
    beforefirstvisit=beforefirstvisit[np.isfinite(beforefirstvisit['Fx'])]
    beforefirstvisit=beforefirstvisit[np.isfinite(beforefirstvisit['Fy'])]
    
    
    
    fig, ax = plt.subplots(figsize=(15,15))
    ax.set_xlim(0,1080)
    ax.set_ylim(0,1080)
    ax.add_patch(plt.Circle((540,540), 60))
    ax.add_patch(plt.Circle((540,540), 30,color='r'))
    ax.add_patch(plt.Circle((540,540), 540,color='grey', alpha=0.1))
    
    ax.set_title("{}_{}hr_{}".format(food, starvation,pick_traj), fontsize=20)
    # Initiate camera
    
    # camera = Camera(fig)
    
    flyindtraj=afterfirstvisit[afterfirstvisit["Flynum"]==pick_traj]
    flyindtraj=flyindtraj.reset_index(drop=True)
    flyindtraj=flyindtraj.truncate(after=trajtime)
    x = list(flyindtraj['Fx'])
    xlist=[]
    
    y = list(flyindtraj['Fy'])
    ylist=[]
    
    particle, = plt.plot([],[], marker='o', color='r', markersize=8, alpha=1)
    traj, = plt.plot([],[], color='b', alpha=1)
    
    metadata=dict(title='AAAA', artist='Naman')
    
    def flytraj(j):
        traj.set_xdata(x[:j+1])
        traj.set_ydata(y[:j+1])
        particle.set_xdata(x[j])
        particle.set_ydata(y[j])
        return traj,particle
    animation2 =FuncAnimation(fig, flytraj, frames=np.arange(1,flyindtraj.shape[0]+1,1),interval=1, blit=True)
    plt.show()
    
    Writer = writers['ffmpeg']
    writer = Writer(fps=24, metadata={'artist': 'Me'}, bitrate=1800)
    
    animation2.save('IndTrajectory_{}_{}_{}.mp4'.format(food, starvation,pick_traj), writer)
except:
    pass
