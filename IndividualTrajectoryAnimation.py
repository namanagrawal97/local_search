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

plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe'

# Animate a single trajectory

# Set up the graph using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
os.chdir('C:\\Users\\Yapicilab\\Dropbox\\Claire\\8maze\\initial experiments')
#Set Parameters here. 

######################################################################################

food="100mM"
starvation="24"
t=10
time_thres=40    
fnames = ["C:\\Users\\Yapicilab\\Documents\\w1118xUASTNT_25_5yeast_10_312022-10-31T16_15_01.csv"]
print(fnames)


    ######################################################################################
    
for u in fnames:#goes thru files in the folder.
    # print(u)
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
    df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
    df['Time']=df['Time']-round(df['Time'][0])
    for i in range(2,len(data_header),2):
        x_coord=list(df[data_header[i]])
        y_coord=list(df[data_header[i+1]])
        # del x_coord[0:split_point_index]
        # del y_coord[0:split_point_index]
        
        rad_dist=list(np.zeros_like(x_coord))
        
        for j in range(0,len(x_coord)-1,1):
            rad_dist[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2)
            
        if len(rad_dist)<time_thres*24:
            pass
        else:
            del rad_dist[time_thres*24:-1]
            del rad_dist[0:(time_thres-1)*24-1]
            rad_dist.pop()
            # radial_distances.append(np.mean(rad_dist))


fig, ax = plt.subplots(figsize=(15,15))
ax.set_xlim(0,1444)
ax.set_ylim(0,1080)
# ax.add_patch(plt.Circle((540,540), 60))
# ax.add_patch(plt.Circle((540,540), 30,color='r'))

# Initiate camera

# camera = Camera(fig)

# flyindtraj=afterfirstvisit[afterfirstvisit["Flynum"]==pick_traj]
# flyindtraj=flyindtraj.reset_index(drop=True)
# flyindtraj=flyindtraj.truncate(after=2000)

x = x_coord
xlist=[]

y = y_coord
ylist=[]

particle, = plt.plot([],[], marker='o', color='r', markersize=3)
traj, = plt.plot([],[], color='b', alpha=0.8)

metadata=dict(title='AAAA', artist='Naman')

def flytraj(j):
    traj.set_xdata(x[:j+1])
    traj.set_ydata(y[:j+1])
    particle.set_xdata(x[j])
    particle.set_ydata(y[j])
    return traj,particle
animation2 =FuncAnimation(fig, flytraj, frames=np.arange(1,len(x_coord)+1,1),interval=1, blit=True)
plt.show()

Writer = writers['ffmpeg']
writer = Writer(fps=15, metadata={'artist': 'Me'}, bitrate=1800)

animation2.save('Line Graph Animation_{}_{}.mp4'.format(food, starvation), writer)