# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:01:18 2022

@author: sinha
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
import sklearn
import numpy.linalg as LA
from matplotlib.animation import FuncAnimation, writers
plt.rcParams['animation.ffmpeg_path'] ='C:\\ffmpeg\\bin\\ffmpeg.exe'

#import plotly.express as px
# import rhinoscriptsyntax as rs
os.chdir('G:\\My Drive\\local_search\\local_search_well')


#foodlist=os.listdir()
#foodlist.remove('desktop.ini')
starvationlist=["0","8","16","24"]
food="yeast"
starvation="24"

fnames = glob.glob(food+'/'+starvation+'/'+'*.csv')

def vec_angle(x,y,j):
    p1=[x[j],y[j]]
    p2=[x[j+1],y[j+1]]
    vec1=np.subtract(p2,p1)
    sumthing=np.dot(vec1,p1)
    norms = LA.norm(vec1) * LA.norm(p1)
    cos = sumthing / norms
    rad=np.arccos(cos)
    #rad = np.arccos(np.clip(cos, -1.0, 1.0))
    
    return 180-np.rad2deg(rad)

for u in fnames: #goes thru files in the folder.
    # print(u)
    df=pd.read_csv(u, header=None)
    df=df.dropna(axis=1, how='all')
    if(df.shape[1]==10):
        data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
        data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
    elif(df.shape[1]==8):
        data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
        data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
    elif(df.shape[1]==6):
         data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2']
         data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2']
    elif(df.shape[1]==4):
         data_header = ['Time', 'Latency', 'Fx1', 'Fy1']
         data_header2 = ['Fx1', 'Fy1']
    df.columns=data_header
    latency=list(df['Latency'])
    latency[0]=0
    for i in range(0,len(data_header2),2):
        empty=pd.DataFrame()
        x_coord=list(df[data_header2[i]])
        y_coord=list(df[data_header2[i+1]])
        time=list(df['Time'])
        inst_vel=np.zeros_like(x_coord)
        rad_dist=np.zeros_like(x_coord)
        heading_vector=np.zeros_like(x_coord)
        #empty['radial_distance']=np.sqrt((df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2)
        for j in range(0,len(x_coord)-1,1):
            rad_dist[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2)
            inst_vel[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2)/latency[j+1]
            heading_vector[j]=vec_angle(x_coord, y_coord, j)

x_coord=x_coord[1:4800]
y_coord=y_coord[1:4800]
rad_dist=list(rad_dist[1:4800])
inst_vel=list(inst_vel[1:4800])
heading_vector=list(heading_vector[1:4800])
time=time[1:4800]

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,15))
ax1.set_xlim(0,1080)
ax1.set_ylim(0,1080)
ax1.add_patch(plt.Circle((540,540), 60))
ax1.add_patch(plt.Circle((540,540), 30,color='r'))
ax1.add_patch(plt.Circle((540,540), 540,color='grey', alpha=0.1))
ax1.set_title("fly_trajectory")

ax2.set_title("radial distance")
ax2.set_xlim(time[0],time[-1])
ax2.set_ylim(0,540)
ax2.axhline(60, ls='--')
# repeat_length=1
ax3.set_title("heading vector")
ax3.set_xlim(time[0],time[-1])
ax3.set_ylim(0,180)

ax4.set_title("inst_vel")
ax4.set_xlim(time[0],2*time[-1])
ax4.set_ylim(0,np.max(inst_vel))

particle, = ax1.plot([],[], marker='o', color='r', markersize=8, alpha=1)
traj, = ax1.plot([],[], color='b', alpha=1)

rad_dist_head, = ax2.plot([],[], marker='o', color='r', markersize=2, alpha=1)
rad_dist_traj, = ax2.plot([],[], color='b', alpha=1)

heading_angle_head, = ax3.plot([],[], marker='o', color='r', markersize=2, alpha=1)
heading_angle_traj, = ax3.plot([],[], color='b', alpha=1)

inst_vel_head, = ax4.plot([],[], marker='o', color='r', markersize=2, alpha=1)
inst_vel_traj, = ax4.plot([],[], color='b', alpha=1)


metadata=dict(title='AAAA', artist='Naman')

def flytraj(j):
    traj.set_xdata(x_coord[:j+1])
    traj.set_ydata(y_coord[:j+1])
    particle.set_xdata(x_coord[j])
    particle.set_ydata(y_coord[j])
    
    rad_dist_traj.set_ydata(rad_dist[:j+1])
    rad_dist_traj.set_xdata(time[:j+1])
    rad_dist_head.set_ydata(rad_dist[j])
    rad_dist_head.set_xdata(time[j])
        
    heading_angle_traj.set_xdata(time[:j+1])
    heading_angle_traj.set_ydata(heading_vector[:j+1])
    heading_angle_head.set_xdata(time[j])
    heading_angle_head.set_ydata(heading_vector[j])
    
    inst_vel_traj.set_xdata(time[:j+1])
    inst_vel_traj.set_ydata(inst_vel[:j+1])
    inst_vel_head.set_xdata(time[:j+1])
    inst_vel_head.set_ydata(inst_vel[:j+1])
    return traj,particle,rad_dist_traj,rad_dist_head,heading_angle_traj,heading_angle_head,inst_vel_traj,inst_vel_head
animation2 =FuncAnimation(fig, flytraj, frames=np.arange(0,4800,1),interval=1, blit=True)

plt.show()

Writer = writers['ffmpeg']
writer = Writer(fps=24, metadata={'artist': 'Me'}, bitrate=1800)

animation2.save('Traj_RD_HA_IV_{}_{}.mp4'.format(food, starvation), writer)
