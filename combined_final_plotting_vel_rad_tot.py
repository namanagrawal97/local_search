# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:19:59 2023

@author: Yapicilab
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import scipy.stats as stats
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib import cm


diameter_of_arena=50.8 #in mm
num_of_pixels=1080 #in both x and y axis

conversion=diameter_of_arena/num_of_pixels

positives_low_sig_list=[
"ss31362"
,"ss31369"
,"ss42603"
,"ss43327"
,"ss43335"
,"ss45693"
,"ss45926"
]
positives_low_std_list=["ss32361","ss39993","ss40950","ss42639","ss44866","ss45929","ss47080","ss49430","ss50701"]
positives_high_list=["ss47305","ss31345","ss45898"]
positives_all=positives_high_list+positives_low_sig_list+positives_low_std_list+["w1118"]

legend_lines = ['Positives (Significant) that stay near food','Positives (1 std) that stay near food', 'Other Lines', 'Positives that go away from food','Mean']
"""
Plotting Difference in Avg Instantaneous velocity after and before eating the food for all genotypes
"""


os.chdir('C:\\Users\\Yapicilab\\Dropbox\\Foraging screen') #SET THIS TO YOUR FOLDER WHERE YOU HAVE KEPT THE DATA FILES
inst_vel_diff_binned_df=pd.read_excel('inst_vel_diff_binned_all_lines.xlsx')
sorted_index_inst_vel_diff_binned = inst_vel_diff_binned_df.mean().sort_values().index
inst_vel_diff_binned_df=inst_vel_diff_binned_df[sorted_index_inst_vel_diff_binned]
inst_vel_diff_binned_df=inst_vel_diff_binned_df.mul(conversion) #converting pix/s to mm/s

my_pal ={}

for i in inst_vel_diff_binned_df.columns:
    if i in positives_low_sig_list:
        my_pal[i]="steelblue"
    elif i in positives_low_std_list:
        my_pal[i]="violet"
    elif i in positives_high_list:
        my_pal[i]="crimson"
    else:
        my_pal[i]="lightgrey"

custom_lines = [Line2D([0], [0], color='steelblue', lw=2),
                Line2D([0], [0], color='violet', lw=2),
                Line2D([0], [0], color='lightgrey', lw=2),
                Line2D([0], [0], color='crimson', lw=2),
                Line2D([0], [0], marker='^', color='w', label='Mean',
                         markerfacecolor='g', markersize=8)]


fig, ax= plt.subplots(figsize=(16, 9))
sns.set_style("white")
ax = sns.boxplot(data=inst_vel_diff_binned_df, palette=(my_pal),showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h", meanprops={"markersize":"2"})
# ax = sns.stripplot(data=inst_vel_diff_binned_df, color=".25",size=2, orient="h")
ax.legend(custom_lines, legend_lines, loc='upper right', prop={'size': 6})
ax.set_xlabel('Instantaneous Velocity Difference (mm/s)')
ax.set_yticklabels(inst_vel_diff_binned_df.columns, fontsize=2)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Difference in Average Instantaneous Velocity After and Before eating the food', fontsize=11)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(False)
ax.axvline(inst_vel_diff_binned_df["w1118"].mean())
ax.xaxis.grid(False)
plt.show()
fig.savefig('inst_vel_diff_all.svg',format='svg', dpi=600, bbox_inches = 'tight')


"""
Plotting Difference in Avg Instantaneous velocity after and before eating the food for positives only

"""

inst_vel_diff_binned_df_positives=inst_vel_diff_binned_df[positives_all]
sorted_index_inst_vel_diff_binned_pos = inst_vel_diff_binned_df_positives.mean().sort_values().index
inst_vel_diff_binned_df_positives=inst_vel_diff_binned_df_positives[sorted_index_inst_vel_diff_binned_pos]

fig, ax= plt.subplots(figsize=(16, 9))
sns.set_style("white")
ax = sns.boxplot(data=inst_vel_diff_binned_df_positives, palette=my_pal, showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=inst_vel_diff_binned_df_positives, color=".25",size=2, orient="h")
ax.legend(custom_lines, legend_lines, loc='upper right', prop={'size': 6})
ax.set_xlabel('Instantaneous Velocity Difference (mm/s)')
ax.set_yticklabels(inst_vel_diff_binned_df_positives.columns, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Difference in Average Instantaneous Velocity After and Before eating the food', fontsize=11)
ax.set_ylabel('Positive Genotypes')
# ax.yaxis.grid(False)
ax.axvline(inst_vel_diff_binned_df_positives["w1118"].mean())
ax.xaxis.grid(False)
plt.show()
fig.savefig('inst_vel_diff_positives.svg',format='svg', dpi=600, bbox_inches = 'tight')


"""
Plotting Avg Radial Distance after eating the food for all genotypes

"""

os.chdir('C:\\Users\\Yapicilab\\Dropbox\\Foraging screen') #SET THIS TO YOUR FOLDER WHERE YOU HAVE KEPT THE DATA FILES
rad_dist_df=pd.read_excel('radial_distance_all_lines.xlsx')
sorted_index_rad_dist = rad_dist_df.mean().sort_values().index
rad_dist_df=rad_dist_df[sorted_index_rad_dist]
rad_dist_df=rad_dist_df.mul(conversion) #converting pix to mm

fig, ax= plt.subplots(figsize=(16, 9))
sns.set_style("white")
ax = sns.boxplot(data=rad_dist_df, palette=my_pal,showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h", meanprops={"markersize":"2"})
# ax = sns.stripplot(data=rad_dist_df, color=".25",size=2, orient="h")
ax.legend(custom_lines, legend_lines, loc='upper right', prop={'size': 6})
ax.set_xlabel('Radial Distance (mm)')
ax.set_yticklabels(rad_dist_df.columns, fontsize=2)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Average Radial Distance in 60 seconds after eating food', fontsize=11)
ax.set_ylabel('Genotypes')
ax.yaxis.grid(False)
ax.axvline(rad_dist_df["w1118"].mean())
ax.xaxis.grid(False)
plt.show()
fig.savefig('radial_distance_all_lines.svg',format='svg', dpi=600, bbox_inches = 'tight')

"""
Plotting Avg Radial Distance after eating the food for positives only

"""

rad_dist_df_positives=rad_dist_df[positives_all]
sorted_index_rad_dist_df = rad_dist_df_positives.mean().sort_values().index
rad_dist_df_positives=rad_dist_df_positives[sorted_index_rad_dist_df]

fig, ax= plt.subplots(figsize=(16, 9))
sns.set_style("white")
ax = sns.boxplot(data=rad_dist_df_positives, palette=my_pal,showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=rad_dist_df_positives, color=".25",size=2, orient="h")
ax.legend(custom_lines, legend_lines, loc='upper right', prop={'size': 6})
ax.set_xlabel('Radial Distance (mm)')
ax.set_yticklabels(rad_dist_df_positives.columns, fontsize=2)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Average Radial Distance in 60 seconds after eating food', fontsize=11)
ax.set_ylabel('Genotypes')
ax.yaxis.grid(False)
ax.axvline(rad_dist_df_positives["w1118"].mean())
ax.xaxis.grid(False)
plt.show()
fig.savefig('radial_distance_positives.svg',format='svg', dpi=600, bbox_inches = 'tight')

"""
Plotting Total Distance after eating the food for all genotypes

"""

os.chdir('C:\\Users\\Yapicilab\\Dropbox\\Foraging screen') #SET THIS TO YOUR FOLDER WHERE YOU HAVE KEPT THE DATA FILES
tot_dist_df=pd.read_excel('total_distance_travelled_all_lines.xlsx')
sorted_index_tot_dist = tot_dist_df.mean().sort_values().index
tot_dist_df=tot_dist_df[sorted_index_tot_dist]
tot_dist_df=tot_dist_df.mul(conversion) #converting pix to mm

fig, ax= plt.subplots(figsize=(16, 9))
sns.set_style("white")
ax = sns.boxplot(data=tot_dist_df, palette=my_pal,showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h", meanprops={"markersize":"2"})
# ax = sns.stripplot(data=tot_dist_df, color=".25",size=2, orient="h")
ax.legend(custom_lines, legend_lines, loc='upper right', prop={'size': 6})
ax.set_xlabel('Total Distance (mm)')
ax.set_yticklabels(tot_dist_df.columns, fontsize=2)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Total Distance Travelled after eating food', fontsize=11)
ax.set_ylabel('Genotypes')
ax.yaxis.grid(False)
ax.axvline(tot_dist_df["w1118"].mean())
ax.xaxis.grid(False)
plt.show()
fig.savefig('total_distance_travelled_all_lines.svg',format='svg', dpi=600, bbox_inches = 'tight')

"""
Plotting Total Distance Travelled after eating the food for positives only

"""

tot_dist_df_positives=tot_dist_df[positives_all]
sorted_index_tot_dist_df = tot_dist_df_positives.mean().sort_values().index
tot_dist_df_positives=tot_dist_df_positives[sorted_index_tot_dist_df]

fig, ax= plt.subplots(figsize=(16, 9))
sns.set_style("white")
ax = sns.boxplot(data=tot_dist_df_positives, palette=my_pal,showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=tot_dist_df_positives, color=".25",size=2, orient="h")
ax.legend(custom_lines, legend_lines, loc='upper right', prop={'size': 6})
ax.set_xlabel('Total Distance (mm)')
ax.set_yticklabels(tot_dist_df_positives.columns, fontsize=2)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Average Total Distance after eating food', fontsize=11)
ax.set_ylabel('Genotypes')
ax.yaxis.grid(False)
ax.axvline(tot_dist_df_positives["w1118"].mean())
ax.xaxis.grid(False)
plt.show()
fig.savefig('total_distance_positives.svg',format='svg', dpi=600, bbox_inches = 'tight')
