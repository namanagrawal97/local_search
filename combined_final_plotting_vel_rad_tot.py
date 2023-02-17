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

positives_low_list=["ss31362","ss45693","ss31369","ss45926","ss43355","ss40950","ss44866","ss43327","ss39993","ss45929","ss42603","ss47080"]
positives_high_list=["ss47305","ss31345","ss45898"]



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
    if i in positives_low_list:
        my_pal[i]="steelblue"
    elif i in positives_high_list:
        my_pal[i]="crimson"
    else:
        my_pal[i]="lightgrey"

custom_lines = [Line2D([0], [0], color='steelblue', lw=4),
                Line2D([0], [0], color='lightgrey', lw=4),
                Line2D([0], [0], color='crimson', lw=4),
                Line2D([0], [0], marker='^', color='w', label='Mean',
                         markerfacecolor='g', markersize=10)]


fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=inst_vel_diff_binned_df, palette=(my_pal),showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h", meanprops={"markersize":"2"})
ax = sns.stripplot(data=inst_vel_diff_binned_df, color=".25",size=2, orient="h")
ax.legend(custom_lines, ['Positives that stay near food', 'Other Lines', 'Positives that go away from food','Mean'])
ax.set_xlabel('Instantaneous Velocity Difference (mm/s)')
ax.set_yticklabels(inst_vel_diff_binned_df.columns, fontsize=2)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Difference in Average Instantaneous Velocity After and Before eating the food', fontsize=11)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.axvline(inst_vel_diff_binned_df["w1118"].mean())
ax.xaxis.grid(True)
plt.show()
fig.savefig('inst_vel_diff_all.svg',format='svg', dpi=600, bbox_inches = 'tight')


"""
Plotting Difference in Avg Instantaneous velocity after and before eating the food for positives only

"""

inst_vel_diff_binned_df_positives=inst_vel_diff_binned_df[["ss31362","ss45693","ss31369","ss45926","ss43355","ss40950",
                                                          "ss44866","ss43327","ss39993","ss45929","ss42603","ss47080",
                                                         "ss47305","ss31345","ss45898","w1118"]]
sorted_index_inst_vel_diff_binned_pos = inst_vel_diff_binned_df_positives.mean().sort_values().index
inst_vel_diff_binned_df_positives=inst_vel_diff_binned_df_positives[sorted_index_inst_vel_diff_binned_pos]

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=inst_vel_diff_binned_df_positives, palette=my_pal, showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h")
ax = sns.stripplot(data=inst_vel_diff_binned_df_positives, color=".25",size=2, orient="h")
ax.legend(custom_lines, ['Positives that stay near food', 'Other Lines', 'Positives that go away from food','Mean'])
ax.set_xlabel('Instantaneous Velocity Difference (mm/s)')
ax.set_yticklabels(inst_vel_diff_binned_df_positives.columns, fontsize=5)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Difference in Average Instantaneous Velocity After and Before eating the food', fontsize=11)
ax.set_ylabel('Positive Genotypes')
# ax.yaxis.grid(True)
ax.axvline(inst_vel_diff_binned_df_positives["w1118"].mean())
ax.xaxis.grid(True)
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

fig, ax= plt.subplots()
sns.set_style("white")
ax = sns.boxplot(data=rad_dist_df, palette=(my_pal),showfliers = False, showmeans=True, linewidth=1, fliersize=3, orient="h", meanprops={"markersize":"2"})
ax = sns.stripplot(data=rad_dist_binned_df, color=".25",size=2, orient="h")
ax.legend(custom_lines, ['Positives that stay near food', 'Other Lines', 'Positives that go away from food','Mean'])
ax.set_xlabel('Instantaneous Velocity Difference (mm/s)')
ax.set_yticklabels(inst_vel_diff_binned_df.columns, fontsize=2)
ax.tick_params(axis='x', labelrotation = 0, size=2)
ax.set_title('Difference in Average Instantaneous Velocity After and Before eating the food', fontsize=11)
ax.set_ylabel('Genotypes')
# ax.yaxis.grid(True)
ax.axvline(inst_vel_diff_binned_df["w1118"].mean())
ax.xaxis.grid(True)
plt.show()
fig.savefig('inst_vel_diff_all.svg',format='svg', dpi=600, bbox_inches = 'tight')

