# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:51:06 2023

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
import easygui as eg

import tkinter as tk
from tkinter import simpledialog

def manual_title_labels():
    msg = "Enter your personal information"
    title = "Labelling of graph"
    fieldNames = ["Graph Title","X-axis Label","Y-axis Label"]
    fieldValues = []  # we start with blanks for the values
    return eg.multenterbox(msg,title, fieldNames)


def manual_title():
    ROOT = tk.Tk()
    
    window_width = 500
    window_height = 300

# get the screen dimension
    screen_width = ROOT.winfo_screenwidth()
    screen_height = ROOT.winfo_screenheight()

# find the center point
    center_x = int(screen_width/2 - window_width / 2)
    center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
    ROOT.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    ROOT.withdraw()
    
    
    # the input dialog
    graph_title = simpledialog.askstring(title="Graph Title",
                                      prompt="Graph Title")
    x_label=simpledialog.askstring(title="X axis label",
                                      prompt="X axis label:")
    y_label=simpledialog.askstring(title="Y axis label",
                                      prompt="Y axis label:")
    return graph_title,x_label,y_label


def box_plot(input_df):
    fig, ax= plt.subplots()
    sns.set_style("white")
    ax = sns.boxplot(data=input_df, palette="Set3", showfliers = False, showmeans=False, linewidth=1, fliersize=3, orient="h")
    ax = sns.stripplot(data=input_df, color=".25",size=2, orient="h")
    graph_title,x_label,y_label=manual_title_labels()
    ax.set_xlabel(x_label)
    ax.set_yticklabels(input_df.columns, fontsize=5)
    ax.tick_params(axis='x', labelrotation = 0, size=2)
    ax.set_title(graph_title, fontsize=11)
    ax.set_ylabel(y_label)
    # ax.yaxis.grid(True)
    ax.axvline(input_df["w1118"].mean())
    ax.xaxis.grid(True)
    plt.show()

def individual_plot(input_df):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.5)
    
    fig.suptitle(manual_title_labels[0], fontsize=18, y=0.95)

    for genotype, ax in zip(list(input_df.columns), axs.ravel()):
        newlist = [x for x in list(input_df[genotype]) if np.isnan(x) == False]

        ax.plot(np.arange(0,len(newlist),1), newlist)
        # ax.plot(np.arange(0,time_thres*24,1), [60]*time_thres*24) 
        ax.set_title("{}".format(genotype),fontsize=9)
        ax.set_ylim(0,650)
        # ax.set_xlim(0,len(input_df[genotype]))
        ax.set_xlabel("")
        ax.axvline(len(newlist)/2, color='red')
        ticks = ax.get_xticks()
        # ax.set_xticklabels(ticks)
def individual_plot_dict(input_dict):
    
    for key in input_dict.keys():
        input_df=input_dict[key]
        
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle(" {} Instantaneous Velocity distribution".format(key), fontsize=18, y=0.95)
    
        for genotype, ax in zip(list(input_df.columns), axs.ravel()):
            newlist = [x for x in list(input_df[genotype]) if np.isnan(x) == False]
    
            ax.plot(np.arange(0,len(newlist),1), newlist)
            # ax.plot(np.arange(0,time_thres*24,1), [60]*time_thres*24) 
            ax.set_title("{}".format(genotype),fontsize=9)
            ax.set_ylim(0,650)
            # ax.set_xlim(0,len(input_df[genotype]))
            ax.set_xlabel("")
            ax.axvline(len(newlist)/2, color='red')
            ticks = ax.get_xticks()
            # ax.set_xticklabels(ticks)
        # fig.savefig('results\\inst_vel_all_flies\\{}_{}_inst_vel_{}_Seconds.png'.format(key,starvation,time_thres),format='png', dpi=600, bbox_inches = 'tight')
        plt.close()