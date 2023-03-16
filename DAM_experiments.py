# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:58:33 2023

@author: Yapicilab
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 11:59:17 2022

@author: Yapicilab
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
import scipy.stats as stats
import dabest
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import mannwhitneyu
from matplotlib.patches import Rectangle

os.chdir('E:\\Dropbox\\DAM_experiments')

df1=pd.read_csv('Monitor1-w1118xUAS-TNT.csv')
df2=pd.read_csv('Monitor2-w1118.csv')
df3=pd.read_csv('Monitor3-UAS-TNT.csv')

def find_index_date(l,t): #This function helps us find indexes
    for j in l:
        if j==t:
            return l.index(j)
            break
        else:
            continue
"w1118xUAS-TNT"
something=find_index_date(list(df1.iloc[:,1]), '8-Mar-23')
ZT_trimmed=df1.drop(df1.index[0:something])
ZT_trimmed=ZT_trimmed.drop(df1.index[27556:27559])

ZT_trimmed.shape
allah=ZT_trimmed.iloc[:,10:]
allah.shape[1]
allah.columns=np.arange(1,33,1)
allah=allah.reset_index(drop=True)

summed_all_flies=allah.sum(axis='columns')    

sums = []
for i in range(0, len(summed_all_flies), 30):
    sums.append(sum(summed_all_flies[i:i+30]))

"W1118"
something=find_index_date(list(df2.iloc[:,1]), '8-Mar-23')

ZT_trimmed_w1118=df2.drop(df2.index[0:something])
ZT_trimmed_w1118=ZT_trimmed_w1118.reset_index(drop=True)
ZT_trimmed_w1118=ZT_trimmed_w1118.drop(df2.index[0:4])

allah_w1118=ZT_trimmed_w1118.iloc[:,10:]
allah_w1118.shape[1]
allah_w1118.columns=np.arange(1,33,1)
allah_w1118=allah_w1118.reset_index(drop=True)
allah_w1118=allah_w1118.dropna(axis=0,how='any')
summed_all_flies_w1118=allah_w1118.sum(axis='columns')    

sums_w1118 = []
for i in range(0, len(summed_all_flies_w1118), 30):
    sums_w1118.append(sum(summed_all_flies_w1118[i:i+30]))

"UAS-TNT"
something=find_index_date(list(df3.iloc[:,1]), '8-Mar-23')

ZT_trimmed_uastnt=df3.drop(df3.index[0:something])
ZT_trimmed_uastnt=ZT_trimmed_uastnt.reset_index(drop=True)
ZT_trimmed_uastnt=ZT_trimmed_uastnt.drop(df3.index[0:4])

ZT_trimmed_uastnt.shape
allah_uastnt=ZT_trimmed_uastnt.iloc[:,10:]
allah_uastnt.shape[1]
allah_uastnt.columns=np.arange(1,33,1)
allah_uastnt=allah_uastnt.reset_index(drop=True)

summed_all_flies_uastnt=allah_uastnt.sum(axis='columns')    

sums_uastnt = []
for i in range(0, len(summed_all_flies_uastnt), 30):
    sums_uastnt.append(sum(summed_all_flies_uastnt[i:i+30]))

legend_lines = ['w1118xUAS-TNT', 'w1118', 'UAS-TNT']

fig,ax=plt.subplots()
ax.plot(np.arange(0,len(sums),1), sums, label=legend_lines[0])
ax.plot(np.arange(0,len(sums_w1118),1), sums_w1118,label=legend_lines[1])
ax.plot(np.arange(0,len(sums_uastnt),1), sums_uastnt,label=legend_lines[2])
locs=ax.get_xticks()
ax.legend(loc='upper right', prop={'size': 6})
ax.set_xticks(np.arange(3,len(sums)-1,24))
ax.set_xticklabels(np.arange(0,len(sums),24))
ax.grid(True)
# ax.add_patch(Rectangle((0, 0), 24, np.max(sums)+10,alpha=0.1,color='y'))
# ax.add_patch(Rectangle((24, 0), 24, np.max(sums)+10,alpha=0.1,color='k'))
# ax.add_patch(Rectangle((48, 0), 24, np.max(sums)+10,alpha=0.1,color='y'))
# ax.add_patch(Rectangle((72, 0), 24, np.max(sums)+10,alpha=0.1,color='k'))
# ax.add_patch(Rectangle((96, 0), 24, np.max(sums)+10,alpha=0.1,color='y'))
# ax.add_patch(Rectangle((120, 0), 24, np.max(sums)+10,alpha=0.1,color='k'))
# ON=ax.add_patch(Rectangle((144, 0), 24, np.max(sums)+10,alpha=0.1,color='y'))
# OFF=ax.add_patch(Rectangle((168, 0), 24, np.max(sums)+10,alpha=0.1,color='k'))