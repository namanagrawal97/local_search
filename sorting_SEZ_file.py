# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:04:08 2022

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
#import plotly.express as px
os.chdir('C:\\Users\\sinha\\Downloads')

df=pd.read_csv('elife-71679-supp1-v2_Haein-edited.csv', header=0)
d={}
new_csv=pd.DataFrame()
ss_number_list=[]
group_id=[]
cell_type_list=[]
for i in range(1,7,1):
    d["dataframe{0}".format(i)] =df[df['Group']==i]
    empty=d["dataframe{0}".format(i)].sort_values(by='SS number')
    ss_number_list.extend(list(empty['SS number']))
    cell_type_list.extend(list(empty['Cell type']))
    group_id.extend(list(empty['Group']))

new_csv['Group id']=group_id
new_csv['SS number']=ss_number_list
new_csv['Cell type']=cell_type_list

dataframe_na= df[df['Group']==0]



# new_csv.to_csv("SEZ_sorted.csv")    
