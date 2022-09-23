# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:17:33 2022

@author: Yapicilab
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
from sklearn.model_selection import train_test_split

#import plotly.express as px
# import rhinoscriptsyntax as rs

os.chdir('C:\\Users\\Yapicilab\\Dropbox\\screening')


# real_food=pd.read_csv('ss34089_food_timing.csv')

#foodlist=os.listdir()
#foodlist.remove('desktop.ini')
starvationlist=["0","8","16","24"]
food="ss15725"
starvation="24"
radial_distance_all=[]
food_index_all=[]
inst_vel_all=[]
heading_vector_all=[]
est_first_food=[]
flyindex=[]
final_df=pd.DataFrame()
thres=75
time_thres=0.5
fnames = sorted(glob.glob(food+'/'+starvation+'/'+'*.csv'))
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

def find_index(l,t):
    for j in l:
        if j>t:
            return l.index(j)
            break
        else:
            continue
def rad_dist_classifier(rad_dist_list,rad_dist_index,thres):
    rad_dist_list=list(rad_dist_list)
    for i in rad_dist_list:
        if i<thres:
            rad_dist_index[rad_dist_list.index(i)]=1
        else:
            continue
def find_split_points(some_list):
    splitlist=[]
    for i in range(0, len(some_list)-1,1):
        if some_list[i+1]-some_list[i]>1:
            splitlist.append(i)
        else:
            continue
    return splitlist
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
    df['Time']=df['Time']-120
    for i in range(2,len(data_header),2):
        # real_food_df=real_food[real_food['Index']==i/2]
        # food_bout_start=list(real_food_df['bout_start_total_time'])
        # food_bout_end=list(real_food_df['bout_end_total_time'])
        x_coord=list(df[data_header[i]])
        y_coord=list(df[data_header[i+1]])
        time=list(df['Time'])
        jumps=[]
        inst_vel=np.zeros_like(x_coord)
        rad_dist=np.zeros_like(x_coord)
        rad_dist_index=np.zeros_like(x_coord)
        food_index=np.zeros_like(x_coord)
        heading_vector=np.zeros_like(x_coord)
        #real_food_df['radial_distance']=np.sqrt((df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2)
        # for k,l in zip(food_bout_start,food_bout_end):
        #     food_index[find_index(time, k):find_index(time, l)+1]=1
        for j in range(0,len(x_coord)-1,1):
            rad_dist[j]=np.sqrt((x_coord[j]-540)**2+(y_coord[j]-540)**2)
            inst_vel[j]=np.sqrt((x_coord[j+1]-x_coord[j])**2+(y_coord[j+1]-y_coord[j])**2)/latency[j+1]
            heading_vector[j]=vec_angle(x_coord, y_coord, j)
        
        rad_dist_classifier(rad_dist, rad_dist_index, thres)
        FRAresidence=list(np.where(rad_dist_index==1)[0])
        splitlist=find_split_points(FRAresidence)
        split_list = [FRAresidence[q: w] for q, w in zip([0] + splitlist, splitlist + [None])]
        # print(i/2, splitlist[0]/24, time[FRAresidence[0]])
        for p in range(0, len(split_list),1):
            print(p)
            if ((len(split_list[p])/24) > time_thres):
                timestamp=split_list[p][0]
                est_first_food.append(time[timestamp])
                flyindex.append(i/2)
                break
            else:
                continue
            radial_distance_all.extend(rad_dist)
        food_index_all.extend(food_index)
        inst_vel_all.extend(inst_vel)
        heading_vector_all.extend(heading_vector)        
final_df['rad_dist']=radial_distance_all
final_df['inst_vel']=inst_vel_all
final_df['food_index']=food_index_all
final_df['heading_vector']=heading_vector_all
final_df=final_df.dropna()
x=final_df[['rad_dist','inst_vel','heading_vector']]
y=final_df['food_index']


"""
Applying Machine Learning Algorithms
"""

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)
# import the class
# x_train = np.array(x_train).reshape(-1, 1)
# x_test = np.array(x_test).reshape(-1, 1)

# from imblearn.under_sampling import RandomUnderSampler
# under_sampler = RandomUnderSampler(random_state=42)
# x_res, y_res = under_sampler.fit_resample(x_train, y_train)
# # x_test,y_test = under_sampler.fit_resample(x_test, y_test)
# from collections import Counter
# print(f"Training target statistics: {Counter(y_res)}")
# print(f"Testing target statistics: {Counter(y_test)}")

# """
# Logistic Regression
# """
# from sklearn.linear_model import LogisticRegression

# # instantiate the model (using the default parameters)
# logreg = LogisticRegression(random_state=16)

# # fit the model with data
# logreg.fit(x_res, y_res)

# y_pred = logreg.predict(x_test)

# # import the metrics class
# from sklearn import metrics

# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# print(cnf_matrix)

# from sklearn.metrics import classification_report
# target_names = ['no food', 'with food']
# print("Logreg", classification_report(y_test, y_pred, target_names=target_names))


# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')

# # # Text(0.5,257.44,'Predicted label');

# # fig, ax=plt.subplots()
# # y_pred_proba = logreg.predict_proba(x_test)[::,1]
# # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
# # auc = metrics.roc_auc_score(y_test, y_pred_proba)
# # plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# # plt.legend(loc=4)
# # plt.show()

# """
# Decision Tree Model

# """
# from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier()

# # Train Decision Tree Classifer
# clf = clf.fit(x_res,y_res)

# #Predict the response for test dataset
# y_pred = clf.predict(x_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("DTM",classification_report(y_test,y_pred))
# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# fig.show()
# """
# SVM

# """
# #Import svm model
# from sklearn import svm

# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel

# #Train the model using the training sets
# clf.fit(x_res, y_res)

# #Predict the response for test dataset
# y_pred = clf.predict(x_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("svm",classification_report(y_test,y_pred))
# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# print(cnf_matrix)
# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label')
# fig.show()
"""
Here we are trying to use our fitted model to predict food eating or not in new data

"""
# Finding mean Instantenous Velocity for Instantenous Velocity Threshold

# """
# k=0
# for i in time:
#     if(i>224):
#         break
#     else:
#         k=k+1
# l=0
# for i in time:
#     if (i>228):
#         break
#     else:
#         l=l+1

# mean_inst_vel=np.mean(inst_vel[k:l])
# mean_rad_dist=np.mean(rad_dist[k:l])

# """
# Now making an index of where we think the fly has eaten food
# """
# w=0

# for r,t in zip(rad_dist, inst_vel):
#     if r<mean_rad_dist and t<mean_inst_vel:
#         food_index[w]=1
#     else:
#         continue
#     print(w)
#     w=w+1


# """""""""
# Plotting Figures

# """""""""
            
# fig,ax=plt.subplots()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(rad_dist,inst_vel,heading_vector, s=3)
# ax.set_xlabel('radial dist')
# ax.set_ylabel('instantenous velocity')
# ax.set_zlabel('Head Angle')
# plt.show()

# fig,ax=plt.subplots()
# ax.scatter(inst_vel,heading_vector, s=3)
# ax.set_xlabel('instantenous velocity')
# ax.set_ylabel('Head Angle')
# plt.show()

# fig,ax=plt.subplots()
# ax.scatter(rad_dist,heading_vector, s=3)
# ax.set_xlabel('rad_dist')
# ax.set_ylabel('Head Angle')
# plt.show()

# """
# Plotting different figures
# """

# fig,(ax1,ax2,ax3)=plt.subplots(3,1, figsize=(20,15))
# ax1.plot(time,rad_dist)
# ax1.axvspan(224, 228, alpha=0.5, color='g')
# ax2.plot(time,heading_vector)
# # ax1.axvspan(224, 228, alpha=0.5, color='g')
# ax2.axvspan(224, 228, alpha=0.5, color='g')

# ax3.plot(time,inst_vel)
# ax3.axvspan(224, 228, alpha=0.5, color='g')

# plt.show()



# fig = px.scatter_3d(df, x=heading_vector, y=rad_dist, z=inst_vel)