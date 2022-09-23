
"""
Created on Fri Apr 15 15:22:38 2022

@author: na488
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
from statsmodels.graphics.tsaplots import plot_acf
os.chdir('G:\\My Drive\\local_search\\local_search_well\\')
#os.chdir('V:\\local_search_new_new\\')
#os.cwd()
#Set Parameters here. 

foodlist=os.listdir()
foodlist.remove('desktop.ini')
starvationlist=["0","8","16","24"]
t=10
trajtime=120     #Insert time in seconds
raddistdf=pd.DataFrame()
raddistdf2=pd.DataFrame()
raddistdf3=pd.DataFrame()
raddistdf4=pd.DataFrame(index=foodlist, columns=starvationlist)
radial_distance_mean=pd.DataFrame()
total_FRAvisitsdf={}
alllist=[]

# food='1M'
# starvation='24'
# fnames = glob.glob('local_search_well/'+food+'/'+starvation+'/'+'*.csv')
#print(fnames)
# data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
# data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']

k=0
for food in foodlist:
    for starvation in starvationlist:
        print(food+starvation)
        total_FRAvisits=0
        fnames = glob.glob(food+'/'+starvation+'/'+'*.csv')
        FRAvisits=pd.DataFrame()#Loads the dataframe
        afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
        beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
        for u in fnames: #goes thru files in the folder.
            #print(u)
            df=pd.read_csv(u, header=None)
            if(df.shape[1]==10):
                data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
                data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3','Fx4','Fy4']
            elif(df.shape[1]==8):
                data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
                data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
            df.columns=data_header#sets the column header
            df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
            for i in range(0,len(data_header2),2):
                empty=pd.DataFrame()
                # empty=df[(df[data_header2[i]]>440) & (df[data_header2[i]] < 640) & (df[data_header2[i+1]] < 640) & (df[data_header2[i+1]] > 440)]
                empty=df[(df[data_header2[i]]-540)**2 + (df[data_header2[i+1]]-540)**2 <= 60**2]
                #here we find when the fly is within 440 and 640 pixels.
                #print(data_header2[i],i)
                #print(data_header2[i+1], 'summer')
                timestamp=empty['Time']
                timestamp=timestamp.astype(int)
                timestamp=timestamp.drop_duplicates()
                timestamp=timestamp.tolist()
                jumps=[]
                #print(len(timestamp), "what")
                #In this for loop, we apply the condition that the fly is in food zone for more than t seconds, we count it as one feeding bout.
                for m in range(0,len(timestamp)-1,1):
                    #print(m)
                    if(timestamp[m+1]-timestamp[m]>t):
                        #print(timestamp[m+1])
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
                    total_FRAvisits=total_FRAvisits+len(jumps)
                    FRAvisits[k]=pd.Series(jumps)
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
        
        flynum=list(afterfirstvisit['Flynum'])
        flynum = list(set(flynum))
        trajectoryafter=pd.DataFrame()
        trajectorybefore=pd.DataFrame()
        realtrajtime=trajtime*24 #Multiplying by the FPS rate
        radial_distance_all=pd.DataFrame()
        for f in flynum:
             #print(f)
             khali1=pd.DataFrame()
             khali2=pd.DataFrame()
             khali1=afterfirstvisit[afterfirstvisit['Flynum']==f]
             khali2=beforefirstvisit[beforefirstvisit['Flynum']==f]
             trajtrunc=khali1.head(realtrajtime)
             trajtrunc['Radial Distance']=np.sqrt(((trajtrunc['Fx']-540)**2)+((trajtrunc['Fy']-540)**2))
             radial_distance_all[f]=list(trajtrunc['Radial Distance'])    
        
        radial_distance_all['mean']=radial_distance_all.mean(axis=1)
        print(radial_distance_all['mean'])
        radial_distance_mean[food+starvation]=list(radial_distance_all['mean'])
        total_FRAvisitsdf[food+starvation]=total_FRAvisits

fig, axx = plt.subplots(nrows=2, ncols=2,
                        figsize=(25.5, 15),
                        gridspec_kw={'wspace': 0.25} # ensure proper width-wise spacing.
                       ,sharey=False)
ax11=axx.flat[0]
ax12=axx.flat[1]
# ax13=axx.flat[2]
ax21=axx.flat[2]
ax22=axx.flat[3]
# ax23=axx.flat[5]

ax11.set_title(radial_distance_mean.columns[0])
ax21.set_title(radial_distance_mean.columns[1])
ax12.set_title(radial_distance_mean.columns[2])
ax22.set_title(radial_distance_mean.columns[3])
# ax13.set_title("500mM_16hr Mean Radial Distance")
# ax23.set_title("500mM_24hr Mean Radial Distance")


ax11.plot(np.arange(0,realtrajtime,1), radial_distance_mean[radial_distance_mean.columns[0]], label=radial_distance_mean.columns[0])
ax21.plot(np.arange(0,realtrajtime,1), radial_distance_mean[radial_distance_mean.columns[1]], label=radial_distance_mean.columns[1])
ax12.plot(np.arange(0,realtrajtime,1), radial_distance_mean[radial_distance_mean.columns[2]], label=radial_distance_mean.columns[2])
ax22.plot(np.arange(0,realtrajtime,1), radial_distance_mean[radial_distance_mean.columns[3]], label=radial_distance_mean.columns[3])
# ax13.plot(np.arange(0,realtrajtime,1), radial_distance_mean['500mM16'], label='500mM16')
# ax23.plot(np.arange(0,realtrajtime,1), radial_distance_mean['500mM24'], label='500mM24')

fig.savefig("Radial Distance combined_screening.png",format='png', dpi=600, bbox_inches = 'tight')

fig4,ax10=plt.subplots()
ax10.set_title("{}_{}hr Radial Distance over time".format(food, starvation))
ax10.set_xlabel("time")
ax10.set_ylabel("Radial Distance from food")

for col in radial_distance_all.columns:
        ax10.plot(np.arange(0,realtrajtime,1), radial_distance_all[col], label=col)
#         # plt.legend(labels=col)
# # ax.plot(np.arange(0,realtrajtime,1), radial_distance_all['mean'], label='mean')
fig4.savefig("{}_{}hr Radial Distance_screening.png".format(food, starvation),format='png', dpi=600, bbox_inches = 'tight')

radial_distance_all2=radial_distance_mean.truncate(before=1000)
fig2, axx2 = plt.subplots(nrows=2, ncols=3,
                        figsize=(25.5, 15),
                        gridspec_kw={'wspace': 0.25} # ensure proper width-wise spacing.
                       ,sharey=False)
ax211=axx2.flat[0]
ax212=axx2.flat[1]
ax213=axx2.flat[2]
ax221=axx2.flat[3]
ax222=axx2.flat[4]
ax223=axx2.flat[5]


plot_acf(x=radial_distance_mean['1M16'], lags=1879, ax=ax211, title='1M16')
plot_acf(x=radial_distance_mean['1M24'], lags=1879, ax=ax221, title='1M24')
plot_acf(x=radial_distance_mean['yeast16'], lags=1879, ax=ax212, title='yeast16')
plot_acf(x=radial_distance_mean['yeast24'], lags=1879, ax=ax222, title='yeast24')
plot_acf(x=radial_distance_mean['500mM16'], lags=1879, ax=ax213, title='500mM16')
plot_acf(x=radial_distance_mean['500mM24'], lags=1879, ax=ax223, title='500mM24')

fig2.suptitle('Autocorrelation Analysis', fontsize=20, y=0.5)
fig2.savefig("Autocorrelation of Radial Distance combined_untrunc_screening.png",format='png', dpi=600, bbox_inches = 'tight')

fig3,ax9=plt.subplots()

names = list(total_FRAvisitsdf.keys())
values = list(total_FRAvisitsdf.values())

ax9.bar(range(len(total_FRAvisitsdf)), values, tick_label=names)
ax9.set_title("Number of revisits to the food")
fig3.savefig("Number of revisits to the food_screening.png",format='png', dpi=600, bbox_inches = 'tight')
# plot_acf(x=radial_distance_all2['mean'], lags=1879)