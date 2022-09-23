import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
from statsmodels.graphics.tsaplots import plot_acf
os.chdir('G:\\My Drive\\local_search\\')
#os.chdir('V:\\local_search_new_new\\')
#os.cwd()
#Set Parameters here. 

foodlist=["1M","yeast","500mM"]
starvationlist=["16","24"] 
t=10
trajtime=120     #Insert time in seconds
raddistdf=pd.DataFrame()
raddistdf2=pd.DataFrame()
raddistdf3=pd.DataFrame()
raddistdf4=pd.DataFrame(index=foodlist, columns=starvationlist)
radial_distance_mean=pd.DataFrame()
total_FRAvisitsdf={}
alllist=[]

food='100mM'
starvation='16'
# fnames = glob.glob('local_search_well/'+food+'/'+starvation+'/'+'*.csv')
#print(fnames)
data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']
data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3']

k=0
# for food in foodlist:
#     for starvation in starvationlist:
print(food+starvation)
total_FRAvisits=0
fnames = glob.glob('local_search_well/'+food+'/'+starvation+'/'+'*.csv')
FRAvisits=pd.DataFrame()#Loads the dataframe
afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
for u in fnames: #goes thru files in the folder.
    #print(u)
    df=pd.read_csv(u, header=None)
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

# columnnames=['fly1','fly2','fly3','fly4','fly5','fly6','fly7','fly8','mean']
columnnames=[]
for fly in range(1,radial_distance_all.shape[1],1):
    flyname="fly{}".format(fly)
    columnnames.append(flyname)
columnnames.append("mean")
radial_distance_all.columns=columnnames
radial_distance_all_trunc=radial_distance_all.truncate(before=1000)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
plt.subplots_adjust(hspace=0.25)
fig.suptitle("Autocorrelation analysis truncated {} {}hr".format(food,starvation), fontsize=18, y=0.95)

# loop through tickers and axes
for ticker, ax in zip(columnnames, axs.ravel()):
    # filter df for ticker and plot on specified axes
    plot_acf(x=radial_distance_all_trunc[ticker], lags=1879, ax=ax)
    # df[df["ticker"] == ticker].plot(ax=ax)

    # chart formatting
    ax.set_title(ticker.upper())
    # ax.get_legend().remove()
    ax.set_xlabel("")

plt.show()
# fig.savefig('G:\\My Drive\\local_search\\autocorr_results\\{}_{} truncated Autocorrelation.png'.format(food, starvation), format='png', bbox_inches='tight',dpi=600)