# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:59:34 2022

@author: na488
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
import math
os.chdir('G:\\My Drive\\local_search\\')

#Set Parameters here. 

food="1M"
starvation="24" 
t=3    
fnames = glob.glob('local_search/'+food+'/'+starvation+'/'+'*.csv')
print(fnames)


data_header = ['Time', 'Latency', 'Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3', 'Fx4', 'Fy4']
data_header2 = ['Fx1', 'Fy1', 'Fx2', 'Fy2', 'Fx3', 'Fy3', 'Fx4', 'Fy4']
FRAvisits=pd.DataFrame()#Loads the dataframe
afterfirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
beforefirstvisit=pd.DataFrame(columns = ['Fx', 'Fy','Flynum','Time'])#generate empty dataframe
k=0

for u in fnames:#goes thru files in the folder.
    print(u)
    df=pd.read_csv(u, header=None)
    df.columns=data_header#sets the column header
    df['Latency'][0] = 0#sets the first value of latency as zero because it is generally very high
    for i in range(0,len(data_header2),2):
        empty=pd.DataFrame()
        empty=df[(df[data_header2[i]]>440) & (df[data_header2[i]] < 640) & (df[data_header2[i+1]] < 640) & (df[data_header2[i+1]] > 440)]
        #here we find when the fly is within 440 and 640 pixels.
        print(data_header2[i],i)
        print(data_header2[i+1], 'summer')
        timestamp=empty['Time']
        #firstvisit=empty.iloc[:,[i+2,i+3]]
        #firstvisit=firstvisit.dropna()
        #firstvisit=firstvisit.astype(int)
        #firstvisit.columns=['Fx','Fy']
        #firstvisit['Flynum']=k
        #afterfirstvisit=pd.concat([afterfirstvisit, firstvisit])
        #timestamp.to_frame()
        timestamp=timestamp.astype(int)
        timestamp=timestamp.drop_duplicates()
        timestamp=timestamp.tolist()
        jumps=[]
        print(len(timestamp), "what")
        #In this for loop, we apply the condition that the fly is in food zone for more than t seconds, we count it as one feeding bout.
        for m in range(0,len(timestamp)-1,1):
            #print(m)
            if(timestamp[m+1]-timestamp[m]>t):
                print(timestamp[m+1])
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


fig6,(ax1,ax2)= plt.subplots(1,2, figsize=(15,7))# ensure proper width-wise spacing.
                       
h=ax1.hist2d(x=afterfirstvisit['Fx'], y=afterfirstvisit['Fy'], bins=(100, 100), cmap=plt.cm.jet, density=True)
h2=ax2.hist2d(x=beforefirstvisit['Fx'], y=beforefirstvisit['Fy'], bins=(100, 100), cmap=plt.cm.jet, density=True)
ax1.set_title("after first eating bout", fontsize=10)
ax2.set_title("before first eating bout", fontsize=10)
#ax1.colorbar(h[3])
#ax2.colorbar(h2[3])
#fig6.colorbar(h2[3])
plt.show()
fig6.suptitle('residence_probability{}{}'.format(food, starvation))
fig6.savefig('residence_probability{}{}.png'.format(food, starvation),format='png', dpi=300)

flynum=list(afterfirstvisit['Flynum'])
keeper=1
# for f in flynum:
#     if(f==keeper):
                

fig7, (ax3,ax4) =plt.subplots(1,2, figsize=(15,7))
ax3=plt.scatter(data=afterfirstvisit, x='Fy', y='Fx', s=1, c='Flynum')
ax4=plt.scatter(data=beforefirstvisit, x='Fy', y='Fx', s=1, c='Flynum')
plt.show()



#fig,ax=plt.subplots()
# joindf = pd.DataFrame()
# for u in fnames:
#     print(u)
#     df=pd.read_csv(u, header=None)
#     df.columns=data_header
#     df['Latency'][0] = 0
#     joindf= pd.concat([joindf, df], axis=1)
# cols = []
# count = 1
# for column in joindf.columns:
#     if column == 'Time':
#         cols.append(f'Time_{count}')
#         count+=1
#         continue
#     elif column == 'Latency':
#         cols.append(f'Latency_{count}')
#         count+=1
#         continue
#     cols.append(column)
# joindf.columns = cols

# joindf=joindf.drop(['Time_3','Latency_4'], axis=1)

# pola=pd.DataFrame(columns = ['Fx', 'Fy'])                                                                                              

# fig,ax=plt.subplots()
# for p in range(0,len(FRAvisits.columns),1):
#     kutta=joindf[joindf['Time_1']>FRAvisits.iat[0,p]]
#     kutta=kutta.iloc[:,[2*p+2,2*p+3]]
#     kutta=kutta.dropna()
#     kutta=kutta.astype(int)
#     kutta.columns=['Fx','Fy']
#     print(kutta.head(10))
#     pola=pd.concat([pola, kutta])
#     #ax=plt.hist2d(x=kutta.iloc[:,0], y=kutta.iloc[:,1], bins=(100, 100), cmap=plt.cm.jet)
# #plt.hist2d(x=kutta['Fx4'], y=kutta['Fy4'], bins=(30, 30), cmap=plt.cm.jet)
# #name="{}h,{}after".format(starvation, food)
# #ax=ax.set_title('{starvation}h,{food}after', fontsize=13)
# plt.hist2d(x=pola['Fx'], y=pola['Fy'], bins=(100, 100), cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()
# fig.savefig('residenceprollity{}{}after.png'.format(food, starvation), dpi=300, bbox_inches='tight')

# pola2=pd.DataFrame(columns = ['Fx', 'Fy'])                                                                                              

# fig2,ax2=plt.subplots()
# for p in range(0,len(FRAvisits.columns),1):
#     kutta2=joindf[joindf['Time_1']<FRAvisits.iat[0,p]]
#     kutta2=kutta2.iloc[:,[2*p+2,2*p+3]]
#     kutta2=kutta2.dropna()
#     kutta2=kutta2.astype(int)
#     kutta2.columns=['Fx','Fy']
#     #print(kutta.head(10))
#     pola2=pd.concat([pola2, kutta2])
#     ax2=plt.hist2d(x=kutta2.iloc[:,0], y=kutta2.iloc[:,1], bins=(100, 100), cmap=plt.cm.jet)
# plt.hist2d(x=pola2['Fx'], y=pola2['Fy'], bins=(100, 100), cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()
# fig2.savefig('residenceprollity{}{}before.png'.format(food, starvation), dpi=300, bbox_inches='tight')
