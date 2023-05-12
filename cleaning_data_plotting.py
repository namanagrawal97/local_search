# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:20:59 2023

@author: Yapicilab
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from conversion_func import *
from plot_func import *
# Load the CSV file into a pandas DataFrame
df = pd.read_csv("https://drive.google.com/uc?id=1M5neKinuQt2P-boDFYdDwLP2BoTbIA1m")
df.columns = df.columns.str.strip()


"""
CLEANING THE DATA . REMOVING OUTLIERS
"""
clean_data = []
compiled_mean_df=pd.DataFrame(columns=['genotype','starvation','rad_dist_mean','rad_dist_var','tot_flies'])

for genotype_starvation, group in df.groupby(['genotype', 'starvation']):
    print(genotype_starvation)
    # Calculate the first and third quartiles of the radial distance
    q1, q3 = np.percentile(group['rad_dist'], [25, 75])
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    # Define the upper and lower bounds as 1.5 times the IQR away from the quartiles
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Remove any outliers from the radial distance for the current genotype and starvation combination
    cleaned_group = group[(group['rad_dist'] >= lower_bound) & (group['rad_dist'] <= upper_bound)]
    # Append the cleaned data to the list of cleaned data
    clean_data.append(cleaned_group)
    compiled_mean_df=compiled_mean_df.append({'genotype':genotype_starvation[0],'starvation':genotype_starvation[1],'rad_dist_mean':np.nanmean(cleaned_group['rad_dist']),'rad_dist_var':np.nanvar(cleaned_group['rad_dist']),'tot_flies':len(cleaned_group['fly_num'])},ignore_index=True)
# Concatenate the cleaned data into a single DataFrame
clean_df = pd.concat(clean_data)


""""
PLOTTING HEATMAP
"""

heatmap_data = clean_df.pivot_table(index=['genotype'], columns=['starvation'], values='rad_dist')
heatmap_data_var = compiled_mean_df.pivot_table(index=['genotype'], columns=['starvation'], values='rad_dist_var') #MAKING DF FOR VARIANCE
heatmap_data_tot_flies = compiled_mean_df.pivot_table(index=['genotype'], columns=['starvation'], values='tot_flies') #making df for total flies

heatmap_data_rad_mean = compiled_mean_df.pivot_table(index=['genotype'], columns=['starvation'], values='rad_dist_mean') #calculating mean rad dist manually as a sanity check that pd.pivottable works accurately

heatmap_data_var=heatmap_data_var.mul(conversion**2)
heatmap_data_rad_mean=heatmap_data_rad_mean.mul(conversion)
heatmap_data=heatmap_data.mul(conversion)
# Reorder the columns to match the desired order
heatmap_data = heatmap_data.loc[["10mM", "100mM", "500mM", "1M"]]
heatmap_data_var=heatmap_data_var.loc[["10mM", "100mM", "500mM", "1M"]]
heatmap_data_tot_flies=heatmap_data_tot_flies.loc[["10mM", "100mM", "500mM", "1M"]]
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu')

def add_var(x):
    return "var=" + str(round(x, 2))

annot_heatmap_data_var = heatmap_data_var.round(2).applymap(add_var)

def add_n(x):
    return "n=" + str(int(x))

annot_heatmap_data_tot_flies = heatmap_data_tot_flies.applymap(add_n)

fig,ax=plt.subplots()
ax=sns.heatmap(heatmap_data,annot=False, cmap='YlOrRd')
ax=sns.heatmap(heatmap_data,annot=annot_heatmap_data_var, annot_kws={'va':'bottom'},fmt="", cbar=False, cmap='YlOrRd')
ax=sns.heatmap(heatmap_data,annot=annot_heatmap_data_tot_flies, annot_kws={'va':'top'},fmt="", cbar=False, cmap='YlOrRd')


# Add axis labels and title
ax.set_xlabel('Food Provided')
ax.set_ylabel('Starvation Level')
ax.set_title('Average Radial Distance (in mm) 60 seconds after eating food')
# fig.savefig('results\\sugar_heatmap_rad_dist.png',format='png', dpi=600, bbox_inches = 'tight')
plt.show()

"""
PLOTTING BOXPLOT
"""
# Create a list of the unique genotypes in the DataFrame
genotypes = clean_df['genotype'].unique()
starvationlist = clean_df['starvation'].unique()
rad_dist_dict={}
# Loop through each genotype and create a box plot
for genotype in genotypes:
    for starvation in starvationlist:
    # Subset the DataFrame to only include the current genotype
        subset = df[df['genotype'] == genotype]
        subset = subset[subset['starvation'] == starvation]
        rad_dist=[]
        rad_dist=list(subset['rad_dist'])
        rad_dist_dict[genotype+str(starvation)]=rad_dist    
   
rad_dist_df=dict_to_df(rad_dist_dict)
rad_dist_df=sort_df_mean(rad_dist_df)
box_plot(rad_dist_df)