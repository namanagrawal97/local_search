# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:50:38 2022

@author: Yapicilab
"""

import scipy.stats as stats
stats.mannwhitneyu(x=sorted_df[sorted_df['Food']=='w1118']['Distance'], y=sorted_df[sorted_df['Food']=='ss41397']['Distance'], alternative = 'greater')
