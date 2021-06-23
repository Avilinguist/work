#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import seaborn as sn
import numpy as np
import os
import json
import jieba

from itertools import combinations

import networkx as nx
import community

#os.chdir('0622bilibili/alcohol/')

titles = []
for i in os.listdir():
    if i[-3:]=='csv':
        df_temp = pd.read_csv(i,index_col = 0)
        titles.extend(df_temp['title'])

titles = [[word for word in jieba.cut(title) if len(word)>1] for title in titles]
titles = [title for title in titles if len(title)>0]

combinations_sum = []
for title in titles:
    for i in combinations(title,2):
        combinations_sum.append(i)

df_combinations = pd.DataFrame(pd.value_counts(combinations_sum))

#df_combinations[0].value_counts().to_csv('temp.csv')


df_combinations_filter = df_combinations.loc[df_combinations[0]>10]

#df_combinations_filter.to_csv('output.csv',encoding = 'utf_8_sig')

new_df = pd.DataFrame()

new_df['word1'] = [elem[0] for elem in df_combinations_filter.index]
new_df['word2'] = [elem[1] for elem in df_combinations_filter.index]


G = nx.from_pandas_edgelist(new_df,'word1','word2')


part = community.best_partition(G)

def cluster_present(cluster):
    for i in part.keys():
        if part[i]==cluster:
            print(i)

cluster_present(1)

