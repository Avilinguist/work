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


# In[2]:


os.chdir('0622bilibili/alcohol/')


# In[3]:


titles = []
for i in os.listdir():
    if i[-3:]=='csv':
        df_temp = pd.read_csv(i,index_col = 0)
        titles.extend(df_temp['title'])


# In[4]:


titles = [[word for word in jieba.cut(title) if len(word)>1] for title in titles]


# In[5]:


titles = [title for title in titles if len(title)>0]


# In[6]:


combinations_sum = []
for title in titles:
    for i in combinations(title,2):
        combinations_sum.append(i)


# In[11]:


df_combinations = pd.DataFrame(pd.value_counts(combinations_sum))


# In[16]:


df_combinations[0].value_counts().to_csv('temp.csv')


# In[97]:


df_combinations_filter = df_combinations.loc[df_combinations[0]>4]


# In[19]:


df_combinations_filter.to_csv('output.csv',encoding = 'utf_8_sig')


# In[98]:


df_combinations = df_combinations.sort_values(0,ascending = False)


# In[82]:


total_set = set()
word_num = []
total_freq = 0
bi_freq = []
for i in df_combinations.index:
    word = set(i)
    total_set = total_set.union(word)
    word_num.append(len(total_set))
for a in df_combinations[0]:
    total_freq += a
    bi_freq.append(total_freq)


# In[99]:


new_df = pd.DataFrame()


# In[100]:


new_df['word1'] = [elem[0] for elem in df_combinations_filter.index]
new_df['word2'] = [elem[1] for elem in df_combinations_filter.index]


# In[101]:


G = nx.from_pandas_edgelist(new_df,'word1','word2')


# In[102]:


nx.draw_spring(G,node_size = 20)


# In[103]:


part = community.best_partition(G)


# In[104]:


def cluster_present(cluster):
    for i in part.keys():
        if part[i]==cluster:
            print(i)


# In[109]:


cluster_present(1)

