#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import selenium
import json
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime


# In[37]:


url_list = []
for i in range(1,51):
    url = 'https://api.bilibili.com/x/web-interface/search/type?context=&page='+str(i)+'&order=&keyword=成都&duration=&tids_1=&tids_2=&from_source=webtop_search&from_spmid=333.851&__refresh__=true&_extra=&search_type=video&highlight=1&single_column=0'
    url_list.append(url)


# In[46]:


a_dic = json.loads(r.text)


# In[64]:


cd_demo_list = []
i = 0
for url in url_list:
    r = requests.get(url)
    temp_dict = json.loads(r.text)
    cd_demo_list.append(temp_dict)
    time.sleep(np.random.randint(1,3))
    i+=1
    if i%10==0:
        print(i)


# In[81]:


cd_dict = {}
for dic in cd_demo_list:
    for video in dic['data']['result']:
        cd_dict.setdefault(video['id'],{})
        cd_dict[video['id']]['title'] = video['title'].replace('<em class="keyword">','').replace('</em>','')
        cd_dict[video['id']]['play'] = video['play']
        cd_dict[video['id']]['favorites'] = video['favorites']
        cd_dict[video['id']]['time'] = datetime.fromtimestamp(video['pubdate'])


# In[83]:


df_cd = pd.DataFrame(cd_dict).T


# In[113]:


#最终封装好的函数
def Bilibili(city):
    url_list = []
    for i in range(1,51):
        url = 'https://api.bilibili.com/x/web-interface/search/type?context=&page='+str(i)+'&order=&keyword='+city+'&duration=&tids_1=&tids_2=&from_source=webtop_search&__refresh__=true&_extra=&search_type=video&highlight=1&single_column=0'
        url_list.append(url)
    temp_list = []
    for url in url_list:
        r = requests.get(url)
        temp_dict = json.loads(r.text)
        temp_list.append(temp_dict)
        time.sleep(np.random.randint(2,6))
    final_dict = {}
    for dic in temp_list:
        for video in dic['data']['result']:
            final_dict.setdefault(video['id'],{})
            final_dict[video['id']]['title'] = video['title'].replace('<em class="keyword">','').replace('</em>','')
            final_dict[video['id']]['play'] = video['play']
            final_dict[video['id']]['favorites'] = video['favorites']
            final_dict[video['id']]['time'] = datetime.fromtimestamp(video['pubdate'])
    df_temp = pd.DataFrame(final_dict).T
    return df_temp


# In[110]:


bj_list = Bilibili('北京')


# In[111]:


final_dict = {}
for dic in bj_list:
    for video in dic['data']['result']:
        final_dict.setdefault(video['id'],{})
        final_dict[video['id']]['title'] = video['title'].replace('<em class="keyword">','').replace('</em>','')
        final_dict[video['id']]['play'] = video['play']
        final_dict[video['id']]['favorites'] = video['favorites']
        final_dict[video['id']]['time'] = datetime.fromtimestamp(video['pubdate'])
df_bj = pd.DataFrame(final_dict).T


# In[112]:


df_bj


# In[116]:


df_qz = Bilibili('泉州')


# In[124]:


df_qz.head()


# In[118]:


df_xa = Bilibili('西安')


# In[ ]:


'''
成都
北京
泉州
西安
上海
哈尔滨
'''


# In[119]:


df_sh = Bilibili('上海')


# In[120]:


df_hrb = Bilibili('哈尔滨')


# In[123]:


df_bj.to_csv('df_bj.csv',encoding = 'utf_8_sig')
df_sh.to_csv('df_sh.csv',encoding = 'utf_8_sig')
df_xa.to_csv('df_xa.csv',encoding = 'utf_8_sig')
df_qz.to_csv('df_qz.csv',encoding = 'utf_8_sig')
df_cd.to_csv('df_cd.csv',encoding = 'utf_8_sig')
df_hrb.to_csv('df_hrb.csv',encoding = 'utf_8_sig')

