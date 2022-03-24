#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import os
import seaborn as sn
import time
import requests
from bs4 import BeautifulSoup


# In[2]:


os.chdir('D:/data/adidas-women/')


# In[3]:


sport_keywords = ['夜跑','晨跑','跑团 运动','接力赛','室内跑','越野跑步','City run','马拉松','铁人三项','短跑',
                   '健身','私教','动感单车','拳击','HIIT','美丽芭蕾','体重训练','功能性训练','举重','CrossFit','TRX','健美','尊巴','普拉提','瑜伽','空瑜',
                 '户外健身','徒步','登山','抱石','攀岩','滑翔伞','跳伞','露营','钓鱼']


# In[23]:


#输入关键词，获得下面所有视频的bvid
def find_all_videos(keyword):
    url = 'https://search.bilibili.com/all?keyword=%s&page=1'%(keyword+'女')
    soup = BeautifulSoup(requests.get(url).text,'html.parser')
    try:
        total_number = int(soup.find('li',class_='page-item last').text.strip('\n').strip())
    except:
        total_number=1
    bvids=[]
    for number in range(1,total_number+1):
        new_url =  'https://search.bilibili.com/all?keyword=%s&page=%d'%(keyword,number)
        soup = BeautifulSoup(requests.get(new_url).text,'html.parser')
        time.sleep(np.random.randint(1,3))
        for a in soup.find_all('a',class_='title'):
            if 'video' in a['href']:
                bvids.append(a['href'].split('/')[4].strip('?from=search'))
    return bvids


# In[24]:


keyword_dic = {}
for keyword in sport_keywords:
    keyword_dic.setdefault(keyword,[])    


# In[25]:


for keyword in keyword_dic:
    if len(keyword_dic[keyword])==0:
        keyword_dic[keyword] = find_all_videos(keyword)
        print('%s已经结束啦'%keyword)


# In[26]:


video_dic = {}
for keyword in keyword_dic.keys():
    video_dic.setdefault(keyword,{})
    for video in keyword_dic[keyword]:
        video_dic[keyword].setdefault(video,{})


# In[ ]:


#获得视频的基础信息和avid
for keyword in video_dic.keys():
    for video in video_dic[keyword].keys():
        try:
            html = 'http://api.bilibili.com/x/web-interface/view?bvid=%s'%video
            page = json.loads(requests.get(html).text)
            video_dic[keyword][video]['aid']=page['data']['aid']
            video_dic[keyword][video]['title']=page['data']['title']
            video_dic[keyword][video]['pubdate']=page['data']['pubdate']
            video_dic[keyword][video]['desc']=page['data']['desc']
            time.sleep(np.random.randint(1,3))
        except:
            continue


# In[33]:


#获取回复
def get_reply(aid):
    url = 'https://api.bilibili.com/x/v2/reply?&jsonp=jsonp&pn=1&type=1&oid=%s&sort=2'%aid
    content_dic = json.loads(requests.get(url).text)
    replies = []
    for person in content_dic['data']['replies']:
        replies.append(person['content']['message'])
    gender = [person['member']['sex'] for person in content_dic['data']['replies']]
    time.sleep(np.random.randint(1,3))
    return replies,gender


# In[34]:


a = 0
for channel in video_dic:
    for video in video_dic[channel]:
        if 'replies' not in video_dic[channel][video]:
            if 'aid' in video_dic[channel][video]:
                aid = video_dic[channel][video]['aid']
                try:
                    output = get_reply(aid)
                    video_dic[channel][video]['replies'] = output[0]
                    video_dic[channel][video]['gender'] = output[1]
                    a +=1
                    if a%1000==0:
                        print(a)
                except:
                    print(aid)


# In[38]:


with open('video_dic.json','w') as f:
    json.dump(video_dic,f)

