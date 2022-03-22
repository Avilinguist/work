#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import sys


# In[2]:


os.chdir('D:/data/1230 - w数据库 - 线上生活志（20位）/线上生活志-20位-20211230')


# In[3]:


#获取所有人
target_persons = []
for person in os.listdir():
    if person[0]=='【':
        target_persons.append(person)


# In[4]:


#以序号第一位为例
g = os.walk(target_persons[0])


# In[5]:


all_pic_path = []
for path,dir_list,file_list in g:
    for file in file_list:
        if file[0] != '.':
            if 'jpg' in file:
                temp_path = os.path.join(path,file)
                all_pic_path.append(temp_path)
            elif 'png' in file:
                temp_path = os.path.join(path,file)
                all_pic_path.append(temp_path)
            elif 'unknown' in file:
                temp_path = os.path.join(path,file)
                all_pic_path.append(temp_path)


# In[6]:


#进行初步清洗处理，切成一个个元素
all_pic_path = [record.replace('\\','/') for record in all_pic_path]
consumer_index = int(all_pic_path[0].split('】')[0].strip('【'))
all_pic_element = [record.split('/')[1:] for record in all_pic_path]


# In[7]:


pic_path_change = []
rest_pic = []
for pic in all_pic_element:
    #走进我的生活
    if '我生活的室外环境' in pic:
        pic_path_change.append([pic[0],pic[1],pic[2],pic[4]])
    elif '展示我的家' in pic:
        pic_path_change.append([pic[0],pic[1],pic[2],pic[4]])
    #介绍我的工作/学习
    elif '公司学校附近' in pic:
        pic_path_change.append([pic[0],pic[2],pic[3],pic[4]])
    elif '建筑内部' in pic:
        pic_path_change.append([pic[0],pic[2],pic[3],pic[-1]])
    elif '上班上学路上' in pic:
        pic_path_change.append([pic[0],pic[2],pic[3],pic[4]])
    #工作日一天/休息日一天
    elif '一日用餐' in pic:
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '一日总结' in pic:
        pic_path_change.append([pic[0],pic[1],pic[2],pic[-1]])
    #我的业务休闲
    elif '我的社交活动' in pic:
        pic_path_change.append([pic[0],pic[1],pic[1],pic[2]])
    elif '我的收藏' in pic:
        pic_path_change.append([pic[0],pic[1],pic[1],pic[2]])
    elif '我追的明星' in pic:
        pic_path_change.append([pic[0],pic[1],pic[1],pic[2]])
    elif '我的兴趣' in pic:
        pic_path_change.append([pic[0],pic[1],pic[1],pic[-1]])
    #最近我看这些
    elif '0-6 最近我“看”这些' in pic:
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '在这些作品中最喜爱的人物角色' in pic:
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    #看看我的手机
    elif '手机APP总览' in pic:
        pic_path_change.append([pic[0],pic[1],pic[2],pic[-1]])
    elif '我的购物车' in pic: #购物车列表新增记录
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '我的社交APP' in pic: #社交APP新增记录
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '我的社区APP' in pic: #社区APP新增记录
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '我的相册' in pic: 
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '我的音乐APP' in pic: #音乐APP新增记录
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '其他我常使用的APP' in pic: #其他APP新增记录
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    #我的游戏人生
    elif '我常玩的游戏' in pic: #游戏新增记录，需要统一到二级标题上
        pic_path_change.append([pic[0],pic[1]+pic[3],pic[2],pic[-1]])
    #我的王者荣耀
    elif '游戏基本信息' in pic: 
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    elif '游戏内社交信息' in pic: 
        pic_path_change.append([pic[0],pic[1],pic[2],pic[-1]])
    elif '王者周边相关' in pic: 
        pic_path_change.append([pic[0],pic[1],pic[3],pic[-1]])
    else:
        rest_pic.append(pic)


# In[8]:


len(rest_pic)


# In[9]:


#做出df_pic的主体
df_pic = pd.DataFrame(pic_path_change)
df_pic['ID'] = consumer_index
df_pic.columns = ['一级维度','二级维度','标题','原始图片标题','ID']
df_pic=df_pic[['ID','一级维度','二级维度','标题','原始图片标题']]


# In[10]:


#把所有的json都提取出来
g = os.walk(target_persons[0])
all_json_path = []
for path,dir_list,file_list in g:
    for file in file_list:
        if file[0] !='.':
            if 'json' in file:
                temp_path = os.path.join(path,file)
                all_json_path.append(temp_path)


# In[11]:


all_json_path = [record.replace('\\','/') for record in all_json_path]


# In[12]:


all_json=[]
for path in all_json_path:
    try:
        with open(path,'r',encoding='utf-8') as f:
            temp = json.load(f)
        all_json.append(temp)
    except:
        print(path)


# In[13]:


#存储所有图片信息
all_dic = {}
for dic in all_json:
    all_dic.update(dic)


# In[14]:


#将有文字的图片信息上传上去
df_pic['intro'] = [str(all_dic[record]).replace("?","").replace("\\","").replace("<",'').replace("\n",'').replace('"','').replace(":","").replace("*","").replace(">","").replace("|","")  if record in all_dic.keys() else None for record in df_pic['原始图片标题'] ]


# In[15]:


#获得所有有文字的图片新名称
new_titles = []
for row in df_pic.index:
    if df_pic.loc[row,'intro'] != None:
        try:
            new_title = "【"+df_pic.loc[row,'标题']+"】"+df_pic.loc[row,'intro']+'.jpg'
        except:
            print(row)
    else:
        new_title = None
    new_titles.append(new_title)


# In[16]:


df_pic['new_titles'] = new_titles


# In[17]:


#开始搞没名字的图片名称
df_pic_none = df_pic.loc[df_pic['new_titles'].isna()]
df_pic_have = df_pic.loc[~df_pic['new_titles'].isna()]


# In[18]:


title_sets  = set(df_pic_none['标题'])


# In[19]:


df_pic_new = pd.DataFrame()
for title in title_sets:
    df_temp = df_pic_none.loc[df_pic_none['标题']==title]
    df_temp['new_titles'] = [title+str(i+1)+'.jpg' for i in range(len(df_temp))]
    df_pic_new = pd.concat([df_pic_new,df_temp])


# In[20]:


#做出来了详情页。下一步需要做的是更改图片名称
df_pic_final = pd.concat([df_pic_new,df_pic_have])
df_pic_final = df_pic_final.sort_index()
df_pic_final['一级维度'] = [record.split(" ")[1] for record in df_pic_final['一级维度']]
df_pic_final.head()


# In[21]:


#定义改名函数
def change_name(temp):
    for path in all_pic_path:
        if temp in path:
            old_path = path
            new_path =list(df_pic_final.loc[df_pic_final['原始图片标题']==temp]['new_titles'])[0]
            new_path = old_path.strip(temp)+new_path
            os.rename(old_path,new_path)


# In[22]:


error_file = []
for file_name in df_pic_final['原始图片标题']:
    try:
        change_name(file_name)
    except:
        error_file.append(file_name)
print(error_file)


# In[23]:


error_file


# In[32]:


for file in error_file:
    print(file)
    print(len(list(df_pic_final[df_pic_final['原始图片标题']==file]['new_titles'])[0]))


# In[31]:


df_pic_final[df_pic_final['原始图片标题']=='tmp_a6dbb05796df5b3ccd4964f43e4810d7.png']['new_titles'][188]


# In[33]:


change_name('tmp_a6dbb05796df5b3ccd4964f43e4810d7.png')

