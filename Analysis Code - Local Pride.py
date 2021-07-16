#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#基础包，基本干啥活都会导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #画图
import seaborn as sn#画图
import os
import json

#NLP分词包
import jieba

#排列组合
from itertools import combinations

#网络分析相关
import networkx as nx
import community


# ### 共现分析
# 
# 背后逻辑：如果两个词同时出现在了同一句话中，证明两个词有一定的关系；共同出现的频率越高，说明两者之间的关系越紧密，越有可能指向某一样事物。

# In[ ]:


#提取所有的分析文本进一个对象（一般为list）
titles = []
for i in os.listdir():
    if i[-3:]=='csv':
        df_temp = pd.read_csv(i,index_col = 0)
        titles.extend(df_temp['title'])#这里，我们将爬取下来的标题存在了csv格式的表格里。也可能是txt

titles = [[word for word in jieba.cut(title) if len(word)>1] for title in titles] #分词。可以引入停止词

titles = [title for title in titles if len(title)>0] #去除掉分词过后空掉的句子

combinations_sum = []
for title in titles:
    for i in combinations(title,2): #组合的写法。combinations(x,y)中，x是组合对象，y是组合的元素数
        combinations_sum.append(i)

df_combinations = pd.DataFrame(pd.value_counts(combinations_sum)) #利用pandas中value_counts函数实现共现组合的计数

df_combinations[0].value_counts() #看一下共现组合的频率，决定筛选哪一个数字以上的共现组合

df_combinations_filter = df_combinations.loc[df_combinations[0]>4] #筛选

#将共现的词语提炼到一个Dataframe
new_df = pd.DataFrame()
new_df['word1'] = [elem[0] for elem in df_combinations_filter.index]
new_df['word2'] = [elem[1] for elem in df_combinations_filter.index]

#网络分析
G = nx.from_pandas_edgelist(new_df,'word1','word2')
part = community.best_partition(G)
def cluster_present(cluster):
    for i in part.keys():
        if part[i]==cluster:
            print(i)


# ### 词性筛选
# 背后逻辑：相比较虚词、动词、形容词而言，名词有时候更重要，可以选择筛选名次等词性来完成相关分析。运用的是jieba中的posseg工具

# In[ ]:


#锁定目标词性
target_type = {'n','nz','an','vn','nw'} #名词，特殊名词，形名词，动名词，其他名词

new_corpus = []
for sentence in corpus: #针对原始语料；不用分词
    new_sentence = set()
    for word,i in pseg.cut(sentence):
        if i in target_type:#i对应的是词性
            if len(word)>1:#词语长度大于1
                if word not in stopwords: #停止词过滤
                    new_sentence.add(word)
    new_corpus.append(list(new_sentence))
#new_corpus是经过词性筛选后的语料


# ### Beer Brand 分析
# 没有太多花头，主要是针对json文件的处理

# In[ ]:


#读入啤酒热度文档
with open('啤酒品牌热度搜索-有记录-210622v3.txt','r') as f:
    bear_list = [record.strip('\n') for record in f.readlines()]

bear_list = [record for record in bear_list if len(record)>0]#去除空白行

#将json文件读到字典对象中
brand_dic = {}
for record in bear_list:
    temp_dic = json.loads(record)
    brand = temp_dic['data']['hot_list'][0]['keyword']
    brand_dic.setdefault(brand,{})
    for date in temp_dic['data']['hot_list'][0]['hot_list']:
        brand_dic[brand][date['datetime']]=int(date['index'])

#将字典变化为dataframe，成为原始数据提取处
df_brand = pd.DataFrame(brand_dic)

#建立方程，对每个品牌算数
def brand_calculation(brand):
    brand_num = df_brand[brand]
    brand_num = [record for record in brand_num if record!=0] #筛选出所有非0的记录。主要是获取收录，有记录的数据
    #从下面一步的四行，是为了去除排名前9的高峰值数据
    brand_num_reverse = np.sort(brand_num)[::-1]
    max_3 = brand_num_reverse[:10]
    for num in max_3:
        brand_num.remove(num)
    
    average_num = np.median(brand_num[-30:])
    
    trend = (np.average(brand_num[-10:])/np.average(brand_num[:11]))*100
    return average_num,trend

#对每个品牌算平均数和趋势
brand_info_dic = {}
for brand in df_brand.columns:
    brand_info_dic.setdefault(brand,{})
    num,trend = brand_calculation(brand)
    brand_info_dic[brand]['num']=num
    brand_info_dic[brand]['trend']=trend
    
df_brand_agg = pd.DataFrame(brand_info_dic).T #将算出来的数转换成dataframe

