import pandas as pd
import numpy as np
import json
import pickle
import seaborn as sn

from gensim import corpora,models,similarities
import community #图分类的算法
import networkx as nx

#导入数据，直接导入为分好词的形式
with open('skincare_corpus.txt','r',encoding='ANSI') as f:
    skincare_corpus = [record.strip('\n').strip(',').split(',') for record in f.readlines()]

df_skincare = pd.read_csv('df_skincare.csv',index_col = 0,low_memory = False)

    
# NLP part
## 相似度
sc_dictionary = corpora.Dictionary(skincare_corpus) #序号化
bow_corpus = [sc_dictionary.doc2bow(text) for text in skincare_corpus]#bag of words
tfidf = models.TfidfModel(bow_corpus) #构建tf-idf模型
corpus_tfidf = tfidf[bow_corpus]#tf-idf模型应用到bow模型上
index = similarities.MatrixSimilarity(corpus_tfidf) #输出tf-idf的相似度矩阵
sims = [index[corpus] for corpus in corpus_tfidf] #similarity matrix变成numpy
##只取下半角的数
new_sims = []
for i in range(len(sims)):
    new_sims.append(sims[i][:i+1])

##一个存储动作
with open('sims.pkl','wb') as f:
    pickle.dump(new_sims,f)

with open('sims.pkl','rb') as f:
    sims = pickle.load(f)


##将 similarity matrix 转至成供网络使用的样子
df = pd.DataFrame(sims)
df_1 = df.stack().reset_index()
df_2 = df_1.loc[(df_1[0]>0)]#去除零相似度
df_3 = df_2.loc[df_2.level_0!=df_2.level_1]#去除自循环

##看囊括的元素数量占比，来判断哪个threshold是最合适的
percentage = []
for i in np.arange(0.2,0.3,0.01):#0.2~0.3是经验值,可调整
    df_temp = df_3.loc[df_3[0]>i]
    all_element = set(df_temp['level_0']).union(set(df_temp['level_1']))
    percentage.append(len(all_element)/32843)
sn.lineplot(np.arange(0.2,0.3,0.01),percentage)


df_4 = df_3.loc[df_3[0]>0.15]#0.15为elbow point，找到合适的threshold进行切割


#Graph part
G = nx.from_pandas_edgelist(df_4,'level_0','level_1')

##存储动作
with open('graph.pkl','wb') as f:
    pickle.dump(G,f)

##分类
part = community.best_partition(G)
###聚类内容储存
reverse_dic = {}
for i in part.values():
    reverse_dic.setdefault(i,[])
    for j in part.keys():
        if part[j]==i:
            reverse_dic[i].append(j)
#总计聚类数量
len(reverse_dic.keys())

total_word_list = [word for sentence in skincare_corpus for word in sentence]
df_total_word = pd.DataFrame(pd.value_counts(total_word_list))
df_total_word.columns = ['num']

ratios = []
for word in df_total_word.index:
    a = 0
    for sentence in skincare_corpus:
        if word in sentence:
            a+=1
    ratios.append(a/len(skincare_corpus))
df_total_word['ratios'] = ratios

def print_topic(number,dic,large_corpus): #打印每个聚类下的内容；之所以分成large_corpus和small_corpus，是可能会对已分好的聚类再次拆分
    corpus_list = dic[number]
    word_list = [] #这一个聚类里
    for i in corpus_list:
        word_list.extend(large_corpus[i])
    df_temp = pd.DataFrame(pd.value_counts(word_list))
    df_temp.columns=['num']        
    
    df_temp['inter_ratio'] = df_temp['num']/len(corpus_list)
    df_temp['outer_ratio'] = [df_total_word.loc[word,'ratios'] for word in df_temp.index]
    df_temp['tf_idf'] = df_temp.apply(lambda x: np.log(x['num']*x['inter_ratio']/x['outer_ratio']),axis=1)
    df_temp = df_temp.sort_values('tf_idf',ascending = False)
    return df_temp.head(5).index.to_list() #查看前五特殊的词汇
