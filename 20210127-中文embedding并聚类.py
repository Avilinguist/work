# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:12:24 2021

@author: Mingcong Li
"""

# Importing the relevant modules
from transformers import BertTokenizer, BertModel #need install
import pandas as pd
import numpy as np
import torch
import os
import umap
import hdbscan
import matplotlib.pyplot as plt
import networkx as nx
import community #图分类的算法
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from gensim import corpora, models


os.chdir(r'D:\桌面的文件夹\实习\睿丛\BERT-for-Rhizome\誊录稿自动化')

# Loading the pre-trained BERT model
# Embeddings will be derived from the outputs of this model
# 在'https://huggingface.co/models'上的model可以自动下载
model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states = True,)

# Setting up the tokenizer
# This is the same tokenizer that was used in the model to generate embeddings to ensure consistency
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# the function to massage the input into the right form
def bert_text_preparation(text, tokenizer, max_len):
    """Preparing the input for BERT

    Takes a string argument and performs pre-processing like adding special tokens, tokenization, tokens to ids, and tokens to segment ids. All tokens are mapped to segment id = 1.

    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object to convert text into BERT-readable tokens and ids

    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids


    """
    text = text[:max_len-2]
    marked_text = "[CLS] " + str(text) + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


# another function to convert the input into embeddings
def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model

    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens] with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens] with segment ids for each token in text
        model (obj): Embedding model to generate embeddings from token and segment ids

    Returns:
        list: List of list of floats of size [n_tokens, n_embedding_dimensions] containing embeddings for each token

    """

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings


# 输入相似度矩阵和threshold，返回一个network G，和网络中包含的nodes数量
def make_graph(similarity_matrix, threshold):
    # 开始用图理论聚类
    graph=list()
    # 控制相似度
    logic_matr = (similarity_matrix>threshold) & (similarity_matrix<0.9999)
    for i in range(len(logic_matr)):
        temp =  logic_matr[logic_matr[i]].index.to_list()
        for item in temp:
            graph.append((i,item))

    # 制作graph
    G = nx.Graph()
    G.add_edges_from(graph)
    # nx.draw(G)
    # 统计有多少node纳入了图中
    num_nodes = len([item for cluster in nx.connected_components(G) for item in cluster])

    return G,num_nodes


# 输入分好词的句子，返回聚类结果
def tfidf_cluster(tokenized_sentences):
    # 计算TFIDF矩阵
    # （1）通过corpora.Dictionary方法建立字典对象
    dictionary = corpora.Dictionary(tokenized_sentences)
    # （2）使用dictionary.doc2bow方法构建词袋(语料库)
    corpus = [dictionary.doc2bow(stu) for stu in tokenized_sentences]  # 元组中第一个元素是词语在词典中对应的id，第二个元素是词语在文档中出现的次数
    # （3）使用models.TfidfModel方法对语料库建模
    tfIdf_model = models.TfidfModel(corpus)


    tfidf = list(tfIdf_model[corpus])

    IDs = set([word_ID[0] for sentence in tfidf for word_ID in sentence])


    # 制作TFIDF矩阵
    tfidf_matrix = pd.DataFrame(index=IDs)

    for i in range(len(tfidf)):
        vector = pd.DataFrame(tfidf[i],columns=['ID','TFIDF'+str(i+1)]).set_index('ID')
        tfidf_matrix = pd.concat([tfidf_matrix,vector],axis=1)

    # 用TFIDF制作余弦相似度矩阵

    tfidf_matrix.fillna(0,inplace=True)
    simi_matr = pd.DataFrame(cosine_similarity(tfidf_matrix.T))


    # 自动寻找最佳的threshold
    elbow=[]
    # 设置步长
    step_size = 0.01  # 经过试验的结果
    # 循环计算结果
    for threshold in np.arange(0.05,0.7,step_size):
        G,num_nodes = make_graph(simi_matr, threshold)
        elbow.append((threshold, num_nodes/len(simi_matr)))
    # 转成df
    elbow_df = pd.DataFrame(elbow,columns=['shreshold','percentage']).set_index('shreshold')

    # 以0.05为间隔抽样
    sampling_interval = 0.05
    elbow_cal = elbow_df.iloc[list(range(0,len(elbow_df),int(sampling_interval/step_size))),:]

    # 差分两次，求拐点，相当于二阶导
    diff_2 = elbow_cal.diff().diff()
    threshold = round(diff_2[diff_2>0].dropna().index[0]-sampling_interval*2,2)  # 减去2倍interval是因为，每做一次差分，y和x的位置会错开一格。例如，一次差分以后，最后一行有数据且第一行没数据；实际上应该是第一行有斜率，最后一行没有斜率。
    # 输出结果：最佳threshold和图像
    elbow_df.plot()
    print('The suggested threshold is', threshold)


    # # 手动设置threshold
    # threshold =

    # 使用选择出来的threshold开始真正做了
    G,num_nodes = make_graph(simi_matr, threshold)

    # 用划分社区的方式聚类
    part = community.best_partition(G)
    part = pd.Series(part)
    num_clu = len(part.value_counts())
    total_out = []
    for i in range(num_clu):
        indexes = part[part==i].index.to_list()
        total_cluster = []
        total_index = []
        for sentence in indexes:
            total_cluster.append(texts[sentence])
            total_index.append(sentence)
        total_out.append(pd.Series(total_cluster,index=total_index))

    print('The number of sentences included in the network:', num_nodes, '/', len(tokenized_sentences))
    return G, total_out












# generate embeddings for the following texts
# Text corpus
# These sentences show the different forms of the word 'bank' to show the value of contextualized embeddings
texts = pd.read_csv('chanel-上海-范范-20201216.csv',header=None) #word to csv
texts = texts[texts[0].str.contains('1：')]
texts = texts[0].to_list()

# Embeddings are generated in the following manner
# Getting embeddings for the target word in all given contexts
all_word_embeddings = []
all_word_tokens = []
i=0

for text in texts:

    tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer,512)

    all_word_tokens.append(tokenized_text)

    # list_token_emdeddings是每一个句子被embedding后的结果，里面每一个元素是一个列表，表示一个token的embedding；每一个元素的长度是768。这个list_token_embedding的长度等于这个句子的单词个数加上2，这个2是cls和sep。
    list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)

    all_word_embeddings.append(list_token_embeddings)

    i+=1
    print(i)


len(all_word_embeddings) # 190个回答
len(all_word_embeddings[-1]) # 最后一个回答有38个字
len(all_word_embeddings[-1][1])  # 一个字被分解为768个维度
len(list_token_embeddings)
len(tokenized_text)
len(all_word_tokens[-1])


# 准备所有的embeddings
embeddings = [word for sentence in all_word_embeddings for word in sentence]
words = [word for sentence in all_word_tokens for word in sentence]

# 降维
umap_embeddings = umap.UMAP(n_neighbors=15,#fine-tune
                            n_components=5,#fine-tune, - 人工调参
                            metric='cosine').fit_transform(embeddings)



# 聚类
cluster = hdbscan.HDBSCAN(min_cluster_size=40,
                          metric='euclidean',
                          cluster_selection_method='eom',
                          core_dist_n_jobs=10).fit(umap_embeddings)


# 查看每个类别的字数

d = Counter(cluster.labels_)
d
freq = sorted(d.items(),key=lambda x:x[1],reverse=True)
freq.pop(0)
total = 0
for item in freq:
    total+=item[1]
total

# 可视化聚类结果
# 先降到二维，方便展示
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# 可视化聚类结果
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()




len(cluster.labels_)
# 把原文的tokenized汉字转换成聚类的类别label：all_word_labels
labels = cluster.labels_.tolist()
all_word_labels = []
for sentence in all_word_tokens:
    sent_len = len(sentence)
    one_sent_label = labels[0:sent_len]
    labels = labels[sent_len:]
    all_word_labels.append(one_sent_label)



len(all_word_labels)

# 看一下把每一个字都label成什么了
# 分回答
list(zip(all_word_labels[1],all_word_tokens[1]))
# 全文
label_class = pd.DataFrame(list(zip(cluster.labels_.tolist(),words)))

# 看一下，每一个label下面都有哪些字
label_class_outcome = label_class.groupby(by=0).apply(lambda x:[','.join(x[1])])
label_class_outcome[149]

# 看一下被标记为离群点的点，的词频统计
label_minus_one=label_class_outcome[-1]
label_minus_one = label_minus_one[0].split(',')
result = Counter(label_minus_one)
print(result)



# 把原来的tokenized汉字转成label。其中非离群点就是label，离群点就是原来的汉字
labels = cluster.labels_.tolist()
words = [word for sentence in all_word_tokens for word in sentence]
all_token_labels = []
all_sent_labels = []
all_word_labels2 = []
for sentence in all_word_tokens:
    one_sent_labels = []
    for word in sentence:
        one_label = labels[0]
        one_word = words[0]
        one_token_label = one_label if one_label != -1 else one_word
        labels = labels[1:]
        words= words[1:]
        one_sent_labels.append(str(one_token_label))
    all_word_labels2.append(one_sent_labels)


# 比较一下全部换成label，和有label用label、离群点用汉字的区别
list(zip(all_word_labels[3],all_word_labels2[-1]))





# 输入为tfidf，返回整个文档的聚类结果total_out
G,total_out=tfidf_cluster(all_word_labels2)



# nx.draw(G)


# 两种需要重新处理的情况：之前没在network里面的，一个cluster里面的句子过多的

# 一个cluster里面句子过多的情况
# 生成cluster size的均值和中位数
len_medi = np.median([len(df) for df in total_out])
len_mean = np.mean([len(df) for df in total_out])
# 选出来cluster size过大的cluster
large_clusters_indexs = [df.index.values for df in total_out if len(df)>len_medi and len(df) > len_mean]

# 制作这些cluster的tokenization
large_clusters_tokens = [[all_word_labels2[sentence] for sentence in cluster] for cluster in large_clusters_indexs]

# 把这些每一个cluster，聚类成sub clusters
sub_clusters=[]
for i in range(len(large_clusters_indexs)):
    G,total_out=tfidf_cluster(large_clusters_tokens[i])
    sub_clusters.append(total_out)





G,total_out=tfidf_cluster(large_clusters_tokens[0])

len(total_out[0])
len(total_out[1])

len(large_clusters_tokens[0][0])
len(large_clusters_tokens[0][1])




# 保存total_out结果到excel


def save_total_out(file_name, total_out):
    total_out[0].to_excel(file_name,sheet_name=('0-'+str(len(total_out[0]))))
    for i in range(1,len(total_out)):
        with pd.ExcelWriter(file_name,
                            mode='a', engine="openpyxl") as writer:
            total_out[i].to_excel(writer, sheet_name=str(i)+'-'+str(len(total_out[i])))

save_total_out('sub_clusters[2].xlsx', sub_clusters[2])

len(sub_clusters[0])
sorted(large_clusters_indexs[0])
sorted([item for sub_cluster in sub_clusters[1] for item in sub_cluster.index])

for item in large_clusters_indexs:
    print(item)

for item in sub_clusters:
    print(item)



# 之前没在network里面的情况
# “所有的句子的序号”与“在network中的句子的序号”的差集
index_outliers = list(set(range(len(texts))).difference(set([index for df in total_out for index in df.index.values])))
# 离群点的index
outlier_tokens = [all_word_labels2[i] for i in index_outliers]
# 离群点的cluster
G,total_out=tfidf_cluster(outlier_tokens)






















# 输出结果，每一个topic下的句子全部放在一个sheet里面，总共输出一个excel文件
file_name='total_out-0204-2-1.xlsx'
total_out[0].to_excel(file_name,sheet_name=('0-'+str(len(total_out[0]))))
for i in range(1,len(total_out)):
    with pd.ExcelWriter(file_name,
                        mode='a', engine="openpyxl") as writer:
        total_out[i].to_excel(writer, sheet_name=str(i)+'-'+str(len(total_out[i])))


# distances between the embeddings for the word bank in different contexts


# Calculating the distance between the embeddings of 'bank' in all the given contexts of the word
list_of_distances = []
for text1, embed1 in zip(texts, target_word_embeddings):
    for text2, embed2 in zip(texts, target_word_embeddings):
        cos_dist = 1 - cosine(embed1, embed2)
        list_of_distances.append([text1, text2, cos_dist])

distances_df = pd.DataFrame(list_of_distances, columns=['text1', 'text2', 'distance'])

# “Context-free” pre-trained embeddings
distances_df[distances_df.text1 == '学']

# “Context-based” pre-trained embeddings
distances_df[distances_df.text1 == '其他语言学起来很难']

# “Context-averaged” pre-trained embeddings
cos_dist = 1 - cosine(target_word_embeddings[0], np.sum(target_word_embeddings, axis=0))
print(f'Distance between context-free and context-averaged = {cos_dist}')
