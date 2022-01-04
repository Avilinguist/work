import jieba
from gensim import corpora
from gensim.models.ldamodel import LdaModel 
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sn

#分词
##有一步是导入中文停止词
with open('stopwords.txt','r') as f:
  stopwords = [word.strip('\n') for word in f.readlines()]

#这一步是导入文本
documents = list(rawdata['clean_lyric'])

#分词
texts = [[word for word in jieba.cut(document) if word not in stopwords] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

#查看对应一个topic数量，其对应的平均组间距离是什么样的
def topic_choosing(number):
    lda_temp = LdaModel(bow_corpus,num_topics=number,id2word=word_dictionary)
    
    temp_dic = {}
    for topic_number in range(len(lda_temp.print_topics())):
        temp_dic.setdefault(topic_number,{})
        for term, frequency in lda_temp.show_topic(topic_number, topn=25):
            temp_dic[topic_number].setdefault(term)
            temp_dic[topic_number][term]=frequency #计算每个分类里前25的词的词频
    X = pd.DataFrame(temp_dic).fillna(0)
    
    pairs = [record for record in combinations(X.columns,2)]
    distance = []
    for pair in pairs:
        x1 = np.asarray(X[pair[0]]).reshape(-1,1)
        x2 = np.asarray(X[pair[1]]).reshape(-1,1)
        distance.append(cosine_distances(x1,x2))#计算不同分类间在词频上的距离
    average_distance = np.average(distance)#计算平均距离
    
    return average_distance

#筛选类别
distance = [topic_choosing(i) for i in range(4,11,1)]
sn.lineplot(range(4,11,1),distance) #7是个好类别

#LDA Topic model
lda = LdaModel(corpus,num_topics=15,id2word=dictionary) #topic的数量可以自己指定；这一步非常算力consuming

#查看lda
def explore_topic(topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
        
    print (u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    for term, frequency in lda.show_topic(topic_number, topn=25):
        print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))

#储存模型
temp_file = datapath("model")
lda.save(temp_file)

#展现Topic
LDAvis_prepared = pyLDAvis.gensim.prepare(lda,corpus,dictionary)
pyLDAvis.display(LDAvis_prepared)
