import jieba
from gensim import corpora
from gensim.models.ldamodel import LdaModel 
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim

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
