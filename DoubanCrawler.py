#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import seaborn as sn
import json
import os

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time


# In[96]:


def Transfer_Clicks(browser):
    time.sleep(5)
    try:
        browser.execute_script("window.scrollBy(0,document.body.scrollHeight)", "")
    except:
        pass
    return "Transfer successfully"


# In[3]:


#prepare
driver = webdriver.Chrome()#你的chromedriver的地址
driver.get('https://www.douban.com/gallery/all')
##需要手动登录注册


# In[43]:


#爬取话题部分，可以略过
topics = driver.find_elements_by_xpath("//*[@href]")
topic_link_dic = {}
##爬取所有话题
while True:
    topics = driver.find_elements_by_xpath("//*[@href]")
    for i in topics:
        link = i.get_attribute('href')
        if 'topic' in link:
            topic_link_dic.setdefault(i.text)
            topic_link_dic[i.text]= link
    time.sleep(np.random.randint(1,3))
    next_page = driver.find_element_by_class_name('load-more-btn')
    print(next_page.text)
    next_page.click()


# In[101]:


#将话题对应链接存到json文件里
with open('topic_link_dic.json','w') as f:
    json.dump(topic_link_dic,f)


# In[ ]:


#读取json文件
with open('topic_link_dic.json','r') as f:
    topic_link_dic = json.load(f)


# In[70]:


#目标topic
target_list = ['如何优雅地度过悠长假期','如何在周五的晚上迎接周末','记录我的周末时光','收集夏天的季节性快乐时刻','属于你的夏夜记忆',
               '分享你的“hygge”时刻','你的愉悦事物清单','哪一个瞬间觉得生活是温柔的？','生活中让你感受到“万物可爱”的瞬间','你生活中cozy的时刻',
               '如何享受庸俗简单的日常生活？','最近一次开怀大笑是因为什么？','那些瞬间让你真切感受到了生活的有趣','热爱生活的一万个理由',
               '我所迷恋的“幼稚”爱好','宅在家里的100种方式','独居生活指南','你为什么迟迟不肯睡觉','如果把看手机时间用来做别的，我可以完成什么？',
               '看到日落时你会想起什么？','那些独在异乡为异客的瞬间','有哪些爱好是为了迎合别人才拥有的？','我的小众冷门爱好',
               '你有哪些看起来与年龄不符的爱好','夏天开始的瞬间','你因为想念一个人而做过什么奇怪的事？','我关于幼儿园的记忆','我的「慢半拍」人生体验',
               '你怎样渡过生活难关的间隙','在城市里，你是如何感受到夏天的来临？''我不想做一个___的人']


# In[109]:


#针对话题抓取的函数
def douban_corpus(topic):
    temp_link  = topic_link_dic[topic]
    driver.get(temp_link)
    temp_list = []
    after = 0
    timeToSleep = 100
    while True:
        before = after
        Transfer_Clicks(driver)
        time.sleep(1)
        elems = driver.find_elements_by_class_name('status-preview')
        new_list = [elem.text for elem in elems]
        temp_list.extend(new_list)
        after = len(set(temp_list))
        if after > before:
            n = 0
        if after == before:
            n = n + 1
        if n == 5:
            print("话题的条数为：%d" % after) 
            break
        if after > timeToSleep:
            timeToSleep = timeToSleep + 100
            time.sleep(np.random.randint(5,10))
    return list(set(temp_list))


# In[105]:


#循环抓取待机。豆瓣不是很稳定，经常容易崩溃，也可手动输入话题名进行抓取。
topic_corpus_dic = {}
for topic in target_list:
    if topic in topic_link_dic.keys():
        topic_corpus_dic.setdefault(topic,[])
        print("现在抓取的话题是：%s"%topic)
        topic_corpus_dic[topic] = douban_corpus(topic)
        time.sleep(np.random.randint(0,2))

