#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 09:26:42 2020

@author: djemi
"""

print('Hello Sam Sam')

# import libraries
import matplotlib.pyplot as plt
import pandas as pd
import json


# import libraries for Natural Language Processing and more
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import bs4 as bs  
import urllib.request  
import re
from datetime import datetime

# importing the html form the url
url_list = pd.read_excel('../Sam_Url_OK_KO_connexion.xlsx')

# starting cleaning htlm pages and adding data in to json file
word_lemmatizer = WordNetLemmatizer()

patents = {'patent', 'patents', 'company', 'application','product', 
                     'products', 'access', 'package', 'brend','companies'}
patents_words = set(w.rstrip() for w in patents)

#print(time_it(start_process_nlp, n=1, start=1,end=2))

def start_process_nlp(article_paragraphs = '', corpus = '', number = 0):
    try:
        data = []
        errors = []
        for ligne in range(len(url_list)):
            try:
                url = url_list.get('OK_Connexion')[ligne]
                if ligne <= 350:
                    print (ligne, url)
                    raw_html = urllib.request.urlopen(url)  
                    raw_html = raw_html.read()
                    article_html = bs.BeautifulSoup(raw_html, 'lxml')
                    article_paragraphs = article_html.find_all('body')
                    corpus = first_clean_text_html_return_corpus(article_paragraphs)
                    if len(corpus) == 0:
                        print('No text or problem to find text in the url :',url)
                        add_error(errors,ligne,url,'requete_html_vide_text')
                        # add list of error url can be use after for analysing
                    else:   
                        make_clean_text(corpus)
                        add_to_data(data, corpus, url, ligne)
                else:
                    break
            except urllib.error.URLError as urlErr:
                print('URLError ',url)
                add_error(errors,ligne,url, str(urlErr))
                pass
            except urllib.error.HTTPError as httpErr:
                print('HTTPError ',url)
                add_error(errors,ligne,url, str(httpErr))
                pass
    except :
        pass
    finally :
        add_data_to_json(data)
        add_errors_to_json(errors)
        
def first_clean_text_html_return_corpus(article_paragraphs):
    article_text = ''
    for para in article_paragraphs:  
        article_text += para.text
    corpus = nltk.sent_tokenize(article_text)
    return corpus

def make_clean_text(corpus):
    for i in range(len(corpus )):
        corpus [i] = corpus [i].lower()
        corpus [i] = re.sub(r'\W',' ',corpus [i])
        corpus [i] = re.sub(r'\s+',' ',corpus [i])
        #corpus [i] = my_tokenizer(corpus [i])


def my_tokenizer(s):
    s = s.lower()
    tokens_new = nltk.tokenize.word_tokenize(s)
    tokens_new = [token for token in tokens_new if len(token) > 2]
    '''
        Definition of :
            lemmatize -> a lemma is the canonical or base form for a set of word  and is 
    '''
    tokens_new = [word_lemmatizer.lemmatize(token) for token in tokens_new]
    
    #tokens_new = [token for token in tokens_new if token not in patents_words]
    tokens_new = [token for token in tokens_new if not any(c.isdigit() for c in token)]
    return tokens_new

def add_to_data(data, corpus, url, number):
    one_patern = {}
    one_patern['number'] = number
    one_patern['date_time_extraction'] = str(datetime.now())
    one_patern['url'] = url
    one_patern['all_text'] = corpus
    data.append(one_patern)

def add_data_to_json(data):
    with open("data_corpus.json", "w") as write_file:
            json.dump(data, write_file)

def add_error(errors, number, url, type_error):
    error = {}
    error['number'] = number
    error['url'] = url
    error['date_time'] = str(datetime.now())
    error['type_error'] = type_error
    errors.append(error)
    
def add_errors_to_json(errors):
    with open("data_errors.json", "w") as write_file:
            json.dump(errors, write_file)

def visualize(corpus):
    words=''
    for msg in corpus:
        msg = msg.lower()
        words += msg + ''
    
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

'''
def time_it(fn, rep=1, *args, ***kwargs):
    start = time.perf_counter()
    for i in range(rep):
        fn(*args,***kwargs)
    end = time.perf_counter()
    return (end-start)/rep
'''
# start runing
start_process_nlp()