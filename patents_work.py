#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:47:45 2020

@author: djemi

I Need this :
    
For sam project I need to use Contextual Recognition and Reolution
-> undestanding natular language and includes both syntactic and semantic-based reasoning.

Text Summarization
-> retains the key points of the collection

Text Categorization
-> is identifying to which category or class a specific document should be placed based on the 
contents of the document.

Steps :
    1. text preprocessing
        a. NLTK -> the natural language toolkit
        b. gensim
        c. textblob
        d. spacy
"""

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud


word_lemmatizer = WordNetLemmatizer()

url_list = pd.read_excel('../Sam_Url_OK_KO_connexion.xlsx')
url = url_list.get('OK_Connexion')[4]

import bs4 as bs  
import urllib.request  
import re

#creating Bag of Words Model
raw_html = urllib.request.urlopen(url)  
raw_html = raw_html.read()
article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('body')

patents = {'patent', 'patents', 'company', 'application','product', 
                     'products', 'access', 'package', 'brend','companies'}
patents_words = set(w.rstrip() for w in patents)

article_text = ''

for para in article_paragraphs:  
    article_text += para.text

corpus = nltk.sent_tokenize(article_text)

'''
If I find the sentance with text patents or patent I keap the all sentance ZD
Natural language : word, sentence, document de plus il faut corpus oeuvre etc..
'''

def clen_sentence(corpus):
    make_clean_text(corpus)

def visualize(corpus):
    words=''
    for msg in corpus:
        msg = msg.lower()
        words += msg + ''
    
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
def check_patent_into_sentence(corpus):
    list_sentence = []
    for sentence in corpus:
        token = my_tokenizer(sentence)
        print("token :{0}".format(token))
        if token in patents_words:
            list_sentence.append(sentence)
    return list_sentence
                    

def make_clean_text(corpus):
    for i in range(len(corpus )):
        corpus [i] = corpus [i].lower()
        corpus [i] = re.sub(r'\W',' ',corpus [i])
        corpus [i] = re.sub(r'\s+',' ',corpus [i])

wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1


def my_tokenizer(s):
    s = s.lower()
    tokens_new = nltk.tokenize.word_tokenize(s)
    tokens_new = [token for token in tokens_new if len(token) > 2]
    '''
        Definition of :
            1. lemmatize -> a lemma is the canonical or base form for a set of word  and is 
            also known as the head word
            2. stem -> a stem for a word is a part of the word to which various affixes can be 
            attached
            3. pow -> This is mainly used to annotate each word with a POS tag indicating
            the part of speech associated with it
    '''
    tokens_new = [word_lemmatizer.lemmatize(token) for token in tokens_new]
    tokens_new = [word_lemmatizer.stem(token) for token in tokens_new]
    tokens_new = [word_lemmatizer.pow(token) for token in tokens_new]
    
            
    tokens_new = [token for token in tokens_new if token not in patents_words]
    tokens_new = [token for token in tokens_new if not any(c.isdigit() for c in token)]
    return tokens_new


word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []

for title in corpus:
    try :
        #title = title.encode('ascii', 'ignore') -> transform in bytes 
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except :
        pass


def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map)) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x


def display_histo(all_tokens, word_index_map):
    N = len(all_tokens)
    D = len(word_index_map)
    X = np.zeros((D,N))
    i = 0
    
    for tokens in all_tokens:
        X[:,i] = tokens_to_vector(tokens)
        i +=1
        
    svd = TruncatedSVD()
    Z = svd.fit_transform(X)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(D):
        plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
    plt.show()


from nltk.corpus import brown

def new_text_cleaning(corpus):
    pass    
