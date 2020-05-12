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
        
Pre-processing techniques:
    Help cleaning and stadardization of the text, whit help in analytical systems
    1. Tokenization
    2. Tagging
    3. Chunking
    4. Stemming
    5. Lemmatization
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
SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.(?<=\./\?/\!)\s)'

TOKEN_PATTERN = r"\w+"

'''
Create a liste static of stopwords with analysing few websites and add words 
without a lot of significance to my list
'''
STOPWORDS_LIST = ['mr','mrs','come','go','get','tell','listen', 'back', 'ask']

stopword_list = nltk.corpus.stopwords.words("english")

def clean_sentence(corpus):
    make_clean_text(corpus)
    '''
        1. regex_st cleaning sentences
        2. regex_wt cleaning senteces this is for me cleaning all non text and
        keep only words.
    '''
    #regex_st = nltk.tokenize.RegexpTokenizer(pattern=SENTENCE_TOKENS_PATTERN,gaps=True)
    treebank_wt = nltk.TreebankWordTokenizer()
    words_001 = treebank_wt.tokenize(article_text)
     
    regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, gaps=False)
    words_002 = treebank_wt.tokenize(article_text)
    word_indices = list(regex_wt.span_tokenize(article_text))
    
def keep_text_characters(texts):
    filtered_words = []
    
    for word in texts:
        if re.search(SENTENCE_TOKENS_PATTERN, word):
            filtered_words.append(word)
    filtered_text = ''.join(filtered_words)
    return filtered_text
    
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

# Analyzing Term Similarity
def vectorize_terms(terms):
    '''
    Help :
        The function takes input list of words and returns the corresponding character vecrtors for the words.
        And we coupute the frequency of each caracter in the word
    '''
    terms = [term.lower() for term in terms]
    terms = [np.array(list(term)) for term in terms]
    terms = [np.array([ord(char) for char in term]) for trem in terms]
    return terms

from scipy.stats import itemfreq

def boc_term_verctors(word_list):
    '''
        Help:
            We take in a list of words or terms and then extract the unique characters form all whe words.
            
    '''
    word_list = [word.lower() for word in word_list]
    unique_chars = np.unique(np.hstack([list(word) for word in word_list]))
    word_list_term_counts = [{char: count for char, count in itemfreq (list(word))} for word in word_list]
    boc_vectors = [np.array([int(word_term_counts.get(char,o))
        for char in unique_chars]) 
            for word_term_counts in word_list_term_counts]
    return list(unique_chars),boc_vectors

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
