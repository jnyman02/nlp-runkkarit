#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:17:05 2020
@author: Yazid BOUNAB
Yazid.Bounab@oulu.fi

"""
import nltk
import numpy as np
from nltk.corpus import genesis
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

'https://www.nltk.org/howto/wordnet.html'
'https://seethesource.wordpress.com/2016/08/07/find-synonyms-and-hyponyms-using-python-nltk-and-wordnet%E2%80%8B/#:~:text=WordNet%20groups%20nouns%2C%20adjectives%2C%20verbs,calls%20them%20synsets%20or%20synonyms.&text=In%20the%20example%20given%2C%20brick,and%20brickwork%20are%20called%20synonyms.'
'https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.X2shcZ-YXM0'

genesis_ic = wn.ic(genesis, False, 0.0)


#car = wn.synsets('car', 'n')
#print(car.name())
#print(car.definition())
#print (car.lemmas()[0].count())

#for i,j in enumerate(wn.synsets('car')):
#    print( "Meaning",i, "NLTK ID:", j.name())
#    print( "Definition:",j.definition())
#    print('_______________')
#    print(j.hypernyms())
#    break

T1='Students feel unhappy today about the class of 2020'
T2='Many students struggled to understand some key concepts about the subject seen in the class of 2020'

def wup(S1, S2):
    return S1.wup_similarity(S2)

def Resnik(S1, S2):
    return S1.res_similarity(S2, genesis_ic)

options = {0 : wup,
           1 : Resnik,
          }

def preProcess(sentence):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    stemmer = SnowballStemmer("english")#we will avoid the stemming because it will give a pbm with sysnset search
    
    words = word_tokenize(sentence)
    #words = [stemmer.stem(word) for word in words] 
    words = [word.lower() for word in words if word.isalpha() and word not in Stopwords] #get rid of numbers and Stopwords
 
    return words



def word_similarity(w1,w2,num):
    S1 = wn.synsets(w1)[0]
    S2 = wn.synsets(w2)[0]
#    print(w1,w2)
#    print(S1,'\n_________________\n',S2)
    if S1 and S2:
       similarity = options[num](S1, S2)
       if similarity:
          return round(similarity,2)
    return 0

def Similarity(T1, T2, num):
    words1 = preProcess(T1)
    words2 = preProcess(T2)
    
    tf = TfidfVectorizer(use_idf=True)
    tf.fit_transform([' '.join(words1), ' '.join(words2)])

    Idf = dict(zip(tf.get_feature_names(), tf.idf_))
    
    Sim = 0
    Sim_score1 = 0
    Sim_score2 = 0
    
    for w1 in words1:
        Max = 0
        for w2 in words2:
            score = word_similarity(w1,w2,num)
            if Max < score:
               Max = score
        Sim_score1 += Max*Idf[w1]
    Sim_score1 /= sum([Idf[w1] for w1 in words1])
    
    print(round(Sim_score1,2))
    for w2 in words2:
        Max = 0
        for w1 in words1:
            score = word_similarity(w1,w2,num)
            if Max < score:
               Max = score
        Sim_score2 += Max*Idf[w2]
        
    Sim_score2 /= sum([Idf[w1] for w2 in words2])
    print(round(Sim_score2,2))

    Sim = (Sim_score1+Sim_score2)/2
    
    return round(Sim,2)



#dog = wn.synset('dog.n.01')
#cat = wn.synset('cat.n.01')
#
#dog.path_similarity(cat)
#
#dog.lch_similarity(cat)
#
#dog.wup_similarity(cat)



print('Wup Similarity(T1, T2) =',Similarity(T1, T2, 0))
#print('Resnik Similarity(T1, T2) =',Similarity(T1, T2, 1))