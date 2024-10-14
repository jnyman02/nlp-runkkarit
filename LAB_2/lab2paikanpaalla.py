import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import string
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr


stop_words = set(stopwords.words('english'))

def readJson(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return tokens
    

# FOR PART 2
def constructInvertedIndex(abstracts):
    invertedIndex = {}
    for index, abstract in enumerate(abstracts):
        for keyword in abstract['keywords']:
            if keyword not in invertedIndex:
                invertedIndex[keyword] = []
            invertedIndex[keyword].append(index)

    return invertedIndex


# FOR PART 2 
def editSimilarity(query, InvertedIndex):
    keywords = []
    querys = query.split(" ")
    for key in InvertedIndex:
        distances = []
        distances.append(nltk.edit_distance(query, key))
        for part in querys:
            distances.append(nltk.edit_distance(key, part))
        distance = min(distances)
        keywords.append((key, distance))

    return keywords

def distancesToAlphabetical(keywordsDistances):
    return sorted(keywordsDistances, key=lambda x: x[0])

def printDistances(keywordsDistances, query):
    print("Distance of each keyword to the query: " + query + "\n")
    for keyword, distance in keywordsDistances:
        print("Keyword: " + keyword + ", Distance: " + str(distance))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def countKeywordOccurrences(documents, keywords):
    M = np.zeros((len(documents), len(keywords)))
    for i, doc in enumerate(documents):
        abstract_tokens = preprocess_text(doc['abstract'])
        for j, keyword in enumerate(keywords):
            M[i, j] = abstract_tokens.count(keyword)
    return M

def booleanModel(query, M, keywords):
    query_terms = nltk.word_tokenize(query.lower())
    relevant_docs = []
    for i, row in enumerate(M):
        for term in query_terms:
            if term in keywords and row[keywords.index(term)] > 0:
                relevant_docs.append(i)
                break
    return relevant_docs

def tfidfModel(documents, query, keywords):
    query_terms = nltk.word_tokenize(query.lower())
    corpus = [doc['abstract'] for doc in documents]
    vectorizer = TfidfVectorizer(vocabulary=keywords)
    X = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([" ".join(query_terms)])
    relevance_scores = X.dot(query_vector.T).toarray()
    relevant_docs = np.argsort(relevance_scores.flatten())[::-1]
    return relevant_docs

def fuzzyWuzzy(documents):
    Z = []
    for doc in documents:
        title = doc['title']
        abstract = doc['abstract']
        score = fuzz.ratio(title, abstract)
        Z.append(score)
    return Z

def main():
    query = "machine learning"
    
    abstracts = readJson('./extracted_articles.json')
    invertedIndex = constructInvertedIndex(abstracts)
    print(invertedIndex)
    keywordsDistances = editSimilarity(query, invertedIndex)
    keywordsDistances = distancesToAlphabetical(keywordsDistances)
    printDistances(keywordsDistances, query)
    
    
    M = countKeywordOccurrences(abstracts, invertedIndex)
    
    relevant_docs = booleanModel(query, M, list(invertedIndex.keys()))
    print(f"\nTotal number of relevant documents using boolean model: {len(relevant_docs)}")
    
    relevant_docs_tfidf = tfidfModel(abstracts, query, list(invertedIndex.keys()))
    print(f"\nTotal number of relevant documents using tf-idf model: {len(relevant_docs_tfidf)}")
    
    Z = fuzzyWuzzy(abstracts)
    print(f"Fuzzy Wuzzy scores: {Z}")




if __name__ == '__main__':
    main()
