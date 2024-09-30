import json
import nltk

def readJson(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def constructInvertedIndex(abstracts):
    invertedIndex = {}
    for index, abstract in enumerate(abstracts):
        for keyword in abstract['keywords']:
            if keyword not in invertedIndex:
                invertedIndex[keyword] = []
            invertedIndex[keyword].append(index)

    return invertedIndex

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

def keywordtoTitleDistances(InvertedIndex, abstracts):
    titleDistances = []

    

    for keyword

def main():
    query = "machine learning"
    abstracts = readJson('./extracted_articles.json')
    invertedIndex = constructInvertedIndex(abstracts)
    print(invertedIndex)
    keywordsDistances = editSimilarity(query, invertedIndex)
    keywordsDistances = distancesToAlphabetical(keywordsDistances)
    printDistances(keywordsDistances, query)
    """titleSimilarities = titleAndKeyowordSimiliarity(abstracts)
    printTitleDistances(titleSimilarities)"""




if __name__ == '__main__':
    main()
