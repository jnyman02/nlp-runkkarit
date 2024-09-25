"""
This code is the work of Jussi Saariniemi and Jesper Nyman.
"""

import json
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Levenshtein import distance
import math

abstracts = {}
abstractsOriginal = {}
keywords = {}
splitAbstracts = {}
invertedFileIndex90percent = {}
keywordsAll = []

# A - Infromation Retrieval 1: task 2)
# Loads the abstracts and the keywords from the given paths.
# Then queries for the keywords given to the method B_task2. Prints 1 if a match was found, 0 if not.
# Also prints the titles of the abstracts that matched the keyword.

def task2(keywords):
    AbstractLoader(r"E:/KOULU 2024-2025/Natural Language Processing/Labs/Lab2/Lab2_files/Abstracts.json")
    for keyword in keywords:
        print(query(keyword))

def query(keyword):
    global abstracts
    global keywords
    print("\nQuerying for keyword: " + "\033[33m" + keyword + "\033[0m")
    keyword = keyword.lower()
    keyword = PorterStemmer().stem(keyword)
    result = []
    boolean = 0
    for key in abstracts:
        if keyword in abstracts[key]:
            result.append(key)
            boolean = 1
    return queryToString(boolean, result)

def queryToString(boolean, result):
    if boolean == 0:
        return "0\n\033[31mNo results found.\033[0m"
    else:
        return "1\n\033[32mResults found:\033[0m " + ', '.join(result)

def AbstractLoader(path):
    global abstracts
    global abstractsOriginal
    with open(path, 'r', encoding='utf-8') as file:
        abstracts = json.load(file)
        abstractsOriginal = abstracts
    for key in abstracts:
        abstracts[key] = abstracts[key].lower()
        abstracts[key] = word_tokenize(abstracts[key])
        abstracts[key] = [PorterStemmer().stem(word) for word in abstracts[key]]
        

def KeywordLoader(path):
    global keywords
    with open(path, 'r', encoding='utf-8') as file:
        keywords = json.load(file)

# 3)
# Loads all the abstracts from separate files and the keywords from a json file.
# Then creates an inverted file index from the keywords and the abstracts.
# Lastly prints the inverted file index.
def task3():
    SeparateAbstractsLoader(r"E:/KOULU 2024-2025/Natural Language Processing/Labs/Lab2/Lab2_files/Abstracts_split")
    KeywordLoader(r"E:/KOULU 2024-2025/Natural Language Processing/Labs/Lab2/Lab2_files/Keywords.json")
    invertedFileIndexer()
    printInvertedFileIndex()

def SeparateAbstractsLoader(path):
    global splitAbstracts

    files = os.listdir(path)
    for file in files:     
        with open(os.path.join(path, file), 'r', encoding='utf-8') as json_file:
                name = os.path.splitext(file)[0]
                temp = json.load(json_file)
                splitAbstracts[name] = next(iter(temp.values()))
                splitAbstracts[name] = splitAbstracts[name].lower()
                splitAbstracts[name] = word_tokenize(splitAbstracts[name])
                #splitAbstracts[name] = [PorterStemmer().stem(word) for word in splitAbstracts[name]]

def KeywordLoader(path):
    global keywords
    with open(path, 'r', encoding='utf-8') as file:
        keywordsRaw = json.load(file)
    for key in keywordsRaw:
        keywords[key] = []
        for element in keywordsRaw[key]:
            list = word_tokenize(element)
            for word in list:
                keywords[key].append(word.lower())
    
def invertedFileIndexer():
    global splitAbstracts
    global keywords
    global invertedFileIndex90percent
    global keywordsAll

    for list in keywords.values():
        for word in list:
            if word not in keywordsAll:
                keywordsAll.append(word)

    """for element in keywordsAll:
        invertedFileIndex[element] = []
        for key in keywords:
            if element in keywords[key]:
                invertedFileIndex[element].append(key)"""
    for element in keywordsAll:
        invertedFileIndex90percent[element] = []
        for key in splitAbstracts:
            if element in splitAbstracts[key]:
                invertedFileIndex90percent[element].append(key)

def printInvertedFileIndex():
    global invertedFileIndex90percent
    print("\nInverted file index")
    for key in invertedFileIndex90percent:
        print("\033[33m" + key + "\033[0m" + ": " + ', '.join(invertedFileIndex90percent[key]))

def task4():
    invertedFileIndexer90percent()
    printInvertedFileIndex90percent()

def invertedFileIndexer90percent():
    global splitAbstracts
    global invertedFileIndex90percent
    global keywordsAll

    for element in keywordsAll:
        invertedFileIndex90percent[element] = []
        for key in splitAbstracts:
            for word in splitAbstracts[key]:
                threshold = int(math.ceil(len(word) * 0.1))
                
                dist = distance(element, word, score_cutoff=threshold)
                if dist <= threshold:
                    """print("threshold: " + str(threshold))
                    print("distance: " + str(dist))
                    print("keyword: "+ element + " matching word: " + word)"""
                    invertedFileIndex90percent[element].append(key)
                    break

def printInvertedFileIndex90percent():
    global invertedFileIndex90percent
    print("\nInverted file index with 90% threshold")
    for key in invertedFileIndex90percent:
        print("\033[33m" + key + "\033[0m" + ": " + ', '.join(invertedFileIndex90percent[key]))

def main():
    # Give this method a list of keywords to query for. Queries the keywords.
    task2(["Encryption", "Fishing", "Privacy", "Philately", "Visualization"])
    
    # Searches the abstract texts rather than the keywords of the abstarcts.
    task3()

    # Searches the abstract texts rather than the keywords of the abstarcts.
    # Calculates the 90% threshold like: word length * 0,1 rounded up.
    # this means that even short words will have a threshold of 1.
    # threshold meaning the maximum levenshtein distance.
    task4()

if __name__ == "__main__":
    main()


"""
Here is what the program prints: 

Querying for keyword: Encryption
1
Results found: CRYSTALS - Kyber: A CCA-Secure Module-Lattice-Based KEM

Querying for keyword: Fishing
0
No results found.

Querying for keyword: Privacy
1
Results found: Machine Unlearning, Membership Inference Attacks Against Machine Learning Models

Querying for keyword: Philately
0
No results found.

Querying for keyword: Visualization
1
Results found: Deep Residual Learning for Image Recognition, Histograms of oriented gradients for human detection, Are we ready for autonomous driving? The KITTI vision benchmark suite

Inverted file index
training: A1, A14, A16, A7
degradation: A7
complexity: A1, A14
theory:
image: A20, A3, A9
recognition: A1, A10, A17, A20, A3, A4
neural: A1, A13, A5, A7, A9
networks: A1, A13, A6, A7, A9
visualization:
segmentation: A1
context: A2
psychology:
databases:
educational:
institutions:
games:
libraries: A15
large-scale: A15, A3, A9
systems: A12, A2, A20, A4, A5, A6
explosions:
internet: A3
robustness: A13, A4
information: A11, A16, A19, A6, A7
retrieval: A19
multimedia: A3
ontologies:
spine:
yolo: A4, A5
performance: A12, A17, A19, A4, A5
evaluation: A14, A4, A5, A7, A9
technological:
innovation:
computer: A20, A3, A4
vision: A20, A3, A4
heuristic:
algorithms: A13, A20, A3, A4, A7, A8, A9
speech: A10
benchmark: A13
testing:
architecture: A5
microprocessors:
object: A1, A17, A20, A3, A4, A5
detection: A1, A14, A17, A20, A4, A5
real-time: A4, A5
pipelines:
recommender: A6
filtering: A6
motion: 
pictures:
social: A11, A6
network: A13, A15, A5
services: A11
science:
international:
collaboration:
electronic: A6
commerce: A6
books:
data: A10, A14, A16, A3, A7
privacy: A16, A7
limiting: A7
transfer: A10, A7
learning: A1, A10, A13, A16, A7, A8, A9
stochastic: A7
processes: A5
deep: A1, A7, A8, A9
artificial:
support:
vector:
machines: A15, A8
neurons:
diseases:
lung:
feature: A17
extraction: A12
x-rays: A9
task: A1, A3
analysis: A1, A11, A3, A6
convolutional: A9
coding:
computational: A10, A4, A7
modeling:
models: A10, A16, A18, A3, A4, A7, A9
system-on-chip:
compaction: A10
processing: A14, A19
europe:
security: A18
licenses:
time-frequency:
modulation: A12
ofdm: A12
transforms:
doppler: A12
effect:
shape:
receivers:
measurement: A6
malware:
resists:
application: A19, A8
software:
credit:
cards:
detectors: A5
constraint:
optimization:
mining:
technology: A10, A18
laboratories:
isolation: A14
astronomy:
costs:
processor:
scheduling: A15
large: A12, A14, A15, A17, A19
language: A15, A19
graphics:
units:
throughput: A12, A15
predictive:
sociology:
statistics:
google: A16
histograms: A17
humans: A19, A9
edge: A10, A17
high: A10, A12, A14, A20
computing: A18, A8
encryption: A18
lattices: A18
mail:
protocols: A18
public:
key: A18, A4
productivity: A19
accuracy: A1, A10, A19, A3, A4, A7, A8
circuits:
and: A1, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A2, A20, A3, A4, A5, A6, A7, A8, A9
buildings:
oral:
communication:
chatbots: A19
user: A19
experience: A19
cameras: A20
optical: A20
imaging:
sensors:

Inverted file index with 90% threshold
training: A1, A14, A16, A7
degradation: A7
complexity: A1, A14
theory:
image: A11, A17, A20, A3, A5, A9
recognition: A1, A10, A17, A20, A3, A4
neural: A1, A13, A5, A7, A9
networks: A1, A13, A15, A5, A6, A7, A9
visualization:
segmentation: A1
context: A2, A6
psychology:
databases: A17, A3
educational:
institutions:
games:
libraries: A15
large-scale: A15, A3, A9
systems: A10, A12, A2, A20, A4, A5, A6, A9
explosions: A3
internet: A3
robustness: A13, A4
information: A11, A16, A19, A6, A7
retrieval: A19
multimedia: A3
ontologies:
spine:
yolo: A4, A5
performance: A12, A17, A19, A4, A5
evaluation: A14, A4, A5, A7, A9
technological:
innovation: A4
computer: A15, A20, A3, A4
vision: A20, A3, A4
heuristic:
algorithms: A13, A14, A20, A3, A4, A7, A8, A9
speech: A10
benchmark: A13, A20, A4
testing:
architecture: A4, A5
microprocessors:
object: A1, A17, A20, A3, A4, A5
detection: A1, A14, A17, A20, A4, A5
real-time: A4, A5
pipelines: A5
recommender: A6
filtering: A6
motion:
pictures:
social: A11, A6
network: A1, A13, A15, A5, A6, A7, A9
services: A11, A16, A7
science:
international:
collaboration: A6
electronic: A6
commerce: A6
books:
data: A10, A14, A16, A3, A7
privacy: A16, A7
limiting: A7
transfer: A10, A7
learning: A1, A10, A13, A16, A7, A8, A9
stochastic: A7
processes: A5
deep: A1, A7, A8, A9
artificial:
support:
vector:
machines: A13, A15, A16, A7, A8
neurons:
diseases: A9
lung: A9
feature: A17, A9
extraction: A12
x-rays: A9
task: A1, A10, A13, A16, A20, A3, A7, A9
analysis: A1, A11, A3, A4, A6
convolutional: A4, A9
coding:
computational: A10, A15, A17, A4, A7
modeling:
models: A10, A11, A15, A16, A18, A19, A3, A4, A5, A6, A7, A9
system-on-chip:
compaction: A10
processing: A14, A19
europe:
security: A18
licenses:
time-frequency:
modulation: A12
ofdm: A12
transforms: A18
doppler: A12
effect: A2
shape: A11
receivers:
measurement: A6
malware:
resists: A13
application: A10, A15, A19, A20, A3, A4, A8
software:
credit:
cards: A20
detectors: A5
constraint:
optimization:
mining:
technology: A10, A18
laboratories:
isolation: A14
astronomy:
costs: A11, A15
processor:
scheduling: A15
large: A12, A14, A15, A17, A19, A3
language: A15, A19
graphics:
units:
throughput: A12, A15
predictive:
sociology:
statistics: A9
google: A16
histograms: A17
humans: A17, A19, A8, A9
edge: A10, A17
high: A10, A12, A14, A20
computing: A18, A8
encryption: A18
lattices: A18
mail: A11, A20
protocols: A18
public:
key: A18, A4
productivity: A19
accuracy: A1, A10, A19, A3, A4, A7, A8
circuits:
and: A1, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A2, A20, A3, A4, A5, A6, A7, A8, A9
buildings: A19
oral:
communication:
chatbots: A19
user: A1, A11, A13, A14, A16, A19, A2, A6, A7, A9
experience: A12, A19
cameras: A20
optical: A20, A4, A9
imaging:
sensors:
"""