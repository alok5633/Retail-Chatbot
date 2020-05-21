import pickle
import json
import random
import tensorflow
import tflearn
import numpy
import re
from nltk.stem.lancaster import LancasterStemmer
import nltk
nltk.download('punkt')
stemmer = LancasterStemmer()

with open("utils/intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    
    if intent["tag"] not in labels:
        labels.append(intent["tag"]) 

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels) 
print(words)
print(labels)   
