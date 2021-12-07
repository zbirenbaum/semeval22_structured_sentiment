import tag
import pandas as pd

df = pd.DataFrame(pd.read_csv("data/short_data.csv", index_col=0))
tagger = tag.Tagger(df.Text)

import nltk.classify.util

from nltk.classify import NaiveBayesClassifier

from nltk.corpus import names

import sys

 

# insert the training data here

positive_vocab = [ 'excellent', 'amazing', 'enjoyed', 'oscar', 'outstanding']

negative_vocab = [ 'batman and superman', 'boring', 'adam sandler']

neutral_vocab = ['ok', 'watchable']

# insert the 'features' function here

def word_feats(words):

            return dict([(word, True) for word in words])

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]

negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]

neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

# train the model here

train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set)

# the following code does the prediction:

neg = 0

pos = 0

sentence = "Here is my outstanding review"

sentence = sentence.lower()

words = sentence.split()

for word in words:

    classResult = classifier.classify( word_feats(word))

    

    if classResult == 'neg':

        neg = neg + 1

    

    if classResult == 'pos':

        pos = pos + 1

     

print('Positive: ' + str(float(pos)/len(words)))

print('Negative: ' + str(float(neg)/len(words)))
