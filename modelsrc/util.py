import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np


def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    return data


def make_tags(txt, use_stop=False, targeted=False):
    stop_words = set(stopwords.words('english'))
    tokenized = sent_tokenize(txt)
    result = []
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        wordsList = [
            w for w in wordsList if not w in stop_words] if use_stop else wordsList
        tagged = nltk.pos_tag(wordsList)
        if targeted:
            tagged = list(filter(lambda c: is_target(c[1]), tagged))
        result.append(tagged)
    return result


def make_txt(txt):
    return ''.join([tups[0] for tups in make_tags(txt, targeted=True)])


def is_verb(token):
    return token in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_noun(token):
    return token in ['NN', 'NNS', 'NNP', 'NNPS']


def is_adj(token):
    return token in ['JJ', 'JJR', 'JJS']


def is_adv(token):
    return token in ['RB', 'RBR', 'RBS']


def is_neg(token):
    return token == "NEG"


def is_target(token):
    return is_verb(token) or is_noun(token) or is_adj(token) or is_adv(token) or is_neg(token)


def embed_matrix(nlp, tokens):
    result = []

    for word in tokens:
        result.append(np.array(nlp(word)[0].vector))

    return np.array(result)
