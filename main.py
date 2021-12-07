from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
#from embedding import Sentencizer
import numpy as np
from tag import make_txt
import spacy


#######################################Prep-rocessing####################################
# df = pd.DataFrame(pd.read_csv("data.csv", encoding='ISO-8859-1',
#                               names=['Target', 'Ids', 'Date', 'Flag', 'User', 'Text']))
# df.drop_duplicates(inplace=True, subset="Text")
# df.Text = df.Text.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
# df.Text.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
# df.Text = df.Text.apply(lambda x: re.sub(r'@mention', '', x))
# df.Text = df.Text.apply(lambda x: re.sub(r'@[A-Za-z0-9_]+', '', x))
# df.Text = df.Text.apply(lambda x: re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", x))
# df.Text = df.Text.apply(lambda x: re.sub(r'\[.*?\]', ' ', x))
# df.to_csv('data_clean.csv')


from sklearn.svm import SVC


from os.path import join
from tag import make_tags
import numpy as np


df=pd.read_csv("data/data_clean1000.csv",encoding='ISO-8859-1',names=np.array(['Target','Ids','Date','Flag','User','Text']))

nlp = spacy.load("en_core_web_lg")


def word_embedding(token):
    return nlp(token).vector


def sentence_embedding(sentence):
    sentence_embed = []
    for token in nlp(sentence):
        if np.fabs(nlp.vocab[token.text].vector).sum() > 0:
            sentence_embed.append(word_embedding(token.text))
    if sentence_embed:
        return embedding_mean(sentence_embed)

    return None


def embedding_mean(sentence_embed):
    result = [0 for i in range(len(sentence_embed[0]))]

    for i in range(len(sentence_embed[0])):
        summ = 0
        for j in range(len(sentence_embed)):
            summ += sentence_embed[j][i]
        result[i] = summ/len(sentence_embed)
    return result

x_train = []
y_train = []
for ind in df.index:
    if ind > 0:
        s = sentence_embedding(df['Text'][ind])
        if s:
            x_train.append(np.array(s))
            y_train.append(df['Target'][ind])


clf = SVC()

x_train = np.array(x_train).T

clf.fit(x_train, y_train)

sample = np.atleast_2d(sentence_embedding(make_txt(
    "good luck  what an exciting day  this is when the REAL fun begins you're going to love being a mommy")))

y_pred = clf.predict(sample)
print(y_pred)
