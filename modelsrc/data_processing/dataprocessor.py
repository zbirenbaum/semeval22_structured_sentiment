#!/usr/bin/env python
# coding: utf-8

# In[23]:


from collections import defaultdict
import re,csv,json
import numpy as np
import pandas as pd

from b4msa.textmodel import TextModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

from os.path import join
from util import load_dataset, make_tags, embed_matrix
from df_builder import build_df

from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
# nlp = spacy.load("en_core_web_lg")


# In[37]:


df=pd.DataFrame(pd.read_csv("training_tweets.csv",encoding='ISO-8859-1',names=['Target','Ids','Date','Flag','User','Text']))
df.drop_duplicates(inplace=True, subset="text")
df.text = df.text.apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
df.text.apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
df.text = df.text.apply(lambda x: re.sub(r'{link}', '', x))
df.text = df.text.apply(lambda x: re.sub(r"\[video\]", '', x))
df.text = df.text.apply(lambda x: re.sub(r'@mention', '', x))
df.text = df.text.apply(lambda x: re.sub(r'@[A-Za-z0-9_]+','',x))
df.text = df.text.apply(lambda x: re.sub(r'[()!?]', ' ', x))
df.text = df.text.apply(lambda x: re.sub(r'\[.*?\]', ' ', x))


# In[47]:



df["text"] = df['text'].str.replace('[^\w\s]','')
X_train=df['text'].to_numpy()
X_train  = X_train.tolist()

Y_train = df['target'].to_numpy().tolist()
Y_train = [0 if i==0 else 1 for i in Y_train]



# In[49]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


# In[ ]:


# embed_path = 'glove.twitter.27B.200d.txt'
# embed_dim = 300
# # Tokenizing the text
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(X_train)
# # Creating the embedding matrix
# embedding = Embeddings(embed_path, embed_dim)
# embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))


# In[17]:


df.to_csv('training_tweets_processed.csv')

