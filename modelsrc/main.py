from collections import defaultdict
from microtc.utils import tweet_iterator
from b4msa.textmodel import TextModel
from df_builder import build_df
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from os.path import join
from util import load_dataset


df = build_df()
data_dict = df.to_dict()
train_data = []

for id, text in data_dict['Polar_expression'].items():
    train_data.append({
        'id': id,
        'klass': data_dict['Polarity'][id],
        'text': text})
tm = TextModel(lang="english", token_list=[-1], stemming=True).fit(train_data)
le = LabelEncoder().fit(['Negative', 'Positive'])
X = tm.transform(train_data)
y = le.transform([x['klass'] for x in train_data])
m = LinearSVC().fit(X, y)

print(le.inverse_transform(
    m.predict(tm.transform(['like this assignment'])))[0])
