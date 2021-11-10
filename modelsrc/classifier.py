import argparse
import json
import pandas as pd

#each dict contains sent id, text, opinion
#['opinions'] contains a list of dicts
#['opinions'] can be of arbitrary len including 0
dataset = "../data/opener_en/train.json"
with open(dataset) as f:
    data=json.load(f)

opinion_labels = ['Polar_expression', 'Intensity', 'Source', 'Polarity', 'Target']
labels = ['sent_id', 'text']

def build_clean_df():
    rows = []
    for datapoint in data:
        basis = {}
        to_append = {}
        for label in labels:
            basis[label] = str(datapoint[label])
            
        if len(datapoint['opinions']) != 0:
            for opinion in datapoint['opinions']:
                for label in opinion_labels:
                    try:
                        to_append[label] = opinion[label]
                    except:
                        to_append[label] = None
                for key in basis.keys():
                    to_append[key] = basis[key]
                rows.append(to_append.copy())
                to_append = basis
        else:
            for label in opinion_labels:
                to_append[label] = None
    df = pd.DataFrame(rows)
    return df

df = build_clean_df()
print(df)
