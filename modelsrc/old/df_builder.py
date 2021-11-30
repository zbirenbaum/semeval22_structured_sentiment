import argparse
import json
import pandas as pd

#each dict contains sent id, text, opinion
#['opinions'] contains a list of dicts
#['opinions'] can be of arbitrary len including 0
def build_df():
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
    relevant = df.filter(['Polar_expression','Polarity', 'Source', 'Target'], axis=1)
    relevant.Polar_expression = relevant.Polar_expression.apply(lambda value : value[0][0])
    relevant.Source = relevant.Source.apply(lambda value : None if not value or not value[0] else value[0][0])
    relevant.Target = relevant.Target.apply(lambda value : None if not value or not value[0] else value[0][0])
    relevant = relevant.dropna()
    return relevant
#relevant.Target = relevant.Target.apply(lambda value : (None, value[0][0])[!value[0]])
#print(relevant.T.dropna())
