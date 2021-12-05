
from microtc.utils import tweet_iterator
from os.path import join
#from EvoMSA.utils import load_model, download, bootstrap_confidence_interval
from nltk import data
#from sklearn.decomposition import PCA
import numpy as np
#from sklearn.metrics import pairwise_distances
#from scipy import stats
from os.path import join
from b4msa.textmodel import TextModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
import json
#probably unnecessary
#self.train_path=train_path
#self.test_path=test_path
PATH = "."

def get_data(data_path):
    fname = join(PATH, data_path)
    data = list(tweet_iterator(fname))
    return data

    
def get_text_data(data):
    text_list = []
    for tweet in data:
        text_list.append(tweet['text'])
    return text_list

class ModelBuilder():
    def __init__(self, train_path, test_path) -> None:
        self.set_datasets(train_path, test_path)
        self.set_data_text() 
        return
    
    def predict(self):
        self.hy = self.m.predict(self.tm.transform(self.test_data))
    
    def build_le(self):
        self.le = LabelEncoder().fit([x['klass'] for x in self.train_data])
    
    def build_tm(self):
        self.tm = TextModel(lang="english", token_list=[-1]).fit(self.train_data)
    
    def build_svc(self):
        X = self.tm.transform(self.train_data)
        y = self.le.transform([x['klass'] for x in self.train_data])
        self.m = LinearSVC().fit(X, y)

    def write_predictions(self):
        output = join(PATH, "predictions.json")
        with open(output, 'w') as fpt:
            for x, y in zip(self.test_data, self.hy):
                pred_class=self.le.inverse_transform([y])[0]
                x['klass'] = str(pred_class)
                print(json.dumps(x), file=fpt)
        return
    
        
    def set_datasets(self, train_path, test_path):
        self.train_data = get_data(train_path)
        self.test_data = get_data(test_path)
        return
        
    def set_data_text(self):
        self.train_text = get_text_data(self.train_data)
        self.test_text = get_text_data(self.test_data)
        return

    def get_data_by_class(self, label):
        y = self.le.transform([x['klass'] for x in self.train_data])
        match = np.array(self.train_data)[np.array([matches for matches in [d['klass']==label for d in self.train_data]])]
        return match

    
    
    def get_sentence_list(self, set="train"):
        data=[]
        sentence_list=[]
        if set == "train":
            data = self.train_data
        elif set == "test":
            data = self.test_data
        else:
            print("Error, expected set={train|test}. Please call with correct parameters")
        for sen_dict in data:
           sentence_list.append(sen_dict['text']) 
        return sentence_list
    
def run_hw4():
    dm = ModelBuilder(train_path="data/semeval2017_En_train.json",test_path="data/test.json")
    print("Building Encoder")
    dm.build_le()
    print("Building Text Model")
    dm.build_tm()
    print("Building SVC")
    dm.build_svc()
    print("Predicting")
    dm.predict()
    print("Writing Predictions")
    dm.write_predictions() 

run_hw4()
