from df_builder import build_df
import torch
from torch import nn
from torch.nn.functional import softmax
from fast_ml.model_development import train_valid_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, sentences=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        if bool(sentences):
            self.encodings = self.tokenizer(self.sentences,
                                            truncation = True,
                                            padding = True)
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        if self.labels == None:
            item['labels'] = None
        else:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.sentences)
    
    
    def encode(self, x):
        return self.tokenizer(x, return_tensors = 'pt').to(DEVICE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (f'Device Availble: {DEVICE}')
df = pd.DataFrame(build_df())
le = LabelEncoder()
df = df.filter(['Polar_expression', 'Polarity'])
df.Polarity = le.fit_transform(df.Polarity) 
print(df)
#data_dict = dict(df.to_dict()) #type: ignore 
#print(data_dict['text'])
(train_texts, train_labels,
 val_texts, val_labels,
 test_texts, test_labels) = train_valid_test_split(df, target = 'Polarity', train_size=0.8, valid_size=0.1, test_size=0.1)


train_texts = train_texts['Polar_expression'].to_list()
train_labels = train_labels.to_list()
val_texts = val_texts['Polar_expression'].to_list()
val_labels = val_labels.to_list()
test_texts = test_texts['Polar_expression'].to_list()
test_labels = test_labels.to_list()

