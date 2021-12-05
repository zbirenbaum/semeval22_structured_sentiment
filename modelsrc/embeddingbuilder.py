import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
#import spacy

from tag import make_tags_from_list, make_tags
class Tagger:
    def __init__(self):
        self.df = self.gen_df()
        #self.nlp = self.get_embed_model()
        self.text_list = self.df.Text.to_list()[1:] #skip label value
        self.tagged_text = self.tag_text_list()

    def tag_sentence(self, sentence):
        return [tup[0] for tup in make_tags(sentence)]

    def tag_text_list(self):
        return [self.tag_sentence(sentence) for sentence in self.text_list]

    def gen_df(self):
        df=pd.read_csv("data/data_clean1000.csv",encoding='ISO-8859-1',names=np.array(['Target','Ids','Date','Flag','User','Text']))
        return df

class Embedder:
    def __init__(self, tagged_words_matrix):
        self.tagged_words_matrix = tagged_words_matrix # each line is a list of tagged words from a sentence
        self.sentence_list = [' '.join(line) for line in tagged_words_matrix]
        self.nlp = spacy.load("en_core_web_lg")
        self.sentence_embeds = [self.get_sentence_embed(sentence) for sentence in self.sentence_list]
        
        # for token in self.nlp("hello my ?.sdf"):
        #     print(self.nlp.vocab[token.text].vector - token.vector) #these are the same
        #self.sentence_embed_list = [self.get_sentence_embed(sentence) for sentence in self.sentence_list]

    # def word_embedding(self, token):
    #     return self.nlp(token).vector
    def normalize_sen_len(self):
        self.longest = max([len(sentence) for sentence in self.sentence_embeds])
        self.pad_matrix = np.zeros(len(self.sentence_embeds[0][0])) #all words same len
        for sentence in self.sentence_embeds:
            sentence.extend([self.pad_matrix for _ in range(self.longest - len(sentence))])
        for sentence in self.sentence_embeds:
            print(len(sentence))

    def get_sentence_embed(self, sentence):
        sentence_embeds = []
        for token in self.nlp(sentence):
            if np.fabs(self.nlp.vocab[token.text].vector).sum() > 0: #type: ignore
            #if np.fabs(np.array(token.vector)).sum() > 0:
                sentence_embeds.append(np.array(token.vector))
        return sentence_embeds.copy()

tagger = Tagger()
tagged_words_matrix = tagger.tagged_text
embedder = Embedder(tagged_words_matrix)
embedder.normalize_sen_len()
#tagslist= embedder.tag_sentence()
#print(embedder.tag_sentence())


