
    def get_embed_model(self):
        nlp = spacy.load("en_core_web_lg")
        return nlp
    def tag_sentence(self):
        return [make_txt(sentence) for sentence in self.df.Text]
            
    def word_embedding(self, token):
        return self.nlp(token).vector
    
    def embedding_mean(self, sentence_embed):
        result = [0 for i in range(len(sentence_embed[0]))]

        for i in range(len(sentence_embed[0])):
            summ = 0
            for j in range(len(sentence_embed)):
                summ += sentence_embed[j][i]
            result[i] = summ/len(sentence_embed)
        return result
    
    def sentence_embedding(self, sentence):
        sentence_embed = []
        for token in self.nlp(sentence):
            if np.fabs(self.nlp.vocab[token.text].vector).sum() > 0:
                sentence_embed.append(word_embedding(token.text))
        if sentence_embed:
            return embedding_mean(sentence_embed)

        return None


