import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
sw = stopwords.words("english")
from sklearn.model_selection import train_test_split

wordnet_lemmatizer = WordNetLemmatizer()

class CountVectorizer:
    def __init__(self):
        self.word_index_map = {}
        self.tokenized_data = []

    def tokenizer(self, data):
        for s in data:
            s = s.lower()
            tokens = nltk.tokenize.word_tokenize(s)
            tokens = [t for t in tokens if len(t) > 2]  
            tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
            tokens = [t for t in tokens if t not in sw]
            self.tokenized_data.append(tokens)

    def create_word_index_map(self):
        index = 0
        word_index_map = {}
        for t in self.tokenized_data:
            for token in t:
                if token not in word_index_map:
                    word_index_map[token] = index
                    index += 1
        self.vector_size = len(word_index_map)
        self.word_index_map = word_index_map
    
    def vectorize_data(self):
        vectorized_list = []
        for tl in self.tokenized_data:
            vect = np.zeros(len(self.word_index_map))
            for token in tl:
                i = self.word_index_map[token]
                vect[i] += 1
            vectorized_list.append(vect)
        self.vectorized_list = vectorized_list

    def fit_transform(self, data):
        self.tokenizer(data)
        self.create_word_index_map()
        self.vectorize_data()
        return self.vectorized_list


