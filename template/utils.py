import string
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from statistics import mode
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
#import warnings
#warnings.simplefilter("ignore")
from sklearn.base import BaseEstimator, TransformerMixin


class StopWordFeature(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.stopwords = set(sw.words('english'))
        self.punct = set(string.punctuation)

    def fit(self, X, y=None):
        return self
     
    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        a= [list(self.extract_sw(doc)) for doc in X]
        return np.asarray([str(i) for i in a])
        
    def extract_sw(self, document):
        for sent in document.split('['):
            for tok in sent.split(','):
                tok = tok.lower()
                tok = tok.strip()
                tok = tok.strip(']')
                if tok in self.stopwords:
                    yield tok


class POSFeature(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.punct = set(string.punctuation)
        self.lem = WordNetLemmatizer()
        self.sw = set(sw.words('english'))

    def fit(self, X, y=None):
        return self
    
    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        a= [list(self.breakDoc(doc)) for doc in X]
        return np.asarray([str(i) for i in a])

    def breakDoc(self, document):
        for s in sent_tokenize(document):
            for tok, tag in pos_tag(wordpunct_tokenize(s)):
                tok = tok.lower()
                tok = tok.strip()
                if tok not in self.sw and not all(i in self.punct for i in tok):
                    l = self.tolem(tok, tag)
                    yield l

    def tolem(self, tok, tag):
        tag = {'N': wn.NOUN,'V': wn.VERB,'R': wn.ADV,'J': wn.ADJ}.get(tag[0], wn.NOUN)
        return self.lem.lemmatize(tok, tag)


class VoteClassifier:
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def predict(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(tuple(v))
        try:
            return mode(votes)
        except:
            return votes[np.random.randint(4)]

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.predict(features)
            votes.append(tuple(v))
        try:
            md = mode(votes)
        except:
            md = votes[np.random.randint(4)]
        choice_votes = votes.count(md)
        conf = choice_votes / len(votes)
        return conf
            
class PipelineShow(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
     
    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):    
        return X
        
def reportGS(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

