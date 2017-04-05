__doc__="""
***********************************  Task 2 ************************************
Improvement methods

In this script, the pipeline makes usage of the FetureUnion module to combine
two features: only word counts without stop-words (POSFeature) and the 
frequency of stop-words (StopWordFeature). Both of these frequency-based 
features were then transformed by tfidf.

The grid search has a limmited ammount of parameters refering to the feature 
extraction methods in order to keep a basis for comparison. Regarding the 
optimisation of the classifiers, cross-validation folds and scoring functions
are kept the same as the baseline.py methods in order to maintain an even base 
for comparison.
"""
import sys
sys.path.append('../')
import numpy as np
import scipy as sc
from time import time
import logging
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from utils import StopWordFeature, POSFeature, PipelineShow
from improved_params import *

clfs = [
    MultinomialNB(),BernoulliNB(),LogisticRegression(),SGDClassifier(),SVC(), 
    LinearSVC(), NuSVC()
]

params = [
    MultinomialNB_params,BernoulliNB_params,LogisticRegression_params,
    SGDClassifier_params,SVC_params, LinearSVC_params, NuSVC_params
]

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics', 
    'sci.space'
]

def KL(x,y):
    return 1-sc.stats.entropy(np.asarray(x)+1,qk=np.asarray(y)+1,base=10)

if __name__ == "__main__":
    print(__doc__)
    print("Loading 20 newsgroups dataset for categories:")
    print(categories)
    
    data = fetch_20newsgroups(subset='all', categories=categories)
    
    print("%d documents" % len(data.filenames))
    print("%d categories:"%len(data.target_names))
    print("\n%s\t%s\t%s\t%s"%tuple(categories))
    print("%d\t\t%d\t\t\t%d\t\t%d"%tuple([data.target.tolist().count(i) for i in range(len(categories))]))
    print('Classifiers used:',[str(i).split('(')[0] for i in clfs])
    print("parameters:")
    
    optimised = {}
    confusion = {'voted':[]}
    
    for i in clfs:
        confusion[str(i).split('(')[0]]=[]
    
    kf = ShuffleSplit(n_splits=10, test_size=0.5, train_size=None, random_state=8101988)
    count = 0
    for train_index, test_index in kf.split(data.data):
        count+=1
        
        print("____Fold_"+str(count)+"_"*70)
        X_train, X_test = np.asarray(data.data)[train_index],np.asarray(data.data)[test_index]
        y_train, y_test = np.asarray(data.target)[train_index], np.asarray(data.target)[test_index]
        
        print('Train size:%d\tTest size:%d'%(len(X_train),len(X_test)))
        for (clf, param_grid) in zip(clfs,params):
            print("Performing grid search for",str(clf).split('(')[0])
            start = time()
        
            parameters = {
                'feats__pos__vect__max_df': [0.5],
                'feats__pos__vect__max_features': [2000],
                'feats__pos__vect__ngram_range': ((1, 1), (1, 2)), 
                'feats__pos__tfidf__use_idf': [True],
                'feats__pos__tfidf__norm': ['l1'],
                'feats__sw__vect__max_df': [0.5],
                'feats__sw__vect__max_features': [2000],
                'feats__sw__vect__ngram_range': ((1, 1), (1, 2)),
                'feats__sw__tfidf__use_idf': [True],
                'feats__sw__tfidf__norm': ['l1']
            }
            
            pipeline = Pipeline([
                ('feats', FeatureUnion([
                    ('pos', Pipeline([
                        ('wpos', POSFeature()),
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer())
                        #('print1', PipelineShow())
                    ])),
                    ('sw', Pipeline([
                        ('swe', StopWordFeature()),
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer())
                        #('print2', PipelineShow())
                    ])),
                  ])),
                ('clf', clf)
            ])
            
            for i in param_grid:
                parameters["clf__"+i]=param_grid[i]
            
            
            grid_search = GridSearchCV(pipeline, parameters, scoring=make_scorer(KL), n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            optimised[str(clf).split('(')[0]] = grid_search
            best_parameters = grid_search.best_estimator_.get_params()
            
            print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
                  % (time() - start, len(grid_search.cv_results_['params'])))
            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
        for i in optimised:
            confusion[str(i).split('(')[0]].append(confusion_matrix(y_test,optimised[i].predict(X_test)))

        for i in confusion:print(i+'\n'+str(confusion[i]))
    np.savez('improved_optimised_classifiers', optimised)
    np.savez('improved_confusion', confusion)