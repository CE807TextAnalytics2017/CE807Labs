"""
*******************************************************************************
********* CE807 - Laboratory Sheet 1, Text pre-processing with Python *********
*******************************************************************************

Basic script from Massimo Poesio, edited by Dimitrios Andreou.

Changes made:
1) Included direct nltk downloads for missing modules within a try-except.
2) Added print statements so that the the script produces output.
3) Changed strings into Python3 strings, i.e. unicode.
4) Removed recurring imports.

Changes that could be made if intended to run as a script:
1) Re-factor and organize multiple imports and variables.
2) Could use pylab instead of matplotlib (ease of plotting).

********************************* SCRIPT OUTPUT *******************************
[u'disk', u'format', u'hard', u'how', u'my', u'problems', u'to']
[[1 1 1 1 1 0 1]
 [1 1 1 0 0 1 0]]
[1 1 1 1 1 0 1]
1
No handlers could be found for logger "sklearn.datasets.twenty_newsgroups"
4690
35788
['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst']
cat
Installing punkt.
[nltk_data] Downloading package punkt to
[nltk_data]     /ufs/servc00/users/dandreb/nltk_data...
[nltk_data]   Unzipping tokenizers/punkt.zip.
Installing dependencies and maxent_treebank_pos_tagger.
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /ufs/servc00/users/dandreb/nltk_data...
[nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
[nltk_data] Downloading package maxent_treebank_pos_tagger to
[nltk_data]     /ufs/servc00/users/dandreb/nltk_data...
[nltk_data]   Unzipping taggers/maxent_treebank_pos_tagger.zip.
[(u'And', 'CC'), (u'now', 'RB'), (u'for', 'IN'), (u'something', 'NN'), (u'completely', 'RB'), (u'different', 'JJ')]
[u'john', u'bought', u'carrots', u'potatoes']
john
bought
carrot
potato
26889
*******************************************************************************
"""

# Python3 version of strings, defaults to unicode.
from __future__ import unicode_literals
import sklearn
import numpy as np
import matplotlib.pyplot as plt

data = np.asarray([[1,2],[2,3],[3,4],[4,5],[5,6]])
x, y = data[:,0], data[:,1]
# Could introduce plt.ion()/ioff(), so that one could see
# interactively what is happening (also does not stop the
# execution of the code if plt.draw() is called and not plt.show()).
plt.scatter(x,y)
plt.grid(True)
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)
content = ['How to format my hard disk', ' Hard disk format problems ']
X = vectorizer.fit_transform(content)
print vectorizer.get_feature_names()
print X.toarray()
print X.toarray()[0]
print X.toarray()[1,2]

from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                   shuffle=True, random_state=42)
vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(twenty_train.data)
# A"u" is placed before quotes, of any type (','',''',","",""",'",...),
# to tell you that the string is unicode; Python 3 switched to using 
# unicode in strings (hence the __future__ import).
# Here is some further info (section 3.3): http://www.nltk.org/book/ch03.html
# And if not enough here is something bigger from the CEO of StackOverflow:
# https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/
print vectorizer.vocabulary_.get('algorithm')
print len(vectorizer.get_feature_names())
vectorizer = CountVectorizer(stop_words='english')
print sorted(vectorizer.get_stop_words())[:20]

import nltk

s = nltk.stem.SnowballStemmer('english')
print s.stem('cats')

from nltk.tokenize import word_tokenize

try:
    text = word_tokenize('And now for something completely different')
except(LookupError):
    print 'Installing punkt.'
    nltk.download('punkt')
    text = word_tokenize('And now for something completely different')

try:
    print nltk.pos_tag(text)
except(LookupError):
    print 'Installing dependencies and maxent_treebank_pos_tagger.'
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_treebank_pos_tagger')
    print nltk.pos_tag(text)

vectorizer = CountVectorizer(stop_words='english')
analyze = vectorizer.build_analyzer()
print analyze('John bought carrots and potatoes')

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        # Both "()" and "[]" work in the lambda statement below; the former
        # creates a generator object while the latter a list i.e., one can
        # be printed using "print XXX" while the other needs to access
        # each element using a loop, as you have done in {*1} below,
        # "for i in XXX:print i".
        #
	# The essential difference lies in the fact that the latter method
	# will construct the whole list in the memory, whereas with a
	# generator, one is more 'lazy' and constructs only one member of the
	# list at a time, thus saving memory.
	#
	# Obviously if the generator is to be used multiple times,
	# constructing the list and keeping it in memory will be more
	# efficient.
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))

stem_vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
stem_analyze = stem_vectorizer.build_analyzer()
Y = stem_analyze('John bought carrots and potatoes')

for tok in Y: # {*1}
    print tok

train_counts = stem_vectorizer.fit_transform(twenty_train.data)
print len(stem_vectorizer.get_feature_names())


