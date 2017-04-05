__doc__="""
***********************************  Task 3 ************************************
Analysis of Results

"""
import sys
import scipy as sc
sys.path.append('../Task1/')
sys.path.append('../Task2/')
sys.path.append('../')
import numpy as np
from improved import __doc__ as doc_imp
from baseline import __doc__ as doc_bas
from baseline import KL, categories
from itertools import combinations as combs
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from utils import reportGS, VoteClassifier

class resultsParser:
    
    def __init__(self,fname='baseline'):
        self.base = dict(np.load(fname+'_confusion.npz')['arr_0'].item())
        self.base_grids = dict(np.load(fname+'_optimised_classifiers.npz')['arr_0'].item())
        k = []
        
        for i in self.base:
            if self.base[i]==[]:
                k.append(i)
        
        for i in k:
            self.base.pop(i)
        
        voted_classifier = VoteClassifier(*[self.base_grids[i] for i in self.base_grids])
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics', 
            'sci.space'
        ]
        data = fetch_20newsgroups(subset='all', categories=categories)
        self.base['vote']=[]
        kf = ShuffleSplit(n_splits=10, test_size=0.5, train_size=None, random_state=8101988)
        count = 0
        print('Evaluating the voted classifier for %s, this may take a while... \n ['%fname, end=' ', flush=True)
        for train_index, test_index in kf.split(data.data):
            count += 1
            X_test = np.asarray(data.data)[test_index]
            y_test = np.asarray(data.target)[test_index]
            print(str(count*10)+'%', end=' ', flush=True)
            self.base['vote'].append(confusion_matrix(y_test,voted_classifier.predict(X_test)))
        print('] done!')
        self.F_conf = lambda f: {i:f(np.asarray(self.base[i])) for i in self.base}
        self.F_ps = lambda f: {i+'vs'+j:f(np.asarray(self.base[i]).flatten(),
                                        np.asarray(self.base[j]).flatten())[1] 
                                        for (i,j) in combs(self.base.keys(),2)}
        #TODO: errors = {i:np.asarray(baseline[i]).sum(0)/10. for i in baseline.keys()}
        

    def conf_matrix(self,f=lambda x:np.mean(x,0)):
        confs = self.F_conf(f)
        print('\n\n%s of confusion matrices over folds:'%(f.__name__))
        for i in self.base:
            print(i)
            for j in confs[i]:
                line = ''
                for k in j:
                    line+= str(int(round(k)))+'\t'
                print(line)
            print()
        
    def abs_acc(self,f=lambda x:np.mean(x,0)):
        confs = self.F_conf(f)
        l = [len(i) for i in confs]
        m_l = max(l)
        print('\n\n%s absolute accuracy over folds:'%f.__name__)
        line=str(m_l*' ')+'\t'
        for i in categories:
            line+=i+'\t'
        for i in self.base:
            line+='\n'+i+str(' '*(m_l-len(i)))+'\t'
            for j in range(len(categories)):
                val = '%.3f'%(100*confs[i][j][j]/sum(confs[i][j]))
                line+=val+str(' '*(len(categories[j])-len(val)))+'\t'
        print(line)

    def sig_test(self,f=sc.stats.ks_2samp):
        ps = self.F_ps(f)
        lp = [len(i) for i in ps]
        m_lp = max(lp)
        print('\n\nSignificance testing of the confusion matrix(%s, both tails <.05, >.95)'%f.__name__+
                '\nover differrent classifiers:')
        for i in ps:
            if ps[i]<.05 or ps[i]>0.95:
                print(i,':'+str(' '*(m_lp-len(i)+1)),ps[i])
    
    def report_gs_results(self):
        print('\n\nGrid search results:')
        for i in self.base_grids:
            print(i)
            reportGS(self.base_grids[i].cv_results_)
    
    def report(self):
        def mean(x):return np.mean(x,0)
        def median(x):return np.median(x,0)
        def sd(x):return np.std(x,0)
        def KS(x,y):return sc.stats.ks_2samp(x,y)
        self.report_gs_results()
        for i in [mean, median, sd]:
            self.conf_matrix(f=i)
            if i is not sd:
                self.abs_acc(f=i)
        try:
            self.sig_test(f=KS)
        except:
            print('Failed to conduct significane testing!')
            
if __name__=='__main__':
    print(doc_bas)
    print(doc_imp)
    print(__doc__)
    res_base = resultsParser(fname='../Task1/baseline')
    res_imp = resultsParser(fname='../Task2/improved')
    print('*'*80,'\nBASELINE\n')
    res_base.report()
    print('*'*80,'\nIMPROVED\n')
    res_imp.report()