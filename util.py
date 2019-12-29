import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn import preprocessing as pre
import json
import os
import time

# load in and preprocess our data
class preprocessor:
    
    def __init__(self):
        self.cols = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 
            'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
            'native-country', 'label']
        self.feats = pd.read_csv('data/train.csv', header=None, names=self.cols)
        self.rmmissing()
        self.labels = self.feats['label']
        del self.feats['label']
        self.ordinalencode()
        self.labelencode()
        
    @property
    def data(self):
        return self.feats, self.labels

    def rmmissing(self):
        ''' just drop all rows w/ a missing value '''
        for c in self.feats.columns:
            self.feats = self.feats[self.feats[c] != ' ?']
            
    def ordinalencode(self):
        ''' encode features '''
        oenc = pre.OrdinalEncoder()
        self.feats = oenc.fit(self.feats).transform(self.feats)
        self.feats = pd.DataFrame(self.feats, columns=self.cols[:-1])
        
    def labelencode(self):
        ''' encode labels '''
        self.labels = (self.labels == ' <=50K').astype(int)
        
# functions to split data
class splitfuncs:
    
    def split(x, y, tr_perc=0.75):
        ''' take first N*tr_perc samples to train on '''
        k = int(x.index.size*tr_perc)
        return x[:k], y[:k], x[k:], y[k:]

    def split2(x, y, tr_perc=0.75):
        ''' take random N*tr_perc samples to train on '''
        y.index = x.index
        loc = np.random.choice(x.index, size=int(x.index.size*tr_perc), replace=False)
        trx, try_ = x.loc[loc], y.loc[loc]
        return trx, try_, x.drop(loc, axis=0), y.drop(loc)
    
# functions to record metrics
class metrics:
    
    def acc(preds, real):
        ''' compute accuracy of predictions '''
        idx = preds == real
        return idx[idx].index.size / idx.index.size
        
# functions to save and load results while handling type conversions (float <--> str, ...)
class resultsmngr:

    def save(results, *args):
        '''
        write results to disk
        @params:
            results: dict, retval of testmodels()
            args[0]: str, filename to write to. defaults to current timestamp
        @return: 
            str, filename
        '''
        os.makedirs('results', exist_ok=True)
        fname = f"results/{int(time.time()) if not args else str(args[0])}.json"
        for m in results:
            results[m]['model'] = str(results[m]['model'])
            for i, p in enumerate(results[m]['params']):
                for j, k in enumerate(p[1]):
                    p[1][k] = str(p[1][k])
        with open(fname, 'w') as fp:
            json.dump(results, fp, indent=4)
        return fname

    def load(path):
        '''
        load in results from disk
        @params:
            path: str, path to results
        @return
            dict, results
        '''
        with open(f"results/{path.split('/')[-1]}", 'r') as fp:
            results = json.loads(fp.read())
        for m in results:
            for p in results[m]['params']:
                for k in p[1]:
                    p[1][k] = float(p[1][k])
        return results

# functions to analyze results
class analyzer:
    
    def bestparams(results):
        '''
        look thru results dict and get the best params
        for each model and their performance
        @params:
            results: dict, retval of testmodels()
        @return 
            dict, mapping models to performance
        '''
        best = {}
        for m in results:
            best_ = max(results[m]['params'], key=lambda x: x[1]['testing-accuracy'])
            best[m] = {
                'params': best_[0],
                'metrics': best_[1]
            }
        return best