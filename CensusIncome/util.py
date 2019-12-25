import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn import preprocessing as pre

# load in and preprocess our data
class preprocessor:
    
    def __init__(self):
        self.cols = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 
            'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
            'native-country', 'label']
        self.feats = pd.read_csv('../data/train.csv', header=None, names=self.cols)
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
class splitfuncs():
    
    def split(x, y, tr_perc=0.75):
        ''' basic split '''
        k = int(x.index.size*tr_perc)
        return x[:k], y[:k], x[k:], y[k:]
    
# functions to record metrics
class metrics():
    
    def acc(preds, real):
        idx = preds == real
        return idx[idx].index.size / idx.index.size
        