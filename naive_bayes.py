import numpy as np
import pandas as pd
from itertools import combinations

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return np.array(new_data).T

def create_test_submission(filename, prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    f = open(filename, 'w')
    f.write('\n'.join(content))
    f.close()
    print 'Saved'

def extract_counts(L):
    """
    Take a 1D numpy array as input and return a dict mapping values to counts
    """
    uniques = set(list(L))
    counts = dict((u, np.sum(L==u)) for u in uniques)
    return counts 
    
class NaiveBayesClassifier(object):
    """
    Naive Bayes Classifier with additive smoothing
    
    Params
        :alpha - hyperparameter for additive smoothing
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __repr__(self):
        return 'NaiveBayesClassifier(alpha=%.1e)' % (self.alpha)
    
    def fit(self, X, y):
        """
        Trains Naive Bayes Classifier on data X with labels y
        
        Input
            :X - numpy.array with shape (num_points, num_features)
            :y - numpy.array with shape (num_points, )
        
        Sets attributes
            :pos_prior - estimate of prior probability of label 1
            :neg_prior - estimate of prior probability of label 0
        """
        self._pos_counts = [extract_counts(L) for L in X[y==1].T]
        self._neg_counts = [extract_counts(L) for L in X[y==0].T]
        self._total_pos = float(sum(y==1))
        self._total_neg = float(sum(y==0))
        total = self._total_pos + self._total_neg
        self.pos_prior = self._total_pos / total
        self.neg_prior = self._total_neg / total
         
    def log_predict(self, X):
        """
        Returns log ((P(c=1) / P(c=0)) * prod_i P(x_i | c=1) / P(x_i | c=0))
        using additive smoothing
        
        Input
            :X - numpy.array with shape (num_points, num_features)
                 num_features must be the same as data used to fit model
        """
        m,n = X.shape
        if n != len(self._pos_counts):
            raise Error('Dimension mismatch: expected %i features, got %i' % (
                         len(self._pos_counts), n))
        alpha = self.alpha
        tot_neg = self._total_neg
        tot_pos = self._total_pos
        preds = np.zeros(m)
        for i, xi in enumerate(X):
            Pxi_neg = np.zeros(n)
            Pxi_pos = np.zeros(n)
            for j, v in enumerate(xi):
                nc = self._neg_counts[j].get(v,0)
                pc = self._pos_counts[j].get(v,0)
                nneg = len(self._neg_counts[j])
                npos = len(self._pos_counts[j])
                # Compute probabilities with additive smoothing
                Pxi_neg[j] = (nc + alpha) / (tot_neg + alpha * nneg)
                Pxi_pos[j] = (pc + alpha) / (tot_pos + alpha * npos)
            # Compute log pos / neg class ratio
            preds[i] = np.log(self.pos_prior) + np.sum(np.log(Pxi_pos)) - \
                       np.log(self.neg_prior) - np.sum(np.log(Pxi_neg))
        return preds 

    def predict(self, X, cutoff=0):
        """
        Returns predicted binary classes for data with decision boundry given
        by cutoff 
        
        Input
            :X - see NaiveBayesClassifier.log_predict 
            :cutoff - decision boundry for log predictions 
        """
        preds = self.log_predict(X)
        print(preds)
        ma=max(preds)
        mi=min(preds)
        print(mi)
        for i in range(len(preds)):
            preds[i]=(preds[i]-mi)/(ma-mi)
        return (preds)
    
def main(train_file='train.csv', test_file='test.csv', output_file='nb_predict.csv'):
    # Load data
    print 'Loading data...'
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    y = np.array(train_data.ACTION)
    X = np.array(train_data.ix[:,1:-1])     # Ignores ACTION, ROLE_CODE
    X_test = np.array(test_data.ix[:,1:-1]) # Ignores ID, ROLE_CODE
    
    # Convert features to triples
    print 'Transforming data...'
    X = group_data(X, degree=2)
    X_test = group_data(X_test, degree=2)
    model = NaiveBayesClassifier(alpha=1e-2)
    
    # Train model 
    print 'Training Naive Bayes Classifier...'
    model.fit(X, y)
    
    # Make prediction on test set
    print 'Predicting on test set...'
    
    preds = model.predict(X_test,170)
    
    print 'Writing predictions to %s...' % (output_file)
    create_test_submission(output_file, preds)

    return model, X, y, X_test
    
if __name__=='__main__':
    args = { 'train_file':  'train.csv',
             'test_file':   'test.csv',
             'output_file': 'nb_predict.csv' }
    model, X, y, X_test = main(**args)
