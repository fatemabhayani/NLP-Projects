import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  
import numpy as np


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    diagonal = C.trace()
    sum_C = C.sum()
    return diagonal / sum_C

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    true_positives = np.diag(C) 
    return true_positives / np.sum(C, axis=1)
def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    true_positives = np.diag(C) 
    return true_positives / np.sum(C, axis=0)


SDG = SGDClassifier()
GNB = GaussianNB()
RFC = RandomForestClassifier(n_estimators=10, max_depth=5)
MLP = MLPClassifier(alpha=0.05)
ABC = AdaBoostClassifier()
models = [SDG, GNB, RFC, MLP, ABC]

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    conf_matrix = [] #matrix in a list
    acc = []
    rec = [] # list ina list
    pre = []# list ina list
    iBest = 0
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i, m in enumerate(models):
        
            outf.write(f'Results for {m.__class__.__name__}:\n')
            m.fit(X_train, y_train)
            y_p = m.predict(X_test) #predicted y
            C = confusion_matrix(y_test, y_p)
            conf_matrix.append(C)
            acc.append(accuracy(C))
        
            outf.write(f'\tAccuracy: {acc[i]:.4f}\n')
            rec.append(recall(C))
            outf.write(f'\tRecall: {[round(item, 4) for item in rec[i]]}\n')
            pre.append(precision(C))
        
            outf.write(f'\tPrecision: {[round(item, 4) for item in pre[i]]}\n')
        
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')
            if i > 0 and acc[i - 1] < acc[i]:
                iBest = i   

    return iBest + 1



def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    '''
    best = models[iBest - 1]
    accuracy_l = []
    size = [1000, 5000, 10000, 20000]
    for s in size:
        ix = np.random.choice(a=X_train.shape[0], size=s)

        X_l = X_train[ix]
        y_l = y_train[ix]

        best.fit(X_l, y_l)
        y_p = best.predict(X_test)
        C = confusion_matrix(y_test, y_p)
        accuracy_l.append(accuracy(C))
        if s == 1000:
            X_1k = X_l
            y_1k = y_l
    
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        for i, s in enumerate(size):
            outf.write(f'{s}: {accuracy_l[i]:.4f}\n')
    
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    best = models[i - 1]
    p_val = []
    
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        for k in [5, 50]:
            selector = SelectKBest(f_classif, k=k)
            X_new = selector.fit_transform(X_train, y_train)
            ix = selector.pvalues_
            ix = ix.argsort()
            p_values = selector.pvalues_[ix]
            p_val.append(p_values)
        # for each number of features k_feat, write the p-values for
        # that number of features:
            outf.write(f'{k} p-values: {[round(pval, 4) for pval in p_values]}\n')
            ix_32k = selector.pvalues_.argsort()[:k]
            outf.write(f'{k} best features: {ix_32k}\n')
            if k == 5:
                # 1k
                X_new5 = selector.fit_transform(X_1k, y_1k)
                best.fit(X_new5, y_1k)
                y_predict1 = best.predict(selector.transform(X_test))
                C = confusion_matrix(y_test, y_predict1)
                accuracy_1k = accuracy(C)
                outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
                
                ix_1k = selector.pvalues_.argsort()[:k]
                # 32 K
                X_new32 = selector.fit_transform(X_train, y_train)
                best.fit(X_new32, y_train)
                y_predict32 = best.predict(selector.transform(X_test))
                C = confusion_matrix(y_test, y_predict32)
                accuracy_full = accuracy(C)
                outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
                
                ix_32k = selector.pvalues_.argsort()[:k]
                print(ix_1k)
                print(ix_32k)
                feature_intersection = set(ix_1k) & set(ix_32k)
                outf.write(f'Chosen feature intersection: {feature_intersection}\n')
                outf.write(f'Top-5 at higher: {ix_32k}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    X = np.vstack((X_train, X_test))   
    y = np.concatenate((y_train, y_test), axis=0)
    kf = KFold(5,True, 401)
    acc_model = [] # list of accuracy across 5 folds inside this list
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        for i, model in enumerate(models):
            acc_fold = []
            for train, test in kf.split(X):
                X_train = X[train]
                X_test = X[test]
                y_train = y[train]
                y_test = y[test]
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                C = confusion_matrix(y_test, y_predict)
                acc_fold.append(accuracy(C))
            acc_model.append(acc_fold)
        mean_accuracy = []
        for x in acc_model:
            mean_accuracy.append(np.mean(x))
        mod1 = [acc_model[0][0], acc_model[1][0], acc_model[2][0], acc_model[3][0], acc_model[4][0]]
        mod2 = [acc_model[0][1], acc_model[1][1], acc_model[2][1], acc_model[3][1], acc_model[4][1]]
        mod3 = [acc_model[0][2], acc_model[1][2], acc_model[2][2], acc_model[3][2], acc_model[4][2]]
        mod4 = [acc_model[0][3], acc_model[1][3], acc_model[2][3], acc_model[3][3], acc_model[4][3]]
        mod5 = [acc_model[0][4], acc_model[1][4], acc_model[2][4], acc_model[3][4], acc_model[4][4]]
        mod =[mod1, mod2, mod3, mod4, mod5]
        for x in mod:
            outf.write(f'Kfold accuracies: {[round(acc, 4) for acc in x]}\n')
        p_values = []
        for a, model in enumerate(models):
            if not np.array_equal(models[i-1], model):
                S = ttest_rel(acc_model[a], acc_model[i - 1])
                p_values.append(S.pvalue)
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')

      
def main(args):
    feats = np.load(args.input)["arr_0"]     
    X = feats[:,0:172]                             
    y = feats[:,173].ravel()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        train_size=0.8, shuffle=True, random_state=401)
    iBest = class31(args.output_dir, x_train, x_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, x_train, x_test, y_train, y_test, iBest)
    class33(args.output_dir, x_train, x_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, x_train, x_test, y_train, y_test, iBest)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    main(args)
    # TODO : complete each classification experiment, in sequence.
