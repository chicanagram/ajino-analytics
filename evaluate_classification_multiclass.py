#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:37:24 2025

@author: charmainechia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024

@author: charmainechia
"""
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 1000)
from variables import data_folder, sampling_rawdata_dict, xvar_list_dict, yvar_list_key
from model_utils import get_classifier_scoring, perform_mean_std_scaling

def sklearn_classifier(X_train, y_train, X_test, y_test, model_dict, class_labels,print_res=False, scale_data=True):

    # get scaled data
    if scale_data:
        mean, std, X_train_, X_test_ = perform_mean_std_scaling(X_train, X_test)
    else:
        X_train_ = X_train
        X_test_ = X_test

    # retrain model if not provided
    model_type = model_dict['model_type']
    yoffset = 0
    if model_type == 'pls':
        from sklearn.linear_model import PLSRegression
        param_name = 'n_components'
        param_val = 100 # model_dict[param_name]
        model = PLSRegression(n_components=param_val)
    elif model_type == 'ridge':
        from sklearn.linear_model import RidgeClassifier
        param_name = 'alpha'
        param_val = 1 # model_dict[param_name]
        model = RidgeClassifier(alpha=param_val)
    elif model_type == 'lasso':
        from sklearn.linear_model import LogisticRegression
        param_name = 'penalty'
        param_val = 'l1' # model_dict[param_name]
        model = LogisticRegression(max_iter=50000, penalty='l1', solver='liblinear')
    elif model_type == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        param_name = 'n_estimators'
        param_val = 100 # model_dict[param_name]
        model = RandomForestClassifier(n_estimators=param_val, random_state=0)
    elif model_type == 'xgb':
        from xgboost import XGBClassifier
        param_name = 'n_estimators'
        param_val = 100 # model_dict[param_name]
        model = XGBClassifier(objective="'multi:softprob'", n_estimators=param_val, random_state=0)
        yoffset = 1

    # get train results
    model.fit(X_train_, y_train+yoffset)
    ypred_train = model.predict(X_train_) - yoffset
    metrics_train = get_classifier_scoring(ypred_train, y_train, model_name=model_type, class_labels=class_labels, plot_roc=False)
    metrics_train.update({'train_or_test':'train'})

    # perform evaluation on test data
    ypred_test = model.predict(X_test_) - yoffset
    metrics_test = get_classifier_scoring(ypred_test, y_test, model_name=model_type, class_labels=class_labels, plot_roc=False)
    metrics_test.update({'train_or_test': 'test'})
    metrics = [metrics_train, metrics_test]

    if print_res:
        print(pd.DataFrame(metrics))

    return metrics, model, ypred_test

def fit_classifier_kfold(X, y, model_dict, class_labels, n_splits=5, scale_data=True):
    from sklearn.model_selection import KFold
    n = len(y)
    ypred = np.zeros((n,))
    metrics_kfold = []
    kFold = KFold(n_splits=n_splits, shuffle=False)
    for i, (train_index, test_index) in enumerate(kFold.split(X)):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        print(f'Split {i+1}/{n_splits}')
        metrics, _, ypred_test = sklearn_classifier(X_train, y_train, X_test, y_test, model_dict, class_labels, print_res=True, scale_data=scale_data)
        metrics_kfold += metrics
        ypred[test_index] = ypred_test
    return metrics_kfold, ypred


#%% 

dataset_name = 'X1Y0'
dataset_suffix = '_norm' # '' # '_norm_with_val-A' # '_avgnorm'
featureset_suffix =  '' #  '_ajinovalidation' # # '_sfs-backward'
yvar_suffix = '_3class'
shuffle_data = True
class_labels = [0,1,2]
n_splits = 5
metrics_list = ['Accuracy', 'F1-score', 'Precision', 'Recall', 'MCC', 'ROC-AUC']
res_cols = ['Featureset', 'p', 'Model', 'train_or_test']+metrics_list
models_to_eval_list = ['ridge']# ['randomforest', 'xgb']
yvar_list = yvar_list_key.copy()
yvar_list_wsuffix = [yvar+yvar_suffix for yvar in yvar_list]
xvar_list = xvar_list_dict[1]
thresholds = {
    'Titer (mg/L)_14': [0.7, 0.96],
    'mannosylation_14': [0.7, 0.96],
    'fucosylation_14': [1.02, 1.1],
    'galactosylation_14': [0.7, 0.96],
    }

# initialize results
res = []
count = 0

# load data and filter rows
print('Fetching labels...')
dfraw = pd.read_csv(data_folder + dataset_name + dataset_suffix + '.csv')

# threshold data to get classification labels
for yvar in yvar_list:
    print(yvar)
    dfraw[yvar+yvar_suffix] = 0
    thres_yvar = thresholds[yvar]
    for class_idx in range(1, len(thres_yvar)+1):
        lbnd = thres_yvar[class_idx-1]
        ubnd = thres_yvar[class_idx] if class_idx<len(thres_yvar) else None
        print(class_idx, lbnd, ubnd)
        if class_idx==1:
            dfraw.loc[(dfraw[yvar]>lbnd) & (dfraw[yvar]<=ubnd), yvar+yvar_suffix] = class_idx
        elif class_idx==2:
            dfraw.loc[(dfraw[yvar]>lbnd), yvar+yvar_suffix] = class_idx   
    labels = dfraw[yvar+yvar_suffix].to_numpy()
    unique, counts = np.unique(labels, return_counts=True)
    for i, label in enumerate(unique): 
        print(label, f'{round(counts[i]/len(dfraw)*100,2)}%')


# get X & y data

df = dfraw[xvar_list + yvar_list_wsuffix].dropna()
X = dfraw[xvar_list].to_numpy()

for i, ylabel in enumerate(yvar_list_wsuffix):
    print(ylabel)
    
    y = df[ylabel].to_numpy()
    num_features = len(xvar_list)
    print('X.shape:', X.shape)
    # shuffle data
    if shuffle_data:
        print('Shuffling the data...')
        shuffle_idx = np.arange(len(y))
        np.random.seed(seed=0)
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx, :]
        y = y[shuffle_idx]
    
    for k, model_type in enumerate(models_to_eval_list):
        print('MODEL TYPE:', model_type)
        model_dict = {'model_type': model_type}
    
        ##########################
        # perform classification #
        ##########################
        print('Fitting classifier on k-folds')
        metrics_kfold, ypred = fit_classifier_kfold(X, y, model_dict, class_labels, n_splits, scale_data=True)
    
        # get average of k-folds
        metrics_kfold = pd.DataFrame(metrics_kfold)
        metrics_kfold_summary = pd.DataFrame([
            metrics_kfold.loc[metrics_kfold.train_or_test == 'train', metrics_list].mean().to_dict(),
            metrics_kfold.loc[metrics_kfold.train_or_test == 'test', metrics_list].mean().to_dict()
        ])
    
        # append metadata cols
        metrics_kfold['Featureset'] = dataset_name+featureset_suffix
        metrics_kfold_summary['Featureset'] = dataset_name+featureset_suffix
        metrics_kfold['p'] = num_features
        metrics_kfold_summary['p'] = num_features
        metrics_kfold['Model'] = model_dict['model_type']
        metrics_kfold_summary['Model'] = model_dict['model_type']
        metrics_kfold_summary.insert(0, 'train_or_test', ['train','test'])
        metrics_kfold = metrics_kfold[res_cols]
        metrics_kfold_summary = metrics_kfold_summary[res_cols]
        print('K-Fold summary')
        print(metrics_kfold_summary)
    
        # update K-fold results
        if count==0:
            metrics_kfold_all = metrics_kfold.round(3).copy()
            metrics_kfold_summary_all = metrics_kfold_summary.round(3).copy()
        else:
            metrics_kfold_all = pd.concat([metrics_kfold_all, metrics_kfold.round(3)], axis=0)
            metrics_kfold_summary_all = pd.concat([metrics_kfold_summary_all, metrics_kfold_summary.round(3)], axis=0)
        count+=1
    
        ##################################
        # OVERALL CLASSIFICATION RESULTS #
        ##################################
        # get overall test results by combining predictions across all folds
        metrics = get_classifier_scoring(ypred, y, model_name=model_dict['model_type'], class_labels=class_labels, plot_roc=False)
        metrics.update({'Featureset':dataset_name+featureset_suffix, 'p':num_features, 'train_or_test':'test'})
        res.append(metrics)

# save overall results
res = pd.DataFrame(res)[res_cols]
res[metrics_list] = res[metrics_list].round(3)
res.to_csv(data_folder+'classification_overall.csv')
print(res)

# save kfold results
metrics_kfold_summary_all.to_csv(data_folder+'classification_kfold_summary.csv')
metrics_kfold_all.to_csv(data_folder + 'classification_kfold.csv')

