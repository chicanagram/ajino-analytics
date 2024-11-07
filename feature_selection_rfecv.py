#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:37:55 2024

@author: charmainechia
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from variables import model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key, xvar_sublist_sets_bymodeltype
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_feature_importance_barplots_bymodel, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
from get_datasets import data_folder, get_XYdata_for_featureset



#%%
featureset_list =  [(1,0)]
models_to_eval_list =  ['randomforest'] # ['randomforest','plsr', 'lasso'] #  
dataset_suffix = ''
f = 1
 
# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_name_wsuffix = dataset_name + dataset_suffix
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    X_df = pd.DataFrame(X, columns=xvar_list)
    print(f'X dataset size: n={X.shape[0]}, p={X.shape[1]}')
        
    # iterate through yvar
    for i, yvar in enumerate(yvar_list_key):     
        print(yvar)
        
        # iterate through model types 
        for model_type in models_to_eval_list: 
            print(model_type)
            
            model_dict = model_params[dataset_name_wsuffix][model_type][yvar][0]
            if model_type=='plsr':
                n_components =  min(model_dict['n_components'], X.shape[1])
                model = PLSRegression(n_components=n_components)
            elif model_type=='lasso':
                max_iter = model_dict['max_iter']
                alpha = model_dict['alpha']
                model = Lasso(max_iter=max_iter, alpha=alpha)
            elif model_type=='randomforest':
                n_estimators = model_dict['n_estimators']
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
            
            y = Y[:,i]
            selector = RFECV(model, step=1, cv=5, scoring='r2')
            selector = selector.fit(X, y)
            print(f'Optimal number of features: {selector.n_features_}')
            print(f'CV score: {round(selector.cv_results_["mean_test_score"][selector.n_features_],2)}')
            print(*[xvar for i, xvar in enumerate(xvar_list) if selector.support_[i]], sep='\n')
            print()
            
            plt.plot(range(1,len(selector.cv_results_['mean_test_score'])+1), selector.cv_results_['mean_test_score'])
            plt.fill_between(range(1,len(selector.cv_results_['mean_test_score'])+1), selector.cv_results_['mean_test_score']-selector.cv_results_['std_test_score'], selector.cv_results_['mean_test_score']+selector.cv_results_['std_test_score'], alpha=0.3)
            plt.xlabel('Number of features selected')
            plt.ylabel('Cross-validation score')
            plt.title(f'[{yvar}] Recursive Feature Elimination with Cross-Validation')
            plt.show()
    
    
    