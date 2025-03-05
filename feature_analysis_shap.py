#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 21:09:54 2024

@author: charmainechia
"""

# Importing Required Libraries
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import shap
from variables import model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key, feature_selections
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_feature_importance_barplots_bymodel, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
from get_datasets import data_folder, get_XYdata_for_featureset


def get_filtered_Xdata(X_trainval, X_test, featureset_suffix, featureset_dict, xvar_list_all, yvar, model_type=None):
    if model_type is None: 
        xvar_list_ = featureset_dict[featureset_suffix][yvar]
    else:
        xvar_list_ = featureset_dict[yvar][model_type][featureset_suffix]
    print(yvar, xvar_list_)
    xvar_idxs_selected = np.array([idx for idx, xvar in enumerate(xvar_list_all) if xvar in xvar_list_])
    X_trainval_ = X_trainval[:, xvar_idxs_selected]
    X_test_ = X_test[:, xvar_idxs_selected]
    print(f'X_trainval size after feature selection: n={X_trainval_.shape[0]}, p={X_trainval_.shape[1]}')       
    return xvar_list_, X_trainval_, X_test_  

#%%
featureset_list =  [(1,0)]
models_to_eval_list = ['randomforest'] # ['plsr']# ['randomforest','plsr', 'lasso'] #  
featureset_suffix = '' # '_ajinovalidation3' #  
dataset_suffix = '_norm_with_val_unshuffled' #'' #   '_norm' # '_avgnorm' # 
yvar_list = yvar_list_key
f = 1

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_name_wsuffix = dataset_name + dataset_suffix
    Y, X, Xscaled, _, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    
    X_df = pd.DataFrame(X, columns=xvar_list)
    print(f'X dataset size: n={X.shape[0]}, p={X.shape[1]}') 
         
    # iterate through model types 
    for model_type in models_to_eval_list: 
        shap_summary = pd.DataFrame(columns=['model_type', 'yvar']+xvar_list, index=list(range(len(models_to_eval_list)*len(yvar_list))))
        idx = 0   
        
        # iterate through yvar
        for i, yvar in enumerate(yvar_list):  
            
            if featureset_suffix!='':
                xvar_list_, X_, X_  = get_filtered_Xdata(X, X, featureset_suffix, feature_selections, xvar_list, yvar, model_type=None)
                X_df_ = X_df.loc[:,[xvar for xvar in xvar_list_ if xvar in X_df]]
                xvar_list_ = X_df_.columns.tolist()
            else: 
                X_ = X.copy()
                X_df_ = X_df.copy()
                xvar_list_ = xvar_list.copy()
        
            
            print(model_type)
            shap_summary_yvar_modeltype = []
            
            model_dict = model_params[dataset_name_wsuffix][model_type][yvar][0]
            if model_type=='plsr':
                n_components =  min(model_dict['n_components'], X_.shape[1])
                model = PLSRegression(n_components=n_components)
            elif model_type=='lasso':
                max_iter = model_dict['max_iter']
                alpha = model_dict['alpha']
                model = Lasso(max_iter=max_iter, alpha=alpha)
            elif model_type=='randomforest':
                n_estimators = model_dict['n_estimators']
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
            elif model_type=='xgb':
                n_estimators = model_dict['n_estimators']
                model = XGBRegressor(n_estimators=n_estimators, random_state=0)
            
            y = Y[:,i]
            model.fit(X_,y)
            ypred = model.predict(X_)
            if model_type in ['randomforest', 'xgb']:
                explainer = shap.TreeExplainer(model, X_df_, feature_perturbation='interventional')
            elif model_type in ['plsr', 'lasso']:
                explainer = shap.LinearExplainer(model, X_df_)
                
            shap_values = explainer(X_, check_additivity=False)
            # shap.plots.bar(shap_values.abs.mean(0), max_display=20)
            # shap.plots.beeswarm(shap_values, max_display=20
            shap.summary_plot(shap_values, max_display=20, feature_names=X_df_.columns.tolist())

            # save SHAP values
            shap_values_df = pd.DataFrame(shap_values.values, columns=xvar_list_)
            shap_values_df.to_csv(f'{data_folder}feature_analysis_SHAPvalues_{dataset_name}{dataset_suffix}{featureset_suffix}_{model_type}_{i}.csv')
            
            # save SHAP summary based on mean absolute shap values
            shap_summary.loc[idx, ['model_type', 'yvar']] = [model_type, yvar]
            shap_summary.loc[idx, xvar_list_] = shap_values.abs.mean(0).values
            idx += 1
    
    shap_summary.to_csv(f'{data_folder}feature_analysis_SHAPsummary_{dataset_name}{dataset_suffix}{featureset_suffix}_{model_type}.csv')
    
    
