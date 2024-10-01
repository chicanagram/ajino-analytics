#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:03:36 2024

@author: charmainechia
"""
import numpy as np
import pandas as pd
from variables import model_params, yvar_sublist_sets, sort_list, yvar_list_key
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder
from get_datasets import data_folder, get_XYdata_for_featureset
from feature_selection_utils import get_feature_to_drop, plot_feature_selection_metrics

# Perform Recursive Feature Elimination
def run_rfe(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_rfe'):    
    
    feature_selection_method = 'rfe'
    p_init = X_init.shape[1]

    # initialize dataframe for recording results
    res_cols = ['model_type', 'yvar', 'num_features', 'num_features_dropped', 'r2_train', 'r2_cv', 'mae_train', 'mae_cv', 'mae_norm_train', 'mae_norm_cv', 'r2_over_mae_norm_train', 'r2_over_mae_norm_cv', 'xvar_to_drop', 'xvar_list']
    res = []
    featureset_opt = {yvar: {model_type: {featureset_suffix: None} for model_type in models_to_evaluate} for yvar in yvar_list}
    
    for model_type in models_to_evaluate:
        print(model_type)
        res_model = []
        
        for i, yvar in enumerate(yvar_list_key):
            # get y and model parameters
            y = Y[:,i]
            model_list = model_params[dataset_name][model_type][yvar].copy()
            res_model_yvar = []

            # get initial X variables
            X = X_init.copy()
            p = p_init
            xvar_list = xvar_list_init.copy()
            # fit model with full feature set
            model_list, metrics = fit_model_with_cv(X, y, yvar, model_list, plot_predictions=False, scale_data=True)
            # get feature to drop
            idx_to_drop, xvar_to_drop = get_feature_to_drop(model_list[0]['model'], xvar_list, model_type) 
            # update dataset
            metrics.update({'num_features_dropped':0, 'xvar_to_drop':xvar_to_drop}) 
            res_model_yvar.append(metrics)

            # iterate over number of features to eliminate
            for k in range(1, p_init): 

                # update Xby dropping selected feature
                p = X.shape[1]
                idx_to_keep = [idx for idx in range(p) if idx != idx_to_drop]
                X = X[:, np.array(idx_to_keep)]
                xvar_list = [xvar_list[idx] for idx in idx_to_keep]
                    
                # evaluate model with dropped feature
                model_list, metrics = fit_model_with_cv(X, y, yvar, model_list, plot_predictions=False)
                # get feature to drop
                idx_to_drop, xvar_to_drop = get_feature_to_drop(model_list[0]['model'], xvar_list, model_type)
                
                # update dataset
                metrics.update({
                    'num_features':p-1, 'num_features_dropped':k, 
                    'r2_over_mae_norm_train': metrics['r2_train']/metrics['mae_norm_train'], 
                    'r2_over_mae_norm_cv': metrics['r2_cv']/metrics['mae_norm_cv'], 
                    'xvar_to_drop':xvar_to_drop, 'xvar_list':', '.join(xvar_list)
                    }) 
                res_model_yvar.append(metrics)
                
                # update n_components for plsr models
                if model_type=='plsr': 
                    model_list[0]['n_components'] = 10
                    
            # select best feature for this model / yvar combination
            res_model_yvar_df = pd.DataFrame(res_model_yvar)[res_cols]
            res_opt = res_model_yvar_df.sort_values(by='r2_over_mae_norm_cv', ascending=False).iloc[0]
            xvar_list_opt = res_opt['xvar_list'].split(', ')
            featureset_opt[yvar][model_type][featureset_suffix] = xvar_list_opt
            r2_train, r2_cv, mae_train, mae_norm_train, mae_cv, mae_norm_cv = res_opt['r2_train'], res_opt['r2_cv'], res_opt['mae_train'], res_opt['mae_norm_train'], res_opt['mae_cv'], res_opt['mae_norm_cv']
            print()
            print(f'{model_type} <> {yvar}')
            print(f'Optimal feature set (p={len(xvar_list_opt)}):')
            print(*xvar_list_opt, sep=', ')
            print(f'R2:{r2_train}, R2 (CV):{r2_cv}, mae (train): {mae_train} ({round(mae_norm_train*100,1)}%), mae (CV):{mae_cv} ({round(mae_norm_cv*100,1)}%)')
            print()

            # update model results
            res_model += res_model_yvar
            
        # aggregate model results dataframe       
        res_model_df = pd.DataFrame(res_model)[res_cols]
        res_model_df = res_model_df[res_model_df.model_type==model_type]
        
        # plot results for model type
        figtitle = f'{model_type} {feature_selection_method} feature selection metrics'
        savefig = f'{figure_folder}feature_selection_{feature_selection_method}_{model_type}_{dataset_name}{dataset_suffix}{kfold_suffix}.png'
        plot_feature_selection_metrics(res_model_df, p_init, yvar_list, figtitle=figtitle, print_features_sequence=True, feature_selection_method=feature_selection_method, savefig=savefig)
        
        # save model results 
        res_model_df.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}_{model_type}.csv')
        
        # update overall results
        res += res_model
        
    # aggregate overall results dataframe
    res = pd.DataFrame(res)[res_cols]
    res.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}.csv')

    return res, featureset_opt

#%% 

# Perform Recursive Feature Elimination
featureset_list =  [(2,0)]
featureset_list =  [(2,0)]
models_to_evaluate = ['plsr'] #  ['randomforest', 'plsr'] # 

# load dataset
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_suffix = '' #'_avg'
    
    Y, X_init, Xscaled_init, yvar_list, xvar_list_init = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    res, featureset_opt = run_rfe(Y, X_init, yvar_list_key, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_rfe')
    
