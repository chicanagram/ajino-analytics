#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:52:31 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variables import model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
from get_datasets import data_folder, get_XYdata_for_featureset


def plot_feature_selection_metrics(res_model, p_init, yvar_list, figtitle=None, print_features_sequence=True, feature_selection_method='SFS', savefig=None):
    fig, ax = plt.subplots(3, len(yvar_list), figsize=(8*len(yvar_list),18))
    for j, yvar in enumerate(yvar_list):
        res_model_yvar = res_model[res_model.yvar==yvar]
        res_model_yvar = res_model_yvar.loc[(res_model_yvar.r2_train>=0) & (res_model_yvar.r2_cv>=0) & (res_model_yvar.mae_norm_cv<10)]
        num_features = res_model_yvar['num_features'].to_numpy()

        # plot R2
        if len(yvar_list)==1: 
            ax[0].plot(num_features, res_model_yvar['r2_train'].to_numpy(), linewidth=3, color='b')
            ax[0].plot(num_features, res_model_yvar['r2_cv'].to_numpy(), linewidth=3, color='r')
            # plot MAE
            ax[1].plot(num_features, res_model_yvar['mae_norm_train'].to_numpy(), linewidth=3, color='b')
            ax[1].plot(num_features, res_model_yvar['mae_norm_cv'].to_numpy(), linewidth=3, color='r')
            # plot R2 over MAE
            ax[2].plot(num_features, res_model_yvar['r2_over_mae_norm_train'].to_numpy(), linewidth=3, color='b')
            ax[2].plot(num_features, res_model_yvar['r2_over_mae_norm_cv'].to_numpy(), linewidth=3, color='r')
            # set titles and labels    
            ax[0].set_title(yvar, fontsize=20)
            for row_idx in range(3):
                ax[row_idx].set_xlabel('Number of features', fontsize=16)
                ax[row_idx].legend(['train', 'loocv'], fontsize=14)
            ax[0].set_ylabel('r2_train', fontsize=16)
            ax[1].set_ylabel('MAE (normalized)', fontsize=16)
            ax[2].set_ylabel('R2 over MAE (norm)', fontsize=16)
        else: 
            ax[0,j].plot(num_features, res_model_yvar['r2_train'].to_numpy(), linewidth=3, color='b')
            ax[0,j].plot(num_features, res_model_yvar['r2_cv'].to_numpy(), linewidth=3, color='r')
            # plot MAE
            ax[1,j].plot(num_features, res_model_yvar['mae_norm_train'].to_numpy(), linewidth=3, color='b')
            ax[1,j].plot(num_features, res_model_yvar['mae_norm_cv'].to_numpy(), linewidth=3, color='r')
            # plot R2 over MAE
            ax[2,j].plot(num_features, res_model_yvar['r2_over_mae_norm_train'].to_numpy(), linewidth=3, color='b')
            ax[2,j].plot(num_features, res_model_yvar['r2_over_mae_norm_cv'].to_numpy(), linewidth=3, color='r')
            # set titles and labels    
            ax[0,j].set_title(yvar, fontsize=20)
            for row_idx in range(3):
                ax[row_idx,j].set_xlabel('Number of features', fontsize=16)
                if j==0:
                    ax[row_idx,j].legend(['train', 'loocv'], fontsize=14)
            ax[0,j].set_ylabel('r2_train', fontsize=16)
            ax[1,j].set_ylabel('MAE (normalized)', fontsize=16)
            ax[2,j].set_ylabel('R2 over MAE (norm)', fontsize=16)
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'{figtitle}', y=ymax*1.06, fontsize=24)  
        
        # print sequence of features added
        if print_features_sequence:
            if feature_selection_method in ['rfe', 'sfs-backward']: 
                colname = 'xvar_to_drop'
            if feature_selection_method in ['sfs-forward']: 
                colname = 'xvar_to_add'
            xvar_seq = res_model_yvar[colname].to_list()
            print(yvar)
            for idx, xvar in enumerate(xvar_seq):
                print(f'{idx+1}: {xvar}')
            print()
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()
    
    
#%% SFS-forward

def run_sfs_forward(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_sfs-forward'):
        
    feature_selection_method='sfs-forward'
    p_init = X_init.shape[1]
    if xvar_idx_end is None:
        xvar_idx_end = p_init

    # initialize dataframe for recording results
    res_cols = ['model_type', 'yvar', 'num_features', 'r2_train', 'r2_cv', 'mae_train', 'mae_cv', 'mae_norm_train', 'mae_norm_cv', 'r2_over_mae_norm_train', 'r2_over_mae_norm_cv', 'xvar_to_add', 'xvar_list']
    res = []
    featureset_opt = {yvar: {model_type: {featureset_suffix: None} for model_type in models_to_evaluate} for yvar in yvar_list}
    
    for model_type in models_to_evaluate:
        res_model = []
        print(model_type)
        
        for i, yvar in enumerate(yvar_list):
            # get y and model parameters
            res_model_yvar = []
            y = Y[:,i]
            model_list = model_params[dataset_name][model_type][yvar].copy()

            # get initial X variables
            xvar_list_available = xvar_list_init.copy()
            xvar_list = []
            xvar_idxs = []

            # iterate over number of features to add
            for k in range(1, xvar_idx_end): 
                
                # iterate over all possible features that can be added
                model_metrics_df_ = []
                for xvar_idx, xvar in enumerate(xvar_list_available):
                    # print(xvar_idx, xvar, end='. ')
                    
                    # refresh temporary xvar_list with added xvar to test
                    xvar_list_, xvar_idxs_ = xvar_list.copy(), xvar_idxs.copy()
                    xvar_list_.append(xvar)
                    xvar_idxs_.append(xvar_list_init.index(xvar))
                    
                    # get dataset
                    X_ = X_init[:, np.array(xvar_idxs_)]
                    
                    # fit model with full feature set
                    model_list, metrics = fit_model_with_cv(X_, y, yvar, model_list, plot_predictions=False, print_output=False, scale_data=True)
                    metrics.update({'xvar_to_add':xvar, 'xvar_list':', '.join(xvar_list_), 'num_features':k})
                    model_metrics_df_.append(metrics)
                    
                # get feature to add by selecting model with the best metrics
                model_metrics_df_ = pd.DataFrame(model_metrics_df_)
                model_metrics_df_['r2_over_mae_norm_cv'] = model_metrics_df_['r2_cv']/model_metrics_df_['mae_norm_cv']
                model_metrics_df_['r2_over_mae_norm_train'] = model_metrics_df_['r2_train']/model_metrics_df_['mae_norm_train']
                metrics_to_add = model_metrics_df_.sort_values(by='r2_over_mae_norm_cv', ascending=False).iloc[0].to_dict()
                xvar_to_add = metrics_to_add['xvar_to_add']
                print(f'FEATURE {k} SELECTED: {xvar_to_add}')
                
                # update xvar_list_available and resuls
                xvar_list_available.remove(xvar_to_add)
                xvar_list.append(xvar_to_add)
                xvar_idxs.append(xvar_list_init.index(xvar_to_add))
                res_model_yvar.append(metrics_to_add)
                
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
            
            # update res_model_df
            res_model += res_model_yvar
            
        # aggregate model results dataframe
        res_model_df = pd.DataFrame(res_model)[res_cols]
        
        # plot results for model type
        figtitle = f'{model_type} {feature_selection_method} feature selection metrics'
        savefig = f'{figure_folder}feature_selection_{feature_selection_method}_{model_type}_{dataset_name}{dataset_suffix}{kfold_suffix}.png'
        plot_feature_selection_metrics(res_model_df, p_init, yvar_list, figtitle=figtitle, print_features_sequence=True, feature_selection_method=feature_selection_method, savefig=savefig)
        # save model results 
        res_model_df.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}_{model_type}{kfold_suffix}.csv')
        
    # aggregate overall results dataframe
    res += res_model
    res = pd.DataFrame(res)[res_cols]
    res.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}{kfold_suffix}.csv')
    
    return res, featureset_opt
    
#%% SFS-backward

def get_feature_to_drop(model, xvar_list, model_type):
    if model_type in ['plsr', 'lasso']: 
        coef_x_list = model.coef_.reshape(-1,)
    elif model_type in ['randomforest']: 
        coef_x_list = model.feature_importances_
    coef_x_list_abs = np.abs(coef_x_list)
    idx_to_drop = np.argmin(coef_x_list_abs)
    xvar_to_drop = xvar_list[idx_to_drop]
    print(f'p={len(coef_x_list_abs)}. Dropping feature {idx_to_drop} ({xvar_to_drop})')
    return idx_to_drop, xvar_to_drop

    
def run_sfs_backward(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_sfs-backward'):
    
    from sklearn.feature_selection import SequentialFeatureSelector
    feature_selection_method='sfs-backward'
    p_init = X_init.shape[1]
    
    # initialize dataframe for recording results
    res_cols = ['model_type', 'yvar', 'num_features', 'num_features_dropped', 'r2_train', 'r2_cv', 'mae_train', 'mae_cv', 'mae_norm_train', 'mae_norm_cv', 'r2_over_mae_norm_train', 'r2_over_mae_norm_cv', 'xvar_to_drop', 'xvar_list']
    res = []
    featureset_opt = {yvar: {model_type: {featureset_suffix: None} for model_type in models_to_evaluate} for yvar in yvar_list}
    
    for model_type in models_to_evaluate:
        res_model = []
        print(model_type)
        
        for i, yvar in enumerate(yvar_list):
            print('*******************')
            print(yvar)
            print('*******************')
            # get y and model parameters
            y = Y[:,i]
            model_list = model_params[dataset_name][model_type][yvar].copy()
            res_model_yvar = []

            # get initial X variables
            X = X_init.copy()
            p = p_init
            xvar_list = xvar_list_init.copy()

            # iterate over number of features to eliminate
            for k in range(1, p_init): 
                
                # fit model with full feature set
                if model_type=='randomforest':
                    from sklearn.ensemble import RandomForestRegressor
                    n_estimators = model_list[0]['n_estimators']
                    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
                elif model_type=='plsr':
                    from sklearn.cross_decomposition import PLSRegression
                    n_components = model_list[0]['n_components']
                    model = PLSRegression(n_components=min(n_components, X.shape[1]-1))
                    
                # perform SFS, backward
                sfs = SequentialFeatureSelector(model, n_features_to_select=p_init-k, tol=None, direction='backward', scoring='r2', cv=5, n_jobs=None)
                sfs.fit(X, y)
                idx_to_drop = np.argwhere(sfs.get_support()*1==0)[0][0]
                xvar_to_drop = xvar_list[idx_to_drop]
                xvar_list = [xvar for xvar in xvar_list if xvar!=xvar_to_drop]
                X = sfs.transform(X)
                p = X.shape[1]
                print(f'{k}/{p+k}', xvar_to_drop, end=': ')

                # evaluate model metrics after dropping feature
                model_list, metrics = fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, scoring='mae', scale_data=True, print_output=False)
                print(metrics['r2_train'], metrics['r2_cv'])
                # update dataset
                metrics.update({
                    'num_features':p, 'num_features_dropped':k, 
                    'r2_over_mae_norm_train': metrics['r2_train']/metrics['mae_norm_train'], 
                    'r2_over_mae_norm_cv': metrics['r2_cv']/metrics['mae_norm_cv'], 
                    'xvar_to_drop':xvar_to_drop, 'xvar_list':', '.join(xvar_list), 
                    }) 
                # print final feature left 
                if len(xvar_list)==1: 
                    print(xvar_list[0])
                    metrics['xvar_to_drop'] += ', ' + xvar_list[0]
                res_model_yvar.append(metrics)
                
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
        res_model_df.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}_{model_type}{kfold_suffix}.csv')
        
        # update overall results
        res += res_model
        
    # aggregate overall results dataframe
    res = pd.DataFrame(res)[res_cols]
    res.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}{kfold_suffix}.csv')
    
    return res, featureset_opt

#%% Recursive Feature Elimination

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
        res_model_df.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}_{model_type}{kfold_suffix}.csv')
        
        # update overall results
        res += res_model
        
    # aggregate overall results dataframe
    res = pd.DataFrame(res)[res_cols]
    res.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}{dataset_suffix}{kfold_suffix}.csv')

    return res, featureset_opt

