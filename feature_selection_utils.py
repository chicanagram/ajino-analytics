#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:52:31 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from variables import model_params, yvar_list_key, features_to_boost_dict, process_features
from model_utils import run_trainval_test, fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx, heatmap
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

def run_sfs_forward(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_sfs-forward', cv=None):
    
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
            print(yvar)
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
                    model_list, metrics = fit_model_with_cv(X_, y, yvar, model_list, plot_predictions=False, print_output=False, cv=cv, scale_data=True)
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

    
def run_sfs_backward(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_sfs-backward', cv=None):
    
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
                sfs = SequentialFeatureSelector(model, n_features_to_select=p_init-k, tol=None, direction='backward', scoring='r2', cv=cv, n_jobs=None)
                sfs.fit(X, y)
                idx_to_drop = np.argwhere(sfs.get_support()*1==0)[0][0]
                xvar_to_drop = xvar_list[idx_to_drop]
                xvar_list = [xvar for xvar in xvar_list if xvar!=xvar_to_drop]
                X = sfs.transform(X)
                p = X.shape[1]
                print(f'{k}/{p+k}', xvar_to_drop, end=': ')

                # evaluate model metrics after dropping feature
                model_list, metrics = fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, scoring='mae', cv=cv, scale_data=True, print_output=False)
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

def run_rfe(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_rfe', cv=None):    
    
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
            model_list, metrics = fit_model_with_cv(X, y, yvar, model_list, plot_predictions=False, cv=cv, scale_data=True)
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
                model_list, metrics = fit_model_with_cv(X, y, yvar, model_list, plot_predictions=False, cv=cv)
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

#%% 

def parse_model_type(f):
    if f.find('randomforest')>-1:
        model_type = 'randomforest'
    elif f.find('lasso')>-1:
        model_type = 'lasso'
    elif f.find('plsr')>-1:
        model_type = 'plsr'
    elif f.find('xgb')>-1:
        model_type = 'xgb'
    return model_type
                
def adapt_dfrow(dfrow_to_adapt, input_suffix, output_suffix, xvar_list_final, xvar_to_ignore=['DO', 'feed %', 'feed vol', 'pH']): 
    dfrow_final = np.zeros((len(xvar_list_final),))
    dfrow_final[:] = np.nan
    for i, xvar_base in enumerate(xvar_list_final): 
        for s in output_suffix:
            xvar_base = xvar_base.replace(s, input_suffix)
        if xvar_base in dfrow_to_adapt and xvar_base not in xvar_to_ignore:
            val = np.abs(dfrow_to_adapt[xvar_base])
            dfrow_final[i] = val
    return dfrow_final

def get_ranking_arr_across_row(val_arr):
    from scipy.stats import rankdata
    # get rankings by row
    ranking_arr = np.zeros_like(val_arr)
    ranking_arr[:] = np.nan
    for row_idx in range(val_arr.shape[0]):
        ranking_arr[row_idx,:] = val_arr.shape[1]+1-rankdata(val_arr[row_idx,:])
    # set nan elements to NaN
    ranking_arr[np.isnan(val_arr)] = np.nan
    return ranking_arr

def get_contents_of_brackets(string): 
    content_list = []
    string_to_search = string
    no_more_brackets = False
    while len(string_to_search)>3 and not no_more_brackets:
        startidx = string_to_search.find('(')
        if startidx > -1: 
            endidx = string_to_search[startidx:].find(')')
            content = string_to_search[startidx+1:startidx+endidx]
            content_list.append(content)
            string_to_search = string_to_search[startidx+endidx:]
        else: 
            no_more_brackets = True
    return content_list

def plot_colorcoded_barplot(arr, yvar, xvar_list, width=0.8, figsize=(10,5), color_list='b', annotate_xvar=None, figtitle=None, savefig=None):
    # plot barplot
    fig, ax = plt.subplots(1,1, figsize=figsize)
    xtickpos = np.arange(len(xvar_list))
    ax.bar(xtickpos, arr, width=width, color=color_list)
    ax.set_xticks(xtickpos, xvar_list, fontsize=7, rotation=90)
    if figtitle is None:
        ax.set_title(yvar, fontsize=20)
    else: 
        ax.set_title(figtitle, fontsize=20)
    ax.set_ylabel('feature importances', fontsize=16)
    # annotate xvar with lasso features, if needed
    if annotate_xvar is not None: 
        (ymin, ymax) = ax.get_ylim()
        ax.scatter(annotate_xvar, arr[annotate_xvar]+(ymax-ymin)/25, color='k', s=2)
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show()
    
def get_color_list_bycluster(cluster_df, xvar_list, num_clusters=24, ncolors_in_palette=10): 
    cluster_labels = cluster_df.loc[num_clusters,xvar_list].to_numpy()
    custom_palette = [plt.cm.tab10(i) for i in range(ncolors_in_palette)]
    custom_palette = custom_palette*int(np.ceil(num_clusters/ncolors_in_palette))
    color_list = [custom_palette[cluster_label] for cluster_label in cluster_labels]
    return color_list, cluster_labels
    

def convert_fstxt_to_dict_list(fs_txt_list, yvar_list, xvar_list):
    fs_all_list = []
    fs_idxs_all_list = []
    fs_dict_list = []
    fs_idxs_dict_list = []
    for k, fs_txt in enumerate(fs_txt_list):
        fs_yvar_list = fs_txt.split('\n')
        fs_dict = {}
        fs_idxs_dict = {}
        fs_all = []
        fs_idxs_all = []
        for i, yvar in enumerate(yvar_list):
            fs_yvar = fs_yvar_list[i][3:].split(', ')
            fs_idxs_yvar = [xvar_list.index(xvar) for xvar in fs_yvar]
            fs_dict[yvar] = fs_yvar
            fs_idxs_dict[yvar] = fs_idxs_yvar
            for idx, xvar in zip(fs_idxs_yvar, fs_yvar):
                if idx not in fs_idxs_all:
                    fs_idxs_all.append(idx)
                    fs_all.append(xvar)
        fs_dict_list.append(fs_dict)
        fs_idxs_dict_list.append(fs_idxs_dict)
        fs_idxs_all.sort()
        fs_idxs_all_list.append(fs_idxs_all)
        fs_all.sort()
        fs_all_list.append(fs_all)
    return fs_all_list, fs_idxs_all_list, fs_dict_list, fs_idxs_dict_list

def get_fs_cluster_avg_performance(fs_df, kmeans_labels, print_top_fs=False): 
    col_to_avg = [ 
        'obj_fn',
        'r2_increase',
        'frac_knowledge_features',
        'frac_allfeatures_selected',
        'frac_paired_features',
        'corr_avg_selected',
     ]
    cluster_res = {}
    unique, counts = np.unique(kmeans_labels, return_counts=True)
    for label, count in zip(unique, counts): 
        cluster_res[label] = {'count': count}
        label_idxs = np.argwhere(kmeans_labels==label)[0]
        fs_df_bylabel = fs_df.iloc[label_idxs][['features_selected']+col_to_avg].sort_values(by='obj_fn')
        if print_top_fs:
            print(f'Cl{label}:', *eval(fs_df_bylabel.iloc[0]['features_selected']))
        fs_df_label_mean = fs_df_bylabel.iloc[1:,:].mean(axis=0).to_dict()
        cluster_res[label].update(fs_df_label_mean)
        # print(fs_df_bylabel.iloc[0][col_to_avg])
    return cluster_res

def get_fs_clusters(fs_df, fs_arr, xvar_list, n_kmeans_clusters=5, random_state=0, print_top_fs=False):
    from sklearn.cluster import KMeans
    n_kmeans_clusters = 5
    kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=random_state, n_init="auto").fit(fs_arr.to_numpy())
    fs_kmeans_df = pd.DataFrame(kmeans.cluster_centers_, columns=xvar_list)
    kmeans_labels = kmeans.labels_
    cluster_res = get_fs_cluster_avg_performance(fs_df, kmeans_labels, print_top_fs=print_top_fs)
    fig, ax = plt.subplots(1,1, figsize=(20,3))
    heatmap(kmeans.cluster_centers_, row_labels=[f'{label} ({label_dict["count"]}), frKnow={round(label_dict["frac_knowledge_features"],2)}, R2inc={round(label_dict["r2_increase"],3)}' for label, label_dict in cluster_res.items()], col_labels=xvar_list, show_gridlines=False)
    plt.show()
    return kmeans, fs_kmeans_df

def plot_fs_grid(fs_idxs_all_list, xvar_list, nrows=6, figtitle=None): 
    # get heatmap
    fs_heatmap = np.zeros((len(xvar_list), len(fs_idxs_all_list)))
    for i in range(len(fs_idxs_all_list)):
        fs_idxs_all = fs_idxs_all_list[i]
        fs_heatmap[np.array(fs_idxs_all), i] = 1
    # plot heatmap
    chunk = int(np.ceil(fs_heatmap.shape[1]/nrows))
    fig, ax = plt.subplots(nrows,1, figsize=(40,len(xvar_list)/chunk*40))
    for k in range(nrows):  
        ax[k].imshow(fs_heatmap[:,chunk*k:min(fs_heatmap.shape[1], chunk*(k+1))])
        ax[k].axis('off')
    if figtitle is not None:
        plt.suptitle(figtitle, y=0.92, fontsize=8)
    plt.savefig(f"{figure_folder}feature_selection_GRIDSEARCH_deduplicated.png", bbox_inches='tight', dpi=600)
    plt.show()
    
    fs_heatmap = pd.DataFrame(fs_heatmap.astype(int), index=xvar_list, columns=range(1, fs_heatmap.shape[1]+1)) 
    return fs_heatmap

#%% 
class FeatureSelectionModule: 
    
    def __init__(self, 
                 dataset_name_wsuffix='X1Y0', 
                 yvar_list=yvar_list_key, 
                 knowledge_features_to_boost=features_to_boost_dict, 
                 plot_scorecards=False,
                 num_clusters=24, 
                 data_folder=data_folder,
                 print_progress=True,
                 save_interval=200,
                 csv_fpath=None
                 ):
        self.dataset_name_wsuffix = dataset_name_wsuffix
        self.data_folder = data_folder
        self.yvar_list = yvar_list        
        self.plot_scorecards = plot_scorecards
        self.print_progress = print_progress
        self.fs_param_names = ['fs_avg_weight', 'lasso_corr_scaling_power', 'lasso_distributed_scores_weight', 'nsrc_scaling_factor', 'knowledge_scaling_factor', 'corr_thres', 'num_features_to_select', 'max_num_highcorr_features']
        
        # get data and knowledge features to boost, by yvar
        self.knowledge_features_dict = {}
        self.knowledge_features_idxs_dict = {}
        self.knowledge_features_dict_binary = {}
        self.knowledge_boost_dict = {}
        self.df_dict = {}
        for k, yvar in enumerate(self.yvar_list): 
            # get df
            df = pd.read_csv(f'{data_folder}feature_analysis_all_{k}_coefvals.png', index_col=0)
            self.df_dict[yvar] = df
            # get xvar list (not including process features)
            self.xvar_list = df.columns.tolist()
            
            # get knowledge features to boost
            features_to_boost = knowledge_features_to_boost[yvar]
            knowledge_features = []
            knowledge_features_idxs = []
            knowledge_boost = np.zeros((len(self.xvar_list),))
            for nutrient, nutrient_score in features_to_boost.items():
                matching_features = []
                matching_idxs = []
                for i, xvar in enumerate(self.xvar_list):
                    if nutrient in xvar:
                        matching_features.append(xvar)
                        matching_idxs.append(i)
                if len(matching_idxs) > 0:
                    knowledge_boost[np.array(matching_idxs)] += nutrient_score
                knowledge_features += matching_features
                knowledge_features_idxs += matching_idxs
            self.knowledge_features_dict[yvar] = knowledge_features
            self.knowledge_features_idxs_dict[yvar] = knowledge_features_idxs
            self.knowledge_boost_dict[yvar] = knowledge_boost

        # get color list by cluster
        cluster_df_sorted = pd.read_csv(f'{self.data_folder}features_by_cluster_corrdist.csv', index_col=0)
        self.color_list, cluster_labels = get_color_list_bycluster(cluster_df_sorted, self.xvar_list, num_clusters, ncolors_in_palette=10)

        # get correlation matrix 
        self.corr_mat = pd.read_csv(f'{self.data_folder}{self.dataset_name_wsuffix}_correlation_matrix.csv', index_col=0)

        # feature selection optimization via objective function
        self.ncalls_count = 0
        self.r2_0 = {'Titer (mg/L)_14': 0.924, 'mannosylation_14': 0.798, 'fucosylation_14': 0.753, 'galactosylation_14': 0.945}
        self.objfn_component_names = ['r2_increase', 'frac_knowledge_features', 'frac_allfeatures_selected', 'frac_paired_features', 'corr_avg_selected']
        self.objfn_component_vals = {yvar: {component:0 for component in self.objfn_component_names} for yvar in self.yvar_list}
        self.objfn_component_wts = {
            'r2_increase': -10, 
            'frac_knowledge_features': -2, 
            'frac_allfeatures_selected': 2, 
            'frac_paired_features': -1, 
            'corr_avg_selected': 1
            }
        self.csv_all = []
        self.csv_avg = []
        self.save_interval = save_interval
        self.csv_columns = ['i', 'features_selected', 'num_unique_features', 'obj_fn'] + self.objfn_component_names + self.fs_param_names
        self.csv_fpath = csv_fpath
        
        
    def fs_settings_arr2dict(self, fs_settings_arr):
        print(fs_settings_arr)
        fs_settings_dict = {}
        fs_settings_dict['fs_avg_weight'] = fs_settings_arr[0]
        fs_settings_dict['lasso_corr_scaling_power'] = fs_settings_arr[1]
        fs_settings_dict['lasso_distributed_scores_weight'] = fs_settings_arr[2]
        fs_settings_dict['nsrc_scaling_factor'] = fs_settings_arr[3]
        fs_settings_dict['knowledge_scaling_factor'] = fs_settings_arr[4]
        fs_settings_dict['corr_thres'] = fs_settings_arr[5]
        fs_settings_dict['num_features_to_select'] = fs_settings_arr[6] 
        fs_settings_dict['max_num_highcorr_features'] = fs_settings_arr[7]
        return fs_settings_dict

    def get_settings(self, fs_settings, fs):
        
        # get feature selection settings
        if isinstance(fs_settings, np.ndarray):
            fs_settings = self.fs_settings_arr2dict(fs_settings)
        self.fs_settings = fs_settings
        self.fs = fs
        self.fs_avg_weight = fs_settings['fs_avg_weight']
        self.lasso_corr_scaling_power = fs_settings['lasso_corr_scaling_power']
        self.lasso_distributed_scores_weight = fs_settings['lasso_distributed_scores_weight']
        self.nsrc_scaling_factor = fs_settings['nsrc_scaling_factor']
        self.knowledge_scaling_factor = fs_settings['knowledge_scaling_factor']
        self.corr_thres= fs_settings['corr_thres']
        self.num_features_to_select_dict = {yvar: fs_settings['num_features_to_select'] for yvar in self.yvar_list}
        self.MAX_NUM_HIGHCORR_FEATURES_DICT = {yvar: fs_settings['max_num_highcorr_features'] for yvar in self.yvar_list}
            
    def add_feature_importance_scores(self, yvar, df, scorecard):
        # average scores from first 4 rows (MC_rf_SHAP, MC_rf_RFE, MC_rf_coef, MC_plsr_coef)
        fs_avg = np.nanmean(df.iloc[:6, :].to_numpy(), axis=0)
        scorecard += fs_avg*self.fs_avg_weight
        if self.plot_scorecards:
            plot_colorcoded_barplot(scorecard, yvar, self.xvar_list, width=0.8, color_list=self.color_list, annotate_xvar=self.knowledge_features_idxs_dict[yvar], figtitle=f'{yvar}: after fs_avg', savefig=None)
        return scorecard        
        
    def add_distributed_lasso_scores(self, yvar, df, scorecard):
        lasso_coefs = df.iloc[6,:]
        lasso_coefs_nonzero = lasso_coefs[lasso_coefs>0]
        lasso_nonzero_xvar = list(lasso_coefs_nonzero.index)
        lasso_distributed_scores = np.zeros((len(lasso_coefs_nonzero), len(self.xvar_list))) 
        for i, xvar in enumerate(lasso_nonzero_xvar): 
            # get coef value
            coefval = lasso_coefs_nonzero[xvar]
            # get correlation coefficient scores with this xvar
            corr_coefs = self.corr_mat.loc[xvar,self.xvar_list]
            # get product of corr_coef and coefval
            product_coefval_corrcoefsq = coefval*(corr_coefs**self.lasso_corr_scaling_power)
            # add to lasso distributed scores array
            lasso_distributed_scores[i,:] = product_coefval_corrcoefsq
        # get max value for each feature down the column
        lasso_distributed_scores = np.max(lasso_distributed_scores, axis=0)
        scorecard += lasso_distributed_scores*self.lasso_distributed_scores_weight
        if self.plot_scorecards:
            plot_colorcoded_barplot(scorecard, yvar, self.xvar_list, width=0.8, color_list=self.color_list, annotate_xvar=self.knowledge_features_idxs_dict[yvar], figtitle=f'{yvar}: after lasso_distributed', savefig=None)
        return scorecard
    
    def boost_with_domain_knowledge(self, scorecard, yvar):
        knowledge_boost_factor = 1 + self.knowledge_boost_dict[yvar]*self.knowledge_scaling_factor
        scorecard *= knowledge_boost_factor
        return scorecard
    
    def boost_with_NSRC_scores(self, yvar, df, scorecard):
        nsrc_vs_cqa_corr = df.loc['NSRC-vs-CQA_corr',:].to_numpy()
        nsrc_boost_factor = 1 + self.nsrc_scaling_factor*nsrc_vs_cqa_corr
        # fill nans
        nsrc_boost_factor[np.isnan(nsrc_boost_factor)] = 1
        scorecard *= nsrc_boost_factor
        if self.plot_scorecards:
            plot_colorcoded_barplot(scorecard, yvar, self.xvar_list, width=0.8, color_list=self.color_list, annotate_xvar=self.knowledge_features_idxs_dict[yvar], figtitle=f'{yvar}: after nsrc_boost', savefig=None)
        return scorecard
    
    def sort_features_using_scorecard(self, scorecard):
        # rank the features by score
        idxs_sorted = np.argsort(scorecard)
        idxs_sorted = idxs_sorted[::-1]
        self.xvar_list_sorted = [self.xvar_list[i] for i in idxs_sorted]
        # reorder correlation matrix by feature rankings, and threshold
        corr_mat_sorted_thres = self.corr_mat.loc[self.xvar_list_sorted, self.xvar_list_sorted]
        corr_mat_sorted_thres = np.tril((corr_mat_sorted_thres>self.corr_thres)*1)
        np.fill_diagonal(corr_mat_sorted_thres, 0)
        self.corr_mat_sorted_thres = pd.DataFrame(corr_mat_sorted_thres, columns=self.xvar_list_sorted, index=self.xvar_list_sorted)
        # plot heatmap
        if self.plot_scorecards:
            fig, ax = plt.subplots(1,1, figsize=(20,20))
            _, _, _ = heatmap(corr_mat_sorted_thres, ax=ax, row_labels=self.xvar_list_sorted, col_labels=self.xvar_list_sorted, show_gridlines=False, labeltop=True)
    
    def select_features_using_scorecard(self, yvar):
        # go down the list and select features, adding to the cluster tally
        pointer = 0
        features_selected = []
        features_discarded = []
        MAX_NUM_HIGHCORR_FEATURES = self.MAX_NUM_HIGHCORR_FEATURES_DICT[yvar]
        num_features_to_select = self.num_features_to_select_dict[yvar]

        while len(features_selected)<num_features_to_select:
            # get next feature to consider
            feature = self.xvar_list_sorted[pointer]
            select_feature = True
            if self.print_progress: print(pointer, feature)
            # get correlations with other previously selected features
            corr_mat_sorted_thres_trunc = self.corr_mat_sorted_thres.loc[features_selected+[feature], features_selected+[feature]]
            sum_highcorr_features_bycol = corr_mat_sorted_thres_trunc.sum(axis=0).to_numpy()
            # find rows where there sum is 2 or more 
            multicorr_cols_idx = np.argwhere(sum_highcorr_features_bycol>=2).reshape(-1,)
            if self.print_progress: print('multicorr_cols_idx:', multicorr_cols_idx)
            if len(multicorr_cols_idx)>0:
                # iterate through highcorr cols, check if we have a high correlation triad (or more) 
                col_idx_idx = 0
                while select_feature and col_idx_idx<len(multicorr_cols_idx):
                    col_idx = multicorr_cols_idx[col_idx_idx]
                    col_sum = sum_highcorr_features_bycol[col_idx]
                    # reject feature if there are 3 or more highly correlated features with the same col feature 
                    if col_sum >= MAX_NUM_HIGHCORR_FEATURES: 
                        select_feature = False
                    else: 
                        if self.print_progress: print(f'Checking col {col_idx} (col_sum={col_sum})')
                        # get slice of col corrs and flip it
                        col_corr_flipped = np.flip(corr_mat_sorted_thres_trunc.iloc[:,col_idx].to_numpy())
                        if self.print_progress: print('col_corr_flipped:', col_corr_flipped)
                        # get indices with 1's
                        row_corr = corr_mat_sorted_thres_trunc.iloc[-1,:len(features_selected)+1].to_numpy()
                        if self.print_progress: print('row_corr:', row_corr)
                        multicorr_rows_idx = np.argwhere(col_corr_flipped==1).reshape(-1,)
                        if self.print_progress: print('multicorr_rows_idx:', multicorr_rows_idx)
                        # get sum of highly correlated features in row (for feature under consideration) 
                        row_sum = corr_mat_sorted_thres_trunc.iloc[-1,multicorr_rows_idx].sum()
                        if self.print_progress: print('Col_sum:', col_sum, 'Row_sum:', row_sum)
                        # sums should be identical to indicate symmetry // mutual correlation of the triad
                        if col_sum==row_sum: 
                            select_feature = False
                    col_idx_idx += 1
            if select_feature:
                features_selected.append(feature)
                if self.print_progress: print(f'Selecting feature {pointer}: {feature}')
            else:
                features_discarded.append(feature)
                if self.print_progress: print(f'Dropping feature {pointer}: {feature}', end='\n')
            pointer += 1
            
        # summary
        if self.print_progress: 
            print(f'{len(features_selected)} features selected:', features_selected, '\n')
            print(f'{len(features_discarded)} features discarded:', features_discarded, '\n')
        return features_selected
    
    def get_corrmat_for_selected_features(self, features_selected):
        # get final correlation map
        corr_mat_selected = self.corr_mat.loc[features_selected, features_selected]
        corr_mat_selected_thres = self.corr_mat_sorted_thres.loc[features_selected, features_selected]
        if self.plot_scorecards:
            fig, ax = plt.subplots(1,1, figsize=(3,3))
            _, _, _ = heatmap(corr_mat_selected, ax=ax, row_labels=features_selected, col_labels=features_selected, show_gridlines=False, labeltop=True)
        return corr_mat_selected, corr_mat_selected_thres
        
    def analyse_features_selected(self, features_selected_dict):
        features_selected_all = []
        print('features_selected_dict = {')
        for yvar, features_selected in features_selected_dict.items():
            features_selected_all += features_selected
            features_selected += process_features
            print(f"'{yvar}': ", end='')
            print(features_selected, end=',')
            print()
        print('}', '\n')
        # get unique features count
        xvars_unique, counts = np.unique(np.array(features_selected_all), return_counts=True)
        idxs = np.argsort(xvars_unique)
        xvars_unique = xvars_unique[idxs]
        counts = counts[idxs]
        xvars_unique_count_dict = {}
        print(len(xvars_unique))
        for xvar, count in zip(xvars_unique, counts):
            print(xvar, count)
            xvars_unique_count_dict[xvar] = count
        return xvars_unique_count_dict

    def evaluate_model_performance(self, features_selected_dict):
        # get dataset
        X_featureset_idx, Y_featureset_idx = self.dataset_name_wsuffix[1], self.dataset_name_wsuffix[3]
        Y, X, _, _, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix='', data_folder=self.data_folder)
        featureset_suffix = self.fs
        _, kfold_metrics_avg, _, _, _ = run_trainval_test(X, Y, yvar_list_key, features_selected_dict, xvar_list_all, self.dataset_name_wsuffix, featureset_suffix, show_plots=self.plot_scorecards, print_progress=False)
        kfold_summary = kfold_metrics_avg.loc[:, ['yvar', 'r2_test_avg']]
        print(kfold_summary)
        kfold_summary = {k: v['r2_test_avg'] for k, v in kfold_summary.set_index('yvar',drop=True).to_dict('index').items()}
        return kfold_summary

    def run_feature_selection(self, 
                            fs_settings, 
                            fs='test', 
                            ): 
        # gets settings
        self.get_settings(fs_settings, fs)
        
        print(f'Implementing setting [{self.fs}]...')
        features_selected_dict = {}
            
        for k, yvar in enumerate(self.yvar_list): 
            print(yvar)
            scorecard = np.zeros((len(self.xvar_list),))
            df = self.df_dict[yvar]
            # add feature importance scores
            scorecard = self.add_feature_importance_scores(yvar, df, scorecard)
            # add lasso distributed scores
            scorecard = self.add_distributed_lasso_scores(yvar, df, scorecard)
            # NSRC features
            scorecard = self.boost_with_NSRC_scores(yvar, df, scorecard)
            # 'domain knowledge' features
            scorecard = self.boost_with_domain_knowledge(scorecard, yvar)
            # plot final scorecard
            if self.plot_scorecards:
                plot_colorcoded_barplot(scorecard, yvar, self.xvar_list, width=0.8, color_list=self.color_list, annotate_xvar=self.knowledge_features_idxs_dict[yvar], figtitle=f'{yvar}: FINAL SCORES',  savefig=None)
            # rank the features by score and reorder correlation matrix
            self.sort_features_using_scorecard(scorecard)

            # go down the sorted scorecard list and select features, checking if there are too many highly-correlated features selected
            features_selected = self.select_features_using_scorecard(yvar)
            corr_mat_selected, corr_mat_selected_thres = self.get_corrmat_for_selected_features(features_selected)
            features_selected_dict[yvar] = features_selected
        
        # feature analysis
        xvars_unique_count_dict = self.analyse_features_selected(features_selected_dict)
            
        # evaluate feature set model performance
        kfold_summary = self.evaluate_model_performance(features_selected_dict)
        
        return features_selected_dict, xvars_unique_count_dict, corr_mat_selected, kfold_summary
    
    def get_frac_paired_features(self, features_selected):
        f_base_count = {}
        for f in features_selected: 
            f_base = f.split('_')[0]
            if f_base not in f_base_count:
                f_base_count[f_base] = 1
            else: 
                f_base_count[f_base] += 1
        num_fbase_paired = 0
        for f, count in f_base_count.items():
            if count > 1: 
                num_fbase_paired += 1
        frac_paired_features = num_fbase_paired*2/len(features_selected)
        return frac_paired_features
    
    def compose_results_df(self, objfun, features_selected_dict, xvars_unique_count_dict):
        
        # initialize row 
        row_dict_avg = {'i': self.ncalls_count}
        row_dict_all = []
        # format features selected cell AND update row_dict_all
        features_txt = ''
        for i, (yvar, flist) in enumerate(features_selected_dict.items()): 
            flist_txt = ', '.join(flist) 
            features_txt += f'{i}: ' + flist_txt
            if i<len(self.yvar_list)-1: 
                features_txt += '\n'
            row_dict_all += [
                {**{'i': f'{self.ncalls_count}-{i}', 'features_selected': flist_txt, 'num_unique_features':len(flist)}, 
                 **self.objfn_component_vals[yvar],
                 **self.fs_settings
                 }]
            
        row_dict_avg.update({'features_selected': features_txt})
        
        # get number of unique features
        row_dict_avg.update({'num_unique_features': len(xvars_unique_count_dict)})
        
        # get overall objective function value 
        row_dict_avg.update({'obj_fn': round(objfun,3)})
        
        # AVERAGE objfn component values
        objfn_component_avg = pd.DataFrame(self.objfn_component_vals).transpose().mean(axis=0).round(4).to_dict()
        row_dict_avg.update({k:v for k, v in objfn_component_avg.items() if k!='obj_fn'})
        
        # get feature selection parameters
        row_dict_avg.update(self.fs_settings)
        
        # append to existing data
        self.csv_avg.append(row_dict_avg)
        self.csv_all += row_dict_all
        
        # append to end of existing csv
        if self.csv_fpath is not None and self.ncalls_count%self.save_interval==0:
            # convert list of dicts to dataframe
            self.csv_avg = pd.DataFrame(self.csv_avg, columns=self.csv_columns)
            self.csv_all = pd.DataFrame(self.csv_all, columns=self.csv_columns)
            if self.ncalls_count==self.save_interval: 
                # start a new CSV if this is the first save
                self.csv_avg.to_csv(self.csv_fpath, index=False)
                self.csv_all.to_csv(self.csv_fpath[:-4]+'_all.csv', index=False)
            else:
                # else append to the end of existing file
                self.csv_avg.to_csv(self.csv_fpath, mode='a', index=False, header=False)
                self.csv_all.to_csv(self.csv_fpath[:-4]+'_all.csv', mode='a', index=False, header=False)
            print('Updated CSV.')
            
            # refresh self.csv_avg for next batch of data
            self.csv_avg = []
            self.csv_all = []
            
        
    def calculate_objective_function(self,
                                    fs_settings, 
                                    fs=None, 
                                    features_selected_dict=None,
                                    xvars_unique_count_dict=None,
                                    corr_mat_selected=None,
                                    r2=None
                                    ): 
        start = time.time()
        self.ncalls_count += 1
        if fs is None:
            fs = self.ncalls_count
        if features_selected_dict is None and xvars_unique_count_dict is None and corr_mat_selected is None and r2 is None:
            features_selected_dict, xvars_unique_count_dict, corr_mat_selected, r2 = self.run_feature_selection(fs_settings, fs)
        # iterate through yvar list and sum
        objfun = 0
        for yvar in yvar_list_key: 
            # get features selected 
            features_selected = features_selected_dict[yvar]
            # fractional r2_decrease (larger = better)
            self.objfn_component_vals[yvar]['r2_increase'] = (r2[yvar] - self.r2_0[yvar])/self.r2_0[yvar]
            # fraction of 'domain knowledge-suggested' features selected (larger = better)
            features_selected_binary = np.array([1 if xvar in features_selected else 0 for xvar in self.xvar_list])
            self.objfn_component_vals[yvar]['frac_knowledge_features'] = np.sum(features_selected_binary*self.knowledge_boost_dict[yvar])/np.sum(self.knowledge_boost_dict[yvar])
            # compactness (smaller = better)
            self.objfn_component_vals[yvar]['frac_allfeatures_selected'] = len(xvars_unique_count_dict)/len(self.xvar_list)
            # fraction of paired (basal + feed) features both selected (larger = better)
            self.objfn_component_vals[yvar]['frac_paired_features'] = self.get_frac_paired_features(features_selected)
            # total correlations amongst selected features (smaller = better)
            self.objfn_component_vals[yvar]['corr_avg_selected'] = np.mean(corr_mat_selected - np.triu(corr_mat_selected))
            print('\n', f'Objective Function Components ({yvar}):', {k: round(v,3) for k, v in self.objfn_component_vals[yvar].items()})
            
            # sum contributions with weights
            objfun_yvar = 0
            for comp in self.objfn_component_names:
                objfun_yvar += self.objfn_component_vals[yvar][comp] * self.objfn_component_wts[comp]
                self.objfn_component_vals[yvar]['obj_fun'] = objfun_yvar
            objfun += objfun_yvar
        
        print(self.ncalls_count, 'OBJECTIVE FUNCTION (SUM):', round(objfun, 3), '\n')
        end = time.time()
        print(f'Time taken: {round(end-start, 2)} s')
        
        # save results
        self.compose_results_df(objfun, features_selected_dict, xvars_unique_count_dict)
        
        return objfun

