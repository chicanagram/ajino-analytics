#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:02:36 2024

@author: charmainechia
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from variables import model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key, xvar_sublist_sets, xvar_sublist_sets_bymodeltype
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
from get_datasets import data_folder, get_XYdata_for_featureset


#%% Evaluate different feature sets and model parameters 
scoring = 'mae'
featureset_list = [(1,0)] # [(1,0), (0,0)] # 
dataset_suffix = '_avg' 
# dataset_suffix = ''
model_params_to_eval = [
    {'model_type': 'randomforest', 'params_to_eval': ('n_estimators', [20,40,60,80,100,120,140,160,180])},
    {'model_type': 'plsr', 'params_to_eval': ('n_components',[2,3,4,5,6,7,8,9,10,11,12,14])},
    {'model_type': 'lasso', 'max_iter':500000, 'params_to_eval': ('alpha', [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100])},
    ]

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_name_wsuffix = dataset_name + dataset_suffix
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    modelparam_metrics_df = []
    
    for i, yvar in enumerate(yvar_list): 
        print(yvar)
        y = Y[:,i]
        
        # get model type and list of parameters to evaluate
        for model_dict in model_params_to_eval:
            model_type = model_dict['model_type']
            print(model_type)
            model_list = [{k:v for k,v in model_dict.items() if k!='params_to_eval'}]
            (param_name, param_val_list) = model_dict['params_to_eval']
            for param_val in param_val_list: 
                print(param_name, param_val, end=': ')
                model_list[0].update({param_name:param_val})
                
                # fit model with selected features and parameters and get metrics 
                model_list, metrics = fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, scoring=scoring, scale_data=True)
                # update metrics dict with param val
                metrics.update({'param_name': param_name, 'param_val': param_val})
                modelparam_metrics_df.append(metrics)
        
    # save model metrics for current dataset
    modelparam_metrics_df = pd.DataFrame(modelparam_metrics_df)
    modelparam_metrics_df = modelparam_metrics_df[['model_type','yvar','param_name', 'param_val', 'r2_train', 'r2_cv', 'mae_norm_train', 'mae_norm_cv','mae_train','mae_cv']]
    modelparam_metrics_df.to_csv(f'{data_folder}modelmetrics_vs_params_{dataset_name}{dataset_suffix}.csv')

    # plot model metric for various parameters for each dataset
    for k, yvar_sublist in enumerate(yvar_sublist_sets):
        print(yvar_sublist)
        for model_type in ['randomforest', 'plsr', 'lasso']: 
            fig, ax = plt.subplots(3,4, figsize=(27,17))
            for i, yvar in enumerate(yvar_sublist):
                # get relevant metrics to plot -- remove data that is out of range
                metrics_filt = modelparam_metrics_df.loc[(modelparam_metrics_df['model_type']==model_type) & (modelparam_metrics_df['yvar']==yvar)]
                metrics_filt = metrics_filt.loc[(metrics_filt.r2_train>=0) & (metrics_filt.mae_norm_cv<10)]
                param_name = str(metrics_filt.iloc[0]['param_name'])
                # plot R2, MAE (CV), MAE (test) for all parameter values
                ax[0][i].plot(metrics_filt['param_val'], metrics_filt['r2_train'], marker='*')
                ax[0][i].plot(metrics_filt['param_val'], metrics_filt['r2_cv'], marker='X')
                ax[0][i].plot(metrics_filt['param_val'], metrics_filt[f'{scoring}_norm_train'], marker='o')
                ax[0][i].plot(metrics_filt['param_val'], metrics_filt[f'{scoring}_norm_cv'], marker='s')
                ax[0][i].legend(['r2_train', 'r2_cv', f'{scoring}_train', f'{scoring}_cv'])
                ax[0][i].set_ylabel('model metrics', fontsize=12)
                ax[0][i].set_xlabel(param_name, fontsize=12)
                ax[0][i].set_title(yvar, fontsize=14)
                if model_type=='lasso': 
                    ax[0][i].set_xscale('log')
                # plot R2 divided by MAE(CV) for all parameter values
                ax[1][i].plot(metrics_filt['param_val'], metrics_filt['r2_train']/metrics_filt[f'{scoring}_norm_cv'], marker='o')
                ax[1][i].set_ylabel(f'r2/{scoring}(loocv)', fontsize=12)
                ax[1][i].set_xlabel(param_name, fontsize=12)
                ax[1][i].set_title(yvar, fontsize=14)
                if model_type=='lasso': 
                    ax[1][i].set_xscale('log')
                # plot MAE (CV) vs MAE (test)
                ax[2][i].plot(metrics_filt[f'{scoring}_norm_train'], metrics_filt[f'{scoring}_norm_cv'], marker='o')
                ax[2][i].set_ylabel(f'{scoring} (loocv)', fontsize=12)
                ax[2][i].set_xlabel(f'{scoring} (train)', fontsize=12)
                ax[2][i].set_title(yvar, fontsize=14)
                for j in range(len(metrics_filt)):
                    datapoint = metrics_filt.iloc[j]
                    param_val = float(datapoint['param_val'])
                    if param_val>=1: param_val = round(param_val)
                    ax[2][i].annotate(str(param_val), (float(datapoint[f'{scoring}_norm_train']), float(datapoint[f'{scoring}_norm_cv'])))
            ymax = ax.flatten()[0].get_position().ymax
            plt.suptitle(f'{model_type} metrics for various model parameters \n{yvar_sublist}', y=ymax*1.07, fontsize=20)    
            fig.savefig(f'{figure_folder}modelmetrics_{model_type}_vs_params_{dataset_name}{dataset_suffix}_{k}.png', bbox_inches='tight')
            plt.show()



#%% Evaluate individual model at a time and get metrics, and feature coefficients / importances 

featureset_list =  [(1,0)] # [(1,0), (0,0)]
models_to_eval_list = ['randomforest','plsr', 'lasso'] # ['randomforest'] # 
dataset_suffix = '_avg'
# dataset_suffix = ''
f = 1
 
# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_name_wsuffix = dataset_name + dataset_suffix
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    print(f'X dataset size: n={X.shape[0]}, p={X.shape[1]}')
    
    # initialize variables for storing results
    model_metrics_df = []
    feature_importance_df = []
    feature_coef_df = []
    ypred_bymodel = {model_type: np.empty_like(Y) for model_type in models_to_eval_list}
    ypred_cv_bymodel = {model_type: np.empty_like(Y) for model_type in models_to_eval_list}
    
    # iterate through model types 
    for model_type in models_to_eval_list: 
        print(model_type)
        feature_order_txt = ['yvar']
        
        # iterate through yvar
        for i, yvar in enumerate(yvar_list):     
            
            model_list = model_params[dataset_name_wsuffix][model_type][yvar]
            y = Y[:,i]
            model_list, metrics = fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, scale_data=True)
            metrics.update({'f':f, 'xvar_list': ', '.join(str(e) for e in xvar_list)})
            # get feature importance and split into COEF and ORDER dicts
            feature_coefs, feature_importances = get_feature_importances(model_list[0], yvar, xvar_list, plot_feature_importances=False)
            # update aggregating dicts
            feature_importance_df += feature_importances
            feature_coef_df += feature_coefs
            model_metrics_df.append(metrics)
            ypred_bymodel[model_type][:, i] = model_list[0]['ypred']
            ypred_cv_bymodel[model_type][:, i] = model_list[0]['ypred_cv']
        print()
        
        # get feature importances for each model
        feature_importance_df_MODEL = pd.DataFrame(feature_importance_df)
        feature_importance_df_MODEL = feature_importance_df_MODEL[feature_importance_df_MODEL['model_type']==model_type]
        feature_coef_df_MODEL = pd.DataFrame(feature_coef_df)
        feature_coef_df_MODEL = feature_coef_df_MODEL[feature_coef_df_MODEL['model_type']==model_type]
        # plot feature importance heatmaps
        arr_importance = plot_feature_importance_heatmap(feature_importance_df_MODEL.set_index('yvar').iloc[:,1:], xvar_list, yvar_list, logscale_cmap=False, scale_vals=False, figtitle=f'Feature importances ({model_type})', savefig=f'feature_importance_{model_type}_{dataset_name}{dataset_suffix}')
        arr_coef = plot_feature_importance_heatmap(feature_coef_df_MODEL.set_index('yvar').iloc[:,1:], xvar_list, yvar_list, logscale_cmap=False, scale_vals=True, figtitle=f'Feature coefficients, normalized ({model_type})', savefig=f'feature_coef_{model_type}_{dataset_name}{dataset_suffix}')
        arr_prominence = arr_importance*np.abs(arr_coef)
        # get feature prominence barplots    
        plot_feature_importance_barplots(arr_prominence, yvar_list, xvar_list, order_by_feature_importance=False, label_xvar_by_indices=True, ncols=4, nrows=3, c=model_cmap[model_type], savefig=f'feature_prominence_barplots_{model_type}_{dataset_name}{dataset_suffix}', figtitle=f'[{model_type}] feature prominences for all y variables')
            
        # order features by importance
        for i, yvar in enumerate(yvar_list):   
            feature_scoring_yvar = arr_prominence[i,:]
            if model_type=='lasso':
                idx_to_keep = [idx for idx, scoring in enumerate(feature_scoring_yvar) if scoring != 0]
                feature_scoring_yvar_toorder = feature_scoring_yvar[np.array(idx_to_keep)]
                xvar_list_toorder = [xvar_list[idx] for idx in idx_to_keep]
            else: 
                feature_scoring_yvar_toorder = feature_scoring_yvar
                xvar_list_toorder = xvar_list
            _, _,feature_ordering_yvar = order_features_by_importance(feature_scoring_yvar_toorder, xvar_list_toorder)
            feature_order_txt_yvar = f'[{yvar}], {", ".join(str(e) for e in feature_ordering_yvar)}'
            feature_order_txt += [feature_order_txt_yvar]
            print(feature_order_txt_yvar, '\n')
        np.savetxt(f'{data_folder}model_{model_type}_feature_ordering_{dataset_name}{dataset_suffix}.csv',  feature_order_txt, fmt='%s', delimiter=', ', newline='\n')
    
    # aggregate all model metrics and save CSV
    model_metrics_df = pd.DataFrame(model_metrics_df)
    model_metrics_df = model_metrics_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    model_metrics_df.to_csv(f'{data_folder}model_metrics_{dataset_name}{dataset_suffix}.csv')
    
    # aggregate all feature importances and save CSV
    feature_importance_df = pd.DataFrame(feature_importance_df)
    feature_importance_df = feature_importance_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    feature_importance_df.to_csv(f'{data_folder}model_feature_importance_{dataset_name}{dataset_suffix}.csv')    
    feature_coef_df = pd.DataFrame(feature_coef_df)
    feature_coef_df = feature_coef_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    feature_coef_df.to_csv(f'{data_folder}model_feature_coef_{dataset_name}{dataset_suffix}.csv')

    # Plot model metrics and predictions
    figtitle = 'Metrics for various models evaluated on all Y variables'
    savefig = f'{figure_folder}modelmetrics_allselectedmodels_allYvar_{dataset_name}{dataset_suffix}.png'
    plot_model_metrics(model_metrics_df, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(30,15), barwidth=0.8, suffix_list=['_train', '_cv'], figtitle=figtitle, savefig=savefig, model_cmap=model_cmap)
    
    # generate scatter plots of ypred vs y
    for k, yvar_sublist in enumerate(yvar_sublist_sets):
    
        # iterate through yvar in sublist
        fig, ax = plt.subplots(3,4, figsize=(27,18))
        for i, yvar in enumerate(yvar_sublist): 
            yvar_idx = k*len(yvar_sublist)+i
            y = Y[:,yvar_idx]
    
            for j, model_type in enumerate(models_to_eval_list): 
                c = model_cmap[model_type]
                ypred = ypred_bymodel[model_type][:,yvar_idx]
                ypred_cv = ypred_cv_bymodel[model_type][:,yvar_idx]
                ax[j][i].scatter(y, ypred, c='k', alpha=0.5, marker='*')
                ax[j][i].scatter(y, ypred_cv, c=c, alpha=1, marker='o', s=16)
                ax[j][i].set_title(f'{model_type} <> {yvar}', fontsize=16)
                ax[j][i].set_ylabel(f'{yvar} (predicted)', fontsize=12)
                ax[j][i].set_xlabel(f'{yvar} (actual)', fontsize=12)          
                (xmin, xmax) = ax[j][i].get_xlim()
                (ymin, ymax) = ax[j][i].get_ylim()
                r2 = float(model_metrics_df[(model_metrics_df.model_type==model_type) & (model_metrics_df.yvar==yvar)].iloc[0].r2_train)
                r2_cv = float(model_metrics_df[(model_metrics_df.model_type==model_type) & (model_metrics_df.yvar==yvar)].iloc[0].r2_cv)
                ax[j][i].text(xmin+(xmax-xmin)*0.05, ymin+(ymax-ymin)*0.85, f'R2 (train): {r2} \nR2 (CV): {r2_cv}', fontsize=14)
                ax[j][i].legend(['train', 'CV'], loc='lower right')
                
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'Model Predicted vs. Actual values for various Y variables\n{yvar_sublist}', y=ymax*1.08, fontsize=24)    
        fig.savefig(f'{figure_folder}modelpredictions_scatterplots_{dataset_name}{dataset_suffix}_{k}.png', bbox_inches='tight')
        plt.show()


#%% Evaluate model performance for different SIZES of feature subset 

scoring = 'mae'
featureset_list = [(0,0)] # [(1,0)]  # 
fraction_of_topfeatures_to_select = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
model_params_to_eval = [
    {'model_type': 'randomforest', 'params_to_eval': ('n_estimators', [50, 100, 150])},
    ]
models_to_eval_list = [model_dict['model_type'] for model_dict in model_params_to_eval]

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
    # get top feature ordering
    overall_ordered_features = pd.read_csv(f'{data_folder}model_features_ordered_AGGREGATED_{dataset_name}.csv', index_col=0)
    modelparam_metrics_df = []    

    for i, yvar in enumerate(yvar_list): 
        print(yvar)
        y = Y[:,i]
        xvar_list_ordered = overall_ordered_features.iloc[i, :].tolist()
    
        # filter X dataset to select a subset of features
        for f in fraction_of_topfeatures_to_select:
            print('Fraction of top features to select:', f)
            X_selected, xvar_list_selected = select_subset_of_X(X, xvar_list, xvar_list_ordered, f)
            
            # get model type and list of parameters to evaluate
            for model_dict in model_params_to_eval:
                model_type = model_dict['model_type']
                model_list = [{k:v for k,v in model_dict.items() if k!='params_to_eval'}]
                (param_name, param_val_list) = model_dict['params_to_eval']

                # get model parameter
                for param_val in param_val_list: 
                    print(param_name, param_val, end=': ')
                    if model_type=='plsr' and param_val>len(xvar_list_selected):
                        param_val = len(xvar_list_selected)
                    model_list[0].update({param_name:param_val})
                    
                    # fit model with selected features and parameters and get metrics 
                    model_list, metrics = fit_model_with_cv(X_selected,y, yvar, model_list, plot_predictions=False, scoring=scoring, scale_data=True)
                    # update metrics dict with param val
                    metrics.update({'param_name': param_name, 'param_val': param_val, 'fraction_topfeatures':f})
                    modelparam_metrics_df.append(metrics)
            
    # save model metrics for current dataset
    modelparam_metrics_df = pd.DataFrame(modelparam_metrics_df)
    modelparam_metrics_df = modelparam_metrics_df[['model_type','yvar','param_name', 'param_val', 'fraction_topfeatures', 'r2_train','mae_norm_train', 'mae_norm_cv','mae_train','mae_cv']]
    modelparam_metrics_df.to_csv(f'{data_folder}modelmetricsvs_featuresubsetsize_{dataset_name}.csv')
    
    # plot model metric for various parameters for current dataset
    for k, yvar_sublist in enumerate(yvar_sublist_sets):
        print(yvar_sublist)
        for model_type in models_to_eval_list: 
            fig, ax = plt.subplots(3,4, figsize=(27,17))

            for i, yvar in enumerate(yvar_sublist):
                # get relevant metrics to plot -- remove data that is out of range
                metrics_filt_allparams = modelparam_metrics_df.loc[(modelparam_metrics_df['model_type']==model_type) & (modelparam_metrics_df['yvar']==yvar)]
                param_name = str(metrics_filt_allparams.iloc[0]['param_name'])
                param_val_list = sort_list(list(set(metrics_filt_allparams['param_val'].tolist())))
                
                # get parameter value
                for param_val in param_val_list:
                    metrics_filt = metrics_filt_allparams[metrics_filt_allparams['param_val']==param_val]
                    metrics_filt = metrics_filt.loc[(metrics_filt.r2_train>=0) & (metrics_filt.mae_norm_cv<10)]
                    
                    # plot R2, MAE (CV), MAE (test) for all parameter values
                    ax[0][i].plot(metrics_filt['fraction_topfeatures'], metrics_filt['r2_train'], marker='*')
                    ax[0][i].plot(metrics_filt['fraction_topfeatures'], metrics_filt[f'{scoring}_norm_train'], marker='o')
                    ax[0][i].plot(metrics_filt['fraction_topfeatures'], metrics_filt[f'{scoring}_norm_cv'], marker='s')
                    ax[0][i].legend(['r2_train', f'{scoring}_train', f'{scoring}_cv'])
                    ax[0][i].set_ylabel('model metrics', fontsize=12)
                    ax[0][i].set_xlabel('fraction of top features', fontsize=12)
                    ax[0][i].set_title(yvar, fontsize=14)
                    if model_type=='lasso': 
                        ax[0][i].set_xscale('log')
                    # plot R2 divided by MAE(CV) for all parameter values
                    ax[1][i].plot(metrics_filt['fraction_topfeatures'], metrics_filt['r2_train']/metrics_filt[f'{scoring}_norm_cv'], marker='o')
                    ax[1][i].set_ylabel(f'r2/{scoring}(loocv)', fontsize=12)
                    ax[1][i].set_xlabel('fraction of top features', fontsize=12)
                    ax[1][i].set_title(yvar, fontsize=14)
                    if model_type=='lasso': 
                        ax[1][i].set_xscale('log')
                    # plot MAE (CV) vs MAE (test)
                    ax[2][i].plot(metrics_filt[f'{scoring}_norm_train'], metrics_filt[f'{scoring}_norm_cv'], marker='o')
                    ax[2][i].set_ylabel(f'{scoring} (loocv)', fontsize=12)
                    ax[2][i].set_xlabel(f'{scoring} (train)', fontsize=12)
                    ax[2][i].set_title(yvar, fontsize=14)
                    for j in range(len(metrics_filt)):
                        datapoint = metrics_filt.iloc[j]
                        fraction_topfeatures = float(datapoint['fraction_topfeatures'])
                        ax[2][i].annotate(str(fraction_topfeatures), (float(datapoint[f'{scoring}_norm_train']), float(datapoint[f'{scoring}_norm_cv'])))
            plt.legend([f'{param_name}={param_val}' for param_val in param_val_list])
            ymax = ax.flatten()[0].get_position().ymax               
            plt.suptitle(f'{model_type} metrics for various feature subset sizes \n{yvar_sublist}', y=ymax*1.07, fontsize=20)    
            fig.savefig(f'{figure_folder}modelmetrics_{model_type}_vs_featuresubsetsize_{dataset_name}_{k}.png', bbox_inches='tight')
            plt.show()                           

#%% Evaluate model performance for SELECTED feature subset and model size

featureset_list = [(1,0)] # [(0,0)] # 
dataset_suffix_list = ['_tier12'] # ['_subset1'] # ['_subset2']# ''
models_to_eval_list = ['randomforest', 'plsr', 'lasso'] # 
select_featuresubset_from_topfraction_or_curatedlist = 1 # 0: topfraction, 1: curated list
plot_feature_importance_heatmaps = False

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx), dataset_suffix in zip(featureset_list, dataset_suffix_list): 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}' 
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    # get top feature ordering
    if select_featuresubset_from_topfraction_or_curatedlist==0:
        overall_ordered_features = pd.read_csv(f'{data_folder}model_features_ordered_AGGREGATED_{dataset_name}.csv', index_col=0)
    
    # initialize variables for storing results
    model_metrics_df = []
    feature_importance_df = []
    feature_coef_df = []
    ypred_bymodel = {model_type: np.empty_like(Y) for model_type in models_to_eval_list}
    ypred_cv_bymodel = {model_type: np.empty_like(Y) for model_type in models_to_eval_list}

    # iterate through model types 
    for model_type in models_to_eval_list: 
        print(model_type)
        feature_order_txt = ['yvar']
        
        # iterate through yvar
        for i, yvar in enumerate(yvar_list_key):  
            
            # get y data
            y = Y[:,i]
            
            # get feature subset --> X dataset
            if select_featuresubset_from_topfraction_or_curatedlist==0:
                f = model_params[dataset_name+dataset_suffix][model_type][yvar][0]['f']
                xvar_list_ordered = overall_ordered_features.iloc[i, :].tolist()
                X_selected, xvar_list_selected = select_subset_of_X(X, xvar_list, xvar_list_ordered, f)
                
            elif select_featuresubset_from_topfraction_or_curatedlist==1:
                # xvar_list_selected
                xvar_list_selected = xvar_sublist_sets_bymodeltype[yvar][model_type]
                # X_selected
                idx_selected = [idx for idx, xvar in enumerate(xvar_list) if xvar in xvar_list_selected]
                X_selected = X[:,np.array(idx_selected)]
                # f
                f = round(len(xvar_list_selected)/len(xvar_list), 2)
                
            print('Fraction of features selected:', f)
            print('Dataset shape:', X_selected.shape)
                
            # get model parameters
            model_list = model_params[dataset_name+dataset_suffix][model_type][yvar]
            # fit model
            model_list, metrics = fit_model_with_cv(X_selected, y, yvar, model_list, plot_predictions=False, scale_data=True)
            # update metrics dict
            if model_type=='randomforest':
                param_name = 'n_estimators'
                metrics.update({'param_name': param_name, 'param_val': model_list[0][param_name], 'f':f, 'xvar_list': ', '.join(str(e) for e in xvar_list_selected)})
            
            # get feature importance and split into COEF and ORDER dicts
            feature_coefs, feature_importances = get_feature_importances(model_list[0], yvar, xvar_list_selected, plot_feature_importances=False, normalize_feature_scoring=True)
            # update aggregating dicts
            feature_importance_df += feature_importances
            feature_coef_df += feature_coefs
            model_metrics_df.append(metrics)
            ypred_bymodel[model_type][:, i] = model_list[0]['ypred']
            ypred_cv_bymodel[model_type][:, i] = model_list[0]['ypred_cv']
        
        # get feature importances for each model
        feature_importance_df_MODEL = pd.DataFrame(feature_importance_df)
        cols_feature_df = [col for col in ['model_type','yvar']+xvar_list if col in feature_importance_df_MODEL]
        feature_importance_df_MODEL = feature_importance_df_MODEL[cols_feature_df]
        feature_importance_df_MODEL = feature_importance_df_MODEL[feature_importance_df_MODEL['model_type']==model_type]
        feature_coef_df_MODEL = pd.DataFrame(feature_coef_df)
        feature_coef_df_MODEL = feature_coef_df_MODEL[cols_feature_df]
        feature_coef_df_MODEL = feature_coef_df_MODEL[feature_coef_df_MODEL['model_type']==model_type]
        
        # plot feature importance heatmaps
        if plot_feature_importance_heatmaps:
            arr_importance = plot_feature_importance_heatmap(feature_importance_df_MODEL.set_index('yvar').iloc[:,1:], cols_feature_df[2:], yvar_list, logscale_cmap=False, scale_vals=False, annotate=False, get_clustermap=False, figtitle=f'Feature importances ({model_type})', savefig=f'feature_importance_{model_type}_{dataset_name}{dataset_suffix}')
            arr_coef = plot_feature_importance_heatmap(feature_coef_df_MODEL.set_index('yvar').iloc[:,1:], cols_feature_df[2:], yvar_list, logscale_cmap=False, scale_vals=True, annotate=False, get_clustermap=False, figtitle=f'Feature coefficients, normalized ({model_type})', savefig=f'feature_coef_{model_type}_{dataset_name}{dataset_suffix}')
            arr_coef_importance_product = arr_importance*np.abs(arr_coef)
        
            # get feature importance barplots
            if model_type=='randomforest':
                plot_feature_importance_barplots(arr_coef, yvar_list, cols_feature_df[2:], label_xvar_by_indices=True, ncols=4, nrows=3, savefig=f'feature_importance_barplots_{model_type}_{dataset_name}{dataset_suffix}', figtitle=f'[{model_type}] feature importances for all y variables')
            else:
                plot_feature_importance_barplots(arr_coef_importance_product, yvar_list, cols_feature_df[2:], label_xvar_by_indices=True, ncols=4, nrows=3, savefig=f'feature_importance_barplots_{model_type}_{dataset_name}{dataset_suffix}', figtitle=f'[{model_type}] feature importances for all y variables')
        
        # order features by importance
        for i, yvar in enumerate(yvar_list):
            feature_scoring_yvar = feature_importance_df_MODEL.loc[feature_importance_df_MODEL.yvar==yvar, cols_feature_df[2:]]
            feature_scoring_yvar = feature_scoring_yvar.dropna(axis=1, how='all')
            xvar_list_selected = feature_scoring_yvar.columns.tolist()
            feature_scoring_yvar = feature_scoring_yvar.to_numpy().reshape(-1,)
            _, _, feature_ordering_yvar = order_features_by_importance(feature_scoring_yvar, xvar_list_selected)
            feature_order_txt_yvar = f'[{yvar}], {", ".join(str(e) for e in feature_ordering_yvar)}'
            feature_order_txt += [feature_order_txt_yvar]
            print(feature_order_txt_yvar, '\n')
        np.savetxt(f'{data_folder}model_{model_type}_feature_ordering_{dataset_name}{dataset_suffix}.csv',  feature_order_txt, fmt='%s', delimiter=', ', newline='\n')

            
    # aggregate all model metrics and save CSV
    model_metrics_df = pd.DataFrame(model_metrics_df)
    model_metrics_df = model_metrics_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    model_metrics_df.to_csv(f'{data_folder}model_metrics_{dataset_name}{dataset_suffix}.csv')
    
    # aggregate all feature importances and save CSV
    feature_importance_df = pd.DataFrame(feature_importance_df)
    feature_importance_df = feature_importance_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    feature_importance_df.to_csv(f'{data_folder}model_feature_importance_{dataset_name}{dataset_suffix}.csv')
    
    # aggregate all feature coefficients and save CSV
    feature_coef_df = pd.DataFrame(feature_coef_df)
    feature_coef_df = feature_coef_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    feature_coef_df.to_csv(f'{data_folder}model_feature_coef_{dataset_name}{dataset_suffix}.csv')      
                
    # generate scatter plots of ypred vs y
    for k, yvar_sublist in enumerate(yvar_sublist_sets[:1]):
    
        # iterate through yvar in sublist
        fig, ax = plt.subplots(3,4, figsize=(27,18))
        for i, yvar in enumerate(yvar_sublist): 
            yvar_idx = k*len(yvar_sublist)+i
            y = Y[:,yvar_idx]
    
            for j, model_type in enumerate(models_to_eval_list): 
                c = model_cmap[model_type]
                ypred = ypred_bymodel[model_type][:,yvar_idx]
                ypred_cv = ypred_cv_bymodel[model_type][:,yvar_idx]
                ax[j][i].scatter(y, ypred, c='k', alpha=0.5, marker='*')
                ax[j][i].scatter(y, ypred_cv, c=c, alpha=1, marker='o', s=16)
                ax[j][i].set_title(f'{model_type} <> {yvar}', fontsize=16)
                ax[j][i].set_ylabel(f'{yvar} (predicted)', fontsize=12)
                ax[j][i].set_xlabel(f'{yvar} (actual)', fontsize=12)          
                (xmin, xmax) = ax[j][i].get_xlim()
                (ymin, ymax) = ax[j][i].get_ylim()
                r2 = float(model_metrics_df[(model_metrics_df.model_type==model_type) & (model_metrics_df.yvar==yvar)].iloc[0].r2_train)
                r2_cv = float(model_metrics_df[(model_metrics_df.model_type==model_type) & (model_metrics_df.yvar==yvar)].iloc[0].r2_cv)
                ax[j][i].text(xmin+(xmax-xmin)*0.05, ymin+(ymax-ymin)*0.85, f'R2 (train): {r2} \nR2 (CV): {r2_cv}', fontsize=14)
                ax[j][i].legend(['train', 'CV'], loc='lower right')
                
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'Model Predicted vs. Actual values for various Y variables\n{yvar_sublist}', y=ymax*1.08, fontsize=24)    
        fig.savefig(f'{figure_folder}modelpredictions_scatterplots_{dataset_name}{dataset_suffix}_{k}.png', bbox_inches='tight')
        plt.show()

#%% pickle model dict

with open(f'{data_folder}models_and_params.pkl', 'wb') as f:
    pickle.dump(model_params, f)
    

#%% Compare model performance for feature SUBSET to FULL featureset
dataset_name = 'X1Y0' # 'X0Y0' # 
model_type = 'randomforest'
dataset_suffix = '_tier12' # '_subset2' # '_subset1' #
# get model metric data
model_metrics_df = pd.read_csv(f'{data_folder}model_metrics_{dataset_name}.csv', index_col=0)
model_metrics_df_featuresubset = pd.read_csv(f'{data_folder}model_metrics_{dataset_name}{dataset_suffix}.csv', index_col=0)

# # Plot model metrics and predictions
nrows=3
ncols=1
figsize=(27,17)
barwidth=0.25
figtitle = 'Metrics for various models evaluated on all Y variables'
savefig = f'{figure_folder}modelmetrics_allselectedmodels_allYvar_{dataset_name}{dataset_suffix}.png'

fig, ax = plt.subplots(nrows,ncols, figsize=figsize)
xtickpos = np.arange(len(yvar_list_key))+1
for i, yvar in enumerate(yvar_list_key):
    c = model_cmap[model_type]
    model_metrics_df_filt = model_metrics_df[(model_metrics_df.yvar==yvar) & (model_metrics_df.model_type==model_type)].iloc[0].to_dict()
    model_metrics_df_featuresubset_filt = model_metrics_df_featuresubset[(model_metrics_df_featuresubset.yvar==yvar) & (model_metrics_df_featuresubset.model_type==model_type)].iloc[0].to_dict()
    f = model_metrics_df_featuresubset_filt['f']
    ## R2
    ax[0].bar(xtickpos[i]-barwidth/2, model_metrics_df_filt['r2_train'], width=barwidth, label='all features', color=c, alpha=0.5)
    ax[0].bar(xtickpos[i]+barwidth/2, model_metrics_df_featuresubset_filt['r2_train'], width=barwidth, label='selected features', color=c)
    ax[0].text(xtickpos[i]+barwidth/4, model_metrics_df_featuresubset_filt['r2_train']*1.003, f'f={f}', fontsize=12)
    ## MAE_norm (train)
    ax[1].bar(xtickpos[i]-barwidth/2, model_metrics_df_filt['mae_norm_train'], width=barwidth, label='all features', color=c, alpha=0.5)
    ax[1].bar(xtickpos[i]+barwidth/2, model_metrics_df_featuresubset_filt['mae_norm_train'], width=barwidth, label='selected features', color=c)
    ## MAE_norm (CV)
    ax[2].bar(xtickpos[i]-barwidth/2, model_metrics_df_filt['mae_norm_cv'], width=barwidth, label='all features', color=c, alpha=0.5)
    ax[2].bar(xtickpos[i]+barwidth/2, model_metrics_df_featuresubset_filt['mae_norm_cv'], width=barwidth, label='selected features', color=c)

for k in range(nrows): 
    ax[k].set_xticks(xtickpos, yvar_list)
    
# set ylim for MAE plots
ymax_mae = np.max(model_metrics_df[['mae_norm_train', 'mae_norm_cv']].to_numpy())
ax[0].set_ylim([0.8,1])
ax[1].set_ylim([0,0.8])
ax[2].set_ylim([0,0.8])
ax[0].set_ylabel('R2', fontsize=16)
ax[1].set_ylabel('MAE, normalized (train)', fontsize=16)
ax[2].set_ylabel('MAE, normalized (LOOCV)', fontsize=16)
ymax = ax.flatten()[0].get_position().ymax
plt.legend(['all features', 'selected_features'], fontsize=16)
if figtitle is not None:
    plt.suptitle(figtitle, y=ymax*1.03, fontsize=24)    
if savefig is not None:
    fig.savefig(savefig, bbox_inches='tight')
plt.show()

