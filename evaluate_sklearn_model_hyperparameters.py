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
from variables import data_folder, model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key, model_cmap
from model_utils import fit_model_with_cv, get_feature_importances, order_features_by_coefficient_importance, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_feature_importance_barplots_bymodel, plot_model_metrics, plot_model_metrics_cv, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, convert_figidx_to_rowcolidx
from utils import get_XYdata_for_featureset

#%% Evaluate different feature sets and model parameters 
scoring = 'mae'
featureset_list = [(1,0)] # [(1,0), (0,0)] # 
# dataset_suffix = '_avg' 
dataset_suffix = ''
model_params_to_eval = [
    {'model_type': 'xgb', 'params_to_eval': ('n_estimators', [20,40,60,80,100,120,140,160,180])},
    # {'model_type': 'randomforest', 'params_to_eval': ('n_estimators', [20,40,60,80,100,120,140,160,180])},
    # {'model_type': 'plsr', 'params_to_eval': ('n_components',[2,3,4,5,6,7,8,9,10,11,12,14])},
    # {'model_type': 'lasso', 'max_iter':500000, 'params_to_eval': ('alpha', [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100])},
    ]

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_name_wsuffix = dataset_name + dataset_suffix
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    modelparam_metrics_df = []
    
    for i, yvar in enumerate(yvar_list_key): 
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
                model_list, metrics = fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, score_type_list=['r2', 'mae', 'rmse'], scale_data=True)
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

