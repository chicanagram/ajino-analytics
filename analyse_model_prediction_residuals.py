#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:54:10 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variables import data_folder, yvar_list_key, feature_selections, model_params, model_cmap, process_features
from model_utils import run_trainval_test, plot_model_metrics_all
from utils import sort_list, get_XYdata_for_featureset
from plot_utils import figure_folder, convert_figidx_to_rowcolidx

def unshuffle_array(arr, shuffle_idx):
    arr_unshuffled = np.zeros_like(arr)
    unshuffle_mapping = {i:idx_orig for i, idx_orig in enumerate(shuffle_idx)}
    
    for i, idx_orig in unshuffle_mapping.items():
        if len(arr.shape)==1:
            arr_unshuffled[idx_orig] = arr[i]
        elif len(arr.shape)==2:
            arr_unshuffled[idx_orig,:] = arr[i,:]
    return arr_unshuffled

#%%
# get data
X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
featureset_suffix =  '_combi1' # '_combi2' # '_compactness-opt'
dataset_name_wsuffix = dataset_name + dataset_suffix
XYdata = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
models_to_eval_list = ['randomforest', 'xgb']
yvar_list = yvar_list_key
features_selected_dict = feature_selections[featureset_suffix]

shuffle_idx = np.arange(len(Y))
np.random.seed(seed=0)
np.random.shuffle(shuffle_idx)
Y_unshuffled = unshuffle_array(Y, shuffle_idx)

#%% 
# get results WITH feature selection
kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS, ypred_train_bymodel, ypred_test_bymodel = run_trainval_test(X, Y, yvar_list, features_selected_dict, xvar_list_all, dataset_name_wsuffix, featureset_suffix, models_to_eval_list=models_to_eval_list, model_cmap=model_cmap)


#%% plot residuals

metadata_cols = ['exp_label', 'Basal medium', 'Feed medium', 'DO', 'pH', 'feed %']
metadata = XYdata.loc[:, metadata_cols].to_dict(orient='list')
metadata_str = [f'{label}: {basal.split("-")[1]}/{feed.split("-")[1]}, {int(DO)}%, {pH}, {int(feed_percentage)}%' for label, basal, feed, DO, pH, feed_percentage in zip(metadata['exp_label'], metadata['Basal medium'], metadata['Feed medium'], metadata['DO'], metadata['pH'], metadata['feed %'])]
model_type = 'ENSEMBLE'

for i, yvar in enumerate(yvar_list[:2]): 
    nrows, ncols = 1,1
    fig, ax = plt.subplots(nrows, ncols, figsize=(20,12))
    # row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
    y = Y_unshuffled[:,i]
    ymean = np.mean(y)
    x = np.arange(len(y))
    ypred_train = unshuffle_array(ypred_train_bymodel[model_type][:,i], shuffle_idx)
    residuals_train = (ypred_train - y)/ymean
    ypred_test = unshuffle_array(ypred_test_bymodel[model_type][:,i], shuffle_idx)
    residuals_test = (ypred_test - y)/ymean
    ax.scatter(x, residuals_train, color='b', alpha=0.7)
    ax.scatter(x, residuals_test, color='r', alpha=0.7)
    ax.set_ylabel(f'{yvar} residuals', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(metadata_str, fontsize=10, rotation=90)
    plt.legend(['train', 'test'])
    plt.savefig(f'{figure_folder}model_prediction_residuals_{dataset_name_wsuffix}_fs={featureset_suffix[1:]}_{model_type}_{i}.png', bbox_inches='tight')
    plt.show()
    
    
#%% plot actual predictions
        
# get exp labels
metadata_cols = ['exp_label', 'Basal medium', 'Feed medium', 'DO', 'pH', 'feed %']
metadata = XYdata.loc[:, metadata_cols].to_dict(orient='list')
metadata_str = [f'{label}: {basal.split("-")[1]}/{feed.split("-")[1]}, {int(DO)}%, {pH}, {int(feed_percentage)}%' for label, basal, feed, DO, pH, feed_percentage in zip(metadata['exp_label'], metadata['Basal medium'], metadata['Feed medium'], metadata['DO'], metadata['pH'], metadata['feed %'])]
model_type = 'ENSEMBLE'

for i, yvar in enumerate(yvar_list[:2]): 
    nrows, ncols = 1,1
    fig, ax = plt.subplots(nrows, ncols, figsize=(20,12))
    # row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
    y = Y_unshuffled[:,i]
    x = np.arange(len(y))
    ypred_test = unshuffle_array(ypred_test_bymodel[model_type][:,i], shuffle_idx)
    ax.scatter(x, y, color='b', alpha=0.7)
    ax.scatter(x, ypred_test, color='r', alpha=0.7)
    ax.set_ylabel(f'{yvar} predictions', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(metadata_str, fontsize=10, rotation=90)
    plt.legend(['actual', 'predicted'])
    # plt.savefig(f'{figure_folder}model_prediction_residuals_{dataset_name_wsuffix}_fs={featureset_suffix[1:]}_{model_type}_{i}.png', bbox_inches='tight')
    plt.show()
    
    
    
    