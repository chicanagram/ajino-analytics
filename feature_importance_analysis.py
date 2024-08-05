#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:54:14 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from variables import sort_list
from model_utils import plot_feature_importance_heatmap, plot_feature_importance_barplots, order_features_by_importance
from plot_utils import figure_folder, convert_figidx_to_rowcolidx, heatmap
from get_datasets import data_folder, get_XYdata_for_featureset


#%% OVERALL Feature importance analysis
featureset_list = [(0,0), (1,0)]
models_to_eval_list = ['randomforest','plsr', 'lasso'] 

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
 
    # open saved CSVs containing metrics, feature coefficient, feature scoring data
    model_metrics_df = pd.read_csv(f'{data_folder}model_metrics_{dataset_name}.csv')
    feature_coef_df = pd.read_csv(f'{data_folder}model_feature_coef_{dataset_name}.csv')
    feature_importance_df = pd.read_csv(f'{data_folder}model_feature_importance_{dataset_name}.csv')
    
    # initialize arrays for calculating overall feature importance
    feature_coefs_abs_arr = np.zeros((len(yvar_list), len(xvar_list), len(models_to_eval_list)))
    feature_importances_arr = np.zeros((len(yvar_list), len(xvar_list), len(models_to_eval_list)))
    r2_arr = np.zeros((len(yvar_list), len(xvar_list), len(models_to_eval_list)))
    
    # iterate through y variables
    for i, yvar in enumerate(yvar_list):
        for k, model_type in enumerate(models_to_eval_list):
            # get model R2 score
            r2_arr[i,:,k] = float(model_metrics_df[(model_metrics_df.model_type==model_type) & (model_metrics_df.yvar==yvar)].iloc[0].r2)
            # get absolute, scaled feature coefficients
            feature_coefs_abs = np.abs(feature_coef_df[(feature_coef_df.model_type==model_type) & (feature_coef_df.yvar==yvar)][xvar_list].to_numpy().reshape(-1,))
            feature_coefs_abs = feature_coefs_abs/np.max(feature_coefs_abs)
            feature_coefs_abs_arr[i,:,k] = feature_coefs_abs
              # get feature importances
            feature_importances_arr[i,:,k] = feature_importance_df[(feature_importance_df.model_type==model_type) & (feature_importance_df.yvar==yvar)][xvar_list].to_numpy().reshape(-1,)
    
    # get weighted arrays
    r2_weightsum = np.sum(r2_arr, axis=2)
    feature_coefs_abs_arr_weightedavg =  np.sum(feature_coefs_abs_arr*r2_arr, axis=2)/r2_weightsum
    feature_importances_arr_weightedavg = np.sum(feature_importances_arr*r2_arr, axis=2)/r2_weightsum
    feature_prominence = feature_coefs_abs_arr_weightedavg*feature_importances_arr_weightedavg
    
    # plot overall weighted feature importances as bar plots
    plot_feature_importance_barplots(feature_prominence, yvar_list, xvar_list, label_xvar_by_indices=True, ncols=4, nrows=3, savefig=f'overall_feature_prominence_barplots_{dataset_name}', figtitle='Overall feature prominence for all y variables')
    
    # plot heatmap
    overall_feature_importance_heatmap = pd.DataFrame(feature_importances_arr_weightedavg, index=yvar_list, columns=xvar_list)
    _ = plot_feature_importance_heatmap(overall_feature_importance_heatmap, xvar_list, yvar_list, logscale_cmap=True, annotate=False, figtitle='Overall feature importances for all y variables', savefig=f'overall_feature_importance_heatmap_{dataset_name}')
    
    # order x variable importance for each y variable using feature_coef_importance_product
    overall_feature_importance = []
    overall_feature_prominence = []
    feature_ordering = {}
    
    for i, yvar in enumerate(yvar_list):
        print(yvar)
        feature_prominence_yvar = feature_prominence[i, :]
        overall_feature_importance_yvar, feature_scoring_yvar, feature_ordering_yvar = order_features_by_importance(feature_prominence_yvar, xvar_list)
        overall_feature_importance_dict = {'yvar': yvar}
        overall_feature_importance_dict.update(overall_feature_importance_yvar)
        overall_feature_importance.append(overall_feature_importance_dict)
        overall_feature_prominence_dict = {'yvar': yvar}
        overall_feature_prominence_dict.update(feature_scoring_yvar)
        overall_feature_prominence.append(overall_feature_prominence_dict)
        feature_ordering[yvar] = feature_ordering_yvar
        
    overall_feature_importance = pd.DataFrame(overall_feature_importance).set_index('yvar')
    overall_feature_prominence = pd.DataFrame(overall_feature_prominence).set_index('yvar')
    overall_ordered_features = pd.DataFrame(feature_ordering).transpose()
    overall_feature_importance.to_csv(f'{data_folder}model_feature_importance_AGGREGATED_{dataset_name}.csv')
    overall_feature_prominence.to_csv(f'{data_folder}model_feature_prominence_AGGREGATED_{dataset_name}.csv')
    overall_ordered_features.to_csv(f'{data_folder}model_features_ordered_AGGREGATED_{dataset_name}.csv')

#%% Get overlapping features 
model_type = 'randomforest'
dataset_name_list = ['X0Y0', 'X1Y0','X0Y0', 'X1Y0']
subset_suffix_list = ['', '', '_subset2', '_subset1']
fraction_of_features_to_include = [0.3, 0.3, 1, 1]
feature_ordering_dict = {}

for dataset_name, subset_suffix in zip(dataset_name_list,subset_suffix_list):
    feature_ordering_dict_dataset = {}
    with open(f'{data_folder}model_{model_type}_feature_ordering_{dataset_name}{subset_suffix}.csv') as f:
        for i, line in enumerate(f):       
            if i>0:
                yvar_endidx = line.find('],')
                yvar = line[1:yvar_endidx]
                xvar_startidx = yvar_endidx + 3
                xvar_list_ordered = line[xvar_startidx:-1].split(', ')
                feature_ordering_dict_dataset.update({yvar:xvar_list_ordered})
    feature_ordering_dict.update({f'{dataset_name}{subset_suffix}': feature_ordering_dict_dataset})

#%% parse to variable counts to base variables
var_count_dict_base_all_byyvar = {}
for dataset_name, subset_suffix, f in zip(dataset_name_list,subset_suffix_list, fraction_of_features_to_include):
    dataset_fullname = f'{dataset_name}{subset_suffix}'
    print(dataset_fullname)
    feature_ordering_dict_dataset = feature_ordering_dict[dataset_fullname]
    var_count_dict_base_byyvar = {}
    # iterate through yvar
    for yvar in yvar_list:
        var_count_dict_base_byyvar[yvar] = {}
        # get xvar_list to include
        xvar_list_ordered = feature_ordering_dict_dataset[yvar]
        num_features_to_include = int(np.ceil(f*len(xvar_list_ordered)))
        xvar_list_ordered_filt = xvar_list_ordered[:num_features_to_include]
        # iterate through xvar
        for xvar in xvar_list_ordered_filt: 
            ## var_count_dict_base ##             
            # remove suffix, if present
            if xvar.find('_0')>-1: 
                xvar_base = xvar[:xvar.find('_0')]
            elif xvar.find('_basal')>-1: 
                xvar_base = xvar[:xvar.find('_basal')]
            elif xvar.find('_feed')>-1: 
                xvar_base = xvar[:xvar.find('_feed')]
            else: 
                xvar_base = xvar
            # update dict
            if xvar_base not in var_count_dict_base_byyvar[yvar]:
                var_count_dict_base_byyvar[yvar].update({xvar_base:1})      
            else: 
                var_count_dict_base_byyvar[yvar][xvar_base] += 1
            
        # sort dict by keys
        var_count_dict_base_byyvar[yvar] = OrderedDict(sorted(var_count_dict_base_byyvar[yvar].items()))
        
    # update aggregate dict
    var_count_dict_base_all_byyvar.update({dataset_fullname:var_count_dict_base_byyvar})
    print(var_count_dict_base_all_byyvar)
    print()
    
#%% find most frequently occurring features between the first 4 yvar (titer, mannosylation, etc.)
key_yvar = ['Titer (mg/L)_14', 'mannosylation_14', 'fucosylation_14', 'galactosylation_14']
var_count_dict_all = {}
var_count_dict_base_all = {}
xvar_base_all = []
for dataset_name, subset_suffix, f in zip(dataset_name_list,subset_suffix_list, fraction_of_features_to_include):
    dataset_fullname = f'{dataset_name}{subset_suffix}'
    print(dataset_fullname)
    # feature_ordering_dict_dataset = feature_ordering_dict[dataset_fullname]
    var_count_dict = {}
    var_count_dict_base = {}
    
    # iterate through yvar
    for yvar in key_yvar:
        
        ## var_count_dict ##
        # get xvar_list to include
        xvar_list_ordered = feature_ordering_dict[dataset_fullname][yvar]
        num_features_to_include = int(np.ceil(f*len(xvar_list_ordered)))
        xvar_list_ordered_filt = xvar_list_ordered[:num_features_to_include]
        # iterate through xvar
        for xvar in xvar_list_ordered_filt: 
            ## var_count_dict ##
            if xvar not in var_count_dict:
                var_count_dict.update({xvar:1})      
            else: 
                var_count_dict[xvar] += 1
                
        ## var_count_dict_base ##    
        var_count_dict_base_byyvar = var_count_dict_base_all_byyvar[dataset_fullname][yvar]
        for xvar_base in var_count_dict_base_byyvar: 
            # update dict
            if xvar_base not in var_count_dict_base:
                var_count_dict_base.update({xvar_base:1})      
            else: 
                var_count_dict_base[xvar_base] += 1
            
        ## sort dicts by keys ##
        var_count_dict = OrderedDict(sorted(var_count_dict.items()))
        var_count_dict_base = OrderedDict(sorted(var_count_dict_base.items()))
        
    # update aggregate dict
    var_count_dict_all.update({dataset_fullname:var_count_dict})
    var_count_dict_base_all.update({dataset_fullname:var_count_dict_base})
    xvar_base_all += list(var_count_dict_base.keys())
    print(var_count_dict)
    print()
    print(var_count_dict_base)
    print()
    
xvar_base_all = sort_list(list(set(xvar_base_all)))
print(len(xvar_base_all), xvar_base_all)
    
# plot heatmap of feature occurrences 
# get all base features represented
ylabels = list(var_count_dict_base_all.keys())
nan_array = np.zeros((len(var_count_dict_base_all), len(xvar_base_all)))
nan_array[:] = np.nan
top_feature_occurrence_df = pd.DataFrame(nan_array, columns=xvar_base_all, index=ylabels)
for dataset_fullname, var_count_dict_base in var_count_dict_base_all.items():
    for xvar_base, xvar_occurrence in var_count_dict_base.items():
        top_feature_occurrence_df.loc[dataset_fullname, xvar_base] = int(xvar_occurrence)

fig, ax = plt.subplots(1,1, figsize=(30, 10))
top_feature_occurrence_arr = top_feature_occurrence_df.to_numpy()
top_feature_occurrence_norm_arr = top_feature_occurrence_arr/np.nansum(top_feature_occurrence_arr, axis=1).reshape(-1,1)
heatmap(top_feature_occurrence_norm_arr, c='viridis', ax=ax, cbar_kw={}, cbarlabel="", datamin=None, datamax=None, logscale_cmap=False, annotate=3, row_labels=ylabels, col_labels=xvar_base_all)
fig.savefig(f'{figure_folder}top_feature_occurrences_keyCQAscombined_bydataset.png', bbox_inches='tight')
plt.show()


#%% find common base features between the two featuresets, for each key CQA
dataset_groupings_list = [('X0Y0', 'X1Y0'), ('X0Y0_subset2', 'X1Y0_subset1'), ('X0Y0', 'X1Y0', 'X0Y0_subset2', 'X1Y0_subset1')]

var_count_dict_base_featuresetagg = {}
for yvar in key_yvar:
    for k, dataset_grouping in enumerate(dataset_groupings_list): 
        featuresetagg_dict = {}
        for dataset_fullname in dataset_grouping: 
            var_count_dict_base_yvar = var_count_dict_base_all_byyvar[dataset_fullname][yvar]
            for xvar in var_count_dict_base_yvar:
                if xvar not in featuresetagg_dict:
                    featuresetagg_dict[xvar] = 1
                else: 
                    featuresetagg_dict[xvar] += 1   
        yvar_featureset_label = f'{yvar}_[{k}]'
        var_count_dict_base_featuresetagg.update({yvar_featureset_label: featuresetagg_dict})
        xvar_base_all += list(featuresetagg_dict.keys())        
    
xvar_base_all = sort_list(list(set(xvar_base_all)))
print(len(xvar_base_all), xvar_base_all)

# plot heatmap of feature occurrences 
# get all base features represented
ylabels = list(var_count_dict_base_featuresetagg.keys())
nan_array = np.zeros((len(var_count_dict_base_featuresetagg), len(xvar_base_all)))
nan_array[:] = np.nan
top_feature_occurrence_df = pd.DataFrame(nan_array, columns=xvar_base_all, index=ylabels)
for yvar_featureset_label, var_count_dict_base in var_count_dict_base_featuresetagg.items():
    for xvar_base, xvar_occurrence in var_count_dict_base.items():
        top_feature_occurrence_df.loc[yvar_featureset_label, xvar_base] = int(xvar_occurrence)

fig, ax = plt.subplots(1,1, figsize=(30, 10))
top_feature_occurrence_arr = top_feature_occurrence_df.to_numpy()
top_feature_occurrence_norm_arr = top_feature_occurrence_arr/np.nansum(top_feature_occurrence_arr, axis=1).reshape(-1,1)
heatmap(top_feature_occurrence_norm_arr, c='viridis', ax=ax, cbar_kw={}, cbarlabel="", datamin=None, datamax=None, logscale_cmap=False, annotate=3, row_labels=ylabels, col_labels=xvar_base_all)
fig.savefig(f'{figure_folder}top_feature_occurrences_keyCQAsindiv_datasetscombined.png', bbox_inches='tight')
plt.show()    


#%% Calculate feature importance fingerprint similarity metric


        
        