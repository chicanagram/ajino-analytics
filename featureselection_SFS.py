#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:03:57 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variables import model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
from get_datasets import data_folder, get_XYdata_for_featureset


def plot_feature_selection_metrics(res_model, p_init, figtitle=None, print_features_sequence=True, feature_selection_method='SFS'):
    fig, ax = plt.subplots(3, 4, figsize=(32,18))
    for j, yvar in enumerate(yvar_list_key):
        res_model_yvar = res_model[res_model.yvar==yvar]
        num_features = res_model_yvar['num_features'].to_numpy()

        # plot R2
        ax[0,j].plot(num_features, res_model_yvar['r2'].to_numpy(), linewidth=3, color='b')
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
        ax[0,j].set_ylabel('R2', fontsize=16)
        ax[1,j].set_ylabel('MAE (normalized)', fontsize=16)
        ax[2,j].set_ylabel('R2 over MAE (norm)', fontsize=16)
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'{figtitle}', y=ymax*1.06, fontsize=24)  
        
        # print sequence of features added
        if print_features_sequence:
            if feature_selection_method=='RFE': 
                colname = 'xvar_to_drop'
            if feature_selection_method=='SFS': 
                colname = 'xvar_to_add'
            xvar_seq = res_model_yvar[colname].to_list()
            print(yvar)
            for idx, xvar in enumerate(xvar_seq):
                print(f'{idx+1}: {xvar}')
            print()
    fig.savefig(f'{figure_folder}feature_selection_{feature_selection_method}_{model_type}_{dataset_name}.png', bbox_inches='tight')
    plt.show()
    
#%% Perform Recursive Feature Elimination
featureset_list =  [(1,0)]
models_to_evaluate = ['plsr', 'randomforest'] # ['plsr'] # 
feature_selection_method = 'SFS'

# load dataset
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    Y, X_init, Xscaled_init, yvar_list, xvar_list_init = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
    n = Xscaled_init.shape[0]
    p_init = X_init.shape[1]

    # initialize dataframe for recording results
    res_cols = ['model_type', 'yvar', 'num_features', 'r2', 'r2_cv', 'mae_train', 'mae_cv', 'mae_norm_train', 'mae_norm_cv', 'r2_over_mae_norm_train', 'r2_over_mae_norm_cv', 'xvar_to_add', 'xvar_list']
    res = []
    
    for model_type in models_to_evaluate:
        print(model_type)
        
        for i, yvar in enumerate(yvar_list_key):
            # get y and model parameters
            y = Y[:,i]
            model_list = model_params[dataset_name][model_type][yvar].copy()

            # get initial X variables
            Xscaled = Xscaled_init.copy()
            p = p_init
            xvar_list_available = xvar_list_init.copy()
            xvar_list = []
            xvar_idxs = []

            # iterate over number of features to add
            for k in range(1, p_init): 
                
                # iterate over all possible features that can be added
                print(model_type, yvar, 'len(xvar_list_available):', len(xvar_list_available))
                model_metrics_df_ = []
                for xvar_idx, xvar in enumerate(xvar_list_available):
                    print(xvar_idx, xvar, end='. ')
                    
                    # refresh temporary xvar_list with added xvar to test
                    xvar_list_, xvar_idxs_ = xvar_list.copy(), xvar_idxs.copy()
                    xvar_list_.append(xvar)
                    xvar_idxs_.append(xvar_list_init.index(xvar))
                    
                    # get dataset
                    Xscaled_ = Xscaled_init[:, np.array(xvar_idxs_)]
                    
                    # fit model with full feature set
                    model_list, metrics = fit_model_with_cv(Xscaled_, y, yvar, model_list, plot_predictions=False, print_output=False)
                    metrics.update({'xvar_to_add':xvar, 'xvar_list':', '.join(xvar_list_), 'num_features':k})
                    model_metrics_df_.append(metrics)
                    
                # get feature to add by selecting model with the best metrics
                model_metrics_df_ = pd.DataFrame(model_metrics_df_)
                model_metrics_df_['r2_over_mae_norm_cv'] = model_metrics_df_['r2_cv']/model_metrics_df_['mae_norm_cv']
                model_metrics_df_['r2_over_mae_norm_train'] = model_metrics_df_['r2']/model_metrics_df_['mae_norm_train']
                metrics_to_add = model_metrics_df_.sort_values(by='r2_over_mae_norm_cv', ascending=False).iloc[0].to_dict()
                xvar_to_add = metrics_to_add['xvar_to_add']
                print('\n', f'FEATURE {k} SELECTED: {xvar_to_add}','\n')
                
                # update xvar_list_available and resuls
                xvar_list_available.remove(xvar_to_add)
                xvar_list.append(xvar_to_add)
                xvar_idxs.append(xvar_list_init.index(xvar_to_add))
                res.append(metrics_to_add)
                
                # update n_components for plsr models
                if model_type=='plsr': 
                    model_list[0]['n_components'] = 5
            
        # aggregate model results dataframe       
        res_model = pd.DataFrame(res)[res_cols]
        res_model = res_model[res_model.model_type==model_type]
        
        # plot results for model type
        figtitle = f'{model_type} {feature_selection_method} feature selection metrics'
        plot_feature_selection_metrics(res_model, p_init, figtitle=figtitle, print_features_sequence=True, feature_selection_method=feature_selection_method)
        
    # aggregate overall results dataframe
    res = pd.DataFrame(res)[res_cols]
    res.to_csv(f'{data_folder}model_metrics_{feature_selection_method}_{dataset_name}.csv')
    
#%% get features retained at optimal point
num_features_selected_RFE = {
    'randomforest': {
        'Titer (mg/L)_14': 22,
        'mannosylation_14': 18,
        'fucosylation_14': 13,
        'galactosylation_14': 19
        },
    'plsr': {
        'Titer (mg/L)_14': 5,
        'mannosylation_14': 5,
        'fucosylation_14': 5,
        'galactosylation_14': 5
        }
    }
# load dataset
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    for model_type in ['randomforest']:
        print(model_type)
        for yvar in yvar_list_key:
            print(yvar)
            num_features = num_features_selected_RFE[model_type][yvar]
            res = pd.read_csv(f'{data_folder}{dataset_name}_RFE.csv', index_col=0)
            res_opt = res[(res.model_type==model_type) & (res.yvar==yvar) & (res.num_features==num_features)].iloc[0]
            xvar_list_selected = str(res_opt.xvar_list)[1:-1].replace("'",'').split(', ')
            for xvar in xvar_list_selected:
                print(xvar)            
            print(f"R2:{res_opt['r2']}, R2 (CV):{res_opt['r2_cv']}, MAE (train): {res_opt['mae_train']}, ({round(res_opt['mae_norm_train']*100,1)}%), MAE (CV): {res_opt['mae_cv']}, ({round(res_opt['mae_norm_cv']*100,1)}%),")
            print()
