#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:11:41 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from variables import model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key, xvar_sublist_sets_bymodeltype
from model_utils import r2_score, get_score, plot_model_metrics_all, fit_model_with_cv
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
from get_datasets import data_folder, get_XYdata_for_featureset


#%% Ensemble: randomforest + plsr

featureset_list = [(1,0), (1,0)] 
subset_suffix_list = ['', '_tier12'] # '_subset2' # '_subset1' #
models_to_ensemble = ['randomforest', 'plsr', 'lasso']


with open(f'{data_folder}models_and_params.pkl', 'rb') as f:
    model_params = pickle.load(f)


for (X_featureset_idx, Y_featureset_idx), subset_suffix in zip(featureset_list, subset_suffix_list): 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}' 
    print(dataset_name+subset_suffix)
    # get relevant dataset with chosen features
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
    
    # get model metric data
    model_metrics_df = pd.read_csv(f'{data_folder}model_metrics_{dataset_name}.csv', index_col=0)
    model_metrics_df_featuresubset = pd.read_csv(f'{data_folder}model_metrics_{dataset_name}{subset_suffix}.csv', index_col=0)
    
    # initialize variables for storing results
    model_metrics_df = []
    feature_importance_df = []
    feature_coef_df = []
        
    # iterate through yvar
    for i, yvar in enumerate(yvar_list_key):  
        print(yvar)
        y = Y[:,i]
        ypred_bymodel = {model_type: np.empty_like(y) for model_type in models_to_ensemble}
        ypred_cv_bymodel = {model_type: np.empty_like(y) for model_type in models_to_ensemble}
        weights_bymodel = {model_type:None for model_type in models_to_ensemble}
        
        # iterate through model types 
        for model_type in models_to_ensemble: 
            
            # get feature set
            if subset_suffix == '':
                xvar_list_selected = xvar_list
                X_selected = X
                f = 1
            else:
                xvar_list_selected = xvar_sublist_sets_bymodeltype[yvar][model_type]
                # Xscaled_selected
                idx_selected = [idx for idx, xvar in enumerate(xvar_list) if xvar in xvar_list_selected]
                X_selected = X[:,np.array(idx_selected)]
                f = round(len(xvar_list_selected)/len(xvar_list), 2)
                
            # get model parameters
            model_list = model_params[dataset_name+subset_suffix][model_type][yvar]
            # fit model
            model_list, metrics = fit_model_with_cv(X_selected, y, yvar, model_list, plot_predictions=False, scale_data=True)
            # get predictions
            ypred_bymodel[model_type] = model_list[0]['ypred']
            ypred_cv_bymodel[model_type] = model_list[0]['ypred_cv']
            
            # get weight for model from R2 (CV) for model
            weights_bymodel[model_type] = round(metrics['r2_cv']**2,3)
            
            # update metrics
            metrics.update({'f':f, 'xvar_list': ', '.join(str(e) for e in xvar_list_selected)})
            model_metrics_df.append(metrics)
            
        # ensemble model 
        ypred_ensemble = np.zeros_like(y)
        ypred_cv_ensemble = np.zeros_like(y)
        weights_sum = 0
        for model_type in models_to_ensemble:
            ypred_ensemble += ypred_bymodel[model_type]*weights_bymodel[model_type]
            ypred_cv_ensemble += ypred_cv_bymodel[model_type]*weights_bymodel[model_type]
            weights_sum += weights_bymodel[model_type]
        ypred_ensemble = ypred_ensemble/weights_sum
        ypred_cv_ensemble = ypred_cv_ensemble/weights_sum
        
        # calculate ensemble R2 (train and CV)
        r2_train_ensemble = round(r2_score(y, ypred_ensemble),2)
        r2_cv_ensemble = round(r2_score(y, ypred_cv_ensemble),2)
        mae_train_ensemble = round(get_score(y, ypred_ensemble, 'mae'),2)
        mae_norm_train_ensemble = round(mae_train_ensemble/np.mean(y),2)
        mae_cv_ensemble = round(get_score(y, ypred_cv_ensemble, 'mae'),2)
        mae_norm_cv_ensemble = round(mae_cv_ensemble/np.mean(y),2)
        ensemble_metrics = {
            'model_type':'ensemble', 'yvar':yvar, 'model_params':weights_bymodel, 
            'r2':r2_train_ensemble, 'r2_cv':r2_cv_ensemble, 
            'mae_train': mae_train_ensemble, 'mae_norm_train': mae_norm_train_ensemble,
            'mae_cv': mae_cv_ensemble, 'mae_norm_cv': mae_norm_cv_ensemble,
            
                }
        model_metrics_df.append(ensemble_metrics)
        print(f'[ENSEMBLE] R2:{r2_train_ensemble}, R2 (CV):{r2_cv_ensemble}, fitted mae:{mae_train_ensemble} ({round(mae_norm_train_ensemble*100,2)}%), mae (CV):{mae_cv_ensemble} ({round(mae_norm_cv_ensemble*100,2)}%)')
        print()
        
    # aggregate all model metrics and save CSV
    model_metrics_df = pd.DataFrame(model_metrics_df)
    model_metrics_df = model_metrics_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    model_metrics_df.to_csv(f'{data_folder}model_metrics_{dataset_name}{subset_suffix}.csv')
            

#%% plot ensemble results
dataset_name = 'X1Y0'
subset_suffix = '_tier12'
model_metrics_df_allfeatures = pd.read_csv(f'{data_folder}model_metrics_{dataset_name}.csv', index_col=0)
model_metrics_df_subset = pd.read_csv(f'{data_folder}model_metrics_{dataset_name}{subset_suffix}.csv', index_col=0)
model_metrics_df_dict = {0:model_metrics_df_allfeatures, 1: model_metrics_df_subset}


    
figtitle = 'Metrics for various models (and ensembles) using all vs. selected features'
savefig = f'{figure_folder}modelmetrics_allmodels_keyYvar_{dataset_name}{subset_suffix}.png'
plot_model_metrics_all(model_metrics_df_dict, ['randomforest','plsr','lasso','ensemble'], yvar_list_key, 
                       nrows=2, ncols=1, figsize=(30,15), barwidth=0.8, 
                       figtitle=figtitle, savefig=savefig, model_cmap={'randomforest':'r', 'plsr':'b', 'lasso':'g', 'ensemble':'orange'})

#%% pickle model dict

with open(f'{data_folder}models_and_params.pkl', 'wb') as f:
    pickle.dump(model_params, f)
