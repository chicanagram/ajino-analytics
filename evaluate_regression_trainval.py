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
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
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

featureset_list =  [(8,0)] # 
models_to_eval_list = ['randomforest', 'lasso'] # ['randomforest', 'xgb', 'plsr', 'lasso'] # ['randomforest']# 
# dataset_suffix = '_avg'
dataset_suffix = ''
f = 1
cv = None # # 
 
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
        for i, yvar in enumerate(yvar_list_key):     
            
            model_list = model_params[dataset_name_wsuffix][model_type][yvar]
            y = Y[:,i]
            model_list, metrics = fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, scale_data=True, cv=cv)
            metrics.update({'f':f, 'xvar_list': ', '.join(str(e) for e in xvar_list)})
            
            # get feature importance and split into COEF and ORDER dicts
            feature_coefs, feature_importances = get_feature_importances(model_list[0], yvar, xvar_list, plot_feature_importances=False)
            xvar_list_sorted, coefs_sorted = order_features_by_coefficient_importance(feature_coefs, xvar_list, filter_out_zeros=True)
            if model_type=='lasso':
                print('xvar_list_sorted:', ', '.join(xvar_list_sorted))
            # update aggregating dicts
            feature_importance_df += feature_importances
            feature_coef_df += feature_coefs
            model_metrics_df.append(metrics)
            ypred_bymodel[model_type][:, i] = model_list[0]['ypred']
            ypred_cv_bymodel[model_type][:, i] = model_list[0]['ypred_cv']
        print()
        
    # plot coefficients
    feature_importance_df = pd.DataFrame(feature_importance_df)
    feature_coef_df = pd.DataFrame(feature_coef_df)
    feature_coef_df_ABS = feature_coef_df.copy()
    feature_coef_df_ABS.iloc[:,2:] = np.abs(feature_coef_df_ABS.iloc[:,2:].to_numpy())
    plot_feature_importance_barplots_bymodel(feature_coef_df_ABS, yvar_list=yvar_list_key, xvar_list=xvar_list, model_list=models_to_eval_list, order_by_feature_importance=False, label_xvar_by_indices=True, model_cmap=model_cmap, savefig=f'feature_prominence_barplots_{dataset_name}{dataset_suffix}', figtitle=f'Feature prominences for all y variables')
    
    
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
    figtitle = 'CV metrics for various models evaluated on all Y variables'
    savefig = f'{figure_folder}modelmetrics_allselectedmodels_allYvar_{dataset_name}{dataset_suffix}.png'
    plot_model_metrics(model_metrics_df, models_to_eval_list, yvar_list_key, nrows=2, ncols=1, figsize=(30,15), barwidth=0.8, suffix_list=['_train', '_cv'], figtitle=figtitle, savefig=savefig, model_cmap=model_cmap, plot_errors=False)

    # generate scatter plots of ypred vs y
    for k, yvar_sublist in enumerate(yvar_sublist_sets[:1]):
    
        # iterate through yvar in sublist
        fig, ax = plt.subplots(len(models_to_eval_list),(len(yvar_sublist)), figsize=(27,18))
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

#%% get feature importance plots
drop_process_features = True

# feature coefficient barplots
feature_importance_df = pd.DataFrame(feature_importance_df)
feature_coef_df = pd.DataFrame(feature_coef_df)
feature_coef_df.iloc[:,2:] = np.abs(feature_coef_df.iloc[:,2:].to_numpy())
plot_feature_importance_barplots_bymodel(feature_coef_df, yvar_list=yvar_list_key, xvar_list=xvar_list, model_list=models_to_eval_list, order_by_feature_importance=False, label_xvar_by_indices=True, model_cmap={'randomforest':'r', 'plsr':'b', 'lasso':'g'}, savefig=f'feature_prominence_barplots_{dataset_name}{dataset_suffix}', figtitle=f'Feature prominences for all y variables')

# feature importance heatmaps
for yvar_idx, yvar in enumerate(yvar_list_key):
    feature_importance_df_YVAR = feature_importance_df[feature_importance_df['yvar']==yvar].sort_values(by='model_type')
    feature_coef_df_YVAR = feature_coef_df[feature_coef_df['yvar']==yvar].sort_values(by='model_type')
    model_list = feature_importance_df_YVAR.model_type.tolist()
    print(model_list)
    # plot feature importance heatmaps
    # arr_importance = plot_feature_importance_heatmap(feature_importance_df_YVAR.iloc[:,2:], xvar_list, model_list, logscale_cmap=False, scale_vals=False, figtitle=f'Feature importances ({yvar})', savefig=f'feature_importance_{yvar_idx}_{dataset_name}{dataset_suffix}')
    # plot feature coefficient
    if drop_process_features:
        arr_coef = plot_feature_importance_heatmap(feature_coef_df_YVAR.iloc[:,2:-3].abs(), np.array(xvar_list)[:-3], model_list, logscale_cmap=False, scale_vals=True, figtitle=f'Feature coefficients ({yvar})', savefig=f'feature_coef_{yvar_idx}_{dataset_name}{dataset_suffix}')
    else:
        arr_coef = plot_feature_importance_heatmap(feature_coef_df_YVAR.iloc[:,2:].abs(), xvar_list, model_list, logscale_cmap=False, scale_vals=True, figtitle=f'Feature coefficients ({yvar})', savefig=f'feature_coef_{yvar_idx}_{dataset_name}{dataset_suffix}')
     
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

