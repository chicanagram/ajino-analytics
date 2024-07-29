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
from variables import model_params, dict_update, yvar_sublist_sets
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, get_order_of_element_sizes, plot_feature_importance_barplots
from plot_utils import figure_folder, convert_figidx_to_rowcolidx

data_folder = '../ajino-analytics-data/'

# load XY array dict
with open(f'{data_folder}XYarrdict.pkl', 'rb') as f:
    XYarr_dict = pickle.load(f)


def get_XYdata_for_featureset(featureset_idx):
    featureset_idx = 0
    XYarr_dict_featureset = XYarr_dict[featureset_idx]
    Y = XYarr_dict_featureset['Y']       
    X = XYarr_dict_featureset['X']
    Xscaled = XYarr_dict_featureset['Xscaled']
    yvar_list = XYarr_dict_featureset['yvar_list'] 
    xvar_list = XYarr_dict_featureset['xvar_list']
    return Y, X, Xscaled, yvar_list, xvar_list

featureset_idx = 0
Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(featureset_idx)


#%% Evaluate different feature sets and model parameters 
scoring = 'mae'
modelparam_metrics_df = []
featureset_list = [0]
model_params_to_eval = [
    {'model_type': 'randomforest', 'params_to_eval': ('n_estimators', [20,40,60,80,100,120,140])},
    {'model_type': 'plsr', 'params_to_eval': ('n_components',[2,3,4,5,6,7,8,9,10,11,12,14,16,18,20])},
    {'model_type': 'lasso', 'max_iter':100000, 'params_to_eval': ('alpha', [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100])},
    ]

# get relevant dataset with chosen features
for featureset_idx in featureset_list: 
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(featureset_idx)
    
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
                print(param_name, param_val, end=':: ')
                model_list[0].update({param_name:param_val})
                
                # fit model with selected features and parameters and get metrics 
                model_list, metrics = fit_model_with_cv(Xscaled,y, yvar, model_list, plot_predictions=False, scoring=scoring)
                # update metrics dict with param val
                metrics.update({'param_name': param_name, 'param_val': param_val})
                modelparam_metrics_df.append(metrics)
        
modelparam_metrics_df = pd.DataFrame(modelparam_metrics_df)
modelparam_metrics_df = modelparam_metrics_df[['model_type','yvar','param_name', 'param_val', 'r2','mae_norm_train', 'mae_norm_cv','mae_train','mae_cv']]
modelparam_metrics_df.to_csv(f'{data_folder}modelmetrics_vs_params.csv')


#%% plot model metric for various parameters

for k, yvar_sublist in enumerate(yvar_sublist_sets):
    print(yvar_sublist)
    for model_type in ['randomforest', 'plsr', 'lasso']: 
        fig, ax = plt.subplots(3,4, figsize=(27,17))
        for i, yvar in enumerate(yvar_sublist):
            metrics_filt = modelparam_metrics_df.loc[(modelparam_metrics_df['model_type']==model_type) & (modelparam_metrics_df['yvar']==yvar)]
            param_name = str(metrics_filt.iloc[0]['param_name'])
            # plot R2, MAE (CV), MAE (test) for all parameter values
            ax[0][i].plot(metrics_filt['param_val'], metrics_filt['r2'], marker='*')
            ax[0][i].plot(metrics_filt['param_val'], metrics_filt[f'{scoring}_norm_train'], marker='o')
            ax[0][i].plot(metrics_filt['param_val'], metrics_filt[f'{scoring}_norm_cv'], marker='s')
            ax[0][i].legend(['r2', f'{scoring}_train', f'{scoring}_cv'])
            ax[0][i].set_ylabel('model metrics', fontsize=12)
            ax[0][i].set_xlabel(param_name, fontsize=12)
            ax[0][i].set_title(yvar, fontsize=14)
            if model_type=='lasso': 
                ax[0][i].set_xscale('log')
            # plot R2 divided by MAE(CV) for all parameter values
            ax[1][i].plot(metrics_filt['param_val'], metrics_filt['r2']/metrics_filt[f'{scoring}_norm_cv'], marker='o')
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
        fig.savefig(f'{figure_folder}modelmetrics_{model_type}_vs_params_{k}.png', bbox_inches='tight')
        plt.show()

#%% Evaluate individual model at a time and get metrics, and feature coefficients / importances 

models_to_eval_list = ['randomforest','plsr', 'lasso'] # 
model_metrics_df = []
feature_importance_df = []
feature_coef_df = []
ypred_bymodel = {model_type: np.empty_like(Y) for model_type in models_to_eval_list}
    
# iterate through model types 
for model_type in models_to_eval_list: 
    print(model_type)
    
    # iterate through yvar
    for i, yvar in enumerate(yvar_list):     
        
        model_list = model_params[model_type][yvar]
        y = Y[:,i]
        model_list, metrics = fit_model_with_cv(Xscaled,y, yvar, model_list, plot_predictions=False)
        # get feature importance and split into COEF and ORDER dicts
        feature_coefs, feature_importances = get_feature_importances(model_list, yvar, xvar_list, plot_feature_importances=False)
        # update aggregating dicts
        feature_importance_df += feature_importances
        feature_coef_df += feature_coefs
        model_metrics_df.append(metrics)
        ypred_bymodel[model_type][:, i] = model_list[0]['ypred']
    
    # get feature importances for each model
    feature_importance_df_MODEL = pd.DataFrame(feature_importance_df)
    feature_importance_df_MODEL = feature_importance_df_MODEL[feature_importance_df_MODEL['model_type']==model_type]
    feature_coef_df_MODEL = pd.DataFrame(feature_coef_df)
    feature_coef_df_MODEL = feature_coef_df_MODEL[feature_coef_df_MODEL['model_type']==model_type]
    # plot feature importance heatmaps
    arr_importance = plot_feature_importance_heatmap(feature_importance_df_MODEL.set_index('yvar').iloc[:,1:], xvar_list, yvar_list, logscale_cmap=False, scale_vals=False, figtitle=f'Feature importances ({model_type})', savefig=f'feature_importance_{model_type}')
    arr_coef = plot_feature_importance_heatmap(feature_coef_df_MODEL.set_index('yvar').iloc[:,1:], xvar_list, yvar_list, logscale_cmap=False, scale_vals=True, figtitle=f'Feature coefficients, normalized ({model_type})', savefig=f'feature_coef_{model_type}')
    arr_coef_importance_product = arr_importance*np.abs(arr_coef)
    # get feature importance barplots
    plot_feature_importance_barplots(arr_coef_importance_product, yvar_list, xvar_list, label_xvar_by_indices=True, ncols=4, nrows=3, savefig=f'feature_importance_barplots_{model_type}', figtitle=f'[{model_type}] feature importances for all y variables')

# aggregate all model metrics and save CSV
model_metrics_df = pd.DataFrame(model_metrics_df)
model_metrics_df = model_metrics_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
model_metrics_df.to_csv(f'{data_folder}model_metrics.csv')

# aggregate all feature importances and save CSV
feature_importance_df = pd.DataFrame(feature_importance_df)
feature_importance_df = feature_importance_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
feature_importance_df.to_csv(f'{data_folder}model_feature_importance.csv')

# aggregate all feature coefficients and save CSV
feature_coef_df = pd.DataFrame(feature_coef_df)
feature_coef_df = feature_coef_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
feature_coef_df.to_csv(f'{data_folder}model_feature_coef.csv')


#%% Plot model metrics and predictions

model_cmap = {'randomforest':'r', 'plsr':'b', 'lasso':'g'}

# plot metrics for all yvar
nrows = 3
ncols = 1
fig, ax = plt.subplots(nrows,ncols, figsize=(27,17))
xtickpos = np.arange(len(yvar_list))+1
barwidth = 0.25
for i, yvar in enumerate(yvar_list):
    for k, model_type in enumerate(models_to_eval_list):
        c = model_cmap[model_type]
        xtickoffset = -barwidth*len(models_to_eval_list)/2 + 0.5*barwidth + k*barwidth
        model_metrics_df_filt = model_metrics_df[(model_metrics_df.yvar==yvar) & (model_metrics_df.model_type==model_type)].iloc[0].to_dict()
        ## R2
        ax[0].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt['r2'], width=barwidth, label=model_type, color=c)
        ## MAE_norm (train)
        ax[1].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt['mae_norm_train'], width=barwidth, label=model_type, color=c)
        ## MAE_norm (CV)
        ax[2].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt['mae_norm_cv'], width=barwidth, label=model_type, color=c)

for k in range(nrows): 
    ax[k].set_xticks(xtickpos, yvar_list)
    
# set ylim for MAE plots
ymax_mae = np.max(model_metrics_df[['mae_norm_train', 'mae_norm_cv']].to_numpy())
ax[1].set_ylim([0,ymax_mae*1.01])
ax[2].set_ylim([0,ymax_mae*1.01])
ax[0].set_ylabel('R2', fontsize=16)
ax[1].set_ylabel('MAE, normalized (train)', fontsize=16)
ax[2].set_ylabel('MAE, normalized (LOOCV)', fontsize=16)
ymax = ax.flatten()[0].get_position().ymax
plt.legend(models_to_eval_list, fontsize=16)
plt.suptitle('Metrics for various models evaluated on all Y variables', y=ymax*1.03, fontsize=24)    
fig.savefig(f'{figure_folder}modelmetrics_allselectedmodels_allYvar.png', bbox_inches='tight')
plt.show()


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
            ax[j][i].scatter(y, ypred, c=c)
            ax[j][i].set_title(f'{model_type} <> {yvar}', fontsize=16)
            ax[j][i].set_ylabel(f'{yvar} (predicted)', fontsize=12)
            ax[j][i].set_xlabel(f'{yvar} (actual)', fontsize=12)          
            (xmin, xmax) = ax[j][i].get_xlim()
            (ymin, ymax) = ax[j][i].get_ylim()
            r2 = float(model_metrics_df[(model_metrics_df.model_type==model_type) & (model_metrics_df.yvar==yvar)].iloc[0].r2)
            ax[j][i].text(xmin+(xmax-xmin)*0.05, ymin+(ymax-ymin)*0.9, f'R2: {r2}', fontsize=14)
            
    ymax = ax.flatten()[0].get_position().ymax
    plt.suptitle(f'Model Predicted vs. Actual values for various Y variables\n{yvar_sublist}', y=ymax*1.08, fontsize=24)    
    fig.savefig(f'{figure_folder}modelpredictions_scatterplots_{k}.png', bbox_inches='tight')
    plt.show()

#%% Feature importance analysis
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
feature_coef_importance_product = feature_coefs_abs_arr_weightedavg*feature_importances_arr_weightedavg

# plot overall weighted feature importances as bar plots
plot_feature_importance_barplots(feature_coef_importance_product, yvar_list, xvar_list, label_xvar_by_indices=True, ncols=4, nrows=3, savefig='overall_feature_importance_barplots', figtitle='Overall feature importances for all y variables')

# plot heatmap
overall_feature_importance_heatmap = pd.DataFrame(feature_importances_arr_weightedavg, index=yvar_list, columns=xvar_list)
_ = plot_feature_importance_heatmap(overall_feature_importance_heatmap, xvar_list, yvar_list, logscale_cmap=True, figtitle='Overall feature importances for all y variables', savefig='overall_feature_importance_heatmap')

# order x variable importance for each y variable using feature_coef_importance_product
overall_feature_importance = []
overall_feature_scoring = []
feature_ordering = {}

for i, yvar in enumerate(yvar_list):
    print(yvar)
    feature_coef_importance_product_yvar = feature_coef_importance_product[i, :]
    for j, (xvar, val) in enumerate(zip(xvar_list, feature_coef_importance_product_yvar)):
        print(j, xvar, round(val,2))
    orders_x_list = get_order_of_element_sizes(feature_coef_importance_product_yvar)
    overall_feature_importance_dict = {'yvar': yvar}
    overall_feature_importance_dict.update({f'{xvar_list[k]}':feature_coef_importance_product_yvar[k] for k in range(len(xvar_list))})
    overall_feature_importance.append(overall_feature_importance_dict)
    overall_feature_scoring_dict = {'yvar': yvar}
    overall_feature_scoring_dict.update({f'{xvar_list[k]}':orders_x_list[k] for k in range(len(xvar_list))})
    overall_feature_scoring.append(overall_feature_scoring_dict)
    
    feature_ordering[yvar] = [xvar_list[idx] for idx in np.argsort(feature_coef_importance_product_yvar)][::-1]
    print(feature_ordering[yvar])
    
overall_feature_importance = pd.DataFrame(overall_feature_importance)
overall_feature_scoring = pd.DataFrame(overall_feature_scoring)

#%%
for i, (xvar, val, score) in enumerate(zip(xvar_list, feature_coef_importance_product_yvar, orders_x_list)):
    print(i, xvar, round(val,2), round(score,1))

#%% Ensemble: randomforest + plsr
ensemble_params = {
    0: {yvar: [
        dict_update(model_params['randomforest'][yvar][0], {'w':0.7}), 
        dict_update(model_params['plsr'][yvar][0], {'w':0.15}), 
        dict_update(model_params['lasso'][yvar][0], {'w':0.15})
        ] for yvar in yvar_list}
    }


print('Ensemble')
ensembles_to_eval_list = [0] 
for ensembles_to_eval in ensembles_to_eval_list: 
    print([m['model_type'] for m in ensemble_params[ensembles_to_eval][yvar_list[0]]])
    for i, yvar in enumerate(yvar_list): 
        model_list = ensemble_params[ensembles_to_eval][yvar]
        y = Y[:,i]
        model_list, metrics = fit_model_with_cv(Xscaled,y, yvar, model_list, plot_predictions=True)
        print()

# %% 
## TO DO 
# plot feature heat maps
# extract top features for each model
# try out different subsets of features


