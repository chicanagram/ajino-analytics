#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:49:46 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from variables import yvar_list_key, xvar_list_dict, model_params, features_selected_dict
from model_utils import fit_model_with_cv, split_data_to_trainval_test, evaluate_model_on_train_test_data, plot_scatter_train_test_predictions, plot_model_metrics, plot_model_metrics_cv, get_feature_coefficients, order_features_by_coefficient_importance
from plot_utils import figure_folder, heatmap
from get_datasets import data_folder, get_XYdata_for_featureset
#%%

# get data
X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
featureset_suffix = ''

# create train-val / test splits using KFold
n_splits = 10
kf_dict = split_data_to_trainval_test(X, n_splits=n_splits, split_type='random')
kfold_metrics = []
scoring = 'mae'
models_to_eval_list = ['randomforest']
model_type = 'randomforest'
save_results = False

# initialize dict for storing trained models
SURROGATE_MODELS = {}
features_selected_sorted_dict = {}
ypred_train_bymodel = {model_type: np.zeros((len(Y), len(yvar_list_key))) for model_type in models_to_eval_list}
ypred_test_bymodel = {model_type: np.zeros((len(Y), len(yvar_list_key))) for model_type in models_to_eval_list}

for k, (train_idx, test_idx) in kf_dict.items():

    # get split data
    X_trainval, Y_trainval = X[train_idx, :], Y[train_idx, :]
    X_test, Y_test = X[test_idx, :], Y[test_idx, :]

    print('\n************************************************')
    print(f'Performing train/test evaluation on Fold {k}...')
    print('************************************************')          
    for i, yvar in enumerate(yvar_list_key): 
        print(f'Evaluating {model_type} <> {yvar}...')
        # get features selected
        features_selected = features_selected_dict[yvar]
        X_trainval_selected = pd.DataFrame(X_trainval, columns=xvar_list_all).loc[:, features_selected].to_numpy()
        X_test_selected = pd.DataFrame(X_test, columns=xvar_list_all).loc[:, features_selected].to_numpy()
        y_trainval = Y_trainval[:,i]
        y_test = Y_test[:,i]
        model_dict  = model_params[dataset_name_wsuffix][model_type][yvar][0]
        
        # get metrics from evaluating on train & test data
        metrics, model = evaluate_model_on_train_test_data(X_test_selected, y_test, X_trainval_selected, y_trainval, model_dict, scoring=scoring)
        metrics.update({'k':k, 'yvar':yvar, 'xvar_sorted': ', '.join(list(features_selected))})
        kfold_metrics.append(metrics)
        
        # get test predictions
        ypred_test_bymodel[model_type][test_idx,i] = metrics['ypred_test']
        
print('\n********************************************************')
print('Getting train performance and coefficients on full dataset')
print('**********************************************************')
for i, yvar in enumerate(yvar_list_key): 
    # get features selected
    features_selected = features_selected_dict[yvar]
    X_selected = pd.DataFrame(X, columns=xvar_list_all).loc[:, features_selected].to_numpy()
    print('X_selected.shape: ', X_selected.shape)
    y = Y[:,i]
    model_dict = model_params[dataset_name_wsuffix][model_type][yvar][0]
    metrics_train, model = evaluate_model_on_train_test_data(X_selected, y, X_selected, y, model_dict, scoring=scoring)
    ypred_train_bymodel[model_type][:,i] = metrics_train['ypred_test']
    SURROGATE_MODELS.update({yvar: model})

    # get features
    coefs = get_feature_coefficients(model, model_type)
    features_selected_sorted, coefs_sorted = order_features_by_coefficient_importance(coefs, features_selected, filter_out_zeros=True)
    features_selected_sorted_dict.update({yvar: features_selected_sorted})
    xticks = np.arange(len(coefs_sorted))
    plt.bar(xticks, coefs_sorted)
    plt.xticks(xticks, features_selected_sorted, rotation=90)
    plt.ylabel('Feature importance')
    plt.title(f'{yvar}: {model_type} feature importances')
    plt.show()
    
# plot scatter predictions
savefig = None # f'{figure_folder}modelpredictions_scatterplots_{dataset_name}{dataset_suffix}{featureset_suffix}.png'
plot_scatter_train_test_predictions(Y[:,:len(yvar_list_key)], ypred_train_bymodel, ypred_test_bymodel, yvar_list_key, models_to_eval_list, savefig=savefig)

    
print('\n********************')
print('PLOT OVERALL RESULTS')
print('**********************')    
 
# collect kfold metrics
kfold_metrics_cols = ['model_type', 'yvar', 'param_name', 'param_val', 'k', 'r2_train', 'r2_test', f'{scoring}_norm_train', f'{scoring}_norm_test', f'{scoring}_train', f'{scoring}_test', 'xvar_sorted']
kfold_metrics = pd.DataFrame(kfold_metrics, columns=kfold_metrics_cols).sort_values(by=['yvar', 'model_type', 'k']).reset_index(drop=True)

# calculate average metrics
metrics_list = ['r2_train', 'r2_test', 'mae_norm_train', 'mae_norm_test', 'mae_train', 'mae_test']
kfold_metrics_avg = []
for yvar in yvar_list_key:
    kfold_metrics_avg_dict = {'model_type':model_type, 'yvar':yvar}
    metrics_avg = kfold_metrics[(kfold_metrics.yvar==yvar) & (kfold_metrics.model_type==model_type)][metrics_list].rename(columns={c:f'{c}_avg' for c in metrics_list}).mean(axis=0).round(3).to_dict()
    metrics_std = kfold_metrics[(kfold_metrics.yvar==yvar) & (kfold_metrics.model_type==model_type)][metrics_list].rename(columns={c:f'{c}_std' for c in metrics_list}).std(axis=0).round(3).to_dict()
    kfold_metrics_avg_dict.update(metrics_std)
    kfold_metrics_avg_dict.update(metrics_avg)
    kfold_metrics_avg.append(kfold_metrics_avg_dict)
kfold_metrics_avg_cols = ['model_type', 'yvar'] + list(metrics_avg.keys()) + list(metrics_std.keys())
kfold_metrics_avg = pd.DataFrame(kfold_metrics_avg, columns=kfold_metrics_avg_cols)
if save_results:
    kfold_metrics.to_csv(f'{data_folder}model_metrics_kfold_{dataset_name}{dataset_suffix}{featureset_suffix}.csv')
    kfold_metrics_avg.to_csv(f'{data_folder}model_metrics_avg_kfold_{dataset_name}{dataset_suffix}{featureset_suffix}.csv')

# plot kfold results
figtitle = f'Model evaluation metrics for {dataset_name}{dataset_suffix}{featureset_suffix}, {n_splits}-fold CV'
savefig = f'{figure_folder}modelmetrics_allselectedmodels_keyYvar_{dataset_name}{dataset_suffix}{featureset_suffix}' if save_results else None
plot_model_metrics(kfold_metrics_avg, models_to_eval_list, yvar_list_key, nrows=2, ncols=1, figsize=(30,15), barwidth=0.7, figtitle=figtitle, savefig=savefig, suffix_list=['_train_avg','_test_avg'], plot_errors=True, annotate_vals=True)
# plot_model_metrics_cv(kfold_metrics_avg, models_to_eval_list, yvar_list_key, nrows=2, ncols=1, figsize=(20,15), barwidth=0.7, figtitle=figtitle, savefig=savefig, suffix_list=['_test_avg'], plot_errors=True, annotate_vals=True)

    
#%%
from skopt import gp_minimize, forest_minimize, dummy_minimize
from skopt.plots import plot_convergence
from functools import partial
  
# get data and scaling constants
df = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
X = df.loc[:, xvar_list_all]
XMEAN = X.mean(axis=0).to_numpy()
XSTD = X.std(axis=0).to_numpy()

# get average X values for starting condition, e.g. Basal-A, Feed-a, pH 7, DO 40, feed % 6
x0 = df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a') & (df['Feed medium']=='Feed-a')].loc[:, xvar_list_all].mean(axis=0)
x0[['DO', 'pH', 'feed vol']] = [40, 7.0, 0.2335]
x0 = x0.to_numpy().reshape(-1)
features_selected_idxs_dict = {}
for yvar in yvar_list_key:    
    features_selected = features_selected_dict[yvar]
    features_selected_idxs_dict[yvar] = [xvar_list_all.index(xvar) for xvar in features_selected]
print('features_selected_idxs_dict:', features_selected_idxs_dict)
    
# get average Y (CQA) values for starting condition
y0 = df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a') & (df['Feed medium']=='Feed-a') & (df['DO']==40) & (df['pH']==7) & (df['feed %']==6)].loc[:, yvar_list_key].mean(axis=0).to_numpy()
res0 = {yvar:val for yvar, val in zip(yvar_list_key, y0)}
print('Starting CQA values:', res0)

# set optimization parameters      
features_opt = ['feed vol', 'Riboflavin_feed', 'Pro_feed', 'Choline_feed', 'Folic acid_basal', 'Co_feed', 'Ca_feed'] # features_selected_sorted_dict['galactosylation_14'][:5] # 
idxs_opt = np.array([xvar_list_all.index(xvar) for xvar in features_opt])
print('idxs_opt:', idxs_opt)

bounds = []
for feature_idx_to_opt in idxs_opt:
    xvar = xvar_list_all[feature_idx_to_opt]
    lbnd = df.loc[:,xvar].min()
    ubnd = df.loc[:,xvar].max()
    print(xvar, lbnd, ubnd)
    bounds.append((lbnd,ubnd))
bounds = tuple(bounds)

def objective_function(res, res0):
    titer_0 = res0['Titer (mg/L)_14']
    man5_0 = res0['mannosylation_14']
    fuc_0 = res0['fucosylation_14']
    gal_0 = res0['galactosylation_14']
    titer = res['Titer (mg/L)_14']
    man5 = res['mannosylation_14']
    fuc = res['fucosylation_14']
    gal = res['galactosylation_14']
    # combine
    obj_fn = -2*(titer-titer_0)/titer_0 + 3*(man5-man5_0)/man5_0 + (fuc-fuc_0)**2/fuc_0 + (gal-gal_0)**2/gal_0
    return obj_fn

def f(x): 
    # X0, features_selected_dict, yvar_list_key
    # idxs_opt for variables under consideration
    print('x:', np.round(x, 4))
    res = {}
    print('{', end='')
    for i, yvar in enumerate(yvar_list_key): 
        # print(yvar)
        # get features used for model (yvar)
        features_selected = features_selected_dict[yvar]
        idxs_features_selected = features_selected_idxs_dict[yvar]
        # get original feature values and update features to optimize with new values to test
        x_input = x0.copy()
        x_input[idxs_opt] = x
        x_input = x_input[idxs_features_selected]
        # print('x_input to MODEL:', np.round(x_input,4))
        # scale data vector
        x_input = (x_input - XMEAN[idxs_features_selected])/XSTD[idxs_features_selected]
        # get model for yvar and perform prediction
        model = SURROGATE_MODELS[yvar]
        ypred = float(model.predict(x_input.reshape(1,-1)))
        res.update({yvar: ypred})
        print(f'{yvar}: {round(ypred,4)}', end=', ')
    print('}')
        
    # calculate objective function
    obj_fn = objective_function(res, res0)
    print('OBJECTIVE FUNCTION:', round(obj_fn,4))
    print()
    
    return obj_fn
   
    
#%% Surrogate minimization
n_initial_points = 20
n_calls = 200
n_iter = 1
x_input_0 = list(x0[idxs_opt])
print(x_input_0)
print('Initial Obj Fn value:', f(x_input_0))
def run(minimizer, n_iter=1):
    return [minimizer(f, bounds, x0=x_input_0, n_calls=n_calls, random_state=n, n_initial_points=n_initial_points) for n in range(n_iter)]

# Gaussian processes
gp_res = run(gp_minimize)
obj_fn_opt = gp_res[0].fun
x_opt = gp_res[0].x
print('Best Obj Fun score:', obj_fn_opt)
print('Best parameter vals:', x_opt)
print(f(x_opt))

# # Extra trees
# et_res = run(partial(forest_minimize, base_estimator="ET"))

   
# PLOT RESULTS
plot = plot_convergence(
    # ("dummy_minimize", dummy_res),
    ("gp_minimize", gp_res),
    # ("forest_minimize('rf')", rf_res),
    # ("forest_minimize('et)", et_res),
    )

plot.legend(loc="best", prop={'size': 6}, numpoints=1)
plt.show()


#%% scikit-learn minimize
from scipy.optimize import minimize

x_input_0 = list(x0[idxs_opt])
print(x_input_0)
print('Initial Obj Fn value:', f(x_input_0))

result = minimize(f, x0=x_input_0, bounds=bounds)
if result.success:
    fitted_params = result.x
    print(fitted_params)
    print('Final Obj Fn val:', f(fitted_params))
else:
    raise ValueError(result.message)

