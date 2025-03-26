#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 21:02:36 2024

@author: charmainechia
"""
import numpy as np
import pandas as pd
from variables import model_params, yvar_list_key
from model_utils import split_data_to_trainval_test, eval_model_over_params, plot_model_param_results, evaluate_model_on_train_test_data, get_feature_importances, plot_feature_importance_barplots_bymodel, get_feature_coefficients, order_features_by_coefficient_importance, plot_model_metrics, plot_model_metrics_cv, plot_scatter_train_test_predictions, order_list_by_frequencies, plot_feature_importance_heatmap
from plot_utils import figure_folder
from get_datasets import data_folder, get_XYdata_for_featureset
from feature_selection_utils import run_sfs_forward, run_sfs_backward, run_rfe

def reformat_model_params_dict(model_params_opt):
    for dataset_name_wsuffix in model_params_opt:
        for model_type in model_params_opt[dataset_name_wsuffix]:
            for yvar, d in model_params_opt[dataset_name_wsuffix][model_type].items():
                model_params_opt[dataset_name_wsuffix][model_type][yvar][0].update({d[0]['param_name']:d[0]['param_val']})

    return model_params_opt

def get_filtered_Xdata(X_trainval, X_test, featureset_suffix, featureset_dict, xvar_list_all, yvar, model_type=None):
    if model_type is None: 
        xvar_list_ = featureset_dict[featureset_suffix][yvar]
    else:
        xvar_list_ = featureset_dict[yvar][model_type][featureset_suffix]
    print(yvar, xvar_list_)
    xvar_idxs_selected = np.array([idx for idx, xvar in enumerate(xvar_list_all) if xvar in xvar_list_])
    X_trainval_ = X_trainval[:, xvar_idxs_selected]
    X_test_ = X_test[:, xvar_idxs_selected]
    print(f'X_trainval size after feature selection: n={X_trainval_.shape[0]}, p={X_trainval_.shape[1]}')       
    return xvar_list_, X_trainval_, X_test_         

def run_modelparam_optimization(Y, X, yvar_list, xvar_list, model_params_to_eval, dataset_name, dataset_suffix, kfold_suffix='', save_results=False):
    modelparam_metrics_df = []
    feature_coef_df = []
    feature_importance_df = []
    model_params_opt = {dataset_name_wsuffix: {model_type: {yvar: None for yvar in yvar_list} for model_type in models_to_eval_list}}
    
    # get model type, list of parameters, and feature subset (if applicable)
    for model_dict in model_params_to_eval:
        model_type = model_dict['model_type']
        modelparam_metrics_bymodel = []
        
        # iterate over yvar
        for i, yvar in enumerate(yvar_list): 
            # get y data
            y = Y[:,i]
            modelparam_metrics_bymodelyvar = []
            trained_model_cache = {}
            # evaluate model over range of params
            trained_model_cache, modelparam_metrics_bymodelyvar = eval_model_over_params(model_dict, yvar, X, y, modelparam_metrics_bymodelyvar, trained_model_cache, scoring=scoring, scale_data=True)
            # choose optimal parameters
            modelparam_metrics_df_bymodel = pd.DataFrame(modelparam_metrics_bymodelyvar)
            modelparam_metrics_df_bymodel['r2_over_mae_norm_cv'] = modelparam_metrics_df_bymodel['r2_cv']/modelparam_metrics_df_bymodel['mae_norm_cv']
            modelparam_metrics_df_bymodel = modelparam_metrics_df_bymodel.sort_values(by='r2_over_mae_norm_cv', ascending=False)
            param_name = modelparam_metrics_df_bymodel.iloc[0]['param_name']
            param_val_opt = modelparam_metrics_df_bymodel.iloc[0]['param_val']
            model_opt = trained_model_cache[param_val_opt]
            model_dict_opt = {'model_type': model_type, 'param_name': param_name, 'param_val':param_val_opt, 'model':model_opt}
            model_params_opt[dataset_name_wsuffix][model_type][yvar] = [model_dict_opt]
            # get feature coefficients in optimal model
            feature_coefs, feature_importances = get_feature_importances(model_dict_opt, yvar, xvar_list, plot_feature_importances=False)
            
            # update modelparam_metrics_bymodel, feature_coef_df, feature_importance_df
            modelparam_metrics_bymodel += modelparam_metrics_bymodelyvar
            feature_coef_df += feature_coefs
            feature_importance_df += feature_importances
        
        # plot evaluation results for this model type over yvar evaluated
        savefig = f'{figure_folder}modelmetrics_{model_type}_vs_params_{dataset_name}{dataset_suffix}_k={k}.png' if save_results else None
        plot_model_param_results(pd.DataFrame(modelparam_metrics_bymodel), yvar_list, dataset_name, scoring=scoring, model_type_list=[model_type], savefig=savefig)
        # update general results
        modelparam_metrics_df += modelparam_metrics_bymodel
    
    model_params_opt = reformat_model_params_dict(model_params_opt)
    print(f'Optimized model parameters: {model_params_opt}')
        
    # save model param optimization metrics for current trainval split
    modelparam_metrics_df = pd.DataFrame(modelparam_metrics_df)
    modelparam_metrics_df = modelparam_metrics_df[['model_type','yvar','param_name', 'param_val', 'r2_train', 'r2_cv', 'mae_norm_train', 'mae_norm_cv','mae_train','mae_cv']]
    # get feature lists
    feature_importance_df = pd.DataFrame(feature_importance_df)
    feature_importance_df = feature_importance_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    feature_importance_df.to_csv(f'{data_folder}model_feature_importance_{dataset_name}{dataset_suffix}{kfold_suffix}.csv')    
    feature_coef_df = pd.DataFrame(feature_coef_df)
    feature_coef_df = feature_coef_df.sort_values(by=['yvar','model_type']).reset_index(drop=True)
    if save_results:
        modelparam_metrics_df.to_csv(f'{data_folder}modelmetrics_vs_params_{dataset_name}{dataset_suffix}{kfold_suffix}.csv')
        feature_coef_df.to_csv(f'{data_folder}model_feature_coef_{dataset_name}{dataset_suffix}{kfold_suffix}.csv')
    # plot feature prominences
    feature_prominence_df = feature_importance_df.copy()
    feature_prominence_df.iloc[:,2:] = feature_importance_df.iloc[:,2:].to_numpy()*np.abs(feature_coef_df.iloc[:,2:].to_numpy())
    savefig = f'feature_prominence_{dataset_name}{dataset_suffix}{kfold_suffix}' if save_results else None
    figtitle = f'Feature prominences ({dataset_name}{dataset_suffix}, k={k})'
    plot_feature_importance_barplots_bymodel(feature_prominence_df, yvar_list=yvar_list, xvar_list=xvar_list, model_list=models_to_eval_list, order_by_feature_importance=False, label_xvar_by_indices=True, savefig=savefig, figtitle=figtitle)
    
    return model_params_opt, modelparam_metrics_df
    
#%% 

# load data
featureset_list = [(1,0)] # [(4,0)] # 
dataset_suffix = '_norm' # '_norm_with_val-A' # '' # '' # '_avgnorm'
f = 1
n_splits = 10 # 5 # 
yvar_list = yvar_list_key.copy()
# yvar_list = yvar_list_key.copy()[:2]
scoring = 'mae'
optimize_model_params = False
optimize_feature_subset = None # False #'sfs-backward' # None # 
featureset_suffix =  '' # '_ajinovalidation3' #  # '_sfs-backward'
save_results = False# True
print_testres_on_each_fold = True
if optimize_model_params:
    model_params_to_eval = [
        # {'model_type': 'randomforest', 'params_to_eval': ('n_estimators', [20,40,60,80,100,120,140,160,180])},
        # {'model_type': 'plsr', 'params_to_eval': ('n_components',[2,3,4,5,6,7,8,9,10,11,12,14])},
        {'model_type': 'lasso', 'max_iter':500000, 'params_to_eval': ('alpha', [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 100])},
        ]
    models_to_eval_list = [model_dict['model_type'] for model_dict in model_params_to_eval]
    
else: 
    model_params_to_eval = None
    models_to_eval_list = ['xgb', 'randomforest', 'plsr', 'lasso'] # ['xgb', 'randomforest'] # ['plsr']# ['plsr', 'randomforest'] #  ['lasso'] # 
    model_params_opt = model_params.copy()
    
# if optimize_feature_subset is None:  
#     featureset_bymodeltype = features_selected_dict.copy()
    
    
#%% Perform training & testing

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_name_wsuffix = dataset_name + dataset_suffix
    Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    
    
    print(f'X.shape={X.shape}')
    print(xvar_list_all)

    # create train-val / test splits using KFold
    kf_dict = split_data_to_trainval_test(X, n_splits=n_splits, split_type='random')
    kfold_metrics = []
    coef_lists_bymodelyvar = {model_type: {yvar: [] for yvar in yvar_list} for model_type in models_to_eval_list}

    # initialize arrays for storing predicted data for scatter plots
    ypred_train_bymodel = {model_type: np.zeros((len(Y), len(yvar_list))) for model_type in models_to_eval_list}
    ypred_test_bymodel = {model_type: np.zeros((len(Y), len(yvar_list))) for model_type in models_to_eval_list}

    for k, (train_idx, test_idx) in kf_dict.items():

        # get split data
        X_trainval, Y_trainval = X[train_idx, :], Y[train_idx, :]
        X_test, Y_test = X[test_idx, :], Y[test_idx, :]

        # get scaling parameters
        # fit X_trainval with LOOCV to choose model parameters
        if optimize_model_params and k==0: 
            print('\n************************************************')
            print(f'Performing hyperparameter search on Fold {k}...')
            print('************************************************')            
            model_params_opt, modelparam_metrics_df = run_modelparam_optimization(Y_trainval, X_trainval, yvar_list, xvar_list_all, model_params_to_eval, dataset_name, dataset_suffix, kfold_suffix=f'_k={k}', save_results=save_results)
            
        if optimize_feature_subset not in [None, False] and k==0: 
            print('\n*********************************************************************************')
            print(f'Performing feature selection ({optimize_feature_subset}) search on Fold {k}...')
            print('***********************************************************************************')
            
            if optimize_feature_subset=='sfs-forward':
                featureset_suffix = '_sfs-forward'
                res, featureset_bymodeltype = run_sfs_forward(Y_trainval, X_trainval, yvar_list_key, xvar_list_all, models_to_eval_list, dataset_name, dataset_suffix, kfold_suffix=f'_k={k}', xvar_idx_end=None, featureset_suffix=featureset_suffix)
            
            elif optimize_feature_subset=='sfs-backward':
                featureset_suffix = '_sfs-backward'    
                res, featureset_bymodeltype = run_sfs_backward(Y_trainval, X_trainval, yvar_list_key, xvar_list_all, models_to_eval_list, dataset_name, dataset_suffix, kfold_suffix=f'_k={k}', xvar_idx_end=None, featureset_suffix=featureset_suffix)

            elif optimize_feature_subset=='rfe':
                featureset_suffix = '_rfe'    
                res, featureset_bymodeltype = run_rfe(Y_trainval, X_trainval, yvar_list_key, xvar_list_all, models_to_eval_list, dataset_name, dataset_suffix, kfold_suffix=f'_k={k}', xvar_idx_end=None, featureset_suffix='_rfe')
                
        
        print('\n************************************************')
        print(f'Performing train/test evaluation on Fold {k}...')
        print('************************************************')
        # evaluate best model on test data
        for model_type in models_to_eval_list:            
            for i, yvar in enumerate(yvar_list): 
                print(f'Evaluating {model_type} <> {yvar}...')
                X_trainval_ = X_trainval.copy()
                X_test_ = X_test.copy()
                
                # SHUFFLE FEATURE VALUES
                rng = np.random.default_rng()
                # X_trainval_ = rng.permuted(X_trainval_, axis=0)
                # X_test_ = rng.permuted(X_test_, axis=0)
                # X_trainval_[:,-1] = rng.permuted(X_trainval_[:,-1], axis=0)
                # X_test_[:,-1] = rng.permuted(X_test_[:,-1], axis=0)
                
                y_trainval = Y_trainval[:,i]
                y_test = Y_test[:,i]
                model_dict  = model_params_opt[dataset_name_wsuffix][model_type][yvar][0]
                xvar_list_ = xvar_list_all.copy()
                
                # get filtered X data
                if featureset_suffix != '':
                    if 'featureset_bymodeltype' in globals(): 
                        xvar_list_, X_trainval_, X_test_  = get_filtered_Xdata(X_trainval_, X_test_, featureset_suffix, featureset_bymodeltype, xvar_list_all, yvar, model_type=model_type)
                    else: 
                        from variables import feature_selections as featureset_byyvar
                        xvar_list_, X_trainval_, X_test_  = get_filtered_Xdata(X_trainval_, X_test_, featureset_suffix, featureset_byyvar, xvar_list_all, yvar, model_type=None)
                metrics, model = evaluate_model_on_train_test_data(X_test_, y_test, X_trainval_, y_trainval, model_dict, scoring=scoring)
            
                # get features
                coefs = get_feature_coefficients(model, model_type)
                xvar_list_sorted, coefs_sorted = order_features_by_coefficient_importance(coefs, xvar_list_, filter_out_zeros=True)
                if model_type=='lasso':
                    coef_lists_bymodelyvar[model_type][yvar] += xvar_list_sorted
                metrics.update({'k':k, 'yvar':yvar, 'xvar_sorted': ', '.join(list(xvar_list_sorted))})
                kfold_metrics.append(metrics)
                
                # get test predictions
                ypred_test_bymodel[model_type][test_idx,i] = metrics['ypred_test']
            
    print('\n*****************************************')
    print('Getting train performance on full dataset')
    print('*******************************************')
    # get train predictions using a model trained on the full dataset
    for model_type in models_to_eval_list:   
        print(model_type)
        for i, yvar in enumerate(yvar_list): 
            print(yvar)
            y = Y[:,i]
            if featureset_suffix != '':
                if 'featureset_bymodeltype' in globals(): 
                    xvar_list_, X_, X_  = get_filtered_Xdata(X.copy(), X.copy(), featureset_suffix, featureset_bymodeltype, xvar_list_all, yvar, model_type=model_type)
                else: 
                    from variables import feature_selections as featureset_byyvar
                    xvar_list_, X_, X_  = get_filtered_Xdata(X.copy(), X.copy(), featureset_suffix, featureset_byyvar, xvar_list_all, yvar, model_type=None)
            else: 
                xvar_list_ = xvar_list_all.copy()
                X_ = X.copy()
            model_dict = model_params_opt[dataset_name_wsuffix][model_type][yvar][0]
            metrics_train, model = evaluate_model_on_train_test_data(X_, y, X_, y, model_dict, scoring=scoring, print_res=print_testres_on_each_fold)
            ypred_train_bymodel[model_type][:,i] = metrics_train['ypred_test']
            
    print('\n********************')
    print('PLOT OVERALL RESULTS')
    print('**********************')    
    # plot predictions as scatter
    savefig = f'{figure_folder}modelpredictions_scatterplots_{dataset_name}{dataset_suffix}{featureset_suffix}.png'
    plot_scatter_train_test_predictions(Y[:,:len(yvar_list)], ypred_train_bymodel, ypred_test_bymodel, yvar_list, models_to_eval_list, savefig=savefig)
                
    # collect kfold metrics
    kfold_metrics_cols = ['model_type', 'yvar', 'param_name', 'param_val', 'k', 'r2_train', 'r2_test', f'{scoring}_norm_train', f'{scoring}_norm_test', f'{scoring}_train', f'{scoring}_test', 'xvar_sorted']
    kfold_metrics = pd.DataFrame(kfold_metrics, columns=kfold_metrics_cols).sort_values(by=['yvar', 'model_type', 'k']).reset_index(drop=True)
    
    # calculate average metrics
    metrics_list = ['r2_train', 'r2_test', 'mae_norm_train', 'mae_norm_test', 'mae_train', 'mae_test']
    kfold_metrics_avg = []
    for yvar in yvar_list:
        for model_type in models_to_eval_list:
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
    plot_model_metrics(kfold_metrics_avg, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(30,15), barwidth=0.7, figtitle=figtitle, savefig=savefig, suffix_list=['_train_avg','_test_avg'], plot_errors=True, annotate_vals=True)
    # plot_model_metrics_cv(kfold_metrics_avg, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(20,15), barwidth=0.7, figtitle=figtitle, savefig=savefig, suffix_list=['_test_avg'], plot_errors=True, annotate_vals=True)
    plot_model_metrics_cv(kfold_metrics_avg, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(20,15), barwidth=0.7, figtitle=figtitle, savefig=savefig, suffix_list=['_test_avg'], plot_errors=False, annotate_vals=True)
    
    # print all lasso coefficients selected for each yvar
    if 'lasso' in models_to_eval_list:
        for yvar in yvar_list: 
            print(f'lasso <> {yvar}')
            print(len(order_list_by_frequencies(coef_lists_bymodelyvar['lasso'][yvar])))
            lst = order_list_by_frequencies(coef_lists_bymodelyvar['lasso'][yvar])
            print(*lst, sep=', ')
            coef_lists_bymodelyvar['lasso'][yvar] = lst


#%% plot feature importance and prominence heatmaps
k=0
drop_process_features = True

for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_name_wsuffix = dataset_name + dataset_suffix
    Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    feature_coef_df = pd.read_csv(f'{data_folder}model_feature_coef_{dataset_name}{dataset_suffix}_k={k}.csv', index_col=0)
    feature_importance_df = pd.read_csv(f'{data_folder}model_feature_importance_{dataset_name}{dataset_suffix}_k={k}.csv', index_col=0)
    yvar_labels = [(x[1],x[2]) for x in list(feature_importance_df[['yvar','model_type']].to_records())]
    imp_arr = feature_importance_df.iloc[:,2:].to_numpy()
    coef_arr = feature_coef_df.iloc[:,2:].to_numpy()
    
    # plot coef heatmap     
    heatmap_df = pd.DataFrame(imp_arr, columns=xvar_list_all, index=yvar_labels)
    if drop_process_features: 
        arr = plot_feature_importance_heatmap(heatmap_df.iloc[:,:-3], np.array(xvar_list_all)[:-3], yvar_labels, logscale_cmap=False, scale_vals=True, annotate=None, get_clustermap=False, figtitle=f'Feature importances (various models) for Fold={k}', savefig=f'feature_importance_heatmap_{dataset_name}{dataset_suffix}_keyCQAs_k={k}.png')
    else: 
        arr = plot_feature_importance_heatmap(heatmap_df, xvar_list_all, yvar_labels, logscale_cmap=False, scale_vals=True, annotate=None, get_clustermap=False, figtitle=f'Feature importances (various models) for Fold={k}', savefig=f'feature_importance_heatmap_{dataset_name}{dataset_suffix}_k={k}.png')
    
    # plot prominence heatmap
    # arr = imp_arr*np.abs(coef_arr)
    arr = np.abs(coef_arr)
    heatmap_df = pd.DataFrame(arr, columns=xvar_list_all, index=yvar_labels)
    if drop_process_features: 
        arr = plot_feature_importance_heatmap(heatmap_df.iloc[:,:-3], np.array(xvar_list_all)[:-3], yvar_labels, logscale_cmap=False, scale_vals=True, annotate=None, get_clustermap=False, figtitle=f'Feature coefficients (various models) for Fold={k}', savefig=f'feature_coef_heatmap_{dataset_name}{dataset_suffix}_keyCQAs_k={k}.png')
    else: 
        arr = plot_feature_importance_heatmap(heatmap_df, xvar_list_all, yvar_labels, logscale_cmap=False, scale_vals=True, annotate=None, get_clustermap=False, figtitle=f'Feature coefficients (various models) for Fold={k}', savefig=f'feature_coef_heatmap_{dataset_name}{dataset_suffix}_k={k}.png')
    

