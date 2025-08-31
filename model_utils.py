#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:49:45 2024

@author: charmainechia
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from variables import model_params, model_cmap
from plot_utils import figure_folder, heatmap, convert_figidx_to_rowcolidx

data_folder = '../ajino-analytics-data/'

#%% cross-validation testing

def get_corr_coef(x, y, corr_to_get=['spearmanr', 'pearsonr']):
    res = {}
    for corrtype in corr_to_get: 
        if corrtype=='spearmanr':
            res['spearmanr'] = round(spearmanr(x, y)[0],3)
        if corrtype=='pearsonr': 
            res['pearsonr'] = round(pearsonr(x, y)[0],3)
    return res


def get_score(y, ypred, scoring, get_norm_score=True):
    if scoring=='r2': 
        score = r2_score(y, ypred)
    elif scoring=='mae':
        score = mean_absolute_error(y, ypred)
    elif scoring=='rmse':
        score = root_mean_squared_error(y, ypred)
    elif scoring=='PearsonR':
        score = pearsonr(y, ypred)[0]
    elif scoring=='SpearmanR':
        score = spearmanr(y, ypred)[0]
    if not get_norm_score:
        return round(score,3)
    else: 
        score_norm = score/np.mean(y)
        return round(score,3), round(score_norm,3)
        

def perform_mean_std_scaling(xtrain, xtest=None):
    mean = np.mean(xtrain, axis=0)
    std = np.std(xtrain, axis=0)
    xtrain_scaled = (xtrain-mean)/std
    if xtest is not None:
        xtest_scaled = (xtest-mean)/std
    else: 
        xtest_scaled = None
    return mean, std, xtrain_scaled, xtest_scaled
    

def adjusted_loocv_with_scoring(X, y, model_dict, score_type_list=['mae', 'r2'], scale_data=False):
    n = len(y)
    ypred_loocv = np.zeros((n,))
    regr = model_dict['model']
    for test_idx in range(n):
        # get test / train sets
        train_idx = list(range(n))
        train_idx.remove(test_idx)
        train_idx = np.array(train_idx)
        test_idx = np.array([test_idx])
        Xtrain, ytrain = X[train_idx,:], y[train_idx]
        Xtest = X[test_idx,:]
        if scale_data:
            _, _, Xtrain, Xtest = perform_mean_std_scaling(Xtrain, Xtest)
        # fit training data
        regr.fit(Xtrain, ytrain)
        # predict on test data
        ypred_loocv[test_idx] = regr.predict(Xtest) 
        
    # update model dict with ypred values
    model_dict.update({'ypred_cv': ypred_loocv})
        
    # calculate test metrics for ypred_loocv 
    for score_type in score_type_list:
        score, score_norm = get_score(y, ypred_loocv, score_type)
        model_dict.update({f'{score_type}_cv':score, f'{score_type}_norm_cv':score_norm})

    return model_dict


def train_test_with_torch_nn(X_train, y_train, X_test, y_test, num_epochs=800, lr=0.0005, batch_size=32, output_scaler=None, layers=[42,24,12], dropout=0.2):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    from nn_model_utils import MLPRegressor, WeightedMSELoss, train_model, evaluate_model 
    
    # model = MLPRegressor(input_size=76, hidden_sizes=[48, 36, 24, 12], dropout_p=0.2, num_outputs=1)
    model = MLPRegressor(input_size=X_train.shape[1], hidden_sizes=layers, dropout_p=dropout, num_outputs=1)
    # model = MLPRegressor(input_size=76, hidden_sizes=[36, 12], dropout_p=0.2, num_outputs=1)
    # model = MLPRegressor(input_size=76, hidden_sizes=[24], dropout_p=0.2, num_outputs=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedMSELoss(weights=[1])
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = None
    # train
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    # test
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_predictions = evaluate_model(model, test_loader)
    return test_predictions
    

def kfold_cv_with_scoring(X, y, model_dict, score_type_list=['r2', 'mae', 'rmse'], n_splits=8, output_scaler=None):
    from sklearn.model_selection import KFold
    model_type = model_dict['model_type']
    n = len(y)
    ypred_cv = np.zeros((n,))
    regr = model_dict['model']
    

    kFold=KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in kFold.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        
        if model_type not in ['mlp', 'cnn']:
            regr.fit(X_train, y_train)
            ypred_test = regr.predict(X_test)
        else:
            ypred_test = train_test_with_torch_nn(X_train, y_train, X_test, y_test, num_epochs=800, lr=0.0005, batch_size=32, output_scaler=output_scaler)
        ypred_cv[test_index] = ypred_test.reshape(-1)

    # update model dict with ypred values
    if output_scaler is not None:
        ypred_cv = output_scaler.inverse_transform(ypred_cv.reshape(-1,1)).reshape(-1)
        y = output_scaler.inverse_transform(y.reshape(-1,1)).reshape(-1)
    model_dict.update({'ypred_cv': ypred_cv})
    
    # calculate test metrics across k folds
    for score_type in score_type_list:
        score, score_norm = get_score(y, ypred_cv, score_type)
        model_dict.update({f'{score_type}_cv':score, f'{score_type}_norm_cv':score_norm})

    return model_dict

def fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, score_type_list=['r2', 'mae', 'rmse'], cv=None, scale_data=False, print_output=True):
    
    model_params = []
    print('score_type_list:', score_type_list)
    input_scaler = None
    output_scaler = None
    
    # get model
    for i, model_dict in enumerate(model_list):
        model_type = model_dict['model_type']
        if model_type=='plsr':
            n_components =  min(model_dict['n_components'], X.shape[1])
            model = PLSRegression(n_components=n_components)
            model_params.append(('n_components',n_components))
        elif model_type=='ridge':
            max_iter = model_dict['max_iter']
            alpha = model_dict['alpha']
            model = Ridge(max_iter=max_iter, alpha=alpha)
            model_params.append(('alpha',alpha))
        elif model_type=='lasso':
            max_iter = model_dict['max_iter']
            alpha = model_dict['alpha']
            model = Lasso(max_iter=max_iter, alpha=alpha)
            model_params.append(('alpha',alpha))
        elif model_type=='randomforest':
            n_estimators = model_dict['n_estimators']
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
            model_params.append(('n_estimators',n_estimators))
        elif model_type=='xgb':
            n_estimators = model_dict['n_estimators']
            model = XGBRegressor(objective="reg:squarederror", n_estimators=n_estimators, random_state=0)
            model_params.append(('n_estimators',n_estimators))
        elif model_type=='gp':
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.0,                # we already model noise via WhiteKernel
                n_restarts_optimizer=10,  # multiple restarts for better optima
                normalize_y=True,         # standardize targets internally
                random_state=42
                )
        elif model_type=='mlp':
            model = None
            input_scaler = MinMaxScaler() 
            output_scaler = MinMaxScaler()
            X = input_scaler.fit_transform(X)
            y = output_scaler.fit_transform(y.reshape(-1,1))  
        elif model_type=='cnn':
            model = None
            input_scaler = MinMaxScaler() 
            output_scaler = MinMaxScaler()
            X = input_scaler.fit_transform(X)
            y = output_scaler.fit_transform(y.reshape(-1,1))  
            
            
        model_dict.update({'model':model})
        
        # perform cross-validation
        if cv is None:
            model_dict = adjusted_loocv_with_scoring(X,y, model_dict, score_type_list=score_type_list, output_scaler=output_scaler)
        else: 
            model_dict = kfold_cv_with_scoring(X,y, model_dict, score_type_list=score_type_list, n_splits=cv, output_scaler=output_scaler)
            
        
        # fit all data and get R2 score
        if scale_data:
            _, _, X_, _ = perform_mean_std_scaling(X, None)
        else: 
            X_ = X.copy()
            
        # train model
        if model_type not in ['mlp','cnn']:
            model.fit(X_, y)
            ypred_train = model.predict(X_)
        else: 
            ypred_test = train_test_with_torch_nn(X_, y, X_, y, num_epochs=800, lr=0.0005, batch_size=32, output_scaler=output_scaler)

        
        model_list[i].update({'model':model, 'ypred':ypred_train})
        for score_type in score_type_list: 
            score_train, score_train_norm = get_score(y, ypred_train, score_type)
            model_list[i].update({f'{score_type}_train': round(score_train,3), f'{score_type}_norm_train': round(score_train_norm,3)})
        if print_output:
            print(f'[{yvar} <> {model_type}]') 
            for score_type in score_type_list:
                print(f'{score_type}: TRAIN {model_dict[f"{score_type}_train"]} ({model_dict[f"{score_type}_norm_train"]}) | TEST {model_dict[f"{score_type}_cv"]} ({model_dict[f"{score_type}_norm_cv"]})')
        
    
    # UPDATE INDIVIDUAL MODEL SCORES
    if len(model_list)==1: 
        metrics = {
            'model_type': model_type, 
            'yvar': yvar, 
            'model_params': model_params
            }
        metrics.update({k:model_dict[k] for k in model_dict if k not in ['model', 'ypred', 'ypred_cv']})
    # UPDATE ENSEMBLE MODEL SCORES
    else: 
        metrics = {
            'model_type': 'ENSEMBLE', 
            'yvar': yvar, 
            'model_params': None
            }
        
        # ensemble predictions
        ypred_ensemble_cv = np.zeros((len(y), len(model_list)))
        ypred_ensemble = np.zeros((len(y), len(model_list)))
        w = 1/len(model_list)
        for i, model_dict in enumerate(model_list):
            if 'w' in model_dict:
                w = model_dict['w']
            ypred_ensemble_cv[:,i] = model_dict['ypred_cv']*w
            ypred_ensemble[:,i] = model_dict['ypred']*w
        # get scores
        for score_type in score_type_list: 
            ypred_ensemble = np.sum(ypred_ensemble, axis=1)
            ypred_ensemble_cv = np.sum(ypred_ensemble_cv, axis=1)
            score_train, score_train_norm = get_score(y, ypred_train, score_type)
            score_cv, score_cv_norm = get_score(y, ypred_train, score_type)
            model_list[i].update({f'{score_type}_ensemble_train': round(score_train,3), f'{score_type}_norm_ensemble_train': round(score_train_norm,3)})
            model_list[i].update({f'{score_type}_ensemble_cv': round(score_cv,3), f'{score_type}_norm_ensemble_cv': round(score_cv_norm,3)})
        if print_output:
            print(f'[{yvar} <> ENSEMBLE]') 
            for score_type in score_type_list:
                print(f'{score_type}: TRAIN {model_dict[f"{score_type}_ensemble_train"]} ({model_dict[f"{score_type}_norm_ensemble_train"]}) | TEST {model_dict[f"{score_type}_ensemble_cv"]} ({model_dict[f"{score_type}_norm_ensemble_cv"]})')

        
    # plot predicted values
    if plot_predictions:
        plt.scatter(y, ypred_ensemble)
        plt.title(yvar)
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.show()
    
    print()
    return model_list, metrics


def get_classifier_scoring(y_pred, y_true, model_name=None, class_labels=[-1,0,1], plot_roc=False):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef,
        roc_auc_score, roc_curve, auc
    )
    from sklearn.preprocessing import label_binarize
    # Binarize labels for multi-class ROC-AUC calculation
    n_classes = len(class_labels)
    y_true_binarized = label_binarize(y_true, classes=class_labels)  # Convert to one-hot encoding
    y_pred_binarized = label_binarize(y_pred, classes=class_labels)  # Convert predictions to one-hot
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred, average="macro"),  # Balanced across classes
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_true, y_pred),  # Balanced metric for multi-class
        "ROC-AUC": roc_auc_score(y_true_binarized, y_pred_binarized, average="macro")  # Multi-class ROC
    }
    # Plot ROC Curve for each class
    if plot_roc:
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_labels[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random chance
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves for {model_name}")
        plt.legend()
        plt.show()
    return metrics


def split_data_to_trainval_test(X, n_splits=5, split_type='random'):
    split_dict = {}
    if split_type=='random':
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, random_state=None, shuffle=False)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            split_dict.update({i: (train_index, test_index)})
    return split_dict
            

def run_trainval_test(X, Y, yvar_list, xvar_selected, xvar_list_all, dataset_name_wsuffix, featureset_suffix='', n_splits=5, scoring='mae', models_to_eval_list=['randomforest'], model_params=model_params, save_results=None, show_plots=True, print_progress=True, model_cmap=model_cmap):

    print(model_cmap)
    # create train-val / test splits using KFold
    kf_dict = split_data_to_trainval_test(X, n_splits=n_splits, split_type='random')
    metrics_list = ['r2_train', 'r2_test', 'mae_norm_train', 'mae_norm_test', 'mae_train', 'mae_test']
    kfold_metrics = []
        
    # initialize dict for storing trained models
    SURROGATE_MODELS = {yvar:{model_type:None for model_type in models_to_eval_list} for yvar in yvar_list}
    features_selected_sorted_dict = {}
    ypred_train_bymodel = {model_type: np.zeros((len(Y), len(yvar_list))) for model_type in models_to_eval_list}
    ypred_test_bymodel = {model_type: np.zeros((len(Y), len(yvar_list))) for model_type in models_to_eval_list}
    
    for k, (train_idx, test_idx) in kf_dict.items():
        print('Fold:', k)
        print(test_idx)
    
        # get split data
        X_trainval, Y_trainval = X[train_idx, :], Y[train_idx, :]
        X_test, Y_test = X[test_idx, :], Y[test_idx, :]
        
        if print_progress:
            print('\n************************************************')
            print(f'Performing train/test evaluation on Fold {k}...')
            print('************************************************')     
        for model_type in models_to_eval_list:     
            for i, yvar in enumerate(yvar_list): 
                if print_progress:
                    print(f'Evaluating {model_type} <> {yvar}...')
                # get features selected
                if isinstance(xvar_selected, list):
                    features_selected = xvar_selected
                else:
                    features_selected = xvar_selected[yvar]
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
                ypred_test_bymodel[model_type][test_idx, i] = metrics['ypred_test']
    
    if print_progress:
        print('\n********************************************************')
        print('Getting train performance and coefficients on full dataset')
        print('**********************************************************') 
    for model_type in models_to_eval_list:  
        for i, yvar in enumerate(yvar_list): 
            # get features selected
            if isinstance(xvar_selected, list):
                features_selected = xvar_selected
            else:
                features_selected = xvar_selected[yvar]
            X_selected = pd.DataFrame(X, columns=xvar_list_all).loc[:, features_selected].to_numpy()
            print('X_selected.shape: ', X_selected.shape)
            y = Y[:,i]
            model_dict = model_params[dataset_name_wsuffix][model_type][yvar][0]
            metrics_train, model = evaluate_model_on_train_test_data(X_selected, y, X_selected, y, model_dict, scoring=scoring)
            ypred_train_bymodel[model_type][:,i] = metrics_train['ypred_test']
            SURROGATE_MODELS[yvar][model_type] = model
        
            # get features
            try: 
                coefs = get_feature_coefficients(model, model_type)
                features_selected_sorted, coefs_sorted = order_features_by_coefficient_importance(coefs, features_selected, filter_out_zeros=True)
                features_selected_sorted_dict.update({yvar: features_selected_sorted})
                xticks = np.arange(len(coefs_sorted))
                if show_plots:
                    plt.figure(figsize=(20,10))
                    plt.bar(xticks, coefs_sorted)
                    plt.xticks(xticks, features_selected_sorted, rotation=90, fontsize=8)
                    plt.ylabel('Feature importance')
                    plt.title(f'{yvar}: {model_type} feature importances')
                    plt.show()
            except: 
                print(f'No feature importance estimates for {model_type} model.')
                
    # get ensemble results
    if len(models_to_eval_list)>1:
        ypred_train_bymodel['ENSEMBLE'] = np.zeros_like(ypred_train_bymodel[models_to_eval_list[0]])
        ypred_test_bymodel['ENSEMBLE'] = np.zeros_like(ypred_test_bymodel[models_to_eval_list[0]])
        for model_type in models_to_eval_list:  
            ypred_train_bymodel['ENSEMBLE'] += ypred_train_bymodel[model_type]
            ypred_test_bymodel['ENSEMBLE'] += ypred_test_bymodel[model_type]
        # average predictions
        ypred_train_bymodel['ENSEMBLE'] /= len(models_to_eval_list)
        ypred_test_bymodel['ENSEMBLE'] /= len(models_to_eval_list)
        # get kfold metrics
        for k, (train_idx, test_idx) in kf_dict.items():
            Y_train = Y[train_idx, :]    
            Y_test = Y[test_idx, :]      
            for i, yvar in enumerate(yvar_list):  
                y_train = Y_train[:, i]
                y_test = Y_test[:, i]
                ypred_train_ensemble = ypred_train_bymodel['ENSEMBLE'][train_idx, i]
                ypred_test_ensemble = ypred_test_bymodel['ENSEMBLE'][test_idx, i]
                r2_train = round(r2_score(y_train, ypred_train_ensemble),2)
                r2_test = round(r2_score(y_test, ypred_test_ensemble),2)
                mae_train = round(mean_absolute_error(y_train, ypred_train_ensemble),2)
                mae_test = round(mean_absolute_error(y_test, ypred_test_ensemble),2)
                mae_norm_train = mae_train/np.mean(y_train)
                mae_norm_test = mae_test/np.mean(y_test)
                metrics = {
                    'model_type': 'ENSEMBLE',
                    'yvar': yvar,
                    'param_name': None,
                    'param_val': None ,
                    'k': k,
                    'r2_train':r2_train,
                    'r2_test':r2_test,
                    'mae_norm_train':mae_norm_train,
                    'mae_norm_test':mae_norm_test,
                    'mae_train':mae_train,
                    'mae_test':mae_test,
                    'xvar_sorted': None,
                 }
                kfold_metrics.append(metrics)
        
        
        models_to_plot_list = models_to_eval_list + ['ENSEMBLE']
    else: 
        models_to_plot_list = models_to_eval_list
        
    # plot scatter predictions
    if show_plots:
        savefig = None # f'{figure_folder}modelpredictions_scatterplots_{dataset_name}{dataset_suffix}{featureset_suffix}.png'
        plot_scatter_train_test_predictions(Y[:,:len(yvar_list)], ypred_train_bymodel, ypred_test_bymodel, yvar_list, models_to_plot_list, savefig=savefig, model_cmap=model_cmap)
    if print_progress:
        print('\n********************')
        print('PLOT OVERALL RESULTS')
        print('**********************')    
     
    # collect kfold metrics
    kfold_metrics_cols = ['model_type', 'yvar', 'param_name', 'param_val', 'k', 'r2_train', 'r2_test', f'{scoring}_norm_train', f'{scoring}_norm_test', f'{scoring}_train', f'{scoring}_test', 'xvar_sorted']
    kfold_metrics = pd.DataFrame(kfold_metrics, columns=kfold_metrics_cols).sort_values(by=['yvar', 'model_type', 'k']).reset_index(drop=True)
    
    # calculate average metrics
    kfold_metrics_avg = []
    for model_type in models_to_plot_list:
        for yvar in yvar_list:
            kfold_metrics_avg_dict = {'model_type':model_type, 'yvar':yvar}
            metrics_avg = kfold_metrics[(kfold_metrics.yvar==yvar) & (kfold_metrics.model_type==model_type)][metrics_list].rename(columns={c:f'{c}_avg' for c in metrics_list}).mean(axis=0).round(3).to_dict()
            metrics_std = kfold_metrics[(kfold_metrics.yvar==yvar) & (kfold_metrics.model_type==model_type)][metrics_list].rename(columns={c:f'{c}_std' for c in metrics_list}).std(axis=0).round(3).to_dict()
            kfold_metrics_avg_dict.update(metrics_std)
            kfold_metrics_avg_dict.update(metrics_avg)
            kfold_metrics_avg.append(kfold_metrics_avg_dict)
        kfold_metrics_avg_cols = ['model_type', 'yvar'] + list(metrics_avg.keys()) + list(metrics_std.keys())
    kfold_metrics_avg = pd.DataFrame(kfold_metrics_avg, columns=kfold_metrics_avg_cols)
    if save_results is not None:
        kfold_metrics.to_csv(f'{data_folder}model_metrics_kfold_{dataset_name_wsuffix}{featureset_suffix}.csv')
        kfold_metrics_avg.to_csv(f'{data_folder}model_metrics_avg_kfold_{dataset_name_wsuffix}{featureset_suffix}.csv') # 
    
    # plot kfold results
    if show_plots:
        figtitle = f'Model evaluation metrics for {dataset_name_wsuffix}{featureset_suffix}, {n_splits}-fold CV'
        savefig = f'{figure_folder}modelmetrics_allselectedmodels_keyYvar_{dataset_name_wsuffix}{featureset_suffix}' if save_results else None
        plot_model_metrics(kfold_metrics_avg, models_to_plot_list, yvar_list, nrows=2, ncols=1, figsize=(7*len(yvar_list),15), barwidth=0.7, figtitle=figtitle, savefig=savefig, suffix_list=['_train_avg','_test_avg'], plot_errors=True, annotate_vals=True, model_cmap=model_cmap)
        # plot_model_metrics_cv(kfold_metrics_avg, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(20,15), barwidth=0.7, figtitle=figtitle, savefig=savefig, suffix_list=['_test_avg'], plot_errors=True, annotate_vals=True)
        
    return kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS, ypred_train_bymodel, ypred_test_bymodel


def eval_model_over_params(model_dict, yvar, X, y, modelparam_metrics_df_bymodelyvar, trained_model_cache, scoring='mae', scale_data=True):
    # get model_list and parameter list
    model_list = [{k:v for k,v in model_dict.items() if k!='params_to_eval'}]
    (param_name, param_val_list) = model_dict['params_to_eval']
    # iterate over parameter values
    for param_val in param_val_list: 
        print(param_name, param_val, end=': ')
        model_list[0].update({param_name:param_val})
        # fit model with selected features and parameters and get metrics 
        model_list, metrics = fit_model_with_cv(X, y, yvar, model_list, plot_predictions=False, scoring=scoring, scale_data=scale_data)
        # update metrics dict with param val
        metrics.update({'param_name': param_name, 'param_val': param_val})
        modelparam_metrics_df_bymodelyvar.append(metrics)
        trained_model_cache.update({param_val: model_list[0]['model']})
        
    return trained_model_cache, modelparam_metrics_df_bymodelyvar
                        
    
def plot_model_param_results(modelparam_metrics_df, yvar_list_to_plot, dataset_name, scoring='mae', model_type_list=['randomforest', 'plsr', 'lasso'], savefig=None):
    
    for model_type in model_type_list: 
        fig, ax = plt.subplots(3,4, figsize=(27,17))
        for i, yvar in enumerate(yvar_list_to_plot):
            # get relevant metrics to plot -- remove data that is out of range
            metrics_filt = modelparam_metrics_df.loc[(modelparam_metrics_df['model_type']==model_type) & (modelparam_metrics_df['yvar']==yvar)]
            metrics_filt = metrics_filt.loc[(metrics_filt.r2_train>=0) & (metrics_filt.mae_norm_cv<5)]
            if len(metrics_filt)>0:
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
        plt.suptitle(f'{model_type} metrics for various model parameters \n{yvar_list_to_plot}', y=ymax*1.07, fontsize=20)   
        if savefig is not None:
            fig.savefig(savefig, bbox_inches='tight')
        plt.show()
                   

def evaluate_model_on_train_test_data(X_test, y_test, X_train, y_train, model_dict, scoring='mae', print_res=False): 
    
    # initialize results dict
    metrics = {}
    
    # get scaled data
    mean, std, X_train_, X_test_ = perform_mean_std_scaling(X_train, X_test)
    ymean = np.mean(y_train)
    
    # retrain model if not provided
    model_type = model_dict['model_type']
    if model_type=='plsr':
        from sklearn.cross_decomposition import PLSRegression
        param_name = 'n_components'
        param_val = model_dict[param_name]
        model = PLSRegression(n_components=min(param_val, X_test.shape[1]))
    elif model_type=='lasso':
        from sklearn.linear_model import Lasso
        param_name = 'alpha'
        param_val = model_dict[param_name]
        model = Lasso(max_iter=50000, alpha=param_val)
    elif model_type=='randomforest':
        from sklearn.ensemble import RandomForestRegressor
        param_name = 'n_estimators'
        param_val = model_dict[param_name]
        model = RandomForestRegressor(n_estimators=param_val, random_state=0)
    elif model_type=='xgb':
        from xgboost import XGBRegressor
        param_name = 'n_estimators'
        param_val = model_dict[param_name]
        model = XGBRegressor(objective="reg:squarederror", n_estimators=param_val, random_state=0)
    elif model_type=='gp':
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        param_name = 'rbf_lengthscale'
        param_val = 1.0
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=param_val, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,                # we already model noise via WhiteKernel
            n_restarts_optimizer=10,  # multiple restarts for better optima
            normalize_y=True,         # standardize targets internally
            random_state=42
        )
    elif model_type=='mlp':
        model = None
        param_name = 'num_epochs'
        param_val = 800
        input_scaler = MinMaxScaler() 
        output_scaler = MinMaxScaler()
        X_ = input_scaler.fit_transform(np.concatenate((X_train_, X_test_), axis=0))
        y = output_scaler.fit_transform(np.concatenate((y_train, y_test), axis=0).reshape(-1,1))
        X_train_ = X_[:X_train.shape[0],:]
        X_test_ = X_[X_train.shape[0]:,:]
        y_train = y[:X_train.shape[0],:]
        y_test = y[X_train.shape[0]:,:]

        
    # train model
    if model_type not in ['mlp','cnn']:
        model.fit(X_train_, y_train)
        ypred_train = model.predict(X_train_)
        ypred_test = model.predict(X_test_)
    else: 
        ypred_train = train_test_with_torch_nn(X_train_, y_train, X_train_, y_train, num_epochs=800, lr=0.0005, batch_size=32, output_scaler=output_scaler)
        ypred_test = train_test_with_torch_nn(X_train_, y_train, X_test_, y_test, num_epochs=800, lr=0.0005, batch_size=32, output_scaler=output_scaler)
        ypred_train = output_scaler.inverse_transform(ypred_train).reshape(-1)
        ypred_test = output_scaler.inverse_transform(ypred_test).reshape(-1)
        y_train = output_scaler.inverse_transform(y_train).reshape(-1)
        y_test = output_scaler.inverse_transform(y_test).reshape(-1)

        
    # calculate MAE score for ypred 
    score_train = round(get_score(y_train, ypred_train, scoring, get_norm_score=False),3)
    score_norm_train = round(score_train/ymean, 3)
    # calculate R2 score for ypred_loocv
    r2_train = round(r2_score(y_train, ypred_train),2)
    
    # calculate MAE score for ypred 
    score_test = round(get_score(y_test, ypred_test, scoring, get_norm_score=False),3)
    score_norm_test = round(score_test/ymean, 3)
    # calculate R2 score for ypred_loocv
    r2_test = round(r2_score(y_test, ypred_test),2)

    # update model dict
    metrics.update({
        'model_type': model_type, 
        'param_name': param_name, 
        'param_val': param_val,
        f'{scoring}_train':score_train, 
        f'{scoring}_norm_train': score_norm_train,
        'r2_train': r2_train,
        f'{scoring}_test':score_test, 
        f'{scoring}_norm_test': score_norm_test,
        'r2_test': r2_test,
        'ypred_test': ypred_test,
        'y_test': y_test
        })
    
    if print_res: 
        print(f'R2:{r2_train}, R2 (CV):{r2_test}, {scoring} (train): {score_train} ({round(score_norm_train*100,1)}%), {scoring} (CV):{score_test} ({round(score_norm_test*100,1)}%)')
    return metrics, model
        

#%% feature analysis

def order_list_by_frequencies(lst):
    arr = np.array(lst)
    vals, counts = np.unique(arr, return_counts=True)
    idxs_counts_sorted = np.argsort(counts)
    idxs_counts_sorted = idxs_counts_sorted[::-1]
    lst_sorted = vals[idxs_counts_sorted]
    lst_sorted = list(lst_sorted)
    return lst_sorted

def get_order_of_element_sizes(arr, invert_importance_order=True): 
    n = len(arr)
    idxs = np.argsort(arr)
    sortdict = {}
    for k in range(n): 
        idx = int(idxs[k])
        arr_k = arr[idx]
        if arr_k == 0:
            if invert_importance_order:
                sortdict.update({arr_k:0})
            else:
                sortdict.update({arr_k:n})
        else: 
            if invert_importance_order:
                sortdict.update({arr_k:k})
            else: 
                sortdict.update({arr_k:n-k})
            
    el_orders = []
    for k in range(n): 
        el = arr[k]
        el_orders.append(sortdict[el]) 
    return el_orders

def get_feature_coefficients(model, model_type):
    if model_type in ['plsr', 'lasso', 'ridge']: 
        coefs = model.coef_.reshape(-1,)
    elif model_type in ['randomforest', 'xgb']:
        coefs = model.feature_importances_
    return coefs

def order_features_by_coefficient_importance(coefs, xvar_list, filter_out_zeros=True):
    sort_idxs = np.argsort(coefs)
    sort_idxs = sort_idxs[::-1]
    xvar_list_sorted = [xvar_list[idx] for idx in sort_idxs]
    coefs_sorted = np.array([coefs[idx] for idx in sort_idxs])
    if filter_out_zeros:
        zero_idxs = np.where(coefs_sorted==0)[0]
        xvar_list_sorted = [xvar for i, xvar in enumerate(xvar_list_sorted) if i not in zero_idxs]
        coefs_sorted = [coef for i, coef in enumerate(coefs_sorted) if i not in zero_idxs]
    return xvar_list_sorted, coefs_sorted
        

def get_feature_importances(model_dict, yvar, xvar_list, plot_feature_importances=True, plot_ordered=True, normalize_feature_scoring=False):
    
    # initialize lists for recording feature coefficients and importance score
    feature_coefs = []
    feature_importances = []
    
    # get coefficients
    model_type = model_dict['model_type']
    model = model_dict['model']
    coef_x_list = get_feature_coefficients(model, model_type)

    coef_x_list_ABS = np.abs(coef_x_list)
    orders_x_list = get_order_of_element_sizes(coef_x_list_ABS)
    feature_coef_dict = {'model_type': model_type, 'yvar': yvar}
    feature_coef_dict.update({f'{xvar_list[k]}':coef_x_list[k] for k in range(len(xvar_list))})
    feature_coefs.append(feature_coef_dict)
    feature_importance_dict = {'model_type': model_type, 'yvar': yvar}
    if normalize_feature_scoring:
        scoring_norm_constant = len(orders_x_list)-1
    else: 
        scoring_norm_constant = 1
    feature_importance_dict.update({f'{xvar_list[k]}':orders_x_list[k]/scoring_norm_constant for k in range(len(xvar_list))})
    feature_importances.append(feature_importance_dict)   
    
    if plot_feature_importances: 
        if plot_ordered: 
            idx_ascending = coef_x_list_ABS.argsort()
            idx_descending = idx_ascending[::-1]  
            coef_x_list_ABS_plot = coef_x_list_ABS[idx_descending]
            xvar_list_plot = [xvar_list[idx] for idx in idx_descending]
        else:
            coef_x_list_ABS_plot = coef_x_list_ABS
            xvar_list_plot = xvar_list
        plt.figure(figsize=(12, 8))
        plt.bar(np.arange(len(xvar_list_plot)), coef_x_list_ABS_plot, color ='maroon', width = 0.4)
        plt.xticks(np.arange(len(xvar_list_plot)), labels=xvar_list_plot, rotation=90, ha='right')
        plt.xlabel("X variables")
        plt.ylabel("Absolute value of PLS coefficients")
        plt.title(yvar)
        plt.show()
    return feature_coefs, feature_importances


def sort_list(lst):
    lst.sort()
    return lst


def plot_feature_importance_heatmap(heatmap_df, xvar_list, yvar_list, logscale_cmap=False, scale_vals=False, annotate=None, get_clustermap=False, figtitle=None, savefig=None, return_fig_handle=False, datamin=None, datamax=None):

    # get array data from feature dataframe
    arr = heatmap_df.to_numpy()
    
    # scale array values if needed
    if scale_vals:
        arr_unscaled = arr.copy()
        for row_idx in range(arr_unscaled.shape[0]):
            scalefactor = np.nanmax(np.abs(arr_unscaled[row_idx,:]))
            arr[row_idx,:] = arr_unscaled[row_idx,:]/scalefactor
        # replace data in heatmap
        heatmap_df = pd.DataFrame(arr, columns=xvar_list, index=yvar_list)
        
    # plot heatmap 
    fig, ax = plt.subplots(1,1, figsize=(25, arr.shape[0]/arr.shape[1]*25))
    _, _, ax = heatmap(arr, c='viridis', ax=ax, cbar_kw={}, cbarlabel="", annotate=annotate, row_labels=yvar_list, col_labels=xvar_list, logscale_cmap=logscale_cmap, show_gridlines=False, datamin=datamin, datamax=datamax)
    if figtitle is not None:
        ax.set_title(figtitle, fontsize=16)
    if savefig is not None:
        fig.savefig(f'{figure_folder}{savefig}.png',  bbox_inches='tight')     
        
    # plot clustermap of heatmap
    if get_clustermap:
        cl = sns.clustermap(heatmap_df, cmap="viridis", figsize=(20, 12))
        cl.fig.suptitle(f'Cluster map of {figtitle}', fontsize=16) 
        # plt.title(f'Cluster map of {figtitle}', fontsize=16, loc='center')
        plt.savefig(f"{figure_folder}{savefig}_clustermap.png",  bbox_inches='tight') 
    
    if return_fig_handle:
        return arr, (fig, ax)
    else:
        return arr


def plot_feature_importance_barplots_bymodel(feature_importance_df, yvar_list=None, xvar_list=None, model_list=None, order_by_feature_importance=False, label_xvar_by_indices=True, model_cmap=model_cmap, savefig=None, figtitle=None):

    """
    Plot bar chart comparing overall feature importance for all x variables for across y variables
    """
    if yvar_list is None:
        yvar_list = list(set(feature_importance_df.yvar.tolist()))
    if xvar_list is None:
        xvar_list = feature_importance_df.columns.tolist()[2:]
    if model_list is None:
        model_list = list(set(feature_importance_df.model_type.tolist()))
    
    nrows = len(model_list)
    ncols = len(yvar_list)
    fig, ax = plt.subplots(nrows, ncols, figsize=(13.5*ncols,7.3*nrows))
    xtickpos = np.arange(len(xvar_list))+1
    xticklabels = [str(idx) for idx in range(len(xvar_list))]
    width = 0.8
    # if too many x variables, label only every other var
    for i, model_type in enumerate(model_list): 
        for j, yvar in enumerate(yvar_list):
            feature_importance_arr_yvar = feature_importance_df[(feature_importance_df.yvar==yvar) & (feature_importance_df.model_type==model_type)].iloc[0, 2:].to_numpy()
            if order_by_feature_importance:
                idx_ordered = np.argsort(feature_importance_arr_yvar)
                idx_ordered = idx_ordered[::-1]
                feature_importance_arr_yvar_ordered = feature_importance_arr_yvar[idx_ordered]
                xticklabels_ordered = [xticklabels[idx] for idx in idx_ordered]
            else: 
                feature_importance_arr_yvar_ordered = feature_importance_arr_yvar
                xticklabels_ordered = xticklabels
            
            if len(xvar_list)>50 and not order_by_feature_importance:
                xticklabels_ordered = [label if k%2==0 else '' for k, label in enumerate(xticklabels_ordered)]
                
            if len(model_list)>1:
                if len(yvar_list)>1:
                    ax[i][j].bar(xtickpos, feature_importance_arr_yvar_ordered, color=model_cmap[model_type], width=width)
                    if order_by_feature_importance:
                        for x, y, xticklabel in zip(xtickpos, feature_importance_arr_yvar_ordered, xticklabels_ordered):
                            ax[i][j].annotate(xticklabel, (x-width*0.75,y*1.01), fontsize=7)
                    else:
                        ax[i][j].set_xticks(xtickpos, xticklabels_ordered, fontsize=8)
                    ax[i][j].set_title(yvar, fontsize=20)
                    ax[i][j].set_ylabel('feature prominences', fontsize=16)
                else: 
                    ax[i].bar(xtickpos, feature_importance_arr_yvar_ordered, color=model_cmap[model_type], width=width)
                    if order_by_feature_importance:
                        for x, y, xticklabel in zip(xtickpos, feature_importance_arr_yvar_ordered, xticklabels_ordered):
                            ax[i].annotate(xticklabel, (x-width*0.75,y*1.01), fontsize=7)
                    else:
                        ax[i].set_xticks(xtickpos, xticklabels_ordered, fontsize=8)
                    ax[i].set_title(yvar, fontsize=20)
                    ax[i].set_ylabel('feature prominences', fontsize=16)
            else: 
                if len(yvar_list)>1:
                    ax[j].bar(xtickpos, feature_importance_arr_yvar_ordered, color=model_cmap[model_type], width=width)
                    if order_by_feature_importance:
                        for x, y, xticklabel in zip(xtickpos, feature_importance_arr_yvar_ordered, xticklabels_ordered):
                            ax[j].annotate(xticklabel, (x-width*0.75,y*1.01), fontsize=7)
                    else:
                        ax[j].set_xticks(xtickpos, xticklabels_ordered, fontsize=8)
                    ax[j].set_title(yvar, fontsize=20)
                    ax[j].set_ylabel('feature prominences', fontsize=16)
                else:
                    ax.bar(xtickpos, feature_importance_arr_yvar_ordered, color=model_cmap[model_type], width=width)
                    if order_by_feature_importance:
                        for x, y, xticklabel in zip(xtickpos, feature_importance_arr_yvar_ordered, xticklabels_ordered):
                            ax.annotate(xticklabel, (x-width*0.75,y*1.01), fontsize=7)
                    else:
                        ax.set_xticks(xtickpos, xticklabels_ordered, fontsize=8)
                    ax.set_title(yvar, fontsize=20)
                    ax.set_ylabel('feature prominences', fontsize=16)
        
    if figtitle is not None: 
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'{figtitle}', y=ymax*1.04, fontsize=24)    
            
    if savefig is not None:
        fig.savefig(f'{figure_folder}{savefig}.png', bbox_inches='tight')
    plt.show()  
    

def plot_feature_importance_barplots(feature_importance_arr, yvar_list, xvar_list, order_by_feature_importance=False, label_xvar_by_indices=True, ncols=4, nrows=3, c='#1f77b4', savefig=None, figtitle=None):

    """
    Plot bar chart comparing overall feature importance for all x variables for across y variables
    """
    fig, ax = plt.subplots(nrows,ncols, figsize=(54,22))
    xtickpos = np.arange(len(xvar_list))+1
    xticklabels = [str(idx) for idx in range(len(xvar_list))]
    width = 0.8
    # if too many x variables, label only every other var
    for i, yvar in enumerate(yvar_list):
        row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
        feature_importance_arr_yvar = feature_importance_arr[i, :]    
        
        if order_by_feature_importance:
            idx_ordered = np.argsort(feature_importance_arr_yvar)
            idx_ordered = idx_ordered[::-1]
            feature_importance_arr_yvar_ordered = feature_importance_arr_yvar[idx_ordered]
            xticklabels_ordered = [xticklabels[idx] for idx in idx_ordered]
        else: 
            feature_importance_arr_yvar_ordered = feature_importance_arr_yvar
            xticklabels_ordered = xticklabels
        
        if len(xvar_list)>50 and not order_by_feature_importance:
            xticklabels_ordered = [label if k%2==0 else '' for k, label in enumerate(xticklabels_ordered)]

        ax[row_idx][col_idx].bar(xtickpos, feature_importance_arr_yvar_ordered, color=c, width=width)
        if order_by_feature_importance:
            for x, y, xticklabel in zip(xtickpos, feature_importance_arr_yvar_ordered, xticklabels_ordered):
                ax[row_idx][col_idx].annotate(xticklabel, (x-width*0.75,y*1.01), fontsize=7)
        else:
            ax[row_idx][col_idx].set_xticks(xtickpos, xticklabels_ordered, fontsize=8)
        ax[row_idx][col_idx].set_title(yvar, fontsize=20)
        ax[row_idx][col_idx].set_ylabel('feature prominences', fontsize=16)
        
    if figtitle is not None: 
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'{figtitle}', y=ymax*1.06, fontsize=24)    
            
    if savefig is not None:
        fig.savefig(f'{figure_folder}{savefig}.png', bbox_inches='tight')
    plt.show()  
    

def order_features_by_importance(feature_scoring_agg_yvar, xvar_list):     
    orders_x_list = get_order_of_element_sizes(feature_scoring_agg_yvar)
    overall_feature_importance_yvar = {f'{xvar_list[k]}':feature_scoring_agg_yvar[k] for k in range(len(xvar_list))}
    feature_scoring_yvar = {f'{xvar_list[k]}':orders_x_list[k] for k in range(len(xvar_list))}    
    feature_ordering_yvar = [xvar_list[idx] for idx in np.argsort(feature_scoring_agg_yvar)][::-1]
    return overall_feature_importance_yvar, feature_scoring_yvar, feature_ordering_yvar


#%%
def plot_model_metrics(model_metrics_df, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(30,15), barwidth=0.8, figtitle=None, savefig=None, suffix_list=['_train','_cv'], plot_errors=False, model_cmap=model_cmap, annotate_vals=False, score_type_list=['r2','mae_norm']):
    fig, ax = plt.subplots(nrows,ncols, figsize=figsize)
    xtickpos = np.arange(len(yvar_list))+1
    legenditems_count = 0
    legenditems_dict = {}
    for i, yvar in enumerate(yvar_list):
        for j, model_type in enumerate(models_to_eval_list):
            c = model_cmap[model_type]
            w = barwidth / (len(models_to_eval_list)*2)
            xtickoffset = -w*len(models_to_eval_list)/2 + 0.5*w + j*w*2
            model_metrics_df_filt = model_metrics_df[(model_metrics_df.yvar==yvar) & (model_metrics_df.model_type==model_type)].iloc[0].to_dict()
            
            for k, score_type in enumerate(score_type_list):
                if k==0:
                    legenditems_dict[legenditems_count] = ax[k].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]], width=w, label=model_type, color=c, alpha=0.5, hatch='/')
                    legenditems_count += 1 
                    legenditems_dict[legenditems_count] = ax[k].bar(xtickpos[i]+xtickoffset+w, model_metrics_df_filt[score_type+suffix_list[1]], width=w, label=model_type, color=c)
                    legenditems_count += 1
                else: 
                    ax[k].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]], width=w, label=model_type, color=c, alpha=0.5, hatch='/')
                    ax[k].bar(xtickpos[i]+xtickoffset+w, model_metrics_df_filt[score_type+suffix_list[1]], width=w, label=model_type, color=c)
                ymax_ax = np.max(model_metrics_df[[score_type+suffix_list[0], score_type+suffix_list[1]]].to_numpy())

                # annotate values
                if annotate_vals:
                    ax[k].annotate(round(model_metrics_df_filt[score_type+suffix_list[0]],3), (xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]]+0.015*ymax_ax))
                    ax[k].annotate(round(model_metrics_df_filt[score_type+suffix_list[1]],3), (xtickpos[i]+xtickoffset+w, model_metrics_df_filt[score_type+suffix_list[1]]+0.015*ymax_ax))
                # plot errors
                if plot_errors:
                    ax[k].errorbar(xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]], yerr=model_metrics_df_filt[score_type+suffix_list[0].replace('avg','std')], color='k', fmt='o', markersize=8, capsize=10, label=None)
                    ax[k].errorbar(xtickpos[i]+xtickoffset+w, model_metrics_df_filt[score_type+suffix_list[1]], yerr=model_metrics_df_filt[score_type+suffix_list[1].replace('avg','std')], color='k', fmt='o', markersize=8, capsize=10, label=None)
                
                # set axis limits 
                ax[k].set_ylim([0,ymax_ax*1.2])
                ax[k].set_ylabel(score_type.upper(), fontsize=20)
                
                # set xticks
                ax[k].set_xticks(xtickpos+xtickoffset/2, yvar_list, fontsize=20)
    
    # set legend
    legend = [f'{model_type}_{train_or_cv}' for model_type in models_to_eval_list for train_or_cv in ['train', 'cv']]
    plt.legend([v for k,v in legenditems_dict.items()], legend, fontsize=16)
    if figtitle is not None:
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(figtitle, y=ymax*1.03, fontsize=24)    
    # save figure
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()
    


def plot_model_metrics_cv(model_metrics_df, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(20,15), barwidth=0.8, figtitle=None, savefig=None, suffix_list=['_cv'], plot_errors=False, model_cmap=model_cmap, annotate_vals=False, score_type_list=['r2','mae_norm']):
    fig, ax = plt.subplots(nrows,ncols, figsize=figsize)
    xtickpos = np.arange(len(yvar_list))+1
    legenditems_count = 0
    legenditems_dict = {}
    for i, yvar in enumerate(yvar_list):
        for j, model_type in enumerate(models_to_eval_list):
            c = model_cmap[model_type]
            w = barwidth / len(models_to_eval_list)
            xtickoffset = -w*len(models_to_eval_list)/2 + 0.5*w + j*w
            model_metrics_df_filt = model_metrics_df[(model_metrics_df.yvar==yvar) & (model_metrics_df.model_type==model_type)].iloc[0].to_dict()
            
            
            for k, score_type in enumerate(score_type_list):
                if k==0:
                    legenditems_dict[legenditems_count] = ax[k].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]], width=w, label=model_type, color=c)
                    legenditems_count += 1 
                else: 
                    ax[k].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]], width=w, label=model_type, color=c)
                ymax_ax = np.max(model_metrics_df[score_type+suffix_list[0]].to_numpy())

                # annotate values
                if annotate_vals:
                    ax[k].annotate(round(model_metrics_df_filt[score_type+suffix_list[0]],3), (xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]]+0.015*ymax_ax))
                # plot errors
                if plot_errors:
                    ax[k].errorbar(xtickpos[i]+xtickoffset, model_metrics_df_filt[score_type+suffix_list[0]], yerr=model_metrics_df_filt[score_type+suffix_list[0].replace('avg','std')], color='k', fmt='o', markersize=8, capsize=10, label=None)
                
                # set axis limits 
                ax[k].set_ylim([0,ymax_ax*1.2])
                ax[k].set_ylabel(score_type.upper(), fontsize=20)
                
                # set xticks
                ax[k].set_xticks(xtickpos+xtickoffset/2, yvar_list, fontsize=20)
     
    for k in range(nrows): 
        ax[k].set_xticks(xtickpos+xtickoffset/2, yvar_list, fontsize=20)
        
    # set ylim for MAE plots
    ymax = ax.flatten()[0].get_position().ymax
    legend = [f'{model_type}_{train_or_cv}' for model_type in models_to_eval_list for train_or_cv in ['cv']]
    plt.legend([v for k,v in legenditems_dict.items()], legend, fontsize=16)
    plt.tight_layout()
    if figtitle is not None:
        plt.suptitle(figtitle, y=1.02, fontsize=24)    
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()
    


# def plot_model_metrics_cv(model_metrics_df, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(20,15), barwidth=0.8, figtitle=None, savefig=None, suffix_list=['_cv'], plot_errors=False, model_cmap=model_cmap, annotate_vals=False):
#     fig, ax = plt.subplots(nrows,ncols, figsize=figsize)
#     xtickpos = np.arange(len(yvar_list))+1
#     legenditems_count = 0
#     legenditems_dict = {}
#     for i, yvar in enumerate(yvar_list):
#         for k, model_type in enumerate(models_to_eval_list):
#             c = model_cmap[model_type]
#             w = barwidth / len(models_to_eval_list)
#             xtickoffset = -w*len(models_to_eval_list)/2 + 0.5*w + k*w
#             model_metrics_df_filt = model_metrics_df[(model_metrics_df.yvar==yvar) & (model_metrics_df.model_type==model_type)].iloc[0].to_dict()
#             ## R2
#             legenditems_dict[legenditems_count] = ax[0].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt['r2'+suffix_list[0]], width=w, label=model_type, color=c)
#             legenditems_count += 1
#             ## MAE_norm
#             ax[1].bar(xtickpos[i]+xtickoffset, model_metrics_df_filt['mae_norm'+suffix_list[0]], width=w, label=model_type, color=c)
#             if annotate_vals:
#                 ax[0].annotate(round(model_metrics_df_filt['r2'+suffix_list[0]],2), (xtickpos[i]+xtickoffset, model_metrics_df_filt['r2'+suffix_list[0]]+0.015))
#                 ax[1].annotate(round(model_metrics_df_filt['mae_norm'+suffix_list[0]],2), (xtickpos[i]+xtickoffset, model_metrics_df_filt['mae_norm'+suffix_list[0]]+0.008))
#             if plot_errors: 
#                 ## R2
#                 ax[0].errorbar(xtickpos[i]+xtickoffset, model_metrics_df_filt['r2'+suffix_list[0]], yerr=model_metrics_df_filt['r2'+suffix_list[0].replace('avg','std')], color='k', fmt='o', markersize=8, capsize=10, label=None)
#                 ## MAE_norm
#                 ax[1].errorbar(xtickpos[i]+xtickoffset, model_metrics_df_filt['mae_norm'+suffix_list[0]], yerr=model_metrics_df_filt['mae_norm'+suffix_list[0].replace('avg','std')], color='k', fmt='o', markersize=8, capsize=10, label=None)
    
#     for k in range(nrows): 
#         ax[k].set_xticks(xtickpos+xtickoffset/2, yvar_list, fontsize=20)
        
#     # set ylim for MAE plots
#     ymax_mae = np.max(model_metrics_df['mae_norm'+suffix_list[0]].to_numpy())
#     ax[0].set_ylim([0,1])
#     ax[1].set_ylim([0,ymax_mae*1.5])
#     ax[0].set_ylabel('R2', fontsize=20)
#     ax[1].set_ylabel('MAE, normalized', fontsize=20)
#     ymax = ax.flatten()[0].get_position().ymax
#     legend = [f'{model_type}_{train_or_cv}' for model_type in models_to_eval_list for train_or_cv in ['cv']]
#     plt.legend([v for k,v in legenditems_dict.items()], legend, fontsize=16)
#     if figtitle is not None:
#         plt.suptitle(figtitle, y=ymax*1.03, fontsize=24)    
#     if savefig is not None:
#         fig.savefig(savefig, bbox_inches='tight')
#     plt.show()


def plot_model_metrics_all(model_metrics_df_dict, models_to_eval_list, yvar_list, nrows=2, ncols=1, figsize=(30,15), barwidth=0.8, figtitle=None, savefig=None, suffix_list=['_train','_cv'],  model_cmap=model_cmap, annotate_vals=False, alpha_dict=None):
    
    fig, ax = plt.subplots(nrows,ncols, figsize=figsize)
    xtickpos = np.arange(len(yvar_list))*2+1
    for i, yvar in enumerate(yvar_list):
        for j in [0,1]:
            model_metrics_df = model_metrics_df_dict[j]
            for k, model_type in enumerate(models_to_eval_list):
                c = model_cmap[model_type]
                w = barwidth / (len(models_to_eval_list)*len(suffix_list))
                xtickoffset = -barwidth/2 + w/2 + k*w*len(suffix_list)
                model_metrics_df_filt = model_metrics_df[(model_metrics_df.yvar==yvar) & (model_metrics_df.model_type==model_type)].iloc[0].to_dict()
                ## R2
                for l, suffix in enumerate(suffix_list):
                    if suffix.find('train')>-1:
                        alpha=0.5
                    else: 
                        alpha=1
                    if alpha_dict is not None: 
                        alpha = alpha_dict[j]
                    if alpha==0.5: 
                        hatch='/'
                    else: 
                        hatch=None
                    ax[0].bar(xtickpos[i]+j+xtickoffset+w*l, model_metrics_df_filt['r2'+suffix], width=w, label=model_type, color=c, alpha=alpha, hatch=hatch)
                    ax[1].bar(xtickpos[i]+j+xtickoffset+w*l, model_metrics_df_filt['mae_norm'+suffix], width=w, label=model_type, color=c, alpha=alpha, hatch=hatch)
                    if annotate_vals:
                        ax[0].annotate(round(model_metrics_df_filt['r2'+suffix],2), (xtickpos[i]+j+xtickoffset+w*l, model_metrics_df_filt['r2'+suffix]))
                        ax[1].annotate(round(model_metrics_df_filt['mae_norm'+suffix],2), (xtickpos[i]+j+xtickoffset+w*l, model_metrics_df_filt['mae_norm'+suffix]))
    
    for row in range(nrows): 
        xtickpos = []
        xticklabels = []
        for i, yvar in enumerate(yvar_list):
            xtickpos += [2*i+1, 2*i+2]
            xticklabels += [f'{yvar}'+'\n'+'ALL FEATURES', f'{yvar}'+'\n'+'SELECTED FEATURES']
        ax[row].set_xticks(xtickpos, xticklabels, fontsize=16)
        
    # set ylim for MAE plots
    ax[0].set_ylim([0.2,1])
    ax[1].set_ylim([0,0.25])
    ax[0].set_ylabel('R2', fontsize=16)
    ax[1].set_ylabel('MAE, normalized (train)', fontsize=16)
    ymax = ax.flatten()[0].get_position().ymax
    # legend = [f'{model_type}{train_or_cv.replace("_avg","")}' for model_type in models_to_eval_list for train_or_cv in suffix_list]
    legend = [f'{model_type} ({all_or_selected})' for all_or_selected in ['all features', 'selected features'] for model_type in models_to_eval_list ]
    plt.legend(legend, fontsize=16)
    if figtitle is not None:
        plt.suptitle(figtitle, y=ymax*1.03, fontsize=24)    
    if savefig is not None:
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()
    
    
def plot_scatter_train_test_predictions(Y, ypred_train_bymodel, ypred_test_bymodel, yvar_list, models_to_eval_list, model_cmap=model_cmap, savefig=None):

    # iterate through yvar in sublist
    fig, ax = plt.subplots(len(models_to_eval_list),len(yvar_list), figsize=(6.75*len(yvar_list),6*len(models_to_eval_list)))
    for i, yvar in enumerate(yvar_list): 
        y = Y[:,i]
        for j, model_type in enumerate(models_to_eval_list): 
            print(model_type)
            c = model_cmap[model_type]
            ypred_train = ypred_train_bymodel[model_type][:,i]
            ypred_test = ypred_test_bymodel[model_type][:,i]
            r2_train = round(pearsonr(y, ypred_train)[0]**2,2)
            r2_test = round(pearsonr(y, ypred_test)[0]**2,2)
            
            if len(models_to_eval_list)>1 and len(yvar_list)>1:
                ax[j][i].scatter(y, ypred_train, c='k', alpha=0.5, marker='*')
                ax[j][i].scatter(y, ypred_test, c=c, alpha=1, marker='o', s=16)
                ax[j][i].set_title(f'{model_type} <> {yvar}', fontsize=16)
                ax[j][i].set_ylabel(f'{yvar} (predicted)', fontsize=12)
                ax[j][i].set_xlabel(f'{yvar} (actual)', fontsize=12)          
                (xmin, xmax) = ax[j][i].get_xlim()
                (ymin, ymax) = ax[j][i].get_ylim()
                ax[j][i].text(xmin+(xmax-xmin)*0.05, ymin+(ymax-ymin)*0.85, f'R2 (train): {r2_train} \nR2 (test) {r2_test}', fontsize=14)
                ax[j][i].legend(['train', 'test'], loc='lower right')
            elif len(models_to_eval_list)==1 and len(yvar_list)>1: 
                ax[i].scatter(y, ypred_train, c='k', alpha=0.5, marker='*')
                ax[i].scatter(y, ypred_test, c=c, alpha=1, marker='o', s=16)
                ax[i].set_title(f'{model_type} <> {yvar}', fontsize=16)
                ax[i].set_ylabel(f'{yvar} (predicted)', fontsize=12)
                ax[i].set_xlabel(f'{yvar} (actual)', fontsize=12)          
                (xmin, xmax) = ax[i].get_xlim()
                (ymin, ymax) = ax[i].get_ylim()
                ax[i].text(xmin+(xmax-xmin)*0.05, ymin+(ymax-ymin)*0.85, f'R2 (train): {r2_train} \nR2 (test) {r2_test}', fontsize=14)
                ax[i].legend(['train', 'test'], loc='lower right')
            elif len(models_to_eval_list)>1 and len(yvar_list)==1: 
                ax[j].scatter(y, ypred_train, c='k', alpha=0.5, marker='*')
                ax[j].scatter(y, ypred_test, c=c, alpha=1, marker='o', s=16)
                ax[j].set_title(f'{model_type} <> {yvar}', fontsize=16)
                ax[j].set_ylabel(f'{yvar} (predicted)', fontsize=12)
                ax[j].set_xlabel(f'{yvar} (actual)', fontsize=12)          
                (xmin, xmax) = ax[j].get_xlim()
                (ymin, ymax) = ax[j].get_ylim()
                ax[j].text(xmin+(xmax-xmin)*0.05, ymin+(ymax-ymin)*0.85, f'R2 (train): {r2_train} \nR2 (test) {r2_test}', fontsize=14)
                ax[j].legend(['train', 'test'], loc='lower right')
            else: 
                ax.scatter(y, ypred_train, c='k', alpha=0.5, marker='*')
                ax.scatter(y, ypred_test, c=c, alpha=1, marker='o', s=16)
                ax.set_title(f'{model_type} <> {yvar}', fontsize=16)
                ax.set_ylabel(f'{yvar} (predicted)', fontsize=12)
                ax.set_xlabel(f'{yvar} (actual)', fontsize=12)          
                (xmin, xmax) = ax.get_xlim()
                (ymin, ymax) = ax.get_ylim()
                ax.text(xmin+(xmax-xmin)*0.05, ymin+(ymax-ymin)*0.85, f'R2 (train): {r2_train} \nR2 (test) {r2_test}', fontsize=14)
                ax.legend(['train', 'test'], loc='lower right')
    
    if len(models_to_eval_list)>1 or len(yvar_list)>1:
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'Model Predicted vs. Actual values for various Y variables', y=ymax*1.08, fontsize=24)  
    if savefig is not None: 
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()

    
    
def select_subset_of_X(X, xvar_list, xvar_list_ordered, f=1):
    p = X.shape[1]
    num_features_to_select = int(np.ceil(f*p))
    xvar_sublist_ordered = xvar_list_ordered[:num_features_to_select]
    xvar_idx_selected = [idx for idx, xvar in enumerate(xvar_list) if xvar in xvar_sublist_ordered]
    xvar_sublist = [xvar_list[idx] for idx in xvar_idx_selected]
    X_selected = X[:, xvar_idx_selected]
    return X_selected, xvar_sublist