#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:49:45 2024

@author: charmainechia
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale, minmax_scale
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import figure_folder, heatmap, convert_figidx_to_rowcolidx


data_folder = '../ajino-analytics-data/'



def get_corr_coef(x, y, corr_to_get=['spearmanr', 'pearsonr']):
    res = {}
    for corrtype in corr_to_get: 
        if corrtype=='spearmanr':
            res['spearmanr'] = round(spearmanr(x, y)[0],3)
        if corrtype=='pearsonr': 
            res['pearsonr'] = round(pearsonr(x, y)[0],3)
    return res


def get_score(y, ypred, scoring):
    if scoring=='mae':
        score = mean_absolute_error(y, ypred)
    elif scoring=='rmse':
        score = root_mean_squared_error(y, ypred)
    return score


def adjusted_loocv_with_scoring(X, y, model_dict, scoring='mae'):
    n = len(y)
    ymean = np.mean(y)
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
        # fit training data
        regr.fit(Xtrain, ytrain)
        # predict on test data
        ypred_loocv[test_idx] = regr.predict(Xtest)        
    # calculate score for ypred_loocv 
    score = round(get_score(y, ypred_loocv, scoring),3)
    score_norm = round(score/ymean, 3)
    model_dict.update({
        f'{scoring}_cv':score, 
        f'{scoring}_norm_cv': score_norm,
        'ypred_cv': ypred_loocv
        })
    return model_dict

def fit_model_with_cv(X,y, yvar, model_list, plot_predictions=False, scoring='mae'):
    
    cv = KFold(n_splits=5)
    cv.get_n_splits(X)
    ymean = np.mean(y)
    
    # get model
    for i, model_dict in enumerate(model_list):
        model_type = model_dict['model_type']
        if model_type=='plsr':
            n_components =  model_dict['n_components']
            model = PLSRegression(n_components=n_components)
        elif model_type=='pca':
            model = make_pipeline(StandardScaler(), PCA(n_components=4), LinearRegression())
        elif model_type=='lasso':
            max_iter = model_dict['max_iter']
            alpha = model_dict['alpha']
            model = Lasso(max_iter=max_iter, alpha=alpha)
        elif model_type=='randomforest':
            n_estimators = model_dict['n_estimators']
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
        elif model_type=='svr':
            model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        elif model_type=='mlp':
            model = MLPRegressor(hidden_layer_sizes=(50, 20, 5), activation='relu', solver='adam', alpha=0.001, random_state=1, max_iter=100)
        model_dict.update({'model':model})
        
        # perform cross-validation
        model_dict = adjusted_loocv_with_scoring(X,y, model_dict, scoring=scoring)
        cv_score = model_dict[f'{scoring}_cv'] 
        cv_score_norm = model_dict[f'{scoring}_norm_cv'] 
        ypred_cv = model_dict['ypred_cv']
        
        # fit all data and get R2 score
        model.fit(X, y)
        ypred = model.predict(X)
        r2 = round(r2_score(y, ypred),2)
        fitted_score = round(get_score(y, ypred, scoring),3)
        fitted_score_norm = round(fitted_score/ymean,3)
        model_list[i].update({'r2': r2, 'fitted_score': fitted_score, 'cv_score': cv_score, 'model':model, 'ypred':ypred})
        print(f'[{yvar} <> {model_type}], R2:{r2}, fitted {scoring}:{fitted_score} ({round(fitted_score_norm*100,1)}%), {scoring} (CV):{cv_score} ({round(cv_score_norm*100,1)}%), ')
        
    # calculate ensemble scores, if applicable
    ypred_ensemble_cv = np.zeros((len(y), len(model_list)))
    ypred_ensemble = np.zeros((len(y), len(model_list)))
    w = 1/len(model_list)
    for i, model_dict in enumerate(model_list):
        if 'w' in model_dict:
            w = model_dict['w']
        ypred_ensemble_cv[:,i] = model_dict['ypred_cv']*w
        ypred_ensemble[:,i] = model_dict['ypred']*w
    if len(model_list)>1:
        ypred_ensemble_cv = np.sum(ypred_ensemble_cv, axis=1)
        cv_score_ensemble = round(get_score(y, ypred_ensemble_cv, scoring),3)
        cv_score_ensemble_norm = round(cv_score_ensemble/ymean,3)
        ypred_ensemble = np.sum(ypred_ensemble, axis=1)
        fitted_score_ensemble = round(get_score(y, ypred_ensemble, scoring),3)
        fitted_score_ensemble_norm = round(fitted_score_ensemble/ymean,3)
        r2_ensemble = round(r2_score(y, ypred_ensemble),2)
        print(f'[{yvar} <> ENSEMBLE], R2:{r2_ensemble}, fitted {scoring}:{fitted_score_ensemble} ({round(fitted_score_ensemble_norm*100,1)}%), CV {scoring}:{cv_score_ensemble} ({round(cv_score_ensemble_norm*100,1)}%), ')
    else: 
        r2_ensemble = r2
        fitted_score_ensemble = fitted_score
        fitted_score_ensemble_norm = round(fitted_score_ensemble/ymean,3)
        cv_score_ensemble = cv_score
        cv_score_ensemble_norm = round(cv_score_ensemble/ymean,3)
        
    # update overall metrics
    metrics = {'model_type': model_type, 'yvar': yvar, 'r2': r2_ensemble, f'{scoring}_train':fitted_score_ensemble, f'{scoring}_norm_train': fitted_score_ensemble_norm,f'{scoring}_cv':cv_score_ensemble, f'{scoring}_norm_cv': cv_score_ensemble_norm}
    # plot predicted values
    if plot_predictions:
        plt.scatter(y, ypred_ensemble)
        plt.title(yvar)
        plt.ylabel('Predicted')
        plt.xlabel('Actual')
        plt.show()
    
    return model_list, metrics
        

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


def get_feature_importances(model_list, yvar, xvar_list, plot_feature_importances=True, plot_ordered=True):
    
    # initialize lists for recording feature coefficients and importance score
    feature_coefs = []
    feature_importances = []
    for model_dict in model_list:
        model_type = model_dict['model_type']
        model = model_dict['model']
        if model_type in ['plsr', 'lasso']: 
            coef_x_list = model.coef_.reshape(-1,)
        elif model_type in ['randomforest']:
            coef_x_list = model.feature_importances_
        
        coef_x_list_ABS = np.abs(coef_x_list)
        orders_x_list = get_order_of_element_sizes(coef_x_list_ABS)
        feature_coef_dict = {'model_type': model_type, 'yvar': yvar}
        feature_coef_dict.update({f'{xvar_list[k]}':coef_x_list[k] for k in range(len(xvar_list))})
        feature_coefs.append(feature_coef_dict)
        feature_importance_dict = {'model_type': model_type, 'yvar': yvar}
        feature_importance_dict.update({f'{xvar_list[k]}':orders_x_list[k] for k in range(len(xvar_list))})
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


def plot_feature_importance_heatmap(heatmap_df, xvar_list, yvar_list, logscale_cmap=False, scale_vals=False, figtitle=None, savefig=None):

    # get array data from feature dataframe
    arr = heatmap_df.to_numpy()
    
    # scale array values if needed
    if scale_vals:
        arr_unscaled = arr.copy()
        for row_idx in range(arr_unscaled.shape[0]):
            scalefactor = np.max(np.abs(arr_unscaled[row_idx,:]))
            arr[row_idx,:] = arr_unscaled[row_idx,:]/scalefactor
        # replace data in heatmap
        heatmap_df = pd.DataFrame(arr, columns=xvar_list, index=yvar_list)
        
    # plot heatmap 
    fig, ax = plt.subplots(1,1, figsize=(25, arr.shape[0]/arr.shape[1]*25))
    _, _, ax = heatmap(arr, c='viridis', ax=ax, cbar_kw={}, cbarlabel="", annotate=True, row_labels=yvar_list, col_labels=xvar_list, logscale_cmap=logscale_cmap)
    if figtitle is not None:
        ax.set_title(figtitle, fontsize=16)
    if savefig is not None:
        fig.savefig(f'{figure_folder}{savefig}.png',  bbox_inches='tight')     
        
    # plot clustermap of heatmap
    cl = sns.clustermap(heatmap_df, cmap="viridis", figsize=(20, 12))
    cl.fig.suptitle(f'Cluster map of {figtitle}', fontsize=16) 
    # plt.title(f'Cluster map of {figtitle}', fontsize=16, loc='center')
    plt.savefig(f"{figure_folder}{savefig}_clustermap.png",  bbox_inches='tight') 
    return arr




def plot_feature_importance_barplots(feature_importance_arr, yvar_list, xvar_list, label_xvar_by_indices=True, ncols=4, nrows=3, savefig=None, figtitle=None):

    """
    Plot bar chart comparing overall feature importance for all x variables for across y variables
    """
    fig, ax = plt.subplots(nrows,ncols, figsize=(40,16))
    xtickpos = np.arange(len(xvar_list))+1
    for i, yvar in enumerate(yvar_list):
        row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
        feature_importance_arr_yvar = feature_importance_arr[i, :]       
        ax[row_idx][col_idx].bar(xtickpos, feature_importance_arr_yvar)
        ax[row_idx][col_idx].set_xticks(xtickpos, [str(idx) for idx in range(len(xvar_list))], fontsize=8)
        ax[row_idx][col_idx].set_title(yvar, fontsize=20)
        ax[row_idx][col_idx].set_ylabel('feature importances', fontsize=16)
        
    if figtitle is not None: 
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'{figtitle}', y=ymax*1.06, fontsize=24)    
            
    if savefig is not None:
        fig.savefig(f'{figure_folder}{savefig}.png', bbox_inches='tight')
    plt.show()  
    
    

