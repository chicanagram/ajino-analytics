#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:25:11 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from variables import yvar_list_key, xvar_sublist_sets_bymodeltype
from plot_utils import figure_folder, heatmap
from get_datasets import data_folder, get_XYdata_for_featureset


# visualization parameters
X_featureset_idx, Y_featureset_idx = 1, 0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
subset_suffix = '_tier12'
model_fpath = f'{data_folder}models_and_params.pkl'
metrics_fpath = f'{data_folder}model_metrics_{dataset_name}{subset_suffix}.csv'
models_to_ensemble = ['randomforest', 'plsr', 'lasso']
mode = 'median'
var_x = 'Riboflavin_feed'
var_y = 'D-glucose_basal'
nx = 50
ny = 50
nrows_fig, ncols_fig = 2,2
figtitle = f'Effect of {var_x} & {var_y} on CQAs'
savefig = f'{figure_folder}visualization_CQAs_vs_{var_x}_{var_y}.png'

def visualize_effect_of_varying_xvars(
        var_x,
        var_y,
        nx = 50,
        ny = 50,
        X_featureset_idx = 1,
        Y_featureset_idx = 0,
        model_fpath = f'{data_folder}models_and_params.pkl',
        metrics_fpath = f'{data_folder}model_metrics_{dataset_name}{subset_suffix}.csv',
        models_to_ensemble = ['randomforest', 'plsr', 'lasso'],
        mode = 'median',
        nrows_fig = 2,
        ncols_fig = 2,
        figtitle = None,
        savefig = None
        ):
    """
    Perform inference using weighted model ensemble and visualize effect of varying x parameters
    """

    # import model
    with open(model_fpath, 'rb') as f:
        model_params = pickle.load(f)
    
    # load data
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
    Xscaled_df = pd.DataFrame(Xscaled, columns=xvar_list)
    X_df = pd.DataFrame(X, columns=xvar_list)
    # get scaling constants
    Xmean = np.mean(X, axis=0)
    Xstd = np.std(X, axis=0)
    
    # get weights for each model type
    model_metrics_df = pd.read_csv(metrics_fpath) 
        
    # initialize dict for storing inference results
    Y_inference = {}
    
    # initialize dict to store grid prediction parameters
    grid_params = {
        'x': {'var': var_x, 'nsteps': nx},
        'y': {'var': var_y, 'nsteps': ny},  
        }

    # compose input vect for each grid axis
    for axis in grid_params:
        var = grid_params[axis]['var']
        var_idx = xvar_list.index(var)
        ax_min = Xscaled_df[var].min()
        ax_max = Xscaled_df[var].max()
        ax_vect_scaled = np.linspace(ax_min, ax_max, num=grid_params[axis]['nsteps']+1)
        ax_vect = ax_vect_scaled * Xstd[var_idx] + Xmean[var_idx]
        grid_params[axis].update({
            'var_idx': var_idx, 'min':ax_min, 'max':ax_max, 
            'vect_scaled':ax_vect_scaled, 'vect': ax_vect
            })
    
    # compose visualization meshgrid and flatten to vector inputs for model prediction
    x_arr_scaled, y_arr_scaled = np.meshgrid(grid_params['x']['vect_scaled'], grid_params['y']['vect_scaled'])
    grid_params['x']['vect_flattened_scaled'] = x_arr_scaled.flatten()
    grid_params['y']['vect_flattened_scaled'] = y_arr_scaled.flatten()
    n = (nx+1)*(ny+1)
    
    # get input X dataset to perform inference on
    if mode=='median': 
        X_inference = np.tile(np.median(Xscaled, axis=0).reshape(1,-1), (n,1))
    # replace values for features to vary
    for axis in grid_params:
        var_idx = grid_params[axis]['var_idx']
        X_inference[:,var_idx] = grid_params[axis]['vect_flattened_scaled']
    
    # initialize figure for visualizing heatmap of inferences
    fig, ax = plt.subplots(nrows_fig, ncols_fig, figsize=(30,28))
        
    # iterate through yvar and perform inference
    for i, yvar in enumerate(yvar_list_key):  
        print(yvar)
        ypred_bymodel = {}
        weights_bymodel = {}
        
        # iterate through model types 
        for model_type in models_to_ensemble: 
            
            # get feature set
            if subset_suffix == '':
                xvar_list_selected = xvar_list
                X_inference_selected = Xscaled
                f = 1
            else:
                xvar_list_selected = xvar_sublist_sets_bymodeltype[yvar][model_type]
                idx_selected = [idx for idx, xvar in enumerate(xvar_list) if xvar in xvar_list_selected]
                X_inference_selected = X_inference[:,np.array(idx_selected)]
                f = round(len(xvar_list_selected)/len(xvar_list), 2)
                
            # get model 
            model = model_params[dataset_name+subset_suffix][model_type][yvar][0]['model']
            
            # perform inference
            ypred_bymodel[model_type] = model.predict(X_inference_selected)
            
            # get weight for model from R2 (CV) for model
            weights_bymodel[model_type] = model_metrics_df[(model_metrics_df.model_type==model_type) & (model_metrics_df.yvar==yvar) & (model_metrics_df.f==f)].iloc[0].r2_cv**2
            
        # ensemble model 
        ypred_ensemble = np.zeros((n,))
        weights_sum = 0
        for model_type in models_to_ensemble:
            ypred_ensemble += ypred_bymodel[model_type]*weights_bymodel[model_type]
            weights_sum += weights_bymodel[model_type]
        ypred_ensemble = ypred_ensemble/weights_sum
        ypred_ensemble_reshaped = ypred_ensemble.reshape((ny+1, nx+1))
        print(ypred_ensemble_reshaped.shape)
        
        # update Y_inference
        Y_inference[yvar] = ypred_ensemble_reshaped
        
        # plot heatmap
        idx_row, idx_col = np.unravel_index(i, (nrows_fig, ncols_fig))
        labels = {}
        for axis in grid_params:
            ax_vect = grid_params[axis]['vect']
            ax_labels = []
            for k, val in enumerate(ax_vect): 
                if k%2==0: 
                    if val > 1:
                        ax_labels.append(str(round(val,0)))
                    else:
                        ax_labels.append("%.3E" % (val))
                else: 
                    ax_labels.append('')
            labels[axis] = ax_labels
        heatmap(ypred_ensemble_reshaped, ax=ax[idx_row,idx_col], show_gridlines=False, row_labels=labels['y'], col_labels=labels['x'])
        ax[idx_row,idx_col].set_xlabel(grid_params['x']['var'], fontsize=16)
        ax[idx_row,idx_col].set_ylabel(grid_params['y']['var'], fontsize=16)
        ax[idx_row,idx_col].set_title(yvar, fontsize=22)
    
    # set figure title
    if figtitle is not None: 
        ymax = ax.flatten()[0].get_position().ymax
        if figtitle is not None:
            plt.suptitle(figtitle, y=ymax*1.04, fontsize=24)
            
    # save figure
    if savefig is not None: 
        plt.savefig(savefig, bbox_inches='tight')
    
    plt.show()
    
    return X_df, Xscaled_df, Y_inference
    
    
#%%
X_df, Xscaled_df, Y_inference = visualize_effect_of_varying_xvars(
                                                                    var_x,
                                                                    var_y,
                                                                    nx = 50,
                                                                    ny = 50,
                                                                    X_featureset_idx = 1,
                                                                    Y_featureset_idx = 0,
                                                                    model_fpath = f'{data_folder}models_and_params.pkl',
                                                                    metrics_fpath = f'{data_folder}model_metrics_{dataset_name}{subset_suffix}.csv',
                                                                    models_to_ensemble = ['randomforest', 'plsr', 'lasso'],
                                                                    mode = 'median',
                                                                    nrows_fig = 2,
                                                                    ncols_fig = 2,
                                                                    figtitle = figtitle,
                                                                    savefig = savefig
                                                                    )
    

    
    
    
    
    