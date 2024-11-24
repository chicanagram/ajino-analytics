#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:04:35 2024

@author: charmainechia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variables import data_folder, yvar_list_key, feature_selections, model_params, model_cmap
from model_utils import run_trainval_test
from utils import sort_list, get_XYdata_for_featureset
from plot_utils import convert_figidx_to_rowcolidx, get_unused_figidx

class GetFeatureEffects:
        
    def __init__(self, 
                features_selected_idxs_dict,
                features_opt,
                idxs_opt, 
                bounds,
                XMEAN,
                XSTD,
                SURROGATE_MODELS,
                res0,
                x0,
                yvar_list=yvar_list_key,
                models_to_evaluate=['randomforest','xgb'],
                num_pts=10
                ):
        
        self.features_selected_idxs_dict = features_selected_idxs_dict
        self.features_opt = features_opt
        self.idxs_opt = idxs_opt      
        self.SURROGATE_MODELS = SURROGATE_MODELS
        self.bounds = bounds
        self.XMEAN = XMEAN
        self.XSTD = XSTD
        self.res0 = res0
        self.x0 = x0
        self.yvar_list = yvar_list_key
        self.models_to_evaluate = models_to_evaluate
        self.num_pts = num_pts
        
    def compose_input(self):
        x_dict = {yvar:{} for yvar in self.yvar_list}
        for yvar in self.yvar_list:
            # get features used for model (yvar)
            idxs_features_selected = self.features_selected_idxs_dict[yvar]
            xmean_yvar = self.XMEAN[idxs_features_selected].reshape(1,-1)
            xstd_yvar = self.XSTD[idxs_features_selected].reshape(1,-1)
            for features, idxs in zip(self.features_opt, self.idxs_opt):
                # initialize array
                x_input = np.tile(x0.copy().reshape(1,-1), (self.num_pts,1))
                for feature, idx in zip(features, idxs):
                    # modify selected feature to optimize
                    x_input[:, idx] = np.linspace(self.bounds[feature][0], self.bounds[feature][1], num=self.num_pts, endpoint=True)
                # filter by features selected for yvar model
                x_input = x_input[:, idxs_features_selected]
                # scale x input
                x_input = (x_input - xmean_yvar)/xstd_yvar
                # update x_dict
                x_dict[yvar][features] = x_input
        self.x_dict = x_dict
        return x_dict
        
        
    def predict_CQAs(self, x_dict=None): 
        if x_dict is None:
            x_dict = self.x_dict
        res_allmodels = {yvar:{feature:np.zeros((self.num_pts,len(self.models_to_evaluate))) for feature in self.features_opt} for yvar in self.yvar_list}
        res = {yvar:{feature:np.zeros((self.num_pts,)) for feature in self.features_opt} for yvar in self.yvar_list}
        
        for i, yvar in enumerate(self.yvar_list):
            for feature in self.features_opt:
                x_input =  x_dict[yvar][feature]
                for j, model_type in enumerate(self.models_to_evaluate):
                    model = self.SURROGATE_MODELS[yvar][model_type]
                    ypred = model.predict(x_input)
                    res_allmodels[yvar][feature][:,j] = ypred
                # average results, if needed
                res[yvar][feature] = np.mean(res_allmodels[yvar][feature], axis=1)
        return res
    
    def plot_predicted_effects(self, res, nrows=5, ncols=6, ylim=[0.4,1.3], show_xticks=False, figsize=None):
        cmap_yvar = {
            'Titer (mg/L)_14': 'b',
            'mannosylation_14': 'r',
            'fucosylation_14': 'g',
            'galactosylation_14': 'purple'
            }
        if figsize is None:
            figsize=(ncols*4, nrows*4)
        fig, ax = plt.subplots(nrows,ncols, figsize=figsize) 
        for figidx, features in enumerate(self.features_opt):
            for yvar in self.yvar_list:
                row_idx, col_idx = convert_figidx_to_rowcolidx(figidx, ncols)
                x = np.arange(self.num_pts)
                y = res[yvar][features] / self.res0[yvar]
                ax[row_idx, col_idx].plot(x,y, color=cmap_yvar[yvar])
                
            ax[row_idx, col_idx].set_title(features)
            ax[row_idx, col_idx].set_ylim(ylim)
            
            # label x axis
            if show_xticks:
                for k, feature in enumerate(features):
                    xticklabels = [round(self.bounds[feature][0],3), round(self.bounds[feature][1],3)]
                    if k==0: 
                        ax[row_idx, col_idx].set_xticks(x[[0,-1]], xticklabels, fontsize=7.5)
                    elif k==1:
                        ax2 = ax[row_idx, col_idx].twiny()
                        ax2.set_xticks(x[[0,-1]], xticklabels, fontsize=7.5)
            else: 
                ax[row_idx, col_idx].set_xticks([])
            
        # remove unused axes
        unused_figidxs = [idx for idx in range(nrows*ncols) if idx>len(self.features_opt)]
        for idx in unused_figidxs:
            i, j = convert_figidx_to_rowcolidx(idx, ncols)
            ax[i][j].set_axis_off()
        return fig, ax

#%% 
# get data
X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
featureset_suffix = '_curated' #  '_compactness-opt' # '_knowledge-opt' # '_model-opt'
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
models_to_eval_list = ['randomforest', 'xgb']

yvar_list = yvar_list_key 
features_selected_dict = feature_selections[featureset_suffix]

# get SURROGATE MODELS with selected feature set
kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS = run_trainval_test(X, Y, yvar_list, features_selected_dict, xvar_list_all, dataset_name_wsuffix, featureset_suffix, models_to_eval_list=models_to_eval_list, model_cmap=model_cmap)

#%% Get base conditions

starting_point_list = ['avg-Basal-A-Feed-a', 'best1', 'best3'] # ['best'] #  # ['best1', 'best2', 'best3'] # 


# get data and scaling constants
df = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
X = df.loc[:, xvar_list_all]
XMEAN = X.mean(axis=0).to_numpy()
XSTD = X.std(axis=0).to_numpy()
df_good = df[(df['Titer (mg/L)_14']>9000) & (df['mannosylation_14']<12)].sort_values(by=['Titer (mg/L)_14'], ascending=False)


starting_point_dict = {
    # Basal-A, Feed-a, pH7, DO40, feed 6%
    'avg-Basal-A-Feed-a': { 
        'x0': df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a')].loc[:, xvar_list_all].mean(axis=0),
        'y0': df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a') & (df['DO']==40) & (df['pH']==7) & (df['feed %']==6)].loc[:, yvar_list_key].mean(axis=0)
        },
    # Basal-D, Feed-a, pH7, DO40, feed 6%
    'avg-Basal-D-Feed-a': { 
        'x0': df[(df['Basal medium']=='Basal-D') & (df['Feed medium']=='Feed-a')].loc[:, xvar_list_all].mean(axis=0),
        'y0': df[(df['Basal medium']=='Basal-D') & (df['Feed medium']=='Feed-a') & (df['DO']==40) & (df['pH']==7) & (df['feed %']==6)].loc[:, yvar_list_key].mean(axis=0)
        },
    # Basal-A, Feed-a, pH7.1, DO60, feed 8%
    'best1': {
        'x0': df_good.iloc[:1,:].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[:1,:].loc[:, yvar_list_key].mean(axis=0)
        },    
    # Basal-A, Feed-a, pH7.0, DO20, feed 6%
    'best2': {
        'x0': df_good.iloc[2:3,:].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[2:3,:].loc[:, yvar_list_key].mean(axis=0)
        },
    # Basal-A, Feed-a, pH7.0, DO20, feed 8%
    'best3': {
        'x0': df_good.iloc[3:,:].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[3:,:].loc[:, yvar_list_key].mean(axis=0)
        }           
    }

#%% 
res0_dict = {}
for starting_point in starting_point_list: 
    print('Starting point:', starting_point)
    
    # get best conditions in dataset
    x0 = starting_point_dict[starting_point]['x0'].to_numpy().reshape(-1)
    y0 = starting_point_dict[starting_point]['y0'].to_numpy()
    features_selected_idxs_dict = {}
    for yvar in yvar_list_key:    
        features_selected = features_selected_dict[yvar]
        features_selected_idxs_dict[yvar] = [xvar_list_all.index(xvar) for xvar in features_selected]
    print('features_selected_idxs_dict:', features_selected_idxs_dict)
    
    # set feature indices to optimize
    process_features = ['pH', 'DO', 'feed vol']
    features_selected_all_ = []
    for yvar, features_selected_yvar in features_selected_dict.items():
        features_selected_yvar_ = [feature for feature in features_selected_yvar if feature not in process_features]
        features_selected_all_ += features_selected_yvar_
    features_selected_all = sort_list(list(set(features_selected_all_))) + process_features # sort and deduplicate list
    features_opt = features_selected_all.copy()
    idxs_opt = np.array([xvar_list_all.index(xvar) for xvar in features_opt])
    print('features_opt:', features_opt)
    print('idxs_opt:', idxs_opt)
    
    # get bounds for optimization of each feature
    bounds_dict = {}
    for feature, feature_idx_to_opt in zip(features_opt, idxs_opt):
        xvar = xvar_list_all[feature_idx_to_opt]
        lbnd = df.loc[:,xvar].min()
        ubnd = df.loc[:,xvar].max()
        bounds_dict[feature] = (lbnd,ubnd)
    
    # get average Y (CQA) values for starting condition
    res0 = {yvar:val for yvar, val in zip(yvar_list_key, y0)}
    res0_dict[starting_point] = res0
    res0_ref = res0_dict['avg-Basal-A-Feed-a']
    print('Actual Starting CQA values:', res0)
    
    # get individual feature effects
    features_opt_INDIVIDUAL = [(f,) for f in features_opt]
    idxs_opt_INDIVIDUAL = [(i,) for i in idxs_opt]
    FeatureEffects = GetFeatureEffects(features_selected_idxs_dict, features_opt_INDIVIDUAL, idxs_opt_INDIVIDUAL, bounds_dict, 
                                        XMEAN, XSTD, SURROGATE_MODELS, res0_ref, x0, yvar_list_key, ['randomforest','xgb'], num_pts=10)
    x_dict = FeatureEffects.compose_input()
    res = FeatureEffects.predict_CQAs(x_dict)
    fig, ax = FeatureEffects.plot_predicted_effects(res, nrows=5, ncols=6, ylim=[0.5,1.25], show_xticks=True)
    fig.suptitle(starting_point, y=0.92, fontsize=16)
    plt.show()
    print()
    
    # get combined basal/feed feature effects
    feature_combi_dict = {}
    feature_idxs_combi_dict = {}
    for feature, idx in zip(features_opt,idxs_opt): 
        feature_base = feature.replace('_basal', '').replace('_feed', '')
        if feature_base not in feature_combi_dict:
            feature_combi_dict[feature_base] = [feature]
            feature_idxs_combi_dict[feature_base] = [xvar_list_all.index(feature)]
        else: 
            feature_combi_dict[feature_base].append(feature)
            feature_idxs_combi_dict[feature_base].append(xvar_list_all.index(feature))
    features_opt_PAIR = [tuple(feature_set) for feature_base, feature_set in feature_combi_dict.items()]
    idxs_opt_PAIR = [tuple(feature_idxs_set) for feature_base, feature_idxs_set in feature_idxs_combi_dict.items()]
    print(features_opt_PAIR)
    print(idxs_opt_PAIR)
    
    FeatureEffects = GetFeatureEffects(features_selected_idxs_dict, features_opt_PAIR, idxs_opt_PAIR, bounds_dict, 
                                        XMEAN, XSTD, SURROGATE_MODELS, res0_ref, x0, yvar_list_key, ['randomforest','xgb'], num_pts=10)
    x_dict = FeatureEffects.compose_input()
    res = FeatureEffects.predict_CQAs(x_dict)
    fig, ax = FeatureEffects.plot_predicted_effects(res, nrows=3, ncols=6, ylim=[0.4,1.3], show_xticks=True, figsize=(24,15))
    fig.suptitle(starting_point, y=0.95, fontsize=16)
    plt.show()
    print()
    
    
    
    
    