#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:43:27 2024

@author: charmainechia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:04:35 2024

@author: charmainechia
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from variables import data_folder, yvar_list_key, feature_selections, model_cmap
from model_utils import run_trainval_test
from utils import sort_list, get_XYdata_for_featureset
from plot_utils import convert_figidx_to_rowcolidx


class GetFeatureEffects:

    def __init__(self,
                 features_selected_dict,
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
                 models_to_evaluate=['randomforest', 'xgb'],
                 num_pts=10,
                 baseline_nutrient_concentrations=None
                 ):

        self.features_selected_dict = features_selected_dict
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
        self.baseline_nutrient_concentrations = baseline_nutrient_concentrations

    def compose_input(self):
        x_dict = {yvar: {} for yvar in self.yvar_list}
        for yvar in self.yvar_list:
            # get features used for model (yvar)
            idxs_features_selected = self.features_selected_idxs_dict[yvar]
            xmean_yvar = self.XMEAN[idxs_features_selected].reshape(1, -1)
            xstd_yvar = self.XSTD[idxs_features_selected].reshape(1, -1)
            for features, idxs in zip(self.features_opt, self.idxs_opt):
                # initialize array
                x_input = np.tile(x0.copy().reshape(1, -1), (self.num_pts, 1))
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
        res_allmodels = {yvar: {feature: np.zeros((self.num_pts, len(
            self.models_to_evaluate))) for feature in self.features_opt} for yvar in self.yvar_list}
        res = {yvar: {feature: np.zeros(
            (self.num_pts,)) for feature in self.features_opt} for yvar in self.yvar_list}

        for i, yvar in enumerate(self.yvar_list):
            print(yvar)
            for features in self.features_opt:
                x_input = x_dict[yvar][features]
                # # print inputs to check 
                # idxs = []
                # for feature in features:
                #     idxs.append(self.features_selected_dict[yvar].index(feature))
                # print(feature, x_input[:,np.array(idxs)])
                
                for j, model_type in enumerate(self.models_to_evaluate):
                    model = self.SURROGATE_MODELS[yvar][model_type]
                    ypred = model.predict(x_input)
                    res_allmodels[yvar][features][:, j] = ypred
                    print(f'ypred ({yvar} <> {features} <> {model_type}):', ypred)
                # average results, if needed
                ypred_avg = np.mean(res_allmodels[yvar][features], axis=1)
                res[yvar][features] = ypred_avg
                print(f'ypred ({yvar} <> {features} <> (AVG)):', ypred_avg, '\n')
        return res

    def plot_predicted_effects(self, res, nrows=5, ncols=6, ylim=[0.4, 1.3], show_xticks=False, figsize=None):
        cmap_yvar = {
            'Titer (mg/L)_14': 'b',
            'mannosylation_14': 'r',
            'fucosylation_14': 'g',
            'galactosylation_14': 'purple'
        }
        if figsize is None:
            figsize = (ncols*4, nrows*4)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        for figidx, features in enumerate(self.features_opt):
            row_idx, col_idx = convert_figidx_to_rowcolidx(figidx, ncols)
            if nrows > 1:
                ax_current = ax[row_idx, col_idx]
            else:
                ax_current = ax[col_idx]
            # plot predictions of feature effect on each yvar
            for yvar in self.yvar_list:
                x = np.arange(self.num_pts)
                y = res[yvar][features] / self.res0[yvar]
                ax_current.plot(x, y, color=cmap_yvar[yvar])
            ax_current.set_title(features)
            ax_current.set_ylim(ylim)

            # annotate baseline nutrient values
            if self.baseline_nutrient_concentrations is not None:
                for k, feature in enumerate(features):
                    baseline_nutrient_conc = self.baseline_nutrient_concentrations[feature]
                    xrange = self.bounds[feature][1] - self.bounds[feature][0]
                    x_annotate = (baseline_nutrient_conc -
                                  self.bounds[feature][0])/xrange*(self.num_pts-1)
                    yrange = ylim[1] - ylim[0]
                    ax_current.axvline(x_annotate, color='0.3', linestyle='--', linewidth=0.5)
                    if k == 0:
                        ax_current.annotate('', xy=(x_annotate, ylim[0]), xytext=(
                            x_annotate, ylim[0]+0.05*yrange), arrowprops=dict(arrowstyle="->"))
                    elif k == 1:
                        ax_current.annotate('', xy=(x_annotate, ylim[1]), xytext=(
                            x_annotate, ylim[1]-0.05*yrange), arrowprops=dict(arrowstyle="->"))

            # label x axis
            if show_xticks:
                for k, feature in enumerate(features):
                    xticklabels = [round(self.bounds[feature][0], 3), round(
                        self.bounds[feature][1], 3)]
                    if k == 0:
                        ax_current.set_xticks(
                            x[[0, -1]], xticklabels, fontsize=7.5)
                    elif k == 1:
                        ax2 = ax_current.twiny()
                        ax2.xaxis.set_ticks(
                            ax2.get_xticks()[[0, -1]], xticklabels, fontsize=7.5)
            else:
                ax_current.set_xticks([])

        # remove unused axes
        unused_figidxs = [idx for idx in range(
            nrows*ncols) if idx >= len(self.features_opt)]
        for idx in unused_figidxs:
            i, j = convert_figidx_to_rowcolidx(idx, ncols)
            if nrows > 1:
                ax[i][j].set_axis_off()
            else:
                ax[j].set_axis_off()
        return fig, ax


# %%
# get data
X_featureset_idx, Y_featureset_idx = 1, 0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
featureset_suffix =  '_curated2' # '_compactness-opt' # '_curated' #'_knowledge-opt' # '_model-opt'
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(
    X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
models_to_eval_list = ['randomforest', 'xgb']

yvar_list = yvar_list_key
features_selected_dict = feature_selections[featureset_suffix]

# get SURROGATE MODELS with selected feature set
kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS, ypred_train_bymodel, ypred_test_bymodel = run_trainval_test(
    X, Y, yvar_list, features_selected_dict, xvar_list_all, dataset_name_wsuffix, featureset_suffix, models_to_eval_list=models_to_eval_list, model_cmap=model_cmap)

# %% Get base conditions

dataset_name_wsuffix = 'X6Y0'

# ['best'] #  # ['best1', 'best2', 'best3'] #
starting_point_list = ['avg-Basal-A-Feed-a'] # , 'best1', 'best3']


# get data and scaling constants
df = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
X = df.loc[:, xvar_list_all]
XMEAN = X.mean(axis=0).to_numpy()
XSTD = X.std(axis=0).to_numpy()
df_good = df[(df['Titer (mg/L)_14'] > 9000) & (df['mannosylation_14'] < 12)].sort_values(by=['Titer (mg/L)_14'], ascending=False)


starting_point_dict = {
    # Basal-A, Feed-a, pH7, DO40, feed 6%
    'avg-Basal-A-Feed-a': {
        'x0': df[(df['Basal medium'] == 'Basal-A') & (df['Feed medium'] == 'Feed-a') & (df['DO'] == 40) & (df['pH'] == 7) & (df['feed %'] == 6)].loc[:, xvar_list_all].mean(axis=0),
        'y0': df[(df['Basal medium'] == 'Basal-A') & (df['Feed medium'] == 'Feed-a') & (df['DO'] == 40) & (df['pH'] == 7) & (df['feed %'] == 6)].loc[:, yvar_list_key].mean(axis=0)
    },
    # Basal-D, Feed-a, pH7, DO40, feed 6%
    'avg-Basal-D-Feed-a': {
        'x0': df[(df['Basal medium'] == 'Basal-D') & (df['Feed medium'] == 'Feed-a')].loc[:, xvar_list_all].mean(axis=0),
        'y0': df[(df['Basal medium'] == 'Basal-D') & (df['Feed medium'] == 'Feed-a') & (df['DO'] == 40) & (df['pH'] == 7) & (df['feed %'] == 6)].loc[:, yvar_list_key].mean(axis=0)
    },
    # Basal-A, Feed-a, pH7.1, DO60, feed 8%
    'best1': {
        'x0': df_good.iloc[:1, :].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[:1, :].loc[:, yvar_list_key].mean(axis=0)
    },
    # Basal-A, Feed-a, pH7.0, DO20, feed 6%
    'best2': {
        'x0': df_good.iloc[2:3, :].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[2:3, :].loc[:, yvar_list_key].mean(axis=0)
    },
    # Basal-A, Feed-a, pH7.0, DO20, feed 8%
    'best3': {
        'x0': df_good.iloc[3:, :].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[3:, :].loc[:, yvar_list_key].mean(axis=0)
    }
}

with open(f'{data_folder}optimization_starting_points.pkl', 'wb') as f:
    pickle.dump(starting_point_dict, f)
    


# get individual and pair (basal/feed) feature effects
res0_dict = {}
num_pts = 12
ylim = [0.3, 1.4]
for starting_point in starting_point_list:
    print('Starting point:', starting_point)

    # get best conditions in dataset
    baseline_nutrient_concentrations = starting_point_dict[starting_point]['x0'].to_dict(
    )
    x0 = starting_point_dict[starting_point]['x0'].to_numpy().reshape(-1)
    y0 = starting_point_dict[starting_point]['y0'].to_numpy()
    features_selected_idxs_dict = {}
    for yvar in yvar_list_key:
        features_selected = features_selected_dict[yvar]
        features_selected_idxs_dict[yvar] = [xvar_list_all.index(xvar) for xvar in features_selected]
    print('features_selected_idxs_dict:', features_selected_idxs_dict)

    # set feature indices to optimize
    process_features = ['DO', 'pH', 'feed vol']
    features_selected_all_ = []
    for yvar, features_selected_yvar in features_selected_dict.items():
        features_selected_yvar_ = [
            feature for feature in features_selected_yvar if feature not in process_features]
        features_selected_all_ += features_selected_yvar_
    features_selected_all = sort_list(list(
        set(features_selected_all_))) + process_features  # sort and deduplicate list
    features_opt = features_selected_all.copy()
    idxs_opt = np.array([xvar_list_all.index(xvar) for xvar in features_opt])
    print('features_opt:', features_opt)
    print('idxs_opt:', idxs_opt)

    # get bounds for optimization of each feature
    bounds_dict = {}
    for feature, feature_idx_to_opt in zip(features_opt, idxs_opt):
        xvar = xvar_list_all[feature_idx_to_opt]
        lbnd = df.loc[:, xvar].min()
        ubnd = df.loc[:, xvar].max()
        bounds_dict[feature] = (lbnd, ubnd)

    # get average Y (CQA) values for starting condition
    res0 = {yvar: val for yvar, val in zip(yvar_list_key, y0)}
    res0_dict[starting_point] = res0
    res0_ref = res0_dict['avg-Basal-A-Feed-a']
    print('Actual Starting CQA values:', res0)

    # get individual feature effects
    features_opt_INDIVIDUAL = [(f,) for f in features_opt]
    idxs_opt_INDIVIDUAL = [(i,) for i in idxs_opt]
    FeatureEffects = GetFeatureEffects(features_selected_dict, features_selected_idxs_dict, features_opt_INDIVIDUAL, idxs_opt_INDIVIDUAL, bounds_dict,
                                       XMEAN, XSTD, SURROGATE_MODELS, res0_ref, x0, yvar_list_key, ['randomforest', 'xgb'], num_pts=num_pts, baseline_nutrient_concentrations=baseline_nutrient_concentrations)
    x_dict = FeatureEffects.compose_input()
    res_indiv = FeatureEffects.predict_CQAs(x_dict)
    if len(features_opt)<10:
        fig, ax = FeatureEffects.plot_predicted_effects(res_indiv, nrows=1, ncols=len(features_opt), ylim=ylim, show_xticks=True)
        fig.suptitle(starting_point, y=1.0, fontsize=16)
    else:
        ncols = 8
        nrows = int(np.ceil(len(features_opt)/ncols))
        fig, ax = FeatureEffects.plot_predicted_effects(res_indiv, nrows=nrows, ncols=ncols, ylim=ylim, show_xticks=True)
        fig.suptitle(starting_point, y=0.92, fontsize=16)

    plt.show()
    print()
    
    # save results as pickle
    with open(f'{data_folder}feature_effects_indiv_{dataset_name_wsuffix}.pkl', 'wb') as f:
        pickle.dump(res_indiv, f)

    # get combined basal/feed feature effects
    feature_combi_dict = {}
    feature_idxs_combi_dict = {}
    for feature, idx in zip(features_opt, idxs_opt):
        feature_base = feature.replace('_basal', '').replace('_feed', '')
        if feature_base not in feature_combi_dict:
            feature_combi_dict[feature_base] = [feature]
            feature_idxs_combi_dict[feature_base] = [
                xvar_list_all.index(feature)]
        else:
            feature_combi_dict[feature_base].append(feature)
            feature_idxs_combi_dict[feature_base].append(
                xvar_list_all.index(feature))
    features_opt_PAIR = [
        tuple(feature_set) for feature_base, feature_set in feature_combi_dict.items()]
    idxs_opt_PAIR = [tuple(feature_idxs_set) for feature_base,
                      feature_idxs_set in feature_idxs_combi_dict.items()]
    print(features_opt_PAIR)
    print(idxs_opt_PAIR)

    FeatureEffects = GetFeatureEffects(features_selected_dict, features_selected_idxs_dict, features_opt_PAIR, idxs_opt_PAIR, bounds_dict,
                                        XMEAN, XSTD, SURROGATE_MODELS, res0_ref, x0, yvar_list_key, ['randomforest', 'xgb'], num_pts=num_pts, baseline_nutrient_concentrations=baseline_nutrient_concentrations)
    x_dict = FeatureEffects.compose_input()
    res_pair = FeatureEffects.predict_CQAs(x_dict)
    if len(features_opt_PAIR)<10:
        fig, ax = FeatureEffects.plot_predicted_effects(res_pair, nrows=1, ncols=len(features_opt_PAIR), ylim=ylim, show_xticks=True, figsize=(24, 3.5))
        fig.suptitle(starting_point, y=1.0, fontsize=16)
    else:
        ncols = 6
        nrows = int(np.ceil(len(features_opt_PAIR)/ncols))
        fig, ax = FeatureEffects.plot_predicted_effects(res_pair, nrows=nrows, ncols=ncols, ylim=ylim, show_xticks=True, figsize=(24,15))
        fig.suptitle(starting_point, y=0.95, fontsize=16)
    plt.show()
    print()
    
    with open(f'{data_folder}feature_effects_pair_{dataset_name_wsuffix}.pkl', 'wb') as f:
        pickle.dump(res_pair, f)



#%% plot scatter plot of experimental data + extreme range predictions

# get predicted extreme data
starting_point = 'avg-Basal-A-Feed-a'
baseline_nutrient_concentrations = starting_point_dict[starting_point]['x0'].to_dict()
x0 = starting_point_dict[starting_point]['x0'].to_numpy().reshape(-1)
y0 = starting_point_dict[starting_point]['y0'].to_numpy()

features_selected_idxs_dict = {}
for yvar in yvar_list_key:
    features_selected = features_selected_dict[yvar]
    features_selected_idxs_dict[yvar] = [
        xvar_list_all.index(xvar) for xvar in features_selected]
print('features_selected_idxs_dict:', features_selected_idxs_dict)

# set feature indices to optimize
features_opt = features_selected_dict['Titer (mg/L)_14']
idxs_opt = np.array([xvar_list_all.index(xvar) for xvar in features_opt])
x0_opt = x0[idxs_opt].reshape((1,-1))
x0_opt_scaled = ((x0-XMEAN)/XSTD)[idxs_opt]
for k, feature in enumerate(features_opt): 
    print(k, feature)

##############
# get inputs #
############## 
# get bounds for optimization of each featurem and input for model
bounds_dict = {}
x_dict = {yvar:{(feature,):np.tile(x0_opt.copy(),(3,1)) for feature in features_opt} for yvar in yvar_list_key}
x_dict_scaled = {yvar:{(feature,):np.tile(x0_opt_scaled.copy(),(3,1)) for feature in features_opt} for yvar in yvar_list_key}
# iterate through features
for k, (feature, feature_idx_to_opt) in enumerate(zip(features_opt, idxs_opt)):
    print(k, feature)
    # get bounds
    xvar = xvar_list_all[feature_idx_to_opt]
    lbnd = df.loc[:, xvar].min()
    ubnd = df.loc[:, xvar].max()
    bounds_dict[feature] = (lbnd, ubnd)
    # update inputs for prediction
    for yvar in yvar_list_key:
        x_dict[yvar][(feature,)][0,k] = lbnd
        x_dict_scaled[yvar][(feature,)][0,k] = (lbnd-XMEAN[feature_idx_to_opt])/XSTD[feature_idx_to_opt]
        x_dict[yvar][(feature,)][2,k] = ubnd
        x_dict_scaled[yvar][(feature,)][2,k] = (ubnd-XMEAN[feature_idx_to_opt])/XSTD[feature_idx_to_opt]

#########################
# get model predictions #
#########################
num_pts = 3
models_to_evaluate = ['randomforest', 'xgb']
res_allmodels = {}
res = {}
for i, yvar in enumerate(yvar_list_key):
    res_allmodels[yvar] = {}
    res[yvar] = {}
    print(yvar)
    for k, feature in enumerate(features_opt):
        x_input = x_dict[yvar][(feature,)]
        x_input_scaled = x_dict_scaled[yvar][(feature,)]
        print(k, feature)
        print('x:', x_input[:,k])
        print('xscaled:', x_input_scaled[:,k])
        res_allmodels[yvar][(feature,)] = np.zeros((num_pts, len(models_to_evaluate)))
        for j, model_type in enumerate(models_to_evaluate):
            model = SURROGATE_MODELS[yvar][model_type]
            ypred = model.predict(x_input_scaled)
            res_allmodels[yvar][(feature,)][:, j] = ypred
            # print(f'ypred ({model_type}):', ypred)
        # average results, if needed
        res[yvar][(feature,)] = np.mean(res_allmodels[yvar][(feature,)], axis=1)
        print(f'{yvar} ypred (AVG):', res[yvar][(feature,)])

# aggregate results
res_df_cols = ['base', 'feature', 'feature_type', 'yvar', 'x_min', 'x_baseline', 'x_max', 'y_baseline', 'ypred_min', 'ypre_baseline', 'ypred_max']
res_df = pd.DataFrame(columns=res_df_cols, index=range(len(features_opt)*len(yvar_list_key)))
row_count = 0
for i, yvar in enumerate(yvar_list_key):
    y_baseline = y0[i]
    for k, feature in enumerate(features_opt):
        x = x_dict[yvar][(feature,)][:,k]
        ypred = res[yvar][(feature,)]
        if feature.find('_basal')>-1:
            feature_type='basal'
        elif feature.find('_feed')>-1:
            feature_type='feed'
        else:
            feature_type='process'
        res_df.iloc[row_count] = [feature.replace('_basal', '').replace('_feed', ''), feature, feature_type, yvar, x[0], x[1], x[2], y_baseline, ypred[0], ypred[1], ypred[2]]
        row_count += 1

res_df['dy% (ubd)'] = (res_df['ypred_max'] - res_df['y_baseline']) / res_df['y_baseline'] * 100
res_df['dy% (lbd)'] = (res_df['ypred_min'] - res_df['y_baseline']) / res_df['y_baseline'] * 100
res_df = res_df.sort_values(by=['feature'])
res_df_basal = res_df[res_df.feature_type=='basal']
res_df_feed = res_df[res_df.feature_type!='basal']
res_df_summary = res_df_basal.merge(res_df_feed, how='outer', on=['base', 'yvar'], suffixes=['_basal', '_feed'])
res_df_summary = res_df_summary[['base','yvar', 'feature_type_feed'] + [f'{f}_{feature_type}' for feature_type in ['basal', 'feed'] for f in ['x_min', 'x_baseline', 'x_max', 'dy% (lbd)', 'dy% (ubd)']]]
res_df_summary = res_df_summary.sort_values(by=['yvar', 'feature_type_feed', 'base'])

# save results
res_df.to_csv(f'{data_folder}predicted_effect_of_varying_indiv_features.csv')
res_df_summary.to_csv(f'{data_folder}predicted_effect_of_varying_indiv_features_summary.csv')

#%%

#####################
# PLOT SCATTER PLOT #
#####################

features_to_plot_sorted = sort_list(features_opt[:-3]) + ['DO', 'pH', 'feed vol']
feed_percentage_list = list(set(df['feed %'].tolist()))
for i, yvar in enumerate(yvar_list_key):
    # initialize figure
    if len(features_opt)<10:
        nrows = 1
        ncols = len(features_opt)
        fig, ax = plt.subplots(nrows, ncols, figsize=(4*len(features_opt),4.2))
    else:
        ncols = 8
        nrows = int(np.ceil(len(features_opt)/ncols))
        fig, ax = plt.subplots(nrows, ncols, figsize=(4*ncols,4.2*nrows))
    for k, feature in enumerate(features_opt):
        # get position in sorted features list 
        fig_idx = features_to_plot_sorted.index(feature)
        row_idx, col_idx = convert_figidx_to_rowcolidx(fig_idx, ncols)
        df_filt = df.copy()
        x = df_filt.loc[:, feature]
        y = df_filt.loc[:, yvar]       
        if nrows==1:
            ax_k = ax[col_idx]
        else:
            ax_k = ax[row_idx, col_idx]
        # plot all experimental values
        ax_k.scatter(x,y, s=20, alpha=0.5)
        
        # highlight experimental baseline values
        df_baseline = df[(df['Basal medium'] == 'Basal-A') & (df['Feed medium'] == 'Feed-a') & (df['DO'] == 40) & (df['pH'] == 7) & (df['feed %'] == 6)]
        ax_k.scatter(df_baseline.loc[:, feature],df_baseline.loc[:, yvar], c='orange', s=30, alpha=1)  

        # # plot baseline  AVERAGE y value
        ax_k.scatter(starting_point_dict[starting_point]['x0'][feature], starting_point_dict[starting_point]['y0'][yvar], marker='*', c='r', s=90, alpha=0.9)
        
        # plot PREDICTED lbnd, ubnd y values
        ax_k.scatter(x_dict[yvar][(feature,)][:,k], res[yvar][(feature,)], marker='^', c='k', s=80, alpha=0.7)
        ax_k.plot(x_dict[yvar][(feature,)][:,k], res[yvar][(feature,)], c='k', linestyle='--', alpha=0.3)
        print(k, feature)
        print('x:', x_dict[yvar][(feature,)][:,k])
        print('xscaled:', x_dict_scaled[yvar][(feature,)][:,k])
        print(f'{yvar} ypred:', res[yvar][(feature,)])
            
        # set xlabel
        ax_k.set_title(feature, fontsize=16)
        
    # remove unused axes
    unused_figidxs = [idx for idx in range(nrows*ncols) if idx >= len(features_opt)]
    for idx in unused_figidxs:
        i, j = convert_figidx_to_rowcolidx(idx, ncols)
        if nrows > 1:
            ax[i][j].set_axis_off()
        else:
            ax[j].set_axis_off()
    
    # set legend
    ax_k.legend(['expt', 'expt - baseline (all)', 'expt - baseline (AVG)', 'model predictions' ], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    # ax_k.legend(feed_percentage_list + ['baseline (exp)', 'predicted'], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    # set title
    plt.suptitle(yvar, y=0.92, fontsize=30)
    # save and show plot
    plt.show()