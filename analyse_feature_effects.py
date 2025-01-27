#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:57:18 2024

@author: charmainechia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from variables import data_folder, yvar_list_key
from utils import get_XYdata_for_featureset
from plot_utils import figure_folder, convert_figidx_to_rowcolidx, heatmap
from model_utils import get_corr_coef

#%% 
# input params & dataset
X_featureset_idx, Y_featureset_idx =  6, 0
dataset_suffix = ''
yvar_list = yvar_list_key
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, Xscaled, _, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
X_df = pd.DataFrame(X, columns=xvar_list)

# get yvar ranges
yvar_ranges = {}
for i, yvar in enumerate(yvar_list_key):
    yvar_ranges[yvar] = Y[:,i].max()-Y[:,i].min()
    
# get starting points
with open(f'{data_folder}optimization_starting_points.pkl', 'rb') as f:
    starting_point_dict = pickle.load(f)

# initialize feature effects summary table
feature_effects_summary_cols = ['lbnd', 'ubnd', 'baseline', 'optval_titer_vs_baseline', 'optval_man5_vs_baseline', 'sim_vs_expBaseline_mean', 'sim_vs_expBaseline_median'] + [f'{meth}_{yvar}' for yvar in yvar_list_key for meth in ['SHAP-randomforest_pearsonr', 'SHAP-randomforest_spearmanr', 'SHAP-xgb_pearsonr', 'SHAP-xgb_spearmanr', 'slope_indiv_variation_mean']] 
feature_effects_summary = {feature:{col:np.nan for col in feature_effects_summary_cols} for feature in xvar_list}

# get feature bounds, baselines
for feature in xvar_list:
    feature_effects_summary[feature]['lbnd'] = X_df.loc[:, feature].min()
    feature_effects_summary[feature]['ubnd'] = X_df.loc[:, feature].max()
    feature_effects_summary[feature]['baseline'] = starting_point_dict['avg-Basal-A-Feed-a']['x0'][feature]

# load SHAP RF & XGB effects
plot_shap_corr = True
shap_values_dict = {}
shap_meanabs_dict = {}
for model_type in ['randomforest', 'xgb']: 
    shap_values_dict[model_type] = {}
    shap_meanabs_dict[model_type] = {}
    shap_summary = pd.read_csv(f'{data_folder}feature_analysis_SHAPsummary_{dataset_name}{dataset_suffix}_{model_type}.csv')
    
    for i, yvar in enumerate(yvar_list_key):
        df = pd.read_csv(f'{data_folder}feature_analysis_SHAPvalues_{dataset_name}{dataset_suffix}_{model_type}_{i}.csv')
        shap_values_dict[model_type][yvar] = df
        
        if plot_shap_corr:
            nrows, ncols = 5, 8
            fig, ax = plt.subplots(nrows, ncols, figsize=(32,22))
        for feature_idx, feature in enumerate(xvar_list):  
            # shap summary
            shap_summary_yvar = shap_summary.loc[shap_summary.yvar==yvar, xvar_list].iloc[0]
            shap_meanabs_dict[model_type][yvar] = (shap_summary_yvar/shap_summary_yvar.sum()).to_dict()
            
            # shap values
            row_idx, col_idx = convert_figidx_to_rowcolidx(feature_idx, ncols)
            feature_values = X[:, feature_idx]
            shap_values = df.loc[:, feature].to_numpy()
            corr_shap = get_corr_coef(feature_values, shap_values, corr_to_get=['spearmanr', 'pearsonr'])
            if plot_shap_corr:
                ax[row_idx, col_idx].scatter(feature_values, shap_values, s=2)
                ax[row_idx, col_idx].set_xlabel('feature value')
                ax[row_idx, col_idx].set_ylabel('SHAP value')
                ax[row_idx, col_idx].set_title(feature)
                ax[row_idx, col_idx].annotate(f'spearmanr:{corr_shap["spearmanr"]} \n pearsonr:{corr_shap["pearsonr"]}', xy=(0.1, 0.8), xycoords='axes fraction')
            
            # update feature effects summary
            feature_effects_summary[feature][f'SHAP-{model_type}_spearmanr_{yvar}'] = corr_shap["spearmanr"]
            feature_effects_summary[feature][f'SHAP-{model_type}_pearsonr_{yvar}'] = corr_shap["pearsonr"]
        if plot_shap_corr:
            plt.suptitle(f'{yvar} <> {model_type}', y=0.92)
            plt.show()

# load single-feature effects
num_pts = 12
with open(f'{data_folder}feature_effects_indiv_{dataset_name_wsuffix}.pkl', 'rb') as f:
    res_indiv = pickle.load(f)

for feature_idx, feature in enumerate(xvar_list):  
    x = np.linspace(feature_effects_summary[feature]['lbnd'], feature_effects_summary[feature]['ubnd'], num=num_pts, endpoint=True)
    exp_data_feature_range = feature_effects_summary[feature]['ubnd'] - feature_effects_summary[feature]['lbnd']
    exp_data_feature_baseline = feature_effects_summary[feature]['baseline']
    for i, yvar in enumerate(yvar_list_key):
        print(yvar, feature)
        res_indiv_featureyvar = res_indiv[yvar][(feature,)]
        print(res_indiv_featureyvar)
        grad_res_indiv_featureyvar = np.diff(res_indiv_featureyvar)
        # get overall gradient (direction of variation)
        feature_effects_summary[feature][f'slope_indiv_variation_mean_{yvar}'] = np.mean(grad_res_indiv_featureyvar)*num_pts/yvar_ranges[yvar]
        # get optimal values
        if yvar=='Titer (mg/L)_14':
            idx_opt = np.argmax(res_indiv_featureyvar)
            x_opt = x[idx_opt]
            print('Opt feature val:', x_opt, f'(idx={idx_opt})')
            feature_effects_summary[feature]['optval_titer_vs_baseline'] = (x_opt - exp_data_feature_baseline)/exp_data_feature_baseline  # exp_data_feature_range
        elif yvar=='mannosylation_14':
            idx_opt = np.argmin(res_indiv_featureyvar)
            x_opt = x[idx_opt]
            print('Opt feature val:', x_opt, f'(idx={idx_opt})')
            feature_effects_summary[feature]['optval_man5_vs_baseline'] = (x_opt - exp_data_feature_baseline)/exp_data_feature_baseline # exp_data_feature_range


# load simulation violin plot effects 
titer_thres = (9000,9000)
man5_thres = (12,9)
# get experiment data
exp_data = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
exp_data = exp_data.loc[(exp_data['Basal medium']=='Basal-A') & (exp_data['Feed medium']=='Feed-a'), yvar_list_key+xvar_list]
# get simulation data
sim1 = pd.read_csv(f'{data_folder}optimizeCQAs_fs=curated2_wts=3-1-0-0.2-0.2_ncalls=1000_annealing_best1_0.csv').drop_duplicates(subset=['obj_fn']+yvar_list_key)
sim2 = pd.read_csv(f'{data_folder}optimizeCQAs_fs=curated2_wts=3-1-0-0.2-0.2_ncalls=1000_annealing_best3_0.csv').drop_duplicates(subset=['obj_fn']+yvar_list_key)
sim_data = pd.concat([sim1, sim2]).drop_duplicates(subset=['obj_fn']+yvar_list_key)
# filter data using selected thresholds
exp_data_filt = exp_data[(exp_data['Titer (mg/L)_14']>titer_thres[0]) & (exp_data['mannosylation_14']<man5_thres[0])].mean(axis=0)
sim_data_filt = sim_data[(sim_data['Titer (mg/L)_14']>titer_thres[1]) & (sim_data['mannosylation_14']<man5_thres[1])]
# iterate through features and calculate difference in distribution
for feature in xvar_list:
    exp_data_feature_range = feature_effects_summary[feature]['ubnd'] - feature_effects_summary[feature]['lbnd']
    exp_data_feature_baseline = feature_effects_summary[feature]['baseline']
    exp_data_feature_best = exp_data_filt[feature]
    sim_data_feature = sim_data_filt.loc[:, feature]
    sim_data_feature_mean = sim_data_feature.mean()
    sim_data_feature_median = sim_data_feature.median()
    feature_effects_summary[feature]['sim_vs_expBaseline_mean'] = (sim_data_feature_mean - exp_data_feature_baseline)/exp_data_feature_baseline # exp_data_feature_range
    feature_effects_summary[feature]['sim_vs_expBaseline_median'] = (sim_data_feature_median - exp_data_feature_baseline)/exp_data_feature_baseline # exp_data_feature_range


# create table
feature_effects_summary_df = []
index_list = []
for feature, feature_effects_dict in feature_effects_summary.items():
    feature_effects_summary_df.append(feature_effects_dict)
    lbnd, ubnd, baseline = feature_effects_dict['lbnd'], feature_effects_dict['ubnd'], feature_effects_dict['baseline']
    index_list.append((feature, round(lbnd,2), round(ubnd,2), round(baseline,2)))
feature_effects_summary_df = pd.DataFrame.from_dict(feature_effects_summary_df)
feature_effects_summary_df.index = xvar_list
feature_effects_summary_df.to_csv(f'{data_folder}feature_effects_summary_{dataset_name_wsuffix}.csv')


#%% get heatmap
feature_effects_summary_df_filt = feature_effects_summary_df.loc[:, [f for f in feature_effects_summary_cols  if f not in ['lbnd', 'ubnd', 'baseline']]]
feature_effects_summary_df_filt.index = index_list
feature_effects_summary_arr = feature_effects_summary_df_filt.to_numpy()
feature_effects_summary_arr_binarized = ((feature_effects_summary_arr>0)*1).astype(float)
nan_idxs = np.where(np.isnan(feature_effects_summary_df_filt))
feature_effects_summary_arr_binarized[nan_idxs[0], nan_idxs[1]] = np.nan

ax_features = {}
ax_features[0] = [f for f in list(feature_effects_summary_df_filt.index) if f[0].find('_basal')>-1]
ax_features[1] = [f for f in list(feature_effects_summary_df_filt.index) if f[0].find('_basal')==-1]

for k in ax_features:
    fig, ax = plt.subplots(1,1, figsize=(20,16))
    if k==0:
        feature_effects_summary_arr_binarized_AX = feature_effects_summary_arr_binarized[:len(ax_features[0]),:]
        feature_effects_summary_arr_AX = feature_effects_summary_arr[:len(ax_features[0]),:]
    else: 
        feature_effects_summary_arr_binarized_AX = feature_effects_summary_arr_binarized[len(ax_features[0]):,:]
        feature_effects_summary_arr_AX = feature_effects_summary_arr[len(ax_features[0]):,:]
    # plot heatmap
    heatmap(feature_effects_summary_arr_binarized_AX, ax=ax, show_gridlines=False, c='cool', col_labels=feature_effects_summary_df_filt.columns.tolist(), row_labels=ax_features[k], labeltop=True, rotation=45, show_colorbar=False, fontsize=10)
    # annotate
    for row_idx in range(feature_effects_summary_arr_AX.shape[0]):
        for col_idx in range(feature_effects_summary_arr_AX.shape[1]):
            val = round(feature_effects_summary_arr_AX[row_idx, col_idx],2)
            if ~np.isnan(val):
                ax.text(x=col_idx-0.35, y=row_idx, s=val, fontsize=10)
    # draw vlines
    for vline in [0, 1, 3, 8, 13, 18]:
        ax.axvline(x=vline+0.5, linestyle='--', color='k', linewidth=2)
    fig.savefig(f'{figure_folder}feature_effects_{dataset_name_wsuffix}_{k}.png', bbox_inches='tight', dpi=300)

#%% get stacked horizontal bar plot of mean abs SHAP values
cmap_yvar = {
    'Titer (mg/L)_14': 'b',
    'mannosylation_14': 'r',
    'fucosylation_14': 'g',
    'galactosylation_14': 'purple'
}


starts = np.zeros((len(xvar_list)))
fig, ax = plt.subplots(1,1, figsize=(10,16))
for yvar in yvar_list_key: 
    vals_avg = np.zeros((2, len(xvar_list)))
    for k, model_type in enumerate(['randomforest', 'xgb']):
        vals = np.array([shap_meanabs_dict[model_type][yvar][feature] for feature in xvar_list])
        vals_avg[k, :] = vals
    vals_avg = np.mean(vals_avg, axis=0)
    ax.barh(xvar_list, vals_avg, left=starts, color=cmap_yvar[yvar], alpha=0.5)
    starts += vals_avg
ax.invert_yaxis()
plt.legend(yvar_list_key)
plt.show()






