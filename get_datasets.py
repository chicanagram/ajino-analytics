#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:36:39 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import pickle
from variables import data_folder, figure_folder, var_dict_all, overall_glyco_cqas, sort_list,  xvar_sublist_sets, yvar_list_key
from sklearn.preprocessing import scale

def get_XYdataset(d, X_featureset_idx, Y_featureset_idx, xvar_list_dict, yvar_list_dict, csv_fname=None, pkl_fname=None, data_folder=data_folder, shuffle_data=True, remove_cols_w_nan_thres=0.1):
    
    # get X Y variable names
    xvar_list_prefilt = xvar_list_dict[X_featureset_idx]
    yvar_list = yvar_list_dict[Y_featureset_idx]
    
    # get XY data - drop rows which contain any NA values      
    xy_var_list = ['exp_label', 'Basal medium', 'Feed medium'] + [var for var in xvar_list_prefilt+yvar_list if var in d]
    XY_df = d[xy_var_list]
    print('Initial dataframe shape:', XY_df.shape)
    
    # remove columns with above a threshold (e.g 0.1) of NaNs
    cols_to_delete = XY_df.columns[XY_df.isnull().sum()/len(XY_df) > remove_cols_w_nan_thres]
    XY_df.drop(cols_to_delete, axis=1, inplace=True)
    print(f'After removing columns with >{remove_cols_w_nan_thres*100}% of NaN data, dataframe shape:', XY_df.shape)
    
    # remove rows with nan data
    XY_df = XY_df.dropna()
    print('After removing rows with NaN data, dataframe shape:', XY_df.shape)
    
    # remove variables with only zeros
    XY_df = XY_df.loc[:,(XY_df!= 0).any(axis=0)]
    xvar_list = [xvar for xvar in xvar_list_prefilt if xvar in XY_df]
    print('After variables with only zero values, dataframe shape:', XY_df.shape)
    print('X variables:', xvar_list)
    print('Y variables:', yvar_list)
    
    # get X & Y numpy arrays
    X = XY_df[xvar_list].to_numpy()
    Y = XY_df[yvar_list].to_numpy()
    
    # randomize samples (shuffle datasets)
    if shuffle_data:
        shuffle_idx = np.arange(len(Y))
        np.random.seed(seed=0)
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx,:]
        Y = Y[shuffle_idx,:]
    
    # scale X dataset
    Xscaled = scale(X)
    print(f'N={len(Y)}, p={X.shape[1]}')
    
    # create dict of all datasets with various feature set combinations
    XYarr_dict = {
        'Y':Y, 'X': X, 'Xscaled':Xscaled, 'yvar_list': yvar_list, 'xvar_list': xvar_list
    }

    # save csv dataframe
    if csv_fname is not None:
        XY_df.to_csv(f'{data_folder}{csv_fname}')
        
    # save numpy datasets
    if pkl_fname is not None:
        with open(f'{data_folder}{pkl_fname}', 'wb') as f:
            pickle.dump(XYarr_dict, f)
            
    return XYarr_dict, XY_df

def get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder):
    with open(f'{data_folder}X{X_featureset_idx}Y{Y_featureset_idx}.pkl', 'rb') as f:
        XYarr_dict = pickle.load(f)
    Y = XYarr_dict['Y']       
    X = XYarr_dict['X']
    Xscaled = XYarr_dict['Xscaled']
    yvar_list = XYarr_dict['yvar_list'] 
    xvar_list = XYarr_dict['xvar_list']
    return Y, X, Xscaled, yvar_list, xvar_list


# open pickle file for main dataset
dataset_name = 'DATA_avg'
pkl_fname = f'{dataset_name}.pkl'
with open(f'{data_folder}{pkl_fname}', 'rb') as handle:
    datadict = pickle.load(handle)
    
# open csv file for media composition
mediacomposition_fname = 'MediaComposition_avg.csv'
media_df = pd.read_csv(f'{data_folder}{mediacomposition_fname}', index_col=0)

#%% GET FULL DATASET    
# initialize dataframe
cols = [
        ['exp_label'] + var_dict_all['inputs'],
        [f'{var}_basal' for var in var_dict_all['media_components']],
        [f'{var}_feed' for var in var_dict_all['media_components']],
        [f'{var}_{day}' for var in var_dict_all['VCD, VIA, Titer, metabolites'] for day in range(15)],
        [f'{var}_{day}' for var in var_dict_all['Glyco'] for day in range(15)],
        [f'{var}_{day}' for var in var_dict_all['AA'] for day in range(15)],
        [f'{var}_{day}' for var in var_dict_all['VT'] for day in range(15)],
        [f'{var}_{day}' for var in var_dict_all['Nuc, Amine'] for day in range(15)],
        [f'{var}_{day}' for var in var_dict_all['MT'] for day in range(15)],
        [f'{var}_{day}' for var in overall_glyco_cqas for day in range(15)]
        ]
cols = sum(cols, [])
n = len(list(datadict.keys()))
arr = np.zeros((n, len(cols)))
arr[:] = None
d = pd.DataFrame(arr, columns=cols)

# update dataframe contents from main dataset
for i, exp_label in enumerate(list(datadict.keys())): 
    print(i, end=' ')
    data_exp = datadict[exp_label]
    d.loc[i, 'exp_label'] = exp_label
    for vargrp in ['inputs', 'VCD, VIA, Titer, metabolites', 'Glyco', 'AA', 'VT', 'Nuc, Amine', 'MT']: 
        for var in var_dict_all[vargrp]: 
            # update dataframe value
            if var in data_exp:
                val = data_exp[var]
                if isinstance(val, dict): 
                    tlist = val['t']
                    ylist = val['y']
                    for t, y in zip(tlist, ylist): 
                        var_day_colname = f'{var}_{int(t)}'
                        d.loc[i, var_day_colname] = y
                else: 
                    if i==0 and isinstance(val, str):
                        d[var] = d[var].astype(str)
                    d.loc[i, var] = val
  
# add columns for Mannosylation, Fucosylation, Galactosylation
for var, glyco_list in overall_glyco_cqas.items():
    for day in range(15):
        var_colname = f'{var}_{day}'
        glyco_colnames_list = [f'{xvar}_{day}' for xvar in glyco_list]
        d[var_colname] = d[glyco_colnames_list].sum(axis=1)


# add columns for Basal and Feed media composition
for i in range(len(d)):
    basal = d.loc[i,'exp_label']
    basal = d.loc[i,'Basal medium']
    basal_composition = media_df.loc[basal, :].to_dict()
    for k, v in basal_composition.items():
        d.loc[i,f'{k}_basal'] = v
    feed = d.loc[i,'Feed medium']
    feed_composition = media_df.loc[feed, :].to_dict()
    for k, v in feed_composition.items():
        d.loc[i,f'{k}_feed'] = v

# remove columns that are all zeros
d = d.loc[:,(d!= 0).any(axis=0)]
# remove columns that are all NaNs
d = d.dropna(axis=1, how='all')

# save dataframe 
csv_fpath = f'{data_folder}{dataset_name}.csv'
d.to_csv(csv_fpath)

#%% GET X Y VARIABLE SETS 

# segmenting the variables
media_inputs = ['Basal medium', 'Feed medium']
media_composition_inputs = sort_list([f'{var}_basal' for var in var_dict_all['media_components']]) + sort_list([f'{var}_feed' for var in var_dict_all['media_components']])
process_inputs = ['DO', 'pH', 'feed %'] # ['DO', 'pH', 'feed %', 'feed vol'] # ['DO', 'pH', 'feed vol'] # 
nutrient_inputs = ['Glucose (g/L)'] + var_dict_all['AA'] + var_dict_all['VT'] + var_dict_all['MT'] + var_dict_all['Nuc, Amine']
cqa_outputs = ['Titer (mg/L)', 'mannosylation', 'fucosylation', 'galactosylation','G0','G0F','G1','G1Fa','G1Fb','G2','G2F','Other']
# generate day-by-day variable names for all time-series data
cqa_dict_byday = {day: [f'{cqa}_{day}' for cqa in cqa_outputs] for day in range(15)}
nutrient_dict_byday = {day: [f'{nutrient}_{day}' for nutrient in nutrient_inputs] for day in range(15)}

# initialize dicts for x and y variable lists
yvar_list_dict = {}
xvar_list_dict = {}

# Y
# Y0: D14 CQAs
yvar_list = [var for var in cqa_dict_byday[14]] 
yvar_list_dict.update({0: yvar_list}) 
print(len(yvar_list), yvar_list)

# X
## X0: DAY 0 nutrient inputs + process inputs
xvar_list_prefilt = sort_list([var for var in nutrient_dict_byday[0]]) + process_inputs
xvar_list_dict.update({0: xvar_list_prefilt})
print(len(xvar_list_prefilt), xvar_list_prefilt)

## X1: Basal and feed media composition inputs
xvar_list_prefilt = media_composition_inputs + process_inputs
xvar_list_dict.update({1: xvar_list_prefilt})
print(len(xvar_list_prefilt), xvar_list_prefilt)


#%% Get X and Y datasets 
for Y_featureset_idx in [0]:
    for X_featureset_idx in [0,1]:
        print('X_featureset_idx:', X_featureset_idx, ';  Y_featureset_idx:', Y_featureset_idx)
        XY_fname = f'X{X_featureset_idx}Y{Y_featureset_idx}'
        XYarr_dict, XY_df = get_XYdataset(d, X_featureset_idx, Y_featureset_idx, xvar_list_dict, yvar_list_dict, csv_fname=f'{XY_fname}.csv', pkl_fname=f'{XY_fname}.pkl', shuffle_data=True, remove_cols_w_nan_thres=0.1)
        
#%% Visualize cross correlation heatmaps

def get_xy_correlation_matrix(X_featureset_idx, Y_featureset_idx): 
    import matplotlib.pyplot as plt
    from plot_utils import heatmap
    
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
    print(f'X dataset size: n={Xscaled.shape[0]}, p={Xscaled.shape[1]}')
    
    # concatenate X and Y (key) variables
    XYarr = np.concatenate((X,Y[:,:4]), axis=1)
    XYarr = pd.DataFrame(XYarr, columns=xvar_list+yvar_list[:4])
    
    # calculate correlation matrix
    corr_mat = XYarr.corr()
    col_labels = corr_mat.columns.tolist()
    row_labels = list(corr_mat.index)
    corr_mat.round(3).to_csv(f'{data_folder}{dataset_name}_correlation_matrix.csv')
    
    # plot correlation matrix
    fig, ax = plt.subplots(1,1, figsize=(20,20))
    heatmap(corr_mat.to_numpy(), datamin=-1, datamax=1, logscale_cmap=False, annotate=None, row_labels=row_labels, col_labels=col_labels)
    fig.savefig(f'{figure_folder}{dataset_name}_correlation_matrix.png', bbox_inches='tight')
    
    return corr_mat

featureset_list = [(1,0)] # [(0,0), (1,0)]

# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    corr_mat = get_xy_correlation_matrix(X_featureset_idx, Y_featureset_idx)

#%% plot correlation heatmap for subset of features

import matplotlib.pyplot as plt
from plot_utils import heatmap
import seaborn as sns

# get data
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
print(f'X dataset size: n={Xscaled.shape[0]}, p={Xscaled.shape[1]}')

# get feature subset
subset_suffix = '_tier01'
xvar_subset_all = []
for yvar in yvar_list_key:
    xvar_subset_all += xvar_sublist_sets[yvar][0]
    xvar_subset_all += xvar_sublist_sets[yvar][1]
xvar_subset_all = sort_list(list(set(xvar_subset_all)))
print(len(xvar_subset_all), xvar_subset_all)

idx_selected = [idx for idx, xvar in enumerate(xvar_list) if xvar in xvar_subset_all]
X_selected = X[:,np.array(idx_selected)]

# concatenate X and Y (key) variables
XYarr = np.concatenate((X_selected,Y[:,:4]), axis=1)
XYarr = pd.DataFrame(XYarr, columns=xvar_subset_all+yvar_list[:4])

# calculate correlation matrix
corr_mat = XYarr.corr()
col_labels = corr_mat.columns.tolist()
row_labels = list(corr_mat.index)
corr_mat.round(3).to_csv(f'{data_folder}{dataset_name}_correlation_matrix.csv')

# plot correlation matrix
fig, ax = plt.subplots(1,1, figsize=(8,8))
heatmap(corr_mat.to_numpy(), datamin=-1, datamax=1, logscale_cmap=False, annotate=None, row_labels=row_labels, col_labels=col_labels)
fig.savefig(f'{figure_folder}{dataset_name}_correlation_matrix{subset_suffix}.png', bbox_inches='tight')


# get clustermap
cl = sns.clustermap(corr_mat, cmap="viridis", figsize=(12,12))
cl.fig.suptitle(f'Cluster map of selected features from {dataset_name}', fontsize=16) 
# plt.title(f'Cluster map of {figtitle}', fontsize=16, loc='center')
plt.savefig(f"{figure_folder}{dataset_name}_correlation_matrix{subset_suffix}_clustermap.png",  bbox_inches='tight') 