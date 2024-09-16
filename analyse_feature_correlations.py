#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 21:25:14 2024

@author: charmainechia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:36:39 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from plot_utils import heatmap
import seaborn as sns
from variables import data_folder, figure_folder, var_dict_all, overall_glyco_cqas, sort_list,  xvar_sublist_sets, yvar_list_key
from get_datasets import get_XYdataset, get_XYdata_for_featureset
from sklearn.preprocessing import scale

 
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
    heatmap(corr_mat.to_numpy(), ax=ax, datamin=-1, datamax=1, logscale_cmap=False, annotate=None, row_labels=row_labels, col_labels=col_labels)
    ax.grid(None)
    fig.savefig(f'{figure_folder}{dataset_name}_correlation_matrix.png', bbox_inches='tight')
    
    return corr_mat

def get_corrmat_corrlist(df, sort_corrlist=True, csv_fname=None, savefig=None, plot_corrmat=True, plot_clustermap=False):
    
    # calculate correlation matrix
    corr_mat = df.corr()
    cols = corr_mat.columns.tolist()
    rows = list(corr_mat.index)
    n = len(cols)
    
    # save correlation matrix as csv
    if csv_fname is not None:
        corr_mat.round(3).to_csv(f'{csv_fname}.csv')
    
    # convert correlation matrix into list of pairs and corresponding correlations
    corr_all = {}
    # iterate through every element of corr_mat 
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            if row != col:
                pair = [row, col]
                pair.sort()
                pair = tuple(pair)
                if pair not in corr_all:  
                    corr_all.update({pair: float(corr_mat.loc[row, col])})
    
    # sort correlation data in descending order of correlations
    index = []
    corrs = []
    for k, v in corr_all.items():     
        index.append(k)
        corrs.append(v)          
    corr_all = pd.DataFrame(corrs, columns=['corr'], index=index) 
    corr_all['corr_abs'] = corr_all['corr'].abs()
    if sort_corrlist:
        corr_all = corr_all.sort_values(by='corr_abs', ascending=False)
        
    # plot correlation matrix
    if plot_corrmat:
        fig, ax = plt.subplots(1,1, figsize=(n/20*8,n/20*8))
        heatmap(corr_mat.to_numpy(), ax=ax, datamin=-1, datamax=1, logscale_cmap=False, annotate=None, row_labels=rows, col_labels=cols)
        ax.grid(None)
        if savefig is not None:
            fig.savefig(f'{savefig}.png', bbox_inches='tight')
    
    # get clustermap
    if plot_clustermap:
        cl = sns.clustermap(corr_mat, cmap="viridis", figsize=(12,12))
        cl.fig.suptitle(f'Cluster map of selected features from {dataset_name}', fontsize=16) 
        # plt.title(f'Cluster map of {figtitle}', fontsize=16, loc='center')
        if savefig is not None:
            plt.savefig(f"{savefig}_clustermap.png",  bbox_inches='tight') 
    
    return corr_mat, corr_all

       
def get_high_correlation_pairs(corr_all, corr_thres, print_out=True):
    # filter by corr_thres
    corr_selected = corr_all[corr_all.corr_abs > corr_thres]
    index_selected = list(corr_selected.index)
    # print list of pairs with high correlations
    if print_out:
        for i in range(len(corr_selected)):
            print(i, f"({index_selected[i][0]}, {index_selected[i][1]}):", round(corr_selected.iloc[i]['corr'],4))
    return corr_selected

def get_dict_of_features_with_highcorr(corr_mat, corr_thres):
    highcorr_vars_dict = {}
    rows = list(corr_mat.index)
    for i, xvar in enumerate(rows):
        corr_mat_row = corr_mat.loc[xvar]
        high_corr_vars = list(corr_mat_row[corr_mat_row.abs()>corr_thres].index)
        high_corr_vars.remove(xvar)
        highcorr_vars_dict[xvar] = high_corr_vars
        print(i, xvar, ':', *high_corr_vars)
    return highcorr_vars_dict

#%%

# get data
X_featureset_idx, Y_featureset_idx = 1, 0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
# dataset_suffix = '_avg'
dataset_suffix = ''
Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
X_df = pd.DataFrame(X, columns=xvar_list)

# get correlation matrix
corr_mat = get_xy_correlation_matrix(X_featureset_idx, Y_featureset_idx)

#%% plot correlation heatmap for subset of features

# get feature subset
subset_suffix = '' 
if subset_suffix == '':
    X_selected = X.copy()
    xvar_selected = xvar_list.copy()
else:
    xvar_subset_all = []
    for yvar in yvar_list_key:
        xvar_subset_all += xvar_sublist_sets[yvar][0]
        xvar_subset_all += xvar_sublist_sets[yvar][1]
    xvar_subset_all = sort_list(list(set(xvar_subset_all)))
    print(len(xvar_subset_all), xvar_subset_all)
    
    idx_selected = [xvar_list.index(xvar) for xvar in xvar_subset_all]
    X_selected = X[:,np.array(idx_selected)]
    xvar_selected = xvar_subset_all.copy()

# concatenate X and Y (key) variables
XYarr = np.concatenate((X_selected,Y[:,:4]), axis=1)
XYarr = pd.DataFrame(XYarr, columns=xvar_selected+yvar_list[:4])

# get correlation matrix
csv_fname = f'{data_folder}{dataset_name}{dataset_suffix}{subset_suffix}_correlation_matrix'
savefig = f'{figure_folder}{dataset_name}{dataset_suffix}_correlation_matrix{subset_suffix}'
corr_mat, corr_all = get_corrmat_corrlist(XYarr, sort_corrlist=True, csv_fname=csv_fname, savefig=savefig, plot_corrmat=True, plot_clustermap=True)

# get pairs with high correlations
corr_selected = get_high_correlation_pairs(corr_all, corr_thres=0.95, print_out=True)
print()

# for each variable, print list of other variables with which it has high correlation
highcorr_vars_dict = get_dict_of_features_with_highcorr(corr_mat, corr_thres=0.95)

#%% get clusters of high correlation features

# highcorr_vars_dict = get_dict_of_features_with_highcorr(corr_mat, corr_thres=0.98)
# for xvar, highcorr_xvar_list in highcorr_vars_dict.items():
    
# highcorr_vars_dict['Fe_basal']
plt.scatter(XYarr['feed vol'], XYarr['feed %'])
plt.show()

