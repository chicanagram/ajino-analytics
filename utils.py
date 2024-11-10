#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:49:45 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import heatmap, annotate_heatmap
import seaborn as sns
from variables import data_folder, figure_folder, var_dict_all, overall_glyco_cqas, sort_list, yvar_list_key, xvar_list_dict_prefilt
from get_datasets import get_XYdata_for_featureset

def get_xy_correlation_matrix(X_featureset_idx, Y_featureset_idx, use_abs_vals=True):
    import matplotlib.pyplot as plt
    from plot_utils import heatmap

    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(
        X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
    print(f'X dataset size: n={Xscaled.shape[0]}, p={Xscaled.shape[1]}')

    # concatenate X and Y (key) variables
    XYarr = np.concatenate((X, Y[:, :4]), axis=1)
    XYarr = pd.DataFrame(XYarr, columns=xvar_list+yvar_list[:4])

    # calculate correlation matrix
    corr_mat = XYarr.corr()
    col_labels = corr_mat.columns.tolist()
    row_labels = list(corr_mat.index)
    corr_mat.round(3).to_csv(
        f'{data_folder}{dataset_name}_correlation_matrix.csv')

    # plot correlation matrix
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    if use_abs_vals:
        heatmap(np.abs(corr_mat.to_numpy()), ax=ax, datamin=0, datamax=1,
                logscale_cmap=False, annotate=None, row_labels=row_labels, col_labels=col_labels)
    else:
        heatmap(corr_mat.to_numpy(), ax=ax, datamin=-1, datamax=1, logscale_cmap=False,
                annotate=None, row_labels=row_labels, col_labels=col_labels)
    ax.grid(None)
    fig.savefig(
        f'{figure_folder}{dataset_name}_correlation_matrix.png', bbox_inches='tight')

    return corr_mat


def get_corrmat_corrlist(df, sort_corrlist=True, csv_fname=None, savefig=None, plot_corrmat=True, plot_clustermap=False, use_abs_vals=True):
    
    # calculate correlation matrix
    corr_mat = df.corr(numeric_only=True)
    if use_abs_vals:
        corr_mat = corr_mat.abs()
    cols = corr_mat.columns.tolist()
    rows = list(corr_mat.index)
    n = len(cols)
    
    # get cols with any NaNs
    vars_w_nan = []
    for col in cols:
        if corr_mat.loc[:, col].isnull().all():
            vars_w_nan.append(col)
    corr_mat = corr_mat.dropna(how='all', axis=1)
    corr_mat = corr_mat.dropna(how='all', axis=0)
    for col in cols:
        if col in corr_mat and corr_mat.loc[:, col].isnull().any():
            vars_w_nan.append(col)        
    
    # remove rows and columns with any NaNs
    cols_nonan = [c for c in cols if c not in vars_w_nan]
    rows_nonan = [r for r in rows if r not in vars_w_nan]
    corr_mat = corr_mat.loc[rows_nonan, cols_nonan]
    cols = cols_nonan
    rows = rows_nonan

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
        fig, ax = plt.subplots(1, 1, figsize=(n/20*8, n/20*8))
        if use_abs_vals:
            heatmap(corr_mat.to_numpy(), ax=ax, datamin=0, datamax=1,
                    logscale_cmap=False, annotate=None, row_labels=rows, col_labels=cols)
        else:
            heatmap(corr_mat.to_numpy(), ax=ax, datamin=-1, datamax=1,
                    logscale_cmap=False, annotate=None, row_labels=rows, col_labels=cols)
        ax.grid(None)
        if savefig is not None:
            fig.savefig(f'{savefig}.png', bbox_inches='tight')

    # get clustermap
    if plot_clustermap:
        cl = sns.clustermap(corr_mat, cmap="viridis", figsize=(12, 12), yticklabels=True, xticklabels=True)
        cl.ax_heatmap.set_yticklabels(cl.ax_heatmap.get_ymajorticklabels(), fontsize=7)
        cl.ax_heatmap.set_xticklabels(cl.ax_heatmap.get_xmajorticklabels(), fontsize=7)
        cl.fig.suptitle('Cluster map of selected features', fontsize=16)
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
            print(i, f"({index_selected[i][0]}, {index_selected[i][1]}):", round(
                corr_selected.iloc[i]['corr'], 4))
    return corr_selected


def get_dict_of_features_with_highcorr(corr_mat, corr_thres):
    highcorr_vars_dict = {}
    rows = list(corr_mat.index)
    for i, xvar in enumerate(rows):
        corr_mat_row = corr_mat.loc[xvar]
        high_corr_vars = list(
            corr_mat_row[corr_mat_row.abs() > corr_thres].index)
        high_corr_vars.remove(xvar)
        highcorr_vars_dict[xvar] = high_corr_vars
        print(i, xvar, ':', *high_corr_vars)
    return highcorr_vars_dict


def cluster_features_based_on_crosscorrelations(corrX, threshold, print_res=False):
    from scipy.cluster import hierarchy
    import scipy.spatial.distance as ssd

    xvar_list = corrX.columns.tolist()
    distances = 1 - corrX.abs().values  # pairwise distnces
    distArray = ssd.squareform(distances)  # scipy converts matrix to 1d array
    # you can use other methods
    hier = hierarchy.linkage(distArray, method="ward")
    # cluster label features
    cluster_labels = hierarchy.fcluster(hier, threshold, criterion="distance")
    print('# of feature clusters:', len(list(set(cluster_labels))))
    cluster_dict_xvar_idxs = {}
    cluster_dict_xvar_names = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_dict_xvar_names:
            cluster_dict_xvar_names[label] = []
            cluster_dict_xvar_idxs[label] = []
        cluster_dict_xvar_names[label].append(xvar_list[i])
        cluster_dict_xvar_idxs[label].append(i)

    if print_res:
        for label, feature_grp in cluster_dict_xvar_names.items():
            print(label, feature_grp)
    return cluster_dict_xvar_names, cluster_dict_xvar_idxs, hier