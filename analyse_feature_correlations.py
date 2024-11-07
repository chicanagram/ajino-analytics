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
from plot_utils import heatmap, annotate_heatmap
import seaborn as sns
from variables import data_folder, figure_folder, var_dict_all, overall_glyco_cqas, sort_list, yvar_list_key, xvar_list_dict_prefilt
from get_datasets import get_XYdataset, get_XYdata_for_featureset
from sklearn.preprocessing import scale
from scipy.cluster import hierarchy
import scipy.spatial.distance as ssd

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

# %% get all XY feature correlations

use_all_data = False

# get data from specific dataset
X_featureset_idx, Y_featureset_idx = 1, 0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
# dataset_suffix = '_avg'
dataset_suffix = ''
Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)

if use_all_data: 
    # get ALL raw data
    XYarr = pd.read_csv(f'{data_folder}DATA.csv', index_col=0)
    XYarr = XYarr[[var for var in xvar_list_dict_prefilt[X_featureset_idx]+yvar_list_key if var in XYarr]]
else: 
    X_df = pd.DataFrame(X, columns=xvar_list)
    # get concatenated XY data
    XYarr = np.concatenate((X, Y[:, :4]), axis=1)
    XYarr = pd.DataFrame(XYarr, columns=xvar_list+yvar_list[:4])

# get correlation matrix
csv_fname = f'{data_folder}{dataset_name}{dataset_suffix}_correlation_matrix'
savefig = f'{figure_folder}{dataset_name}{dataset_suffix}_correlation_matrix'
corr_mat, corr_all = get_corrmat_corrlist(XYarr, sort_corrlist=True, csv_fname=csv_fname,
                                          savefig=savefig, plot_corrmat=True, plot_clustermap=True, use_abs_vals=True)


# %% get high correlation pairs / clusters

# set correlation threshold
corr_thres = 0.9

# get pairs with high correlations
corr_selected = get_high_correlation_pairs(
    corr_all, corr_thres=corr_thres, print_out=True)
print()

# for each variable, print list of other variables with which it has high correlation
highcorr_vars_dict = get_dict_of_features_with_highcorr(
    corr_mat, corr_thres=corr_thres)

# %% get clusters of high correlation features using scipy hierarchical clustering
thres_to_visualize = 0.1
corrX = corr_mat.iloc[:-4, :-4]
thres_arr = np.arange(0.001, 2.5, 0.001)
num_clusters_arr = np.zeros((len(thres_arr),))
cluster_dict_bynumclusters = {}

for i, thres in enumerate(thres_arr):
    print(i, thres)
    cluster_dict_xvar_names, cluster_dict_xvar_idxs, hier = cluster_features_based_on_crosscorrelations(
        corrX, threshold=thres, print_res=False)
    num_clusters = len(cluster_dict_xvar_names)
    num_clusters_arr[i] = num_clusters
    if num_clusters not in cluster_dict_bynumclusters:
        cluster_dict_bynumclusters[num_clusters] = {
            'xvar_names': cluster_dict_xvar_names,
            'xvar_idxs': cluster_dict_xvar_idxs,
            'thres': thres
        }
    if thres == thres_to_visualize:
        dend = hierarchy.dendrogram(
            hier, truncate_mode="level", p=30, color_threshold=1.5)
        plt.show()


plt.plot(thres_arr, num_clusters_arr)
plt.title(
    'Number of clusters vs. Distance threshold for grouping correlated features')
plt.xlabel('Distance threshold for grouping correlated features')
plt.ylabel('Number of feature clusters')
plt.show()

# get num_cluster to threshold dict
ncluster_to_thres_dict = {}
for ncluster, thres in zip(num_clusters_arr, thres_arr):
    ncluster_to_thres_dict[int(ncluster)] = round(thres, 3)
print(ncluster_to_thres_dict)

# %% investigate one particular threshold
thres = 0.1
xvar_list = corrX.columns.tolist()
distances = 1 - corrX.abs().values  # pairwise distnces
distArray = ssd.squareform(distances)  # scipy converts matrix to 1d array
hier = hierarchy.linkage(distArray, method="ward")  # you can use other methods
dend = hierarchy.dendrogram(
    hier, truncate_mode="level", p=30, color_threshold=1.5)
plt.show()
# cluster label features
cluster_labels = hierarchy.fcluster(hier, thres, criterion="distance")
unique_cluster_labels = np.unique(cluster_labels)
for c in unique_cluster_labels:
    print(c, *[xvar for i, xvar in enumerate(xvar_list)
          if i in np.argwhere(cluster_labels == c).flatten()])

# %%
# create dataframe based on results
num_clusters_list = list(cluster_dict_bynumclusters.keys())
num_clusters_list.sort()
cluster_df = pd.DataFrame(index=num_clusters_list,
                          columns=corrX.columns.tolist())
for num_clusters in num_clusters_list:
    xvar_names_dict = cluster_dict_bynumclusters[num_clusters]['xvar_names']
    for cluster_label in range(1, num_clusters+1):
        cluster_df.loc[num_clusters,
                       xvar_names_dict[cluster_label]] = cluster_label

# sort columns by labels
col_idx_sorted = np.argsort(cluster_df.iloc[-1, :].to_numpy())
colormap = 'gist_ncar'  # 'viridis' #

# plot raw labels
fig, ax = plt.subplots(1, 1, figsize=(32, 28))
heatmap(cluster_df.to_numpy(), ax=ax, logscale_cmap=False, annotate=None,
        row_labels=num_clusters_list, col_labels=corrX.columns.tolist(), show_gridlines=False, c=colormap)
plt.show()

# normalize each row
cluster_df_norm = cluster_df.copy()
for i in range(len(cluster_df_norm)):
    cluster_df_norm.iloc[i, :] = cluster_df_norm.iloc[i,
                                                      :]/cluster_df_norm.iloc[i, :].max()
# plot raw labels normalized along each row
fig, ax = plt.subplots(1, 1, figsize=(32, 28))
heatmap(cluster_df_norm.to_numpy(), ax=ax, logscale_cmap=False, annotate=None,
        row_labels=num_clusters_list, col_labels=corrX.columns.tolist(), show_gridlines=False, c=colormap)
plt.show()

# plot raw labels normalized along each row, sorted by label
cluster_df_sorted = cluster_df.copy().iloc[:, col_idx_sorted]
cluster_df_norm_sorted = cluster_df_norm.copy().iloc[:, col_idx_sorted]
xvar_list_sorted = [xvar_list[idx] for idx in col_idx_sorted]
fig, ax = plt.subplots(1, 1, figsize=(32, 28))
heatmap(cluster_df_norm_sorted.to_numpy(), ax=ax, logscale_cmap=False, annotate=None,
        row_labels=num_clusters_list, col_labels=xvar_list_sorted, show_gridlines=False, c=colormap)
annotate_heatmap(cluster_df_sorted.to_numpy(), ax, ndecimals=0)
plt.show()

# save cluster_df_sorted
cluster_df_sorted.to_csv(f'{data_folder}features_by_cluster_corrdist.csv')
