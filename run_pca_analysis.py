#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:20:01 2025

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from variables import data_folder, var_dict_all
from plot_utils import convert_figidx_to_rowcolidx
from model_utils import get_corr_coef

#%% 
# load data
csv_fpath = data_folder + 'DATA.csv'
df = pd.read_csv(csv_fpath)
print(df.shape)

# feature to name mapping
name_to_feature_mapping_dict = {
    'CPPs': {
        'VCD': 'VCD (E6 cells/mL)',
        'Via': 'Viability (%)',
        'Titer': 'Titer (mg/L)',
        'Gluc': 'Glucose (g/L)',
        'Lac': 'Lac (g/L)',
        'NH4+': 'NH4+ (mM)',
        'Osm': 'Osmolarity (mOsm/kg)',
    },
    # 'CQAs': {
    #     'Titer': 'Titer (mg/L)',
    #     'Man5': 'mannosylation',
    #     'Fuc': 'fucosylation',
    #     'Gal': 'galactosylation',
    # },
    'CPPs/CQAs': {
        'VCD': 'VCD (E6 cells/mL)',
        'Via': 'Viability (%)',
        'Titer': 'Titer (mg/L)',
        'Fuc': 'fucosylation',
        'Gal': 'galactosylation',
        'Gluc_14': 'Glucose (g/L)',
        'Lac_14': 'Lac (g/L)',
        'NH4+_14': 'NH4+ (mM)',
        'Osm_14': 'Osmolarity (mOsm/kg)',
        'Man5_14': 'mannosylation',
    },
    'CQAs/metabolites': {
        'VCD': 'VCD (E6 cells/mL)_14',
        'Via': 'Viability (%)_14',
        'Titer': 'Titer (mg/L)_14',
        'Man5': 'mannosylation_14',
        'Fuc': 'fucosylation_14',
        'Gal': 'galactosylation_14',
        'Gluc_7': 'Glucose (g/L)_7',
        'Lac_7': 'Lac (g/L)_7',
        'NH4+_7': 'NH4+ (mM)_7',
        'Osm_7': 'Osmolarity (mOsm/kg)_7',
        # 'Gluc_14': 'Glucose (g/L)_14',
        # 'Lac_14': 'Lac (g/L)_14',
        # 'NH4+_14': 'NH4+ (mM)_14',
        # 'Osm_14': 'Osmolarity (mOsm/kg)_14',
    },
}

key_var_mappings = {        
    'Titer': 'Titer (mg/L)_14',
    'Man5': 'mannosylation_14'
    }

for k, (pca_set_name, name_to_feature_mapping) in enumerate(name_to_feature_mapping_dict.items()):
    if pca_set_name == 'CQAs/metabolites':
        var_to_pca = list(name_to_feature_mapping.values())
        varnames_to_pca_init = [v for v in list(name_to_feature_mapping.keys())]
    else:
        var_to_pca = [v+'_14' for v in list(name_to_feature_mapping.values())]
        print(var_to_pca)
        varnames_to_pca_init = [v for v in list(name_to_feature_mapping.keys())]
    # add features to eval if not in PCA list
    varnames_to_pca = varnames_to_pca_init.copy()
    for var in ['Titer', 'Man5']: 
        if var not in varnames_to_pca_init: 
            var_to_pca.append(key_var_mappings[var])
            varnames_to_pca.append(var)
    
    # get data
    X = df[var_to_pca].to_numpy()
    # get titer and man5
    titer = X[:, varnames_to_pca.index('Titer')]
    man5 = X[:, varnames_to_pca.index('Man5')]
    for var in ['Man5', 'Titer']: 
        if var not in varnames_to_pca_init: 
            varnames_to_pca.remove(var)
    
    # Standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
        
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Get loadings (feature directions)
    loadings = pca.components_.T  # Shape: (n_features, n_components)
    
    # Plot the PCA-transformed data
    color_feature = titer/man5
    legend_suffix = 'Titer / Man5'
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c=color_feature, cmap='viridis', label='Samples')
    
    # Plot feature directions
    ncolors_in_palette = 10
    cmap = [plt.cm.Dark2(i) for i in range(ncolors_in_palette)]
    cmap = cmap * int(np.ceil(len(varnames_to_pca) / ncolors_in_palette))
    
 
    for i, feature in enumerate(varnames_to_pca):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color=cmap[i], head_width=0.05, length_includes_head=True)
        plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, feature, color=cmap[i], fontsize=8)
    
    # Decorate the plot
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.title(f'PCA Plot for key {pca_set_name}' )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend([f'All samples, color coded by {legend_suffix}'])
    plt.grid(False)
    plt.show()
    
#%% plot correlations between key CPPs
day = 14
plot_correlations = [
    (f'Glucose (g/L)_{day}', 'Titer (mg/L)_14'),
    (f'Lac (g/L)_{day}', 'Titer (mg/L)_14'),
    (f'NH4+ (mM)_{day}', 'Titer (mg/L)_14'),
    (f'VCD (E6 cells/mL)_{day}', 'Titer (mg/L)_14'),
    (f'Glucose (g/L)_{day}', 'mannosylation_14'),
    (f'Lac (g/L)_{day}', 'mannosylation_14'),
    (f'NH4+ (mM)_{day}', 'mannosylation_14'),
    ('NH4+ (mM)_7', 'mannosylation_14'),
    (f'Glucose (g/L)_{day}', f'Lac (g/L)_{day}'),
    (f'NH4+ (mM)_{day}', f'Lac (g/L)_{day}'),
    (f'Glucose (g/L)_{day}', f'NH4+ (mM)_{day}'),
    (f'Glucose (g/L)_7', f'NH4+ (mM)_7'),
    ]


nrows, ncols = 3, 4
fig, ax = plt.subplots(nrows, ncols, figsize=(6*ncols,5.5*nrows))
for i, (xvar, yvar) in enumerate(plot_correlations):
    row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
    x = df[xvar].to_numpy()
    y = df[yvar].to_numpy()
    corr = get_corr_coef(x, y, corr_to_get=['spearmanr', 'pearsonr'])
    ax[row_idx, col_idx].scatter(x, y, s=8, alpha=0.8)
    ax[row_idx, col_idx].set_xlabel(xvar)
    ax[row_idx, col_idx].set_ylabel(yvar)
    ax[row_idx, col_idx].set_title(f'{yvar} vs {xvar}', fontsize=12)
    ax[row_idx, col_idx].annotate(f'spearmanr:{corr["spearmanr"]} \n pearsonr:{corr["pearsonr"]}', xy=(0.1, 0.8), xycoords='axes fraction')
unused_axs = [i for i in range(nrows*ncols) if i>=len(plot_correlations)]
for idx in unused_axs:
    i, j = convert_figidx_to_rowcolidx(idx, ncols)
    ax[i][j].set_axis_off()
    
#%% 
plot_correlations = [
    ('mannosylation_14', 'fucosylation_14'),
    ('mannosylation_14', 'galactosylation_14'),
    ('fucosylation_14', 'galactosylation_14'),
    ]
nrows, ncols = 1, len(plot_correlations)
fig, ax = plt.subplots(nrows, ncols, figsize=(6*ncols,5.5*nrows))
for i, (xvar, yvar) in enumerate(plot_correlations):
    x = df[xvar].to_numpy()
    y = df[yvar].to_numpy()
    ax[i].scatter(x,y)
    ax[i].set_xlabel(xvar)
    ax[i].set_ylabel(yvar)
    corr = get_corr_coef(x, y, corr_to_get=['spearmanr', 'pearsonr'])
    ax[i].annotate(f'spearmanr:{corr["spearmanr"]} \n pearsonr:{corr["pearsonr"]}', xy=(0.6, 0.8), xycoords='axes fraction')
    ax[i].set_title(f'{yvar} vs {xvar}', fontsize=12)
plt.plot()
