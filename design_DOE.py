#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 23:17:39 2025

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from utils import get_corrmat_corrlist, sort_list
import math

num_levels = 40
num_actual_levels = 4
corr_thres = 0.75
combis = [np.array(sort_list(list(row))) for row in list(set(combinations(range(num_levels), num_actual_levels)))]
base_all = []
for combi in combis:
    combi = combi-combi[0]
    combi = list(combi)
    if combi not in base_all:
        base_all.append(combi)
num_base_combis = len(base_all)
base_maxcorr_df = []
num_perms = math.factorial(num_actual_levels)
triu_mask = np.triu(np.ones((num_perms,num_perms)))
triu_mask[triu_mask==0] = np.nan
np.fill_diagonal(triu_mask,np.nan)
num_corr_all = len(triu_mask[triu_mask==1])
for i, base in enumerate(base_all): 
    print(f'{i+1}/{num_base_combis}')
    perms = np.array([list(row) for row in list(set(permutations(base, num_actual_levels)))])
    arr = pd.DataFrame(perms, columns=range(num_actual_levels)).transpose()
    corr_mat = arr.corr().abs().to_numpy()
    corr_mat_triu = corr_mat * triu_mask
    max_corr = np.nanmax(corr_mat_triu)
    median_corr = np.nanmedian(corr_mat_triu)
    mean_corr = np.nanmean(corr_mat_triu)
    frac_above_corr_thres = len(corr_mat_triu[corr_mat_triu>corr_thres])/num_corr_all
    base_maxcorr_df.append({'base':base, 'max_corr':max_corr, 'median_corr':median_corr, 'mean_corr':mean_corr, 'frac_above_corr_thres':frac_above_corr_thres})
  
base_maxcorr_df = pd.DataFrame(base_maxcorr_df)
base_maxcorr_df = base_maxcorr_df.sort_values(by='max_corr')

fig, ax = plt.subplots(2,1)
ax[0].scatter(base_maxcorr_df['max_corr'], base_maxcorr_df['median_corr'], s=4, alpha=0.7)
ax[1].scatter(base_maxcorr_df['max_corr'], base_maxcorr_df['frac_above_corr_thres'], s=4, alpha=0.7)
for i in [0,1]:
    ax[i].set_xlabel('max_corr')
ax[0].set_ylabel('median_corr')
ax[1].set_ylabel(f'frac_above_corr_thres ({corr_thres})')
plt.show()

#%%
num_actual_levels = 5
corr_thres = 1
combis = list(permutations(range(num_actual_levels)) )
print(len(combis))
arr = pd.DataFrame(combis, columns=range(num_actual_levels)).transpose()
corr_mat, corr_all = get_corrmat_corrlist(arr, sort_corrlist=True, csv_fname=None, savefig=None, plot_corrmat=True, plot_clustermap=False, use_abs_vals=True)
highcorr_pairs = list(corr_all[corr_all['corr_abs']>=corr_thres].index)
samples_to_remove = []
for (i,j) in highcorr_pairs:
    samples_to_remove.append(j)
combis_filt = [combi for i, combi in enumerate(combis) if i not in samples_to_remove]
print(len(combis_filt))
arr_filt = pd.DataFrame(combis_filt, columns=range(num_actual_levels)).transpose()
corr_mat_filt, corr_all_filt = get_corrmat_corrlist(arr_filt, sort_corrlist=True, csv_fname=None, savefig=None, plot_corrmat=True, plot_clustermap=False, use_abs_vals=True)
print('Max corr after filtering:',  corr_all_filt['corr_abs'].max())

#%%
base_maxcorr_df_best = base_maxcorr_df[(base_maxcorr_df['max_corr']<0.92) & (base_maxcorr_df['frac_above_corr_thres']<0.20)]
print(base_maxcorr_df_best)
base = base_maxcorr_df_best.iloc[0]['base']
perms = np.array([list(row) for row in list(set(permutations(base, num_actual_levels)))])
arr = pd.DataFrame(perms, columns=range(num_actual_levels)).transpose()
corr_mat, corr_all = get_corrmat_corrlist(arr, sort_corrlist=True, csv_fname=None, savefig=None, plot_corrmat=True, plot_clustermap=False, use_abs_vals=True)

#%%
