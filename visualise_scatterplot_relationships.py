#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:11:44 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from variables import var_dict_all, yvar_sublist_sets
from plot_utils import figure_folder, convert_figidx_to_rowcolidx, remove_nandata, heatmap
from scipy.stats import spearmanr, pearsonr
data_folder = '../ajino-analytics-data/'

# set filenames to load
dataset_fname = 'DATA_avg.csv'
mediacomposition_fname = 'MediaComposition_avg.csv'

d = pd.read_csv(f'{data_folder}{dataset_fname}')
m = pd.read_csv(f'{data_folder}{mediacomposition_fname}')

# bioreactor_CQAs = ['VCD (E6 cells/mL)', 'Viability (%)', 'Titer (mg/L)'] 
media_inputs = ['Basal medium', 'Feed medium']
process_inputs = ['DO', 'pH', 'feed %']
nutrient_inputs = ['Glucose (g/L)'] + var_dict_all['AA'] + var_dict_all['VT'] + var_dict_all['MT'] + var_dict_all['Nuc, Amine']
cqa_outputs = ['Titer (mg/L)'] + var_dict_all['Glyco']

cqa_dict_byday = {day: [f'{cqa}_{day}' for cqa in cqa_outputs] for day in range(15)}
nutrient_dict_byday = {day: [f'{nutrient}_{day}' for nutrient in nutrient_inputs] for day in range(15)}
        
#%% plot nutrients vs final CQAs

nrows = 6
ncols = 8

d_filt = d[((d['DO']==40) & (d['pH']==7.0) & (d['feed %']==6))]
figtitle_suffix = '(DO=40%, pH=7, feed % = 6%)'
corr_res_df = []

for cqa_colname in cqa_dict_byday[14]: 
    y = d_filt[cqa_colname].to_numpy()
    
    for input_day in [0, 5]:
        fig, ax = plt.subplots(nrows, ncols, figsize=(28,20))
        nutrient_col_list = nutrient_dict_byday[input_day]
        nutrient_col_list = [c for c in nutrient_col_list if c in d]
        for figidx, nutrient_colname in enumerate(nutrient_col_list):
            x = d_filt[nutrient_colname].to_numpy()
            x_nonan, y_nonan = remove_nandata(x, y, on_var='xy')
            corr_s = np.round(spearmanr(x_nonan, y_nonan)[0],3)
            corr_p = np.round(pearsonr(x_nonan, y_nonan)[0],3)
            corr_avg = (corr_s+corr_p)/2
            n = len(x_nonan)
            row_idx, col_idx = convert_figidx_to_rowcolidx(figidx, ncols)
            ax[row_idx][col_idx].scatter(x_nonan, y_nonan)
            ax[row_idx][col_idx].set_xlabel(nutrient_colname)
            (xmin, xmax) = ax[row_idx][col_idx].get_xlim()
            (ymin, ymax) = ax[row_idx][col_idx].get_ylim()
            text = 'spearman R:' + str(corr_s) + '\n pearson R:' + str(corr_p)
            ax[row_idx][col_idx].text(xmin+(xmax-xmin)*0.4, ymin+(ymax-ymin)*0.8, text)
            corr_res_df.append({'x_var': nutrient_colname, 'y_var': cqa_colname, 'spearmanr': corr_s, 'pearsonr': corr_p, 'corr_avg': corr_avg, 'n':n})
        ymax = ax.flatten()[0].get_position().ymax
        plt.suptitle(f'{cqa_colname} vs nutrients on day {input_day} {figtitle_suffix}', y=ymax*1.02, fontsize=20)
        if cqa_colname.find(' ')>-1:
            cqa_colname_trunc = cqa_colname[:cqa_colname.find(' ')]
        else: 
            cqa_colname_trunc = cqa_colname
        figname = f'scatter_{cqa_colname_trunc}_VS_nutrients_{input_day}'
        fig.savefig(f'{figure_folder}{figname}.png',  bbox_inches='tight')
        plt.show()
        
corr_res_df = pd.DataFrame(corr_res_df)
corr_res_df = corr_res_df.sort_values(by='corr_avg', ascending=False)

corr_res_df.to_csv(f'{data_folder}corr_cqa_vs_nutrients.csv')

#%% plot correlation heatmap

# remove datapoints where n is too small
corr_res_df_filt = corr_res_df[corr_res_df['n']>10]
# remove rows that have all NaN correlation values
corr_res_df_filt = corr_res_df_filt.dropna(axis=0, how='all', subset='corr_avg')
xvar_list_alphabetical = corr_res_df_filt['x_var'].tolist()
xvar_list_alphabetical = list(set(xvar_list_alphabetical))
xvar_list_alphabetical.sort()
yvar_list = cqa_dict_byday[14]

heatmap_arr = np.zeros((len(yvar_list), len(xvar_list_alphabetical)))
heatmap_arr[:] = np.nan
for y_idx, yvar in enumerate(yvar_list):
    for x_idx, xvar in enumerate(xvar_list_alphabetical): 
        val = corr_res_df_filt.loc[((corr_res_df_filt['x_var']==xvar) & (corr_res_df_filt['y_var']==yvar)), 'corr_avg']
        heatmap_arr[y_idx, x_idx] = val

# plot heatmap
fig, ax = plt.subplots(1,1, figsize=(30, heatmap_arr.shape[0]/heatmap_arr.shape[1]*30))
heatmap(heatmap_arr, c='viridis', ax=None, cbar_kw={}, cbarlabel="", datamin=-1, datamax=1, logscale_cmap=False, annotate=False, row_labels=yvar_list, col_labels=xvar_list_alphabetical)
fig.savefig(f'{figure_folder}corr_heatmap.png',  bbox_inches='tight')


#%% get correlation dataframe with AVERAGE of correlation metric from day 0 and 5 input variables 

corr_res_dayavg = [] 
for y_var in cqa_dict_byday[14]: 
    for x_var in nutrient_inputs:
        df_filt = corr_res_df.loc[((corr_res_df['y_var']==y_var) & (corr_res_df['x_var'].isin([f'{x_var}_0', f'{x_var}_5']))), ['corr_avg', 'n']]
        if len(df_filt)==2:
            corr_avg = np.nanmean(df_filt['corr_avg'].to_numpy())
            n = int(np.nanmean(df_filt['n'].to_numpy()))
            corr_res_dayavg.append({'x_var': x_var, 'y_var': y_var, 'corr_avg': corr_avg, 'n':n})
corr_res_dayavg = pd.DataFrame(corr_res_dayavg)

# remove rows with too few samples
corr_res_df_filt = corr_res_dayavg[corr_res_dayavg['n']>10]
# remove rows that have all NaN correlation values
corr_res_df_filt = corr_res_df_filt.dropna(axis=0, how='all', subset='corr_avg')
xvar_list_alphabetical = corr_res_df_filt['x_var'].tolist()
xvar_list_alphabetical = list(set(xvar_list_alphabetical))
xvar_list_alphabetical.sort()
yvar_list = cqa_dict_byday[14]

# get heatmap data
heatmap_arr = np.zeros((len(yvar_list), len(xvar_list_alphabetical)))
heatmap_arr[:] = np.nan
for y_idx, yvar in enumerate(yvar_list):
    for x_idx, xvar in enumerate(xvar_list_alphabetical): 
        val = corr_res_df_filt.loc[((corr_res_df_filt['x_var']==xvar) & (corr_res_df_filt['y_var']==yvar)), 'corr_avg']
        heatmap_arr[y_idx, x_idx] = val

# plot heatmap
fig, ax = plt.subplots(1,1, figsize=(30, heatmap_arr.shape[0]/heatmap_arr.shape[1]*30))
heatmap(heatmap_arr, c='viridis', ax=None, cbar_kw={}, cbarlabel="", datamin=-1, datamax=1, logscale_cmap=False, annotate=True, row_labels=yvar_list, col_labels=xvar_list_alphabetical)
fig.savefig(f'{figure_folder}corr_inputdayAVG_heatmap.png',  bbox_inches='tight')     

#%% get clustermap

heatmap_df = pd.DataFrame(heatmap_arr, columns=xvar_list_alphabetical, index=yvar_list)
sns.clustermap(heatmap_df, cmap="viridis", figsize=(20, 10))
plt.savefig(f"{figure_folder}corr_inputdayAVG_clustermap.png") 



# %% plot selected scatterplots 
xy_var_list = [
    ('Asp_0', 'G2F_14'),
    ('Fe_0', 'G2F_14'),
    ('Riboflavin_0', 'Man5_14'),
    ('Lys_0', 'Man5_14'),
    ('Pro_0', 'G0F_14'),
    ('Cyanocobalamin_0', 'G0F_14')
    ]

nrows, ncols = 2,3
fig, ax = plt.subplots(nrows, ncols, figsize=(20,12))
for i, (xvar, yvar) in enumerate(xy_var_list): 
    y = d_filt[yvar].to_numpy()
    x = d_filt[xvar].to_numpy()
    x_nonan, y_nonan = remove_nandata(x, y, on_var='xy')
    print(i, xvar, yvar, len(y_nonan))
    corr_s = np.round(spearmanr(x_nonan, y_nonan)[0],3)
    corr_p = np.round(pearsonr(x_nonan, y_nonan)[0],3)
    corr_avg = (corr_s+corr_p)/2
    n = len(x_nonan)
    row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
    ax[row_idx][col_idx].scatter(x_nonan, y_nonan)
    ax[row_idx][col_idx].set_xlabel(xvar)
    ax[row_idx][col_idx].set_ylabel(yvar)
    (xmin, xmax) = ax[row_idx][col_idx].get_xlim()
    (ymin, ymax) = ax[row_idx][col_idx].get_ylim()
    text = 'spearman R:' + str(corr_s) + '\n pearson R:' + str(corr_p)
    ax[row_idx][col_idx].text(xmin+(xmax-xmin)*0.4, ymin+(ymax-ymin)*0.8, text)
    # figname = f'scatter_{cqa_colname_trunc}_VS_nutrients_{input_day}'
    # fig.savefig(f'{figure_folder}{figname}.png',  bbox_inches='tight')
plt.show()


