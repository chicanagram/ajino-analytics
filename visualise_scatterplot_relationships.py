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
from get_datasets import get_XYdata_for_featureset
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
cqa_outputs = ['Titer (mg/L)', 'mannosylation', 'fucosylation', 'galactosylation',] + var_dict_all['Glyco'][1:]

cqa_dict_byday = {day: [f'{cqa}_{day}' for cqa in cqa_outputs] for day in range(15)}
nutrient_dict_byday = {day: [f'{nutrient}_{day}' for nutrient in nutrient_inputs] for day in range(15)}
        
#%% plot AAL nutrients vs final CQAs

nrows = 6
ncols = 8

d_filt = d[((d['DO']==40) & (d['pH']==7.0) & (d['feed %']==6))]
figtitle_suffix = '(DO=40%, pH=7, feed % = 6%)'
corr_res_df = []

for cqa_colname in cqa_dict_byday[14][:4]: 
    y = d_filt[cqa_colname].to_numpy()
    
    for input_day in [0]:
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
        # fig.savefig(f'{figure_folder}{figname}.png',  bbox_inches='tight')
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
nrows, ncols = 3,4
yvar_list_toplot = ['Titer (mg/L)_14', 'mannosylation_14', 'fucosylation_14', 'galactosylation_14']
xvar_process_list = ['feed %', 'pH', 'DO']
xvar_nutrients_list = [
    'Asn', 'Choline', 'Cyanocobalamin', 
    'Glucose (g/L)', 'Pro', 'Met', 
    # 'D-glucose', 'Pro', 'Met', 
    'Nicotinamide', 'Trp', 'Riboflavin', 
    'Folic acid', 'Ca', 'Mn', 
    'Na', 'Phe', 'His',
    'Biotin', 'K', 'Pantothenic acid'
                  ]
# nutrients_suffix_list = ['_basal', '_feed']
nutrients_suffix_list = ['_0']
featureset_name = 'featureset0'
xvar_nutrients_wsuffix_list = []
num_sets_of_xvars = int(np.ceil(len(xvar_nutrients_list)/nrows))
for i in range(num_sets_of_xvars):
    xvar_base_sublist = xvar_nutrients_list[i*nrows:(i+1)*nrows]
    xvar_nutrients_wsuffix_sublist = [f'{xvar_base}{suffix}' for suffix in nutrients_suffix_list for xvar_base in xvar_base_sublist]
    xvar_nutrients_wsuffix_list += xvar_nutrients_wsuffix_sublist
        
xvar_list_toplot = xvar_process_list + xvar_nutrients_wsuffix_list
print(len(xvar_list_toplot), xvar_list_toplot)

# plot scatter
num_plots = int(np.ceil(len(xvar_list_toplot)/nrows))
for plot_num in range(num_plots):
    xvar_sublist_toplot_startidx = plot_num*nrows
    xvar_sublist_toplot_endidx = (plot_num+1)*nrows
    xvar_sublist_toplot = xvar_list_toplot[xvar_sublist_toplot_startidx:xvar_sublist_toplot_endidx]
    # get yx_var_list for this plot
    yx_var_list = []
    for xvar in xvar_sublist_toplot: 
        for yvar in yvar_list_toplot: 
            yx_var_list.append((yvar, xvar))
            
    fig, ax = plt.subplots(nrows, ncols, figsize=(20,12))
    for i, (yvar, xvar) in enumerate(yx_var_list): 
        # get data to plot
        d_filt = d.copy()
        if xvar in ['feed %', 'DO', 'pH']: 
            d_filt = d[(d['Basal medium']=='Basal-A') & (d['Feed medium']=='Feed-a')].copy()
        else: 
            d_filt = d[(d['feed %']==6) & (d.pH==7.0) & (d.DO==40)].copy()
        if xvar in d_filt:
            y = d_filt[yvar].to_numpy()
            x = d_filt[xvar].to_numpy()
            x_nonan, y_nonan = remove_nandata(x, y, on_var='xy')
            # plot scatter data
            if len(y_nonan)>2:
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
    # set general axis labels and titles
    ax[0][0].set_title('Titer', fontsize=16, c='b')
    ax[0][1].set_title('Mannosylation', fontsize=16, c='b')
    ax[0][2].set_title('Fucosylation', fontsize=16, c='b')
    ax[0][3].set_title('Galactosylation', fontsize=16, c='b')
    ax_dict = {}
    ax_dict[0] = ax[0][3].twinx()
    ax_dict[1] = ax[1][3].twinx()
    ax_dict[2] = ax[2][3].twinx()
    for xvar_idx, xvar in enumerate(xvar_sublist_toplot):
        ax_dict[xvar_idx].set_ylabel(xvar, fontsize=16, c='r')
        ax_dict[xvar_idx]
        ax_dict[xvar_idx].tick_params(labelright=False, length=0)
    # set overall plot title
    ymax = ax.flatten()[0].get_position().ymax
    plt.suptitle('Relationship between Key CQAs and Selected Process or Nutrient Inputs', y=ymax*1.06, fontsize=24)    
    # save plot      
    figname = f'scatter_keyCQAs_VS_inputs_{featureset_name}_{plot_num}'
    fig.savefig(f'{figure_folder}{figname}.png',  bbox_inches='tight')
    plt.show()          

#%% Correlation between Day 0 Nutrients and Basal Media composition

data_AVG = pd.read_csv(f'{data_folder}data_AVG.csv', index_col=0)
df_X0Y0 = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
df_X1Y0 = pd.read_csv(f'{data_folder}X1Y0.csv', index_col=0)

# get pairs of x variables to plot
xvarpair_list = []
xvar_list_0 = df_X0Y0.columns.tolist()[3:]
xvar_list_1 = df_X1Y0.columns.tolist()[3:]
for idx_0, xvar_0 in enumerate(xvar_list_0): 
    if xvar_0.find('_0')>-1:
        xvar = xvar_0[:-2]
        xvar_basal = xvar + '_basal'
        if xvar_basal in xvar_list_1: 
            xvarpair_list.append(xvar)

print(len(xvarpair_list), xvarpair_list)

# get scatter plots 
nrows, ncols = 5,7
# fig, ax = plt.subplots(nrows, ncols, figsize=(34,34))
fig, ax = plt.subplots(nrows, ncols, figsize=(42,27))
for k, xvar in enumerate(xvarpair_list):
    row_idx, col_idx = convert_figidx_to_rowcolidx(k, ncols)
    xdata = data_AVG[f'{xvar}_0'].to_numpy()
    ydata = data_AVG[f'{xvar}_basal'].to_numpy()
    ax[row_idx][col_idx].scatter(xdata, ydata)
    ax[row_idx][col_idx].set_title(xvar, fontsize=16)
    ax[row_idx][col_idx].set_xlabel(f'D0 {xvar} conc.', fontsize=12)
    ax[row_idx][col_idx].set_ylabel(f'Basal {xvar} conc.', fontsize=12)
ymax = ax.flatten()[0].get_position().ymax
plt.suptitle(f'Correlation between Nutrient concentrations in Basal Media vs. D0 sample', y=ymax*1.03, fontsize=24)    
fig.savefig(f'{figure_folder}scatterplot_corr_nutrients_basalVSday0.png', bbox_inches='tight')
plt.show()


