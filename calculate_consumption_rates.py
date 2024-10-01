#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:41:35 2024

@author: charmainechia
"""
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from variables import data_folder, var_dict_all
from plot_utils import figure_folder, remove_nandata,convert_figidx_to_rowcolidx, heatmap

#%%
nutrient_inputs = ['Glucose (g/L)'] + var_dict_all['AA'] + var_dict_all['VT'] + var_dict_all['MT'] + var_dict_all['Nuc, Amine']
nutrients_calculated_list = []

# load data
with open(f'{data_folder}DATA.pkl', 'rb') as handle:
    d = pickle.load(handle)
    
# iterate through experiments
for exp_idx, d_exp in d.items(): 
    print('[EXP_IDX]', exp_idx)
    # get IVCD array
    t_integrated = d_exp['VCD (E6 cells/mL)']['t_imputed']
    vcd_integrated = d_exp['VCD (E6 cells/mL)']['y_integrated']
    feed_days = [int(d) for d in d_exp['feed day'].split(',')]
    feed_days_to_sample = [day for day in feed_days if (day>=5 and day<=11)]
    feed_interval = feed_days_to_sample[1]-feed_days_to_sample[0]
    feed_days_to_sample.append(feed_days_to_sample[-1]+feed_interval)
    print(feed_days_to_sample)
    
    # iterate through nutrients
    for nutrient in nutrient_inputs:
        if (nutrient not in d_exp) or ('y_imputed' not in d_exp[nutrient]):
            pass
        else: 
            if nutrient not in nutrients_calculated_list:
                nutrients_calculated_list.append(nutrient)
            sampling_days_list = d_exp[nutrient]['t_rounded']
            t_imputed = d_exp[nutrient]['t_imputed']
            y_imputed = d_exp[nutrient]['y_imputed']
            
            # iterate through feed days
            rates = []
            for i, day in enumerate(feed_days_to_sample[:-1]): 
                # get start and end days
                t_start = day
                t_end = feed_days_to_sample[i+1]
                # get closest matching sampling times
                idx_start_adj = np.where(sampling_days_list==t_start)[0][-1]
                t_start_adj = t_imputed[idx_start_adj]
                try: 
                    idx_end_adj = np.where(sampling_days_list==t_end)[0][0]
                    t_end_adj = t_imputed[idx_end_adj]
                except:
                    t_end_adj = t_end
                # get difference in IVCDs
                ivcd_diff_denominator = np.interp(t_end_adj, t_integrated, vcd_integrated) - np.interp(t_start_adj, t_integrated, vcd_integrated)                
                # get difference in substrate concentrations
                y_diff_numerator = y_imputed[idx_end_adj] - y_imputed[idx_start_adj]
                # calculate consumption rate for each feed day
                rate = y_diff_numerator / ivcd_diff_denominator
                rates.append(rate)
            rates = np.array(rates)
            d_exp[nutrient]['rate'] = rates
            # get the average rate
            d_exp[nutrient]['rate_avg'] = np.mean(rates)
            # print(nutrient, d_exp[nutrient]['rate_avg'])
            
        # update overall dict
        d.update({exp_idx: d_exp})


# save data
with open(f'{data_folder}DATA.pkl', 'wb') as handle:
    pickle.dump(d, handle)
    
#%% visualize relationships

cqas_to_analyse = ['Titer (mg/L)', 'mannosylation', 'fucosylation', 'galactosylation']
nutrients_to_analyse = nutrients_calculated_list.copy()
res = {}
for nutrient in nutrients_to_analyse:
    res_nutrient = {'rate_avg': []}
    for cqa in cqas_to_analyse: 
        res_nutrient.update({cqa:[]})
    res.update({nutrient: res_nutrient})

# iterate through experiments
for exp_idx, d_exp in d.items(): 
    # iterate through nutrients
    for nutrient in nutrients_to_analyse:
        # filter out samples
        # if nutrient in d_exp and 'rate_avg' in d_exp[nutrient] and exp_idx not in ['0.1','0.2'] and exp_idx.find('1.')>-1:
        if nutrient in d_exp and 'rate_avg' in d_exp[nutrient] and exp_idx not in ['0.3','0.4','0.5','0.6','0.7','0.8']:
            res[nutrient]['rate_avg'].append(d_exp[nutrient]['rate_avg'])
            for cqa in cqas_to_analyse:
                res[nutrient][cqa].append(d_exp[cqa]['y'][-1])

# initiialize dict to record correlations
cqa_vs_rate_corr = {}
corr_arr = np.zeros((len(cqas_to_analyse), len(nutrients_to_analyse)))
corr_df_all = pd.DataFrame(corr_arr, columns=nutrients_to_analyse, index=cqas_to_analyse)

# plot correlations between average consumption rate and CQAs (titer, glycosylation)
for k, cqa in enumerate(cqas_to_analyse):
    print('\n*********************')
    print(cqa)
    print('*********************\n')
    
    # initialize plot
    nrows, ncols = 6,7
    fig, ax = plt.subplots(nrows, ncols, figsize=(40,24))
    corr_cqa = []
    
    for i, nutrient in enumerate(nutrients_to_analyse):
        row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
        try:
            x = np.array(res[nutrient]['rate_avg'])
            y = np.array(res[nutrient][cqa])
            x, y = remove_nandata(x,y, on_var='xy')
            if len(x)>4:
                corr_p = round(pearsonr(x,y)[0],3)
                corr_s = round(spearmanr(x,y)[0],3)
                # print(f'{nutrient} <> {cqa} ({len(x)}): {corr_p}, {corr_s}')
                ax[row_idx, col_idx].scatter(x, y, s=7)
                ax[row_idx, col_idx].set_xlabel(f'Specific rate of change of {nutrient} conc.')
                ax[row_idx, col_idx].set_ylabel(cqa)
                (xmin, xmax) = ax[row_idx][col_idx].get_xlim()
                (ymin, ymax) = ax[row_idx][col_idx].get_ylim()
                text = 'spearman R:' + str(corr_s) + '\n pearson R:' + str(corr_p)
                ax[row_idx][col_idx].text(xmin+(xmax-xmin)*0.4, ymin+(ymax-ymin)*0.8, text)
                # update results
                corr_cqa.append({'nutrient':nutrient, 'cqa':cqa, 'pearsonR': corr_p, 'spearmanR':corr_s, 'abs_avg':np.abs((corr_p+corr_s)/2)})
                corr_df_all.loc[cqa, nutrient] = corr_p
                
        except:
            print(f"Couldn't plot results for {nutrient}")
            
    # set overall plot title
    ymax = ax.flatten()[0].get_position().ymax
    plt.suptitle(f'{cqa} (Day 14) vs average specific rate of change for various nutrients', y=ymax*1.025, fontsize=24)
    plt.savefig(f'{figure_folder}scatter_cqa-{k}_vs_nutrientSpecificRate.png', bbox_inches='tight')
    plt.show()

    # update result
    corr_df = pd.DataFrame(corr_cqa)
    cqa_vs_rate_corr.update({cqa:corr_df})

    # identify nutrients with highest correlations
    corr_df = corr_df.sort_values(by='abs_avg', ascending=False)
    print(corr_df[['nutrient', 'pearsonR', 'spearmanR']].set_index('nutrient', drop=True))
    
# save results
corr_df_all.to_csv(f'{data_folder}corr_cqa_nutrient_specific_rate_of_change.csv')
    
#%% 

fig, ax = plt.subplots(figsize=(32,12))
arr = heatmap(corr_df_all.to_numpy(), c='viridis', ax=ax, cbar_kw={}, cbarlabel="", datamin=None, datamax=None, logscale_cmap=False, annotate=2, row_labels=list(corr_df_all.index), col_labels=corr_df_all.columns.tolist(), show_gridlines=False)

fig, ax = plt.subplots(figsize=(32,12))
arr = heatmap(np.abs(corr_df_all.to_numpy()), c='viridis', ax=ax, cbar_kw={}, cbarlabel="", datamin=None, datamax=None, logscale_cmap=False, annotate=2, row_labels=list(corr_df_all.index), col_labels=corr_df_all.columns.tolist(), show_gridlines=False)
