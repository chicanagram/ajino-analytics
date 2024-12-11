#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:36:39 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import pickle
from variables import data_folder, figure_folder, var_dict_all, overall_glyco_cqas, sort_list, yvar_list_key
from sklearn.preprocessing import scale
from utils import get_XYdataset, get_XYdata_for_featureset

# %% Get  data dict from pickle

# open pickle file for main dataset
dataset_name = 'DATA'
dataset_suffix = ''
# dataset_suffix = '_avg'
pkl_fname = f'{dataset_name}{dataset_suffix}.pkl'
with open(f'{data_folder}{pkl_fname}', 'rb') as handle:
    datadict = pickle.load(handle)

# open csv file for media composition
mediacomposition_fname = 'MediaComposition_avg.csv'
media_df = pd.read_csv(f'{data_folder}{mediacomposition_fname}', index_col=0)

# %% GET FULL DATASET
# initialize dataframe
avg_media_composition_vals = False
cols = [
    ['exp_label'] + var_dict_all['inputs'],
    [f'{var}_basal' for var in var_dict_all['media_components']],
    [f'{var}_feed' for var in var_dict_all['media_components']],
    [f'{var}_{day}' for var in var_dict_all['VCD, VIA, Titer, metabolites']
        for day in range(15)],
    [f'{var}_{day}' for var in var_dict_all['Glyco'] for day in range(15)],
    [f'{var}_{day}' for var in var_dict_all['AA'] for day in range(15)],
    [f'{var}_{day}' for var in var_dict_all['VT'] for day in range(15)],
    [f'{var}_{day}' for var in var_dict_all['Nuc, Amine']
        for day in range(15)],
    [f'{var}_{day}' for var in var_dict_all['MT'] for day in range(15)],
    [f'{var}_NSRC' for var in var_dict_all['AA']],
    [f'{var}_NSRC' for var in var_dict_all['VT']],
    [f'{var}_NSRC' for var in var_dict_all['Nuc, Amine']],
    [f'{var}_NSRC' for var in var_dict_all['MT']],
]
cols = sum(cols, [])
n = len(list(datadict.keys()))
arr = np.zeros((n, len(cols)))
arr[:] = None
d = pd.DataFrame(arr, columns=cols)
d['exp_label'] = d['exp_label'].astype(str)

# update dataframe contents from main sampling dataset
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
                    if i == 0 and isinstance(val, str):
                        d[var] = d[var].astype(str)
                    d.loc[i, var] = val

# add columns for Mannosylation, Fucosylation, Galactosylation
for var, glyco_list in overall_glyco_cqas.items():
    for day in range(15):
        var_colname = f'{var}_{day}'
        glyco_colnames_list = [f'{xvar}_{day}' for xvar in glyco_list]
        d[var_colname] = d[glyco_colnames_list].sum(axis=1)


# add columns for Basal and Feed media composition
if avg_media_composition_vals:
    # update dataframe from averaged media composition csv
    for i in range(len(d)):
        basal = d.loc[i, 'exp_label']
        basal = d.loc[i, 'Basal medium']
        basal_composition = media_df.loc[basal, :].to_dict()
        for k, v in basal_composition.items():
            d.loc[i, f'{k}_basal'] = v
        feed = d.loc[i, 'Feed medium']
        feed_composition = media_df.loc[feed, :].to_dict()
        for k, v in feed_composition.items():
            d.loc[i, f'{k}_feed'] = v
else:
    # update dataframe contents from actual media composition values
    for i, exp_label in enumerate(list(datadict.keys())):
        print(i, end=' ')
        data_exp = datadict[exp_label]
        d.loc[i, 'exp_label'] = exp_label
        # find all k, v pairs where k ends with '_basal' or '_feed'
        for k, v in data_exp.items():
            if (k.find('_basal') > -1 or k.find('_feed') > -1) and k in cols:
                d.loc[i, k] = v
    # go through dataframe to update any missing values
    for i in list(d.index):
        for col in [f'{var}_basal' for var in var_dict_all['media_components']] + [f'{var}_feed' for var in var_dict_all['media_components']]:
            if np.isnan(d.loc[i, col]):
                media_type = {
                    'basal': d.loc[i, 'Basal medium'], 'feed': d.loc[i, 'Feed medium']}
                [nutrient, basal_or_feed] = col.split('_')
                v_avg = media_df.loc[media_type[basal_or_feed], nutrient]
                d.loc[i, col] = v_avg
                print(i, nutrient, basal_or_feed, v_avg, end='\t')

# add columns for average Nutrient Specific Rate of Change (NSRCavg)
for i, exp_label in enumerate(list(datadict.keys())):
    print(i, end=' ')
    data_exp = datadict[exp_label]
    d.loc[i, 'exp_label'] = exp_label
    for vargrp in ['AA', 'VT', 'Nuc, Amine', 'MT']:
        for var in var_dict_all[vargrp]:
            # update dataframe value
            if var in data_exp:
                if 'rate_avg' in data_exp[var]:
                    d.loc[i, f'{var}_NSRC'] = data_exp[var]['rate_avg']


# remove columns that are all zeros
d = d.loc[:, (d != 0).any(axis=0)]
# remove columns that are all NaNs
d = d.dropna(axis=1, how='all')

# save dataframe
csv_fpath = f'{data_folder}{dataset_name}{dataset_suffix}.csv'
d.to_csv(csv_fpath)

# %% GET X Y VARIABLE SETS

# segmenting the variables
media_inputs = ['Basal medium', 'Feed medium']
media_composition_inputs = [f'{var}_basal' for var in var_dict_all['media_components']] + [
    f'{var}_feed' for var in var_dict_all['media_components']]
process_inputs = ['DO', 'pH', 'feed %', 'feed vol']
nutrient_inputs = ['Glucose (g/L)'] + var_dict_all['AA'] + \
    var_dict_all['VT'] + var_dict_all['MT'] + var_dict_all['Nuc, Amine']
cqa_outputs = ['Titer (mg/L)', 'mannosylation', 'fucosylation', 'galactosylation',
               'G0', 'G0F', 'G1', 'G1Fa', 'G1Fb', 'G2', 'G2F', 'Other']
# generate day-by-day variable names for all time-series data
cqa_dict_byday = {day: [f'{cqa}_{day}' for cqa in cqa_outputs]
                  for day in range(15)}
nutrient_dict_byday = {
    day: [f'{nutrient}_{day}' for nutrient in nutrient_inputs] for day in range(15)}
# get rate variables
NSRC_avg = []
for nutrient_type in ['AA', 'VT', 'MT', 'Nuc, Amine']:
    NSRC_avg += [f'{nutrient}_NSRC' for nutrient in var_dict_all[nutrient_type]]

# initialize dicts for x and y variable lists
yvar_list_dict = {}
xvar_list_dict = {}

# Y
# Y0: D14 CQAs
yvar_list = [var for var in cqa_dict_byday[14]]
yvar_list_dict.update({0: yvar_list})
print('Y0:', len(yvar_list), yvar_list)
print()

# X
# X0: Basal and feed media composition inputs + process inputs
xvar_list_prefilt = media_composition_inputs + process_inputs
xvar_list_dict.update({0: xvar_list_prefilt})
print('X0:', len(xvar_list_prefilt))
print("'", end='')
print(*xvar_list_prefilt, sep="'\n'", end="'")
print('\n')


# X1: Basal and feed media composition inputs + process inputs (w/o feed %)
xvar_list_prefilt = media_composition_inputs + process_inputs
xvar_list_prefilt.remove('feed %')
xvar_list_dict.update({1: xvar_list_prefilt})
print('X1:', len(xvar_list_prefilt))
print("'", end='')
print(*xvar_list_prefilt, sep="'\n'", end="'")
print('\n')

# X2: Basal and feed media composition inputs (w/o feed vol)
xvar_list_prefilt = media_composition_inputs + process_inputs
xvar_list_prefilt.remove('feed vol')
xvar_list_dict.update({2: xvar_list_prefilt})
print('X2:', len(xvar_list_prefilt))
print("'", end='')
print(*xvar_list_prefilt, sep="'\n'", end="'")
print('\n')

# X3: DAY 0 nutrient inputs + process inputs
xvar_list_prefilt = sort_list(
    [var for var in nutrient_dict_byday[0]]) + process_inputs
xvar_list_prefilt.remove('feed %')
xvar_list_dict.update({3: xvar_list_prefilt})
print('X3:', len(xvar_list_prefilt))
print("'", end='')
print(*xvar_list_prefilt, sep="'\n'", end="'")
print('\n')

# X4: Average Nutrient Specific Rate of Change + process inputs
xvar_list_prefilt = NSRC_avg + process_inputs
xvar_list_prefilt.remove('feed %')
xvar_list_dict.update({4: xvar_list_prefilt})
print('X4:', len(xvar_list_prefilt))
print("'", end='')
print(*xvar_list_prefilt, sep="'\n'", end="'")
print('\n')

# X5: Average Nutrient Specific Rate of Change + process inputs
MC_shortlist = ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Ca',
                'Pyridoxine', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Zn', 'Tyr']
xvar_list_prefilt = [f'{MC}_basal' for MC in MC_shortlist] + \
    [f'{MC}_feed' for MC in MC_shortlist] + process_inputs
xvar_list_prefilt.remove('feed %')
xvar_list_dict.update({5: xvar_list_prefilt})
print('X5:', len(xvar_list_prefilt))
print("'", end='')
print(*xvar_list_prefilt, sep="'\n'", end="'")
print('\n')

# X6: Average Nutrient Specific Rate of Change + process inputs
MC_shortlist = ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Fe', 'Mn', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Tyr', 'Glu', ]
xvar_list_prefilt = [f'{MC}_basal' for MC in MC_shortlist] + \
    [f'{MC}_feed' for MC in MC_shortlist] + process_inputs
xvar_list_prefilt.remove('feed %')
xvar_list_dict.update({6: xvar_list_prefilt})
print('X6:', len(xvar_list_prefilt))
print("'", end='')
print(*xvar_list_prefilt, sep="'\n'", end="'")
print('\n')


# %% Get X and Y datasets
suffix = ''
remove_cols_w_nan_thres = 0.07  # 0.1 #  0.25 #

for Y_featureset_idx in [0]:
    for X_featureset_idx in [1]:
        print('X_featureset_idx:', X_featureset_idx,
              ';  Y_featureset_idx:', Y_featureset_idx)
        XY_fname = f'X{X_featureset_idx}Y{Y_featureset_idx}{suffix}'
        XYarr_dict, XY_df, nan_df = get_XYdataset(d, X_featureset_idx, Y_featureset_idx, xvar_list_dict, yvar_list_dict,
                                          csv_fname=f'{XY_fname}.csv', pkl_fname=f'{XY_fname}.pkl', shuffle_data=True, remove_cols_w_nan_thres=remove_cols_w_nan_thres)
        print()

# %%

# print(XY_df.iloc[XY_df['Titer (mg/L)_14'].argmax()][['exp_label', 'Basal medium', 'Feed medium', 'DO', 'pH', 'feed vol', 'Titer (mg/L)_14', 'mannosylation_14']])
# print(XY_df.iloc[XY_df['mannosylation_14'].argmin()][['exp_label', 'Basal medium', 'Feed medium', 'DO', 'pH', 'feed vol', 'Titer (mg/L)_14', 'mannosylation_14']])
