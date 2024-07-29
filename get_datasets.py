#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:36:39 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import pickle
from variables import var_dict_all, overall_glyco_cqas, sort_list
from sklearn.preprocessing import scale, minmax_scale
data_folder = '../ajino-analytics-data/'

# set dataset name
dataset_name = 'DATA_avg'
pkl_fname = f'{dataset_name}.pkl'

# open pickle file
with open(f'{data_folder}{pkl_fname}', 'rb') as handle:
    datadict = pickle.load(handle)

#%% GET FULL DATASET    
# initialize dataframe
cols = [
        ['exp_label'] + var_dict_all['inputs'],
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

# get dataframe
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

# remove columns that are all zeros
d = d.loc[:,(d!= 0).any(axis=0)]

# remove columns that are all NaNs
d = d.dropna(axis=1, how='all')

# save dataframe 
csv_fpath = f'{data_folder}{dataset_name}.csv'
d.to_csv(csv_fpath)

#%% GET DATASETS OF X (DAY 0) AND Y (DAY 14) ONLY

# bioreactor_CQAs = ['VCD (E6 cells/mL)', 'Viability (%)', 'Titer (mg/L)'] 
media_inputs = ['Basal medium', 'Feed medium']
process_inputs = ['DO', 'pH', 'feed %']
nutrient_inputs = ['Glucose (g/L)'] + var_dict_all['AA'] + var_dict_all['VT'] + var_dict_all['MT'] + var_dict_all['Nuc, Amine']
cqa_outputs = ['Titer (mg/L)', 'mannosylation', 'fucosylation', 'galactosylation','G0','G0F','G1','G1Fa','G1Fb','G2','G2F','Other']

# generate variable names for all days
cqa_dict_byday = {day: [f'{cqa}_{day}' for cqa in cqa_outputs] for day in range(15)}
nutrient_dict_byday = {day: [f'{nutrient}_{day}' for nutrient in nutrient_inputs] for day in range(15)}

# get initial x variable list (day 0)
xvar_list_all = [var for var in nutrient_dict_byday[0] + process_inputs]
xvar_list_all = sort_list(xvar_list_all[:-len(process_inputs)]) + process_inputs
print(len(xvar_list_all), xvar_list_all)

# get y variable list (day 14)
yvar_list = [var for var in cqa_dict_byday[14]]  
print(len(yvar_list), yvar_list)

#%% Get X and Y datasets 

# get XY data - drop rows which contain any NA values      
xy_var_list = [var for var in xvar_list_all+yvar_list if var in d]
XY_df = d[xy_var_list]
XY_df = XY_df.dropna()

# remove variables with only zeros
XY_df = XY_df.loc[:,(XY_df!= 0).any(axis=0)]
xvar_list = [xvar for xvar in xvar_list_all if xvar in XY_df]
XY_df.to_csv(f'{data_folder}X0_Y14_dataset.csv')
print(len(xvar_list), xvar_list)
print(len(yvar_list), yvar_list)

# get X & Y arrays
X = XY_df[xvar_list].to_numpy()
Y = XY_df[yvar_list].to_numpy()

# randomize samples (shuffle datasets)
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
    0: {'Y':Y, 'X': X, 'Xscaled':Xscaled, 'yvar_list': yvar_list, 'xvar_list': xvar_list},
}

# save numpy datasets
with open(f'{data_folder}XYarrdict.pkl', 'wb') as f:
    pickle.dump(XYarr_dict, f)
        



