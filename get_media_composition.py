#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:56:08 2024

@author: charmainechia
"""

import pandas as pd
import numpy as np
import pickle
from get_raw_data import append_to_datadict, data_folder

#%% read sheet: 'Media composition'

rawdata_dict = {
    0: {'fname': '240402_20RP06-19_data for Astar_4conditions_v2.xlsx', 'skiprows':1, 'usecols':'A:F', 'cqa_startcol':1, },
    1: {'fname': '240704_22DX05-12_media combination_AJI-Astar_v1.xlsx', 'skiprows':2, 'usecols':'B:J', 'cqa_startcol':1, },
    2: {'fname': '240710_22DX05-12_process combination_AJI-Astar_v2.xlsx', 'skiprows':2, 'usecols':'B:D', 'cqa_startcol':1, },
    }

sheet_name = 'Media composition'

for dataset_num, dataset_meta in rawdata_dict.items():
    fname, skiprows, usecols, cqa_startcol = dataset_meta['fname'], dataset_meta['skiprows'], dataset_meta['usecols'], dataset_meta['cqa_startcol']
    fpath = f'{data_folder}{fname}'
    df = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols)
    # remove empty rows
    df = df.dropna(axis=0, how='all')
    # remove rows with redundant headers
    col0 = df.iloc[:, 0].tolist()
    idx_to_keep = [i for i, el in enumerate(col0) if (el.find('unit')==-1 and el.find('AA')==-1 and el.find('VT')==-1 and el.find('MT')==-1 and el.find('Sugar')==-1)]
    df = df.iloc[idx_to_keep,:]
    # reset index
    df = df.reset_index(drop=True)
    col0 = df.iloc[:, 0].tolist()
    # replace non float values with np.nan
    df = df[df.map(np.isreal)]
    df.iloc[:,0] = col0
    
    if dataset_num == 0: 
        df_merge = df.copy()
    if dataset_num > 0:
        df_merge = df_merge.merge(df, on='unit: mM', suffixes=[f'_{dataset_num-1}', f'_{dataset_num}'])

col0 = df_merge.iloc[:, 0].tolist()
colnames = df_merge.columns.tolist()[1:]
colnames_base = []
for c in colnames: 
    if c.find('_')>-1: 
        c = c[:c.find('_')]
    colnames_base.append(c)
colnames_base = list(set(colnames_base))
colnames_base.sort()

#%%  
# initialize df for averaged variables
arr = np.zeros((len(df_merge), len(colnames_base)+1))
arr[:] = np.nan
df_avg = pd.DataFrame(arr, columns=['var']+colnames_base)
df_avg['var'] = col0
for colbase in colnames_base: 
    print(colbase)
    col_list = [c for c in colnames if c.find(colbase)>-1]
    df_col_avg = df_merge[col_list].mean(axis=1)
    print(df_col_avg)
    df_avg[colbase] = df_col_avg.tolist()

# set var col as index
df_avg = df_avg.set_index('var')

# remove rows that are all zeros
df_avg = df_avg.dropna(axis=0, how = 'all')

# translate dataframe so that nutrient components are the columns, and media labels are the index
df_avg = df_avg.transpose()

# save dataframe to csv
df_avg.to_csv(f'{data_folder}MediaComposition_avg.csv')
