#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:50:03 2024

@author: charmainechia
"""

import numpy as np
import pickle
import pandas as pd
from variables import data_folder, baseline_norm

data_folder = '../ajino-analytics-data/'


def get_full_ordered_seq(agg_tlist): 
    t_all = []
    for tlist in agg_tlist: 
        t_all += list(tlist)
    tseq = [float(t) for t in list(set(t_all))]
    return tseq

def get_idxs_matching_elements(lst1, lst2):
    arr1 = np.array(lst1)
    arr2 = np.array(lst2)
    res = np.where(np.isin(arr1, arr2))[0]
    return res
    
def align_ydata_by_timpoints(vardata_agg_t, vardata_agg_y):
    # get full ordered time sequence 
    tseq = get_full_ordered_seq(vardata_agg_t)
    # initialize y array to aggregate ydata
    vardata_agg_y_ = np.zeros((len(vardata_agg_y), len(tseq)))
    vardata_agg_y_[:] = np.nan
    # align y data into aggregating array
    for k, (tlist, ylist) in enumerate(zip(vardata_agg_t, vardata_agg_y)):
        matching_idxs = get_idxs_matching_elements(tseq, tlist)
        vardata_agg_y_[k,matching_idxs] = np.array(ylist)
    return np.array(tseq), vardata_agg_y_


#%%  Perform averaging on pickle file

dataset_name = 'DATA'
suffix = ''

# open pickle file
with open(f'{data_folder}{dataset_name}{suffix}.pkl', 'rb') as handle:
    datadict = pickle.load(handle)

# initialize dict for averaged data
datadict_averaged = {}

# get unique experiment conditions
explabel_to_datakey_dict = {}
for k in datadict:
    exp_label = datadict[k]['exp_label']
    if exp_label not in explabel_to_datakey_dict:
        explabel_to_datakey_dict.update({exp_label:[]})
    explabel_to_datakey_dict[exp_label].append(k)

# get unique variables
var_list = list(datadict[k].keys())

# iterate through exp_list and average data replicates
for i, (exp_label, key_list) in enumerate(explabel_to_datakey_dict.items()):
    print(exp_label)
    # initialize nested dict
    datadict_averaged[exp_label] = {}
    
    # iterate through variables
    for var in var_list:   
        print(var, end=' ')
        vardata_agg_t = []
        vardata_agg_y = []
        for key in key_list: 
            # aggregate time series data
            if var in datadict[key] and isinstance(datadict[key][var], dict):
                vardata_agg_t.append(datadict[key][var]['t'])
                vardata_agg_y.append(datadict[key][var]['y'])
            # aggregate non-time series data
            elif var in datadict[key] and isinstance(datadict[key][var], float):
                vardata_agg_y.append(datadict[key][var])
        
        # average data
        if len(vardata_agg_y) > 0:
            if len(vardata_agg_t) > 0:
                vardata_avg_t, vardata_avg_y = align_ydata_by_timpoints(vardata_agg_t, vardata_agg_y)
                vardata_avg_y = np.nanmean(np.array(vardata_avg_y),axis=0)
                datadict_averaged[exp_label][var] = {'t':vardata_avg_t, 'y': vardata_avg_y}
            else: 
                vardata_avg_y = np.mean(np.array(vardata_agg_y))
                datadict_averaged[exp_label][var] = float(vardata_avg_y)
        elif var in datadict[key]:
            if var == 'n':
                datadict_averaged[exp_label][var] = len(key_list)
            else:
                datadict_averaged[exp_label][var] = datadict[key][var]
    print()

# save averaged data
pkl_fname_avg = f'{dataset_name}{suffix}_avg.pkl'
with open(f'{data_folder}{pkl_fname_avg}', 'wb') as handle:
    pickle.dump(datadict_averaged, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% Perform averaging on csv file
dataset_name = 'DATA'
dataset_suffix = ''
df_raw = pd.read_csv(f'{data_folder}{dataset_name}{dataset_suffix}.csv', index_col=0)
cqa_list = ['Titer (mg/L)', 'mannosylation', 'fucosylation', 'galactosylation']
use_baseline_norm_for_runs_wo_baseline = False

# get unique id exp#_basal_feed_pH_DO_feed%
exp_feed_procparams_id = []
exp_num_list = []
for i in range(len(df_raw)):
    l = df_raw.iloc[i][['exp_label' , 'Basal medium', 'Feed medium', 'pH', 'DO', 'feed %']].tolist()
    lstr = f'{str(l[0])[0]}_{l[1][-1]}_{l[2][-1]}_{l[3]}_{int(l[4])}_{int(l[5])}'
    exp_feed_procparams_id.append(lstr)
    exp_num_list.append(str(l[0])[0])
df_raw['id'] = exp_feed_procparams_id
df_raw['exp_num'] = exp_num_list

unique_exp_num =  list(set(exp_num_list))
unique_exp_num.sort()
unique_id =  list(set(exp_feed_procparams_id))
unique_id.sort()
print(unique_exp_num)
print(len(unique_id), unique_id)

# get columns with numeric data
col_w_txt = ['exp_label', 'Basal medium', 'Feed medium', 'feed day', 'id', 'exp_num']
col_to_avg = [c for c in df_raw.columns.tolist() if c not in col_w_txt]

# average replicates
df_avg = []
for exp_id in unique_id: 
    df_raw_id = df_raw[df_raw['id']==exp_id]
    df_avg_id = df_raw_id.iloc[0][['exp_num', 'id', 'exp_label', 'Basal medium', 'Feed medium']].to_dict()
    df_avg_id.update(df_raw_id[col_to_avg].mean(axis=0, skipna=True).to_dict())
    df_avg.append(df_avg_id)
df_avg = pd.DataFrame(df_avg)
df_avg.to_csv(f'{data_folder}{dataset_name}{dataset_suffix}_avg.csv')

#######################################
# normalize averaged data by CTRL exp #
#######################################

print('Normalizing averaged data...')
cqa_cols_dict = {cqa: [c for c in df_avg.columns.tolist() if c.find(cqa)>-1] for cqa in cqa_list}
print(cqa_cols_dict)

count = 0
for exp_num in unique_exp_num:
    df_avg_expnum = df_avg[df_avg['exp_num']==exp_num]
    ref_id = f'{exp_num}_A_a_7.0_40_6'
    df_avg_expnum_REF = df_avg_expnum[df_avg_expnum['id']==ref_id]
    print(f'{len(df_avg_expnum_REF)} reference experiments found in run {exp_num}')
    
    if len(df_avg_expnum_REF)>0:
        CQAnorm = {cqa: round(df_avg_expnum_REF.iloc[0][f'{cqa}_14'],4) for cqa in cqa_list}
    else: 
        CQAnorm = baseline_norm.copy()
    print(CQAnorm)
    df_avgnorm_expnum = df_avg_expnum.copy()
    print(CQAnorm)
    # get columns containing cqa, normalize them by rerence D14 values
    if len(df_avg_expnum_REF)>0 or use_baseline_norm_for_runs_wo_baseline:
        for cqa in cqa_list:
            cqa_cols = cqa_cols_dict[cqa]
            df_avgnorm_expnum.loc[:, cqa_cols] = df_avgnorm_expnum.loc[:, cqa_cols]/CQAnorm[cqa]
        # update overall dataframe
        if count == 0:
            df_avgnorm = df_avgnorm_expnum.copy()
        else:
            df_avgnorm = pd.concat([df_avgnorm, df_avgnorm_expnum], axis=0)
        count += 1
print('df_avgnorm.shape:', df_avgnorm.shape)
df_avgnorm.to_csv(f'{data_folder}{dataset_name}{dataset_suffix}_avgnorm.csv')

##########################################
# normalize un-averaged data by CTRL exp #
##########################################
print('Normalizing raw data...')
cqa_cols_dict = {cqa: [c for c in df_raw.columns.tolist() if c.find(cqa)>-1] for cqa in cqa_list}
print(cqa_cols_dict)

count = 0
for exp_num in unique_exp_num:
    df_avg_expnum = df_avg[df_avg['exp_num']==exp_num]
    ref_id = f'{exp_num}_A_a_7.0_40_6'
    df_avg_expnum_REF = df_avg_expnum[df_avg_expnum['id']==ref_id]
    print(f'{len(df_avg_expnum_REF)} reference experiments found in run {exp_num}')
    
    if len(df_avg_expnum_REF)>0:
        CQAnorm = {cqa: round(df_avg_expnum_REF.iloc[0][f'{cqa}_14'],4) for cqa in cqa_list}
    else: 
        CQAnorm = baseline_norm.copy()
    print(CQAnorm)
    # get columns containing cqa, normalize them by rerence D14 values
    if len(df_avg_expnum_REF)>0 or use_baseline_norm_for_runs_wo_baseline:
        df_rawnorm_expnum = df_raw[df_raw['exp_num']==exp_num]
        for cqa in cqa_list:
            cqa_cols = cqa_cols_dict[cqa]
            df_rawnorm_expnum.loc[:, cqa_cols] = df_rawnorm_expnum.loc[:, cqa_cols]/CQAnorm[cqa]
        # update overall dataframe
        if count == 0:
            df_rawnorm = df_rawnorm_expnum.copy()
        else:
            df_rawnorm = pd.concat([df_rawnorm, df_rawnorm_expnum], axis=0)
        count += 1
print('df_rawnorm.shape:', df_rawnorm.shape)
df_rawnorm.to_csv(f'{data_folder}{dataset_name}{dataset_suffix}_norm.csv')
        
#%% get dataset with target features
from variables import xvar_list_dict_prefilt, yvar_list_key
from utils import get_XYdataset

suffix = '_norm' # '_avgnorm'
remove_cols_w_nan_thres = 0.07  # 0.1 #  0.25 #

for Y_featureset_idx in [0]:
    for X_featureset_idx in [1]:
        print('X_featureset_idx:', X_featureset_idx, ';  Y_featureset_idx:', Y_featureset_idx)
        XY_fname = f'X{X_featureset_idx}Y{Y_featureset_idx}{suffix}'
        XYarr_dict, XY_df, nan_df = get_XYdataset(df_rawnorm, X_featureset_idx, Y_featureset_idx, xvar_list_dict_prefilt, {Y_featureset_idx:yvar_list_key},
                                          csv_fname=f'{XY_fname}.csv', pkl_fname=f'{XY_fname}.pkl', shuffle_data=True, remove_cols_w_nan_thres=remove_cols_w_nan_thres)
        print()
        
        for i, f in enumerate(XYarr_dict['xvar_list']):
            print(i, f)       
    

        

                     
    
                
        