#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:50:03 2024

@author: charmainechia
"""

import numpy as np
import pickle

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
    return tseq, vardata_agg_y_


#%% 
# set dataset name
# dataset_num = 2
# dataset_name = f'dataset{dataset_num}'
dataset_name = 'DATA_ALL'
pkl_fname = f'{dataset_name}.pkl'

# open pickle file
with open(f'{data_folder}{pkl_fname}', 'rb') as handle:
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
for exp_label, key_list in explabel_to_datakey_dict.items():
    # initialize nested dict
    datadict_averaged[exp_label] = {}
    
    # iterate through variables
    for var in var_list:   
        print(var)
        vardata_agg_t = []
        vardata_agg_y = []
        for key in key_list: 
            if var in datadict[key] and isinstance(datadict[key][var], dict):
                    vardata_agg_t.append(datadict[key][var]['t'])
                    vardata_agg_y.append(datadict[key][var]['y'])
        
        # average data
        if len(vardata_agg_y) > 0:
            vardata_avg_t, vardata_avg_y = align_ydata_by_timpoints(vardata_agg_t, vardata_agg_y)
            vardata_avg_y = np.nanmean(np.array(vardata_avg_y),axis=0)
            datadict_averaged[exp_label][var] = {'t':vardata_avg_t, 'y': vardata_avg_y}
        elif var in datadict[key]:
            if var == 'n':
                datadict_averaged[exp_label][var] = len(key_list)
            else:
                datadict_averaged[exp_label][var] = datadict[key][var]

# save averaged data
pkl_fname_avg = f'{dataset_name}_avg.pkl'
with open(f'{data_folder}{pkl_fname_avg}', 'wb') as handle:
    pickle.dump(datadict_averaged, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
    
                
        