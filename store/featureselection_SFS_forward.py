#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:03:57 2024

@author: charmainechia
"""

from variables import yvar_list_key
from get_datasets import data_folder, get_XYdata_for_featureset
from feature_selection_utils import run_sfs_forward
    
# Perform Sequential Feature Selection
featureset_list =  [(2,0)]
models_to_evaluate = ['plsr', 'randomforest'] # ['randomforest'] # ['plsr'] # 
feature_selection_method = 'SFS-forward'

# load dataset
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_suffix = '' #'_avg'
    Y, X_init, Xscaled_init, yvar_list, xvar_list_init = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    res, featureset_opt = run_sfs_forward(Y, X_init, yvar_list_key, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, xvar_idx_end=None)
    print(featureset_opt)
    