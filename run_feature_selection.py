

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:03:36 2024

@author: charmainechia
"""
from variables import yvar_list_key
from get_datasets import data_folder, get_XYdata_for_featureset
from feature_selection_utils import run_sfs_forward, run_sfs_backward, run_rfe

#%% 
# Perform Recursive Feature Elimination
featureset_list =  [(1,0)]
models_to_evaluate = ['randomforest']
feature_selection_method = 'rfe' # 'sfs-forward' # 
yvar_list = yvar_list_key
cv = 8 # None
xvar_idx_end = None

# load dataset
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_suffix = '' #'_avg'
    Y, X_init, Xscaled_init, _, xvar_list_init = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    

    if feature_selection_method=='sfs-forward':
        res, featureset_bymodeltype = run_sfs_forward(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=xvar_idx_end, featureset_suffix='_'+feature_selection_method, cv=cv)
    
    elif feature_selection_method=='sfs-backward':
        res, featureset_bymodeltype = run_sfs_backward(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=xvar_idx_end, featureset_suffix='_'+feature_selection_method, cv=cv)

    elif feature_selection_method=='rfe':
        res, featureset_bymodeltype = run_rfe(Y, X_init, yvar_list, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=xvar_idx_end, featureset_suffix='_'+feature_selection_method, cv=cv)
    

print(featureset_bymodeltype)