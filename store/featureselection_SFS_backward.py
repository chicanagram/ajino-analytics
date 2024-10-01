#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:50:33 2024

@author: charmainechia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:03:36 2024

@author: charmainechia
"""
import pandas as pd
from variables import yvar_list_key
from get_datasets import data_folder, get_XYdata_for_featureset
from feature_selection_utils import run_sfs_forward, run_sfs_backward, run_rfe

# Perform Recursive Feature Elimination
featureset_list =  [(2,0)]
models_to_evaluate = ['plsr'] #  ['randomforest', 'plsr'] # 


# load dataset
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    dataset_suffix = '' #'_avg'
    
    Y, X_init, Xscaled_init, yvar_list, xvar_list_init = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
    res, featureset_opt = run_sfs_backward(Y, X_init, yvar_list_key, xvar_list_init, models_to_evaluate, dataset_name, dataset_suffix, kfold_suffix='', xvar_idx_end=None, featureset_suffix='_sfs-backward')
    

