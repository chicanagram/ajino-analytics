#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:50:07 2025

@author: charmainechia
"""
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from variables import data_folder, var_dict_all, yvar_list_key, process_features, model_params, nutrients_list_all
from nn_model_utils import MLPRegressor, CNNRegressor, train_model, evaluate_model, cross_validate_model_with_combined_predictions, get_dataloaders
from model_utils import fit_model_with_cv

#%% GET TIME SERIES DATASET

# set parameters
sampling_days = [11] # [0,3,5,7,11] # [0,3,5,7,11,14] # 
xvar_list_base_prefilt = var_dict_all['VCD, VIA, Titer, metabolites'] + nutrients_list_all
xvar_list_base_prefilt = [xvar for xvar in xvar_list_base_prefilt if xvar not in ['Titer (mg/L)', 'Adenosine', 'Thiamin', 'D-glucose', 'Uridine']]
yvar_list  = yvar_list_key

# load full data
d = pd.read_csv(data_folder + 'DATA.csv', index_col=0)
sample_idx_list = list(range(8,len(d))) # get all except first 8 samples
y_arr = d.iloc[sample_idx_list][yvar_list].to_numpy()

# initialize numpy array store data
X_arr = np.zeros((len(sample_idx_list), len(sampling_days), len(xvar_list_base_prefilt))) # samples, sequence_length, channels 
X_arr[:] = np.nan
process_params = d.iloc[sample_idx_list][process_features].to_numpy()

# fill array 
feature_day_list = []
feature_name_base_list = []
feature_day_missed = []
col_idx_list = []
features_missed = []
features_nan = []
for j, xvar in enumerate(xvar_list_base_prefilt):
    for i, day in enumerate(sampling_days):
        colname = f'{xvar}_{day}'
        if colname in d:
            val_vect = d.iloc[sample_idx_list][colname].to_numpy()
            X_arr[:, i, j] = val_vect
            feature_day_list.append(colname)
            if any(np.isnan(val_vect)):
                nan_idxs = np.where(np.isnan(val_vect))[0]
                features_nan.append((colname, nan_idxs))
            if xvar not in feature_name_base_list:
                feature_name_base_list.append(xvar)
        else:
            feature_day_missed.append(colname)
print('feature_day_missed:', feature_day_missed)
print('features_nan:', features_nan)

# get missed features
# features_missed = [f for f in xvar_list_base_prefilt if f not in feature_name_base_list]
features_missed = list(set([f.split('_')[0] for f in feature_day_missed]))
print('features missed:', features_missed)
features_to_keep_idxs = [idx for idx, f in enumerate(xvar_list_base_prefilt) if f not in features_missed]
features_to_keep = [f for idx, f in enumerate(xvar_list_base_prefilt) if f not in features_missed]

# filter X_arr to drop features missed
X_arr = X_arr[:,:,features_to_keep_idxs]
print('X_arr.shape', X_arr.shape)

# get flatteed array with process parameters
X_arr_flattened = np.reshape(X_arr, (X_arr.shape[0], X_arr.shape[1]*X_arr.shape[2]))
X_arr_flattened_wPP = np.concatenate((X_arr_flattened, process_params), axis=1)
print('X_arr_flattened.shape', X_arr_flattened.shape)
print('X_arr_flattened_wPP.shape', X_arr_flattened_wPP.shape)

#%% Evaluate NN models
model_type = 'mlp'


# get data parameters
if model_type=='cnn':
    # get data loaders
    X_arr_in = X_arr.copy()
    num_samples = X_arr_in.shape[0]
    sequence_length = X_arr_in.shape[1]
    num_channels = X_arr_in.shape[2]
    dropout = 0.2 # 0.2
    layers = [50, 50] # [24,48]
    kernels = [3,5]
    fc_out = [72]
    model_class = CNNRegressor
elif model_type=='mlp':
    # X_arr_in = X_arr_flattened.copy()
    X_arr_in = X_arr_flattened_wPP.copy()
    num_samples = X_arr_in.shape[0]
    sequence_length = X_arr_in.shape[1]
    num_channels = None
    dropout = 0.1 # 0.2 # 
    layers = [36,24] # [24] # [100,40,20] # 
    kernels = None
    fc_out = None
    model_class = MLPRegressor


print(model_type)
for i, yvar in enumerate(yvar_list_key):
    print(f"=== {yvar} ===")
    y = y_arr[:, i].reshape(-1, 1)
    print('X_arr_in shape:', X_arr_in.shape)
    
    avg_r2 = cross_validate_model_with_combined_predictions(
        X_arr=X_arr_in,
        y_arr=y,
        sequence_length=sequence_length,
        num_channels=num_channels,
        model_type=model_type,
        model_class=model_class,
        n_splits=5,
        num_epochs=500,
        dropout=dropout,
        layers=layers,
        kernels=kernels,
        fc_out=fc_out
    )

#%% Evaluate sklearn models
models_to_eval_list = ['randomforest'] # ['randomforest', 'xgb', 'plsr', 'lasso', 'mlp'] # ['randomforest']# 
cv = 5
score_type_list = ['r2'] # ['r2', 'SpearmanR','mae', 'rmse']

for model_type in models_to_eval_list:
    print(model_type)

    for i, yvar in enumerate(yvar_list_key):
        print(f"=== {yvar} ===")
        y = y_arr[:, i]
        model_list = model_params['sampling'][model_type][yvar]
        # model_list, metrics = fit_model_with_cv(X_arr_flattened, y, yvar, model_list, score_type_list=score_type_list, plot_predictions=False, scale_data=False, cv=cv)
        model_list, metrics = fit_model_with_cv(X_arr_flattened_wPP, y, yvar, model_list, score_type_list=score_type_list, plot_predictions=False, scale_data=False, cv=cv)

        
        
        
