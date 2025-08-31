import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from variables import data_folder, var_dict_all, yvar_list_key, process_features
from nn_model_utils import MLPRegressor, CNNRegressor, train_model, evaluate_model, cross_validate_model_with_combined_predictions, get_dataloaders
from model_utils import fit_model_with_cv

#%% GET TIME SERIES DATASET

# set parameters
sampling_days = [7] # [0,3,5,7,11,14] # [0,3,5,7,11] # 
xvar_list_base_prefilt = var_dict_all['VCD, VIA, Titer, metabolites']
xvar_list_base_prefilt = [xvar for xvar in xvar_list_base_prefilt if xvar.find('Titer')==-1]
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
for j, xvar in enumerate(xvar_list_base_prefilt):
    for i, day in enumerate(sampling_days):
        colname = f'{xvar}_{day}'
        if colname in d:
            X_arr[:, i, j] = d.iloc[sample_idx_list][colname].to_numpy()
            feature_day_list.append(colname)
            if xvar not in feature_name_base_list:
                feature_name_base_list.append(xvar)
        else:
            feature_day_missed.append(colname)

# get missed features
# features_missed = [f for f in xvar_list_base_prefilt if f not in feature_name_base_list]
features_missed = list(set([f.split('_')[0] for f in feature_day_missed]))
print('features missed:', features_missed)
features_to_keep_idxs = [idx for idx, f in enumerate(xvar_list_base_prefilt) if f not in features_missed]
features_to_keep = [f for idx, f in enumerate(xvar_list_base_prefilt) if f not in features_missed]

# filter X_arr to drop features missed
X_arr = X_arr[:,:,features_to_keep_idxs]
print(X_arr.shape)
X_arr_ = X_arr.copy()

# get flatteed array with process parameters
X_arr_flattened_wPP = np.reshape(X_arr, (X_arr.shape[0], X_arr.shape[1]*X_arr.shape[2]))
X_arr_flattened_wPP = np.concatenate((X_arr_flattened_wPP, process_params), axis=1)
print('X_arr_flattened_wPP.shape', X_arr_flattened_wPP.shape)


#%% ALTERNATIVE (ORIGINAL) CODE -- same performance as code above

# Train the model
num_epochs = 500
model_type = 'mlp' # 'cnn' # 
get_trainval_plot = False
n_splits = 5
shuffle = True

# get data parameters
if model_type=='cnn':
    # get data loaders
    X_arr_in = X_arr.copy()
    num_samples = X_arr_in.shape[0]
    sequence_length = X_arr_in.shape[1]
    num_channels = X_arr_in.shape[2]
elif model_type=='mlp':
    X_arr_in = X_arr_flattened_wPP.copy()
    num_samples = X_arr_in.shape[0]
    sequence_length = X_arr_in.shape[1]
    num_channels = None


metrics = {}

# Initialize the model, loss function, and optimizer
for i, yvar in enumerate(yvar_list_key):
    print(yvar)
    # split data
    metrics_yvar = {'r2':[]}    
    trainval_dataloader_list, train_dataloader_list, val_dataloader_list, test_dataloader_list, y_trainval_scaled_list, y_test_scaled_list, trainval_index_list, test_index_list,  output_scaler = get_dataloaders(X_arr_in, y_arr[:,i].reshape(-1,1), n_splits, num_samples, sequence_length, num_channels, 1, shuffle=shuffle)
    
    # go through iterations
    for k, (trainval_loader, train_loader, val_loader, test_loader, y_trainval_scaled, y_test_scaled) in enumerate(zip(trainval_dataloader_list, train_dataloader_list, val_dataloader_list, test_dataloader_list, y_trainval_scaled_list, y_test_scaled_list)):
        # initialize model
        if model_type=='cnn':
            model = CNNRegressor(in_channels=num_channels, out_channels=[25,50], kernel_size=[3,5], fc_out=[72], dropout_p=0.2, num_outputs=1, sequence_length=sequence_length)
        elif model_type=='mlp':
            model = MLPRegressor(input_size=sequence_length, hidden_sizes=[24], dropout_p=0.1, num_outputs=1)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        # train model
        if get_trainval_plot:
            # train model
            train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
            # Plot the training and validation loss
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.show()
        else:
            train_losses, val_losses = train_model(model, trainval_loader, None, criterion, optimizer, num_epochs)
        
            # Evaluate on training sets
            trainval_predictions_scaled = evaluate_model(model, trainval_loader)
            # trainval_predictions = output_scaler.inverse_transform(trainval_predictions_scaled)  # Inverse transform predictions
            trainval_r2 = r2_score(trainval_predictions_scaled, y_trainval_scaled)
            # Evaluate test data
            test_predictions_scaled = evaluate_model(model, test_loader)
            # test_predictions = output_scaler.inverse_transform(test_predictions_scaled)  # Inverse transform predictions
            test_r2 = r2_score(test_predictions_scaled, y_test_scaled)
            
            metrics_yvar['r2'].append(test_r2)
            # print(k, f"R^2 (TRAINVAL): {trainval_r2:.3f}", f"; R^2 (TEST): {test_r2:.3f}")
    
    metrics[yvar] = metrics_yvar
    r2_avg = np.mean(np.array(metrics_yvar['r2']))
    print(f"R^2 avg (TEST):", round(r2_avg,3))
    
