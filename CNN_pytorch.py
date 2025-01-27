#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:50:07 2025

@author: charmainechia
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from variables import data_folder, var_dict_all, nutrients_list_all, yvar_list_key

#%% GET TIME SERIES DATASET

# set parameters
sampling_days = [0,3,5,7,11,14]# [0,3,5,7,9,11,14]
xvar_list_base_prefilt = var_dict_all['VCD, VIA, Titer, metabolites'] # + nutrients_list_all
yvar_list  = yvar_list_key

# load full data
d = pd.read_csv(data_folder + 'DATA.csv', index_col=0)
sample_idx_list = list(range(8,len(d))) # get all except first 8 samples
y_arr = d.iloc[sample_idx_list][yvar_list].to_numpy()

# initialize numpy array store data
X_arr = np.zeros((len(sample_idx_list), len(sampling_days), len(xvar_list_base_prefilt))) # samples, sequence_length, channels 
X_arr[:] = np.nan

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

#%% remove XVAR FEATURES with above a threshold (e.g 0.1) of NaNs
features_to_remove = []
feature_list = []
features_idxs = []
for j, xvar in enumerate(features_to_keep):
    feature_2Dslice = X_arr[:, :, j]
    isnan_feature_2Dslice = np.isnan(feature_2Dslice)
    num_nan = len(np.argwhere(isnan_feature_2Dslice))
    frac_nan = round(num_nan/feature_2Dslice.size,3)
    # if there are some nans, examine fractions by Sample and Day
    if frac_nan > 0: 
        print(j, xvar, frac_nan)
        print('Frac NaN for each Day (summed over samples):', np.round(np.sum(isnan_feature_2Dslice*1, axis=0)/isnan_feature_2Dslice.shape[0],2))
        print('Frac NaN for each Sample (summed over days):', np.round(np.sum(isnan_feature_2Dslice*1, axis=1)/isnan_feature_2Dslice.shape[1],2))
        features_to_remove.append(xvar)
    else: 
        feature_list.append(xvar)
        features_idxs.append(i)
        
# filter X_arr to drop features missed
X_arr = X_arr[:,:,features_idxs]
print(X_arr.shape)
print(feature_list)
    
#%% 
num_samples = X_arr.shape[0]
sequence_length = X_arr.shape[1]
num_channels = X_arr.shape[2]
num_outputs = len(yvar_list)

# Scale the input data
# INPUT
input_scaler = MinMaxScaler()
X = X_arr.reshape(-1, num_channels)  # Reshape for scaling
X = input_scaler.fit_transform(X)
X = X.reshape(num_samples, sequence_length, num_channels)  # Reshape back
# OUTPUT
output_scaler = MinMaxScaler()
y = output_scaler.fit_transform(y_arr)  # Scale the outputs

# Split data into training and validation sets
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Further split trainval set into train and val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32).permute(0, 2, 1)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # PyTorch: (batch, channels, length)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders for training and validation
trainval_dataset = TensorDataset(X_trainval_tensor, y_trainval_tensor)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# create dataloaders
trainval_loader = DataLoader(trainval_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the CNN model
class CNNRegressor(nn.Module):
    def __init__(self, in_channels=50, out_channels=[25,25], kernel_size=[3,3], fc_out=[72], dropout_p=0.2, num_outputs=4, sequence_length=6):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels=out_channels[0], kernel_size=kernel_size[0], padding=int((kernel_size[0]-1)/2))
        self.bn1 = nn.BatchNorm1d(out_channels[0])
        self.conv2 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=kernel_size[1], padding=int((kernel_size[1]-1)/2))
        self.bn2 = nn.BatchNorm1d(out_channels[1])
        self.dropout = nn.Dropout(dropout_p)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(out_channels[1]*sequence_length, fc_out[0])
        # self.fc1 = nn.Linear(out_channels[0]*sequence_length, fc_out[0])
        self.fc2 = nn.Linear(fc_out[0], num_outputs)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training and validation loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


# Predict outputs and calculate R^2 score
def evaluate_model(model, data_loader, true_values, scaler):
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            preds = model(X_batch).cpu().numpy()
            predictions.append(preds)
    
    predictions = np.vstack(predictions)
    predictions = scaler.inverse_transform(predictions)  # Inverse transform predictions
    true_values = scaler.inverse_transform(true_values)  # Inverse transform true values
    r2_list = []
    for i in range(predictions.shape[1]):
        r2 = round(r2_score(true_values[:,i], predictions[:,i]),2)
        r2_list.append(r2)
        # print('y_pred', np.round(predictions[:,i],2))
        # print('y', np.round(true_values[:,i],2))
        
    print(r2_list)
    return predictions, r2_list


#%%
hyperparams = {
    0: {'num_channels':50, 'lr':0.001, 'num_epochs':1000, 'out_channels':[25,12], 'kernel_size':[3,3], 'fc_out':[72], 'dropout_p':0.3}, # 0.82, 0.55, 0.53, 0.65
    1: {'num_channels':7, 'lr':0.001, 'num_epochs':1000, 'out_channels':[25,25], 'kernel_size':[3,3], 'fc_out':[72], 'dropout_p':0.3}, # 0.82, 0.55, 0.53, 0.65
    2: {'num_channels':7, 'lr':0.001, 'num_epochs':1000, 'out_channels':[25,50], 'kernel_size':[3,5], 'fc_out':[72], 'dropout_p':0.4}, # 0.82, 0.55, 0.53, 0.65
    }

# Train the model
num_epochs = 1000

# Initialize the model, loss function, and optimizer
model = CNNRegressor(in_channels=num_channels, out_channels=[25,50], kernel_size=[3,5], fc_out=[72], dropout_p=0.2, num_outputs=len(yvar_list), sequence_length=sequence_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Plot the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate on training and validation sets
trainval_predictions, trainval_r2 = evaluate_model(model, trainval_loader, y_trainval, output_scaler)
test_predictions, test_r2 = evaluate_model(model, test_loader, y_test, output_scaler)

for i, yvar in enumerate(yvar_list): 
    print(yvar)
    print(f"R^2 Score on Train-Val Set: {trainval_r2[i]:.4f}")
    print(f"R^2 Score on Test Set: {test_r2[i]:.4f}")

#%% 

# Titer (mg/L)_14
# R^2 Score on Train-Val Set: 0.9500
# R^2 Score on Test Set: 0.8700
# mannosylation_14
# R^2 Score on Train-Val Set: 0.9400
# R^2 Score on Test Set: 0.5600
# fucosylation_14
# R^2 Score on Train-Val Set: 0.9300
# R^2 Score on Test Set: 0.5100
# galactosylation_14
# R^2 Score on Train-Val Set: 0.9400
# R^2 Score on Test Set: 0.7400

# Titer (mg/L)_14
# R^2 Score on Train-Val Set: 0.9500
# R^2 Score on Test Set: 0.8800
# mannosylation_14
# R^2 Score on Train-Val Set: 0.9500
# R^2 Score on Test Set: 0.6200
# fucosylation_14
# R^2 Score on Train-Val Set: 0.9500
# R^2 Score on Test Set: 0.6100
# galactosylation_14
# R^2 Score on Train-Val Set: 0.9300
# R^2 Score on Test Set: 0.6800