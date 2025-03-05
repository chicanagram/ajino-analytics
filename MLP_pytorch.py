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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from variables import data_folder, var_dict_all, nutrients_list_all, yvar_list_key, xvar_list_dict
from utils import get_XYdata_for_featureset

#%% GET TIME SERIES DATASET

# set parameters
xvar_list = xvar_list_dict[1] # + nutrients_list_all
yvar_list  = yvar_list_key

X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = '' #  '_norm_with_val' # '_norm' # '_avgnorm' # 
featureset_suffix = '' # '_ajinovalidation' # 
models_to_eval_list = ['randomforest'] # ['plsr']# ['randomforest','plsr', 'lasso'] #  
dataset_suffix = '' #  '_norm_with_val' # '_norm' # '_avgnorm' # 
yvar_list = yvar_list_key
dataset_name_wsuffix = dataset_name + dataset_suffix
y_arr, X, Xscaled, _, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)

    
#%% 

# Define the MLP model
class MLPRegressor(nn.Module):
    def __init__(self, input_size=76, hidden_sizes=[76, 32, 12], dropout_p=0.4, num_outputs=4):
        super(MLPRegressor, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input Layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden Layers
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Dropout Layer
        self.dropout = nn.Dropout(dropout_p)
        
        # Output Layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_outputs)
        
    def forward(self, x):
        # convert x to float
        x = x.float()
        
        # Flatten input for MLP
        x = torch.flatten(x, start_dim=1)
        
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Output Layer (no activation for regression)
        x = self.output_layer(x)
        return x


# Custom Weighted MSE Loss Class
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        """
        :param weights: Tensor of shape (num_outputs,) specifying the weight for each output dimension
        """
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
    def forward(self, outputs, targets):
        """
        :param outputs: Model predictions of shape (batch_size, num_outputs)
        :param targets: Ground truth labels of shape (batch_size, num_outputs)
        """
        # Ensure weights are on the same device as the outputs
        weights = self.weights.to(outputs.device)
        
        # Calculate squared errors
        squared_errors = (outputs - targets) ** 2
        
        # Apply weights to each output dimension
        weighted_errors = squared_errors * weights
        
        # Return the mean of the weighted errors
        return torch.mean(weighted_errors)
    

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

        # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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
        # print('y_pred', np.round(predictions[:,i],2))"
        # print('y', np.round(true_values[:,i],2))
        
    return predictions, r2_list


#%%
hyperparams = {
    0: {'lr':0.001, 'num_epochs':1000, 'hidden_sizes':[76, 32, 12], 'dropout_p':0.3},
    1: {'lr':0.001, 'num_epochs':1000, 'hidden_sizes':[76, 32, 12], 'dropout_p':0.3}, 
    2: {'lr':0.001, 'num_epochs':1000, 'hidden_sizes':[76, 32, 12], 'dropout_p':0.4}, 
    }


for i, yvar in enumerate(yvar_list_key):
    num_samples = X.shape[0]
    input_size = X.shape[1]
    y = y_arr[:,i].reshape(-1,1)
    num_outputs = y.shape[1]
    
    # Scale the input data
    # INPUT
    input_scaler = MinMaxScaler()
    X = input_scaler.fit_transform(X)
    # OUTPUT
    output_scaler = MinMaxScaler()
    y = output_scaler.fit_transform(y)  # Scale the outputs
    
    # Split data into training and validation sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Further split trainval set into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)
    
    # Convert data to PyTorch tensors
    X_trainval_tensor = torch.tensor(X_trainval)
    X_train_tensor = torch.tensor(X_train) # PyTorch: (batch, channels, length)
    X_val_tensor = torch.tensor(X_val)
    X_test_tensor = torch.tensor(X_test)
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
    
    # Train the model
    num_epochs = 800
    
    # Initialize the model, loss function, and optimizer
    model = MLPRegressor(input_size=76, hidden_sizes=[36, 12], dropout_p=0.2, num_outputs=num_outputs)
    # Define weights for each output dimension
    output_weights = [1] # [1, 2, 2, 1]  # Example: 4 output dimensions with different weights
    # Instantiate the custom loss function
    criterion = WeightedMSELoss(weights=output_weights)
    # criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # TRAIN MODEL
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
    
    print(yvar)
    print(f"R^2 Score on Train-Val Set: {trainval_r2[0]:.4f}")
    print(f"R^2 Score on Test Set: {test_r2[0]:.4f}")

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