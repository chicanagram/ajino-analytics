#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:50:07 2025

@author: charmainechia
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# define MLP model
class MLPRegressor(nn.Module):
    def __init__(self, input_size=50, hidden_sizes=[76, 32, 12], dropout_p=0.4, num_outputs=4):
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

        if val_loader is not None:
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
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actual = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            predictions.append(model(X_batch).cpu().numpy())
            actual.append(y_batch.cpu().numpy())
    
    predictions = np.vstack(predictions)
    actual = np.vstack(actual)
    r2 = r2_score(predictions, actual)
    
    return predictions
    

def cross_validate_model_with_combined_predictions(X_arr, y_arr, sequence_length, num_channels, model_type, model_class, n_splits=5, num_epochs=500, dropout=0.1, layers=[24], kernels=None, fc_out=[72]):
    """
    Performs k-fold cross-validation and returns average and overall R^2 score.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_r2_scores = []

    # Store combined predictions and ground truth
    all_preds = []
    all_true = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Scale input
        input_scaler = MinMaxScaler()
        if num_channels is not None:
            X = X_train.reshape(-1, num_channels)
            X = input_scaler.fit_transform(X)
            X_train_scaled = X.reshape(len(X_train), sequence_length, num_channels)

            X = X_test.reshape(-1, num_channels)
            X_test_scaled = input_scaler.transform(X).reshape(len(X_test), sequence_length, num_channels)
        else:
            X_train_scaled = input_scaler.fit_transform(X_train)
            X_test_scaled = input_scaler.transform(X_test)

        # Scale output
        output_scaler = MinMaxScaler()
        y_train_scaled = output_scaler.fit_transform(y_train)
        y_test_scaled = output_scaler.transform(y_test)

        # Convert to tensors
        if num_channels is not None:
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)
        else:
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

        # DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Model init
        if model_type == 'cnn':
            model = model_class(in_channels=num_channels, out_channels=layers, kernel_size=kernels,
                                fc_out=fc_out, dropout_p=dropout, num_outputs=1, sequence_length=sequence_length)
        elif model_type == 'mlp':
            model = model_class(input_size=sequence_length, hidden_sizes=layers,
                                dropout_p=dropout, num_outputs=1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        # Train
        train_model(model, train_loader, None, criterion, optimizer, num_epochs)

        # Evaluate
        test_preds_scaled = evaluate_model(model, test_loader)
        r2 = r2_score(test_preds_scaled, y_test_scaled)
        test_r2_scores.append(r2)

        # Store for overall R^2
        all_preds.extend(test_preds_scaled)
        all_true.extend(y_test_scaled)

        # print(f"Fold {fold + 1}/{n_splits}", f"R^2 (TEST): {r2:.3f}")

    # Compute overall R^2 across all test predictions
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_true = np.array(all_true).reshape(-1, 1)
    overall_r2 = r2_score(all_true, all_preds)
    avg_r2 = np.mean(test_r2_scores)

    # print(f"Average R^2 (TEST) across {n_splits} folds: {avg_r2:.3f}")
    print(f"Overall R^2 (combined predictions): {overall_r2:.3f}")
    return avg_r2, overall_r2



def get_dataloaders(X_arr, y_arr, n_splits, num_samples, sequence_length, num_channels, num_outputs, shuffle=True):

    # Scale the input data
    # INPUT
    input_scaler = MinMaxScaler()
    if num_channels is not None:
        X = X_arr.reshape(-1, num_channels)  # Reshape for scaling
        X = input_scaler.fit_transform(X)
        X = X.reshape(num_samples, sequence_length, num_channels)  
    else: 
        X = input_scaler.fit_transform(X_arr)
        
    # OUTPUT
    output_scaler = MinMaxScaler()
    y = output_scaler.fit_transform(y_arr)  # Scale the outputs
    
    # get kfold splits
    kFold=KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
    
    trainval_dataloader_list = []
    train_dataloader_list = []
    val_dataloader_list = []
    test_dataloader_list = []
    y_trainval_scaled_list = []
    y_test_scaled_list = []
    trainval_index_list = []
    test_index_list = []

    
    # cross-validation
    for trainval_index, test_index in kFold.split(X_arr):
        trainval_index_list.append(trainval_index)
        test_index_list.append(test_index)
        
        # Split data into training and validation sets
        X_trainval, X_test, y_trainval, y_test = X[trainval_index], X[test_index], y[trainval_index], y[test_index]
        y_trainval_scaled_list.append(y_trainval)
        y_test_scaled_list.append(y_test)
        
        # Further split trainval set into train and val
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        if num_channels is not None:
            X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32).permute(0, 2, 1)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # PyTorch: (batch, channels, length)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
        else: 
            X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
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
        
        # append to dataloader list
        trainval_dataloader_list.append(trainval_loader)
        train_dataloader_list.append(train_loader)
        val_dataloader_list.append(val_loader)
        test_dataloader_list.append(test_loader)
        
    
    return trainval_dataloader_list, train_dataloader_list, val_dataloader_list, test_dataloader_list, y_trainval_scaled_list, y_test_scaled_list, trainval_index_list, test_index_list, output_scaler
    
