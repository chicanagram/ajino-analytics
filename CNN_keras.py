#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:51:27 2025

@author: charmainechia
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

def build_1d_multichannel_cnn(input_shape, dropout_rate=0.5):
    """
    Builds a simple 1D CNN with dropout for multi-channel time-series input.

    Args:
        input_shape (tuple): Shape of one sample, e.g. (seq_length, num_channels).
        dropout_rate (float): Probability of dropping units in the Dropout layer.
    
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    model = Sequential()
    
    # 1D Convolution layer 1
    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu',
                     input_shape=input_shape))
    # Dropout
    model.add(Dropout(rate=dropout_rate))
    
    # 1D Convolution layer 2
    model.add(Conv1D(filters=16,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    # Optional max pooling to reduce sequence length
    model.add(MaxPooling1D(pool_size=2))
    
    # Flatten to feed into dense layers
    model.add(Flatten())
    
    # Dense layer(s) for classification or regression
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Example: binary classification
    
    # Compile the model (using binary crossentropy as an example)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Example usage:
    seq_length = 100   # e.g., 100 time steps
    num_channels = 3   # e.g., 3 features (multi-channel) per time step
    
    # Build the model
    model = build_1d_multichannel_cnn(input_shape=(seq_length, num_channels), 
                                      dropout_rate=0.5)
    
    # Print a summary
    model.summary()
    
    # Generate some dummy data
    import numpy as np
    X_dummy = np.random.rand(10, seq_length, num_channels)  # batch of 10 samples
    y_dummy = np.random.randint(0, 2, size=(10,))           # binary targets

    # Train on dummy data (just as an example)
    model.fit(X_dummy, y_dummy, epochs=3, batch_size=2)