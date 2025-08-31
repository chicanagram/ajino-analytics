#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:17:29 2025

@author: charmainechia
"""

import pandas as pd

# Load the Excel file
data_folder = '../GSK-BTI/'
data_subfolder = 'DoE'
doe_fname = '8factors_4levels_54exps.xlsx'
doe_fpath = data_folder + data_subfolder + '/' + doe_fname
df = pd.read_excel(doe_fpath)

# Define the DoE variables of interest
doe_vars = ['Temperature', 'pH', 'DO', 'Nutrient1', 'Nutrient2', 'Nutrient3', 'Nutrient4', 'Nutrient5']

# Subset the dataframe to include only those columns
doe_df = df[doe_vars]

# Calculate the correlation matrix (Pearson by default)
correlation_matrix = doe_df.corr().round(4)

# Save the correlation matrix to a CSV file
correlation_csv_path = data_folder + data_subfolder + '/Correlations_' + doe_fname.replace('.xlsx', '.csv')
correlation_matrix.to_csv(correlation_csv_path)
