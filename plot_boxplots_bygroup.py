#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:50:05 2024

@author: charmainechia
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from plot_utils import data_folder, figure_folder, plot_data_panel
from variables import nutrients_calculated_list

    
#%% get data

# set dataset name
dataset_name = 'DATA'
dataset_suffix =  '' 
dataset_name_wsuffix = f'{dataset_name}{dataset_suffix}'

df = pd.read_csv(data_folder+dataset_name_wsuffix+'.csv', index_col=0)
df = df.sort_values(by='exp_label')
# plot_suffixes={'_sample_adj':'--', '_imputed':'-'}

var_to_plot = [
    'VCD (E6 cells/mL)',
    'Titer (mg/L)',
    'mannosylation',
    'fucosylation',
    'galactosylation'
    ]

#%% Effect of Basal A/B/C/D 

primary_effect_to_vary = 'Basal medium'
secondary_effect_to_vary = 'Feed medium'
xvar_to_fix = {'pH':7.0, 'DO':40, 'feed %':6}
suptitle = f'Effect of {primary_effect_to_vary} -- Various {secondary_effect_to_vary} (filter: {xvar_to_fix})'

# filter by xvar_to_fix
df_filt = df.copy()
for xvar, xval in xvar_to_fix.items():
    df_filt = df_filt[df_filt[xvar]==xval].copy()
    
# get all primary effect groups
primary_effect_group_list = list(set(df_filt[primary_effect_to_vary].tolist()))
primary_effect_group_list.sort()
num_primary_effect_groups = len(primary_effect_group_list)
print(primary_effect_group_list)

# get all secondary effect groups
secondary_effect_group_list_ = list(set(df_filt[secondary_effect_to_vary].tolist()))
secondary_effect_group_list = []
for secondary_effect in secondary_effect_group_list_:
    df_filt_sec = df_filt[df_filt[secondary_effect_to_vary]==secondary_effect].copy()
    if len(list(set(df_filt_sec[primary_effect_to_vary].tolist()))) > 1: 
        secondary_effect_group_list.append(secondary_effect)
secondary_effect_group_list.sort()
num_secondary_effect_groups = len(secondary_effect_group_list)
print(num_secondary_effect_groups, secondary_effect_group_list)

# initialize plot 
fig, ax = plt.subplots(num_secondary_effect_groups,len(var_to_plot), figsize=(num_primary_effect_groups*1.6*len(var_to_plot), num_secondary_effect_groups*3.6))

# iterate throguh yvar
for col_idx, yvar in enumerate(var_to_plot): 
    yvar_col = f'{yvar}_14'
    ymin = df_filt[yvar_col].min()*0.9
    ymax = df_filt[yvar_col].max()*1.1
    
    # iterate through secondary effects
    for row_idx, secondary_effect in enumerate(secondary_effect_group_list):
        df_filt_byseceffect = df_filt[df_filt[secondary_effect_to_vary]==secondary_effect].copy()
        df_filt_byseceffect.boxplot(column=yvar_col, by=primary_effect_to_vary, ax=ax[row_idx, col_idx], fontsize=12, grid=False, rot=0)
        ax[row_idx, col_idx].set_ylabel(yvar, fontsize=14)
        ax[row_idx, col_idx].set_xlabel('', fontsize=12)
        ax[row_idx, col_idx].set_title(secondary_effect, fontsize=12)
        ax[row_idx, col_idx].set_ylim([ymin, ymax])
    plt.suptitle(suptitle, fontsize=24, y=0.93)
plt.show()
plt.savefig(f"{figure_folder}Boxplots_{primary_effect_to_vary.replace(' ','-')}_{secondary_effect_to_vary.replace(' ','-')}.png", bbox_inches='tight')
    
#%% Effect of feed a/b/c/d/e/f

primary_effect_to_vary = 'Feed medium'
secondary_effect_to_vary = 'Basal medium'
xvar_to_fix = {'pH':7.0, 'DO':40, 'feed %':6}
suptitle = f'Effect of {primary_effect_to_vary} -- Various {secondary_effect_to_vary} (filter: {xvar_to_fix})'

# filter by xvar_to_fix
df_filt = df.copy()
for xvar, xval in xvar_to_fix.items():
    df_filt = df_filt[df_filt[xvar]==xval].copy()
    
# get all primary effect groups
primary_effect_group_list = list(set(df_filt[primary_effect_to_vary].tolist()))
primary_effect_group_list.sort()
num_primary_effect_groups = len(primary_effect_group_list)
print(primary_effect_group_list)

# get all secondary effect groups
secondary_effect_group_list_ = list(set(df_filt[secondary_effect_to_vary].tolist()))
secondary_effect_group_list = []
for secondary_effect in secondary_effect_group_list_:
    df_filt_sec = df_filt[df_filt[secondary_effect_to_vary]==secondary_effect].copy()
    if len(list(set(df_filt_sec[primary_effect_to_vary].tolist()))) > 1: 
        secondary_effect_group_list.append(secondary_effect)
secondary_effect_group_list.sort()
num_secondary_effect_groups = len(secondary_effect_group_list)
print(num_secondary_effect_groups, secondary_effect_group_list)

# initialize plot 
fig, ax = plt.subplots(num_secondary_effect_groups,len(var_to_plot), figsize=(num_primary_effect_groups*1.6*len(var_to_plot), num_secondary_effect_groups*4))

# iterate throguh yvar
for col_idx, yvar in enumerate(var_to_plot): 
    yvar_col = f'{yvar}_14'
    ymin = df_filt[yvar_col].min()*0.9
    ymax = df_filt[yvar_col].max()*1.1
    
    # iterate through secondary effects
    for row_idx, secondary_effect in enumerate(secondary_effect_group_list):
        df_filt_byseceffect = df_filt[df_filt[secondary_effect_to_vary]==secondary_effect].copy()
        df_filt_byseceffect.boxplot(column=yvar_col, by=primary_effect_to_vary, ax=ax[row_idx, col_idx], fontsize=12, grid=False, rot=0)
        ax[row_idx, col_idx].set_ylabel(yvar, fontsize=14)
        ax[row_idx, col_idx].set_xlabel('', fontsize=12)
        ax[row_idx, col_idx].set_title(secondary_effect, fontsize=12)
        ax[row_idx, col_idx].set_ylim([ymin, ymax])
    plt.suptitle(suptitle, fontsize=24, y=0.93)
plt.show()
plt.savefig(f"{figure_folder}Boxplots_{primary_effect_to_vary.replace(' ','-')}_{secondary_effect_to_vary.replace(' ','-')}.png", bbox_inches='tight')

#%% Effect of Process conditions: pH

primary_effect_to_vary = 'pH'
secondary_effect_to_vary = ['feed %','DO']
all_effects_list = [primary_effect_to_vary]+secondary_effect_to_vary
xvar_to_fix = {'Basal medium':'Basal-A', 'Feed medium':'Feed-a'}
suptitle = f'Effect of {primary_effect_to_vary} -- Various {secondary_effect_to_vary} (filter: {xvar_to_fix})'

# filter by xvar_to_fix
df_filt = df.copy()
for xvar, xval in xvar_to_fix.items():
    df_filt = df_filt[df_filt[xvar]==xval].copy()
df_filt = df_filt.sort_values(by=all_effects_list)
  
# get all primary effect groups
primary_effect_group_list = list(set(df_filt[primary_effect_to_vary].tolist()))
primary_effect_group_list.sort()
num_primary_effect_groups = len(primary_effect_group_list)
print(primary_effect_group_list)

# get all secondary effect groups
secondary_effect_group_list__ = list(df_filt[secondary_effect_to_vary].to_records(index=False))
# deduplicate groups
secondary_effect_group_list_ = []
for grp in secondary_effect_group_list__:
    if grp not in secondary_effect_group_list_:
        secondary_effect_group_list_.append(grp)
# filter to remove effects with only one primary effect group
secondary_effect_group_list = []
for secondary_effects in secondary_effect_group_list_:
    df_filt_byseceffect = df_filt.copy()
    for eff_idx, secondary_effect in enumerate(secondary_effects):
        df_filt_byseceffect = df_filt_byseceffect[df_filt_byseceffect[secondary_effect_to_vary[eff_idx]]==secondary_effect].copy()
    if len(list(set(df_filt_byseceffect[primary_effect_to_vary].tolist()))) > 1: 
        secondary_effect_group_list.append(secondary_effects)
num_secondary_effect_groups = len(secondary_effect_group_list)
print(num_secondary_effect_groups, secondary_effect_group_list)

# initialize plot 
fig, ax = plt.subplots(num_secondary_effect_groups,len(var_to_plot), figsize=(num_primary_effect_groups*2*len(var_to_plot), num_secondary_effect_groups*4))

# iterate throguh yvar
for col_idx, yvar in enumerate(var_to_plot): 
    yvar_col = f'{yvar}_14'
    ymin = df_filt[yvar_col].min()*0.9
    ymax = df_filt[yvar_col].max()*1.1
    
    # iterate through secondary effects
    for row_idx, secondary_effects in enumerate(secondary_effect_group_list):
        df_filt_byseceffect = df_filt.copy()
        for eff_idx, secondary_effect in enumerate(secondary_effects):
            df_filt_byseceffect = df_filt_byseceffect[df_filt_byseceffect[secondary_effect_to_vary[eff_idx]]==secondary_effect].copy()
        df_filt_byseceffect.boxplot(column=yvar_col, by=primary_effect_to_vary, ax=ax[row_idx, col_idx], fontsize=12, grid=False, rot=0)
        ax[row_idx, col_idx].set_ylabel(yvar, fontsize=14)
        ax[row_idx, col_idx].set_xlabel('', fontsize=12)
        ax[row_idx, col_idx].set_title(f'{secondary_effect_to_vary}: {secondary_effects}', fontsize=12)
        ax[row_idx, col_idx].set_ylim([ymin, ymax])
    plt.suptitle(suptitle, fontsize=24, y=0.93)
plt.show()
plt.savefig(f"{figure_folder}Boxplots_{primary_effect_to_vary.replace(' ','-')}_{'-'.join(secondary_effect_to_vary)}.png", bbox_inches='tight')

#%% Effect of Process conditions: DO

primary_effect_to_vary = 'DO'
secondary_effect_to_vary = ['feed %','pH']
all_effects_list = [primary_effect_to_vary]+secondary_effect_to_vary
xvar_to_fix = {'Basal medium':'Basal-A', 'Feed medium':'Feed-a'}
suptitle = f'Effect of {primary_effect_to_vary} -- Various {secondary_effect_to_vary} (filter: {xvar_to_fix})'

# filter by xvar_to_fix
df_filt = df.copy()
for xvar, xval in xvar_to_fix.items():
    df_filt = df_filt[df_filt[xvar]==xval].copy()
df_filt = df_filt.sort_values(by=all_effects_list)
  
# get all primary effect groups
primary_effect_group_list = list(set(df_filt[primary_effect_to_vary].tolist()))
primary_effect_group_list.sort()
num_primary_effect_groups = len(primary_effect_group_list)
print(primary_effect_group_list)

# get all secondary effect groups
secondary_effect_group_list__ = list(df_filt[secondary_effect_to_vary].to_records(index=False))
# deduplicate groups
secondary_effect_group_list_ = []
for grp in secondary_effect_group_list__:
    if grp not in secondary_effect_group_list_:
        secondary_effect_group_list_.append(grp)
# filter to remove effects with only one primary effect group
secondary_effect_group_list = []
for secondary_effects in secondary_effect_group_list_:
    df_filt_byseceffect = df_filt.copy()
    for eff_idx, secondary_effect in enumerate(secondary_effects):
        df_filt_byseceffect = df_filt_byseceffect[df_filt_byseceffect[secondary_effect_to_vary[eff_idx]]==secondary_effect].copy()
    if len(list(set(df_filt_byseceffect[primary_effect_to_vary].tolist()))) > 1: 
        secondary_effect_group_list.append(secondary_effects)
num_secondary_effect_groups = len(secondary_effect_group_list)
print(num_secondary_effect_groups, secondary_effect_group_list)

# initialize plot 
fig, ax = plt.subplots(num_secondary_effect_groups,len(var_to_plot), figsize=(num_primary_effect_groups*2*len(var_to_plot), num_secondary_effect_groups*4))

# iterate throguh yvar
for col_idx, yvar in enumerate(var_to_plot): 
    yvar_col = f'{yvar}_14'
    ymin = df_filt[yvar_col].min()*0.9
    ymax = df_filt[yvar_col].max()*1.1
    
    # iterate through secondary effects
    for row_idx, secondary_effects in enumerate(secondary_effect_group_list):
        df_filt_byseceffect = df_filt.copy()
        for eff_idx, secondary_effect in enumerate(secondary_effects):
            df_filt_byseceffect = df_filt_byseceffect[df_filt_byseceffect[secondary_effect_to_vary[eff_idx]]==secondary_effect].copy()
        df_filt_byseceffect.boxplot(column=yvar_col, by=primary_effect_to_vary, ax=ax[row_idx, col_idx], fontsize=12, grid=False, rot=0)
        ax[row_idx, col_idx].set_ylabel(yvar, fontsize=14)
        ax[row_idx, col_idx].set_xlabel('', fontsize=12)
        ax[row_idx, col_idx].set_title(f'{secondary_effect_to_vary}: {secondary_effects}', fontsize=12)
        ax[row_idx, col_idx].set_ylim([ymin, ymax])
    plt.suptitle(suptitle, fontsize=24, y=0.93)
plt.show()
plt.savefig(f"{figure_folder}Boxplots_{primary_effect_to_vary.replace(' ','-')}_{'-'.join(secondary_effect_to_vary)}.png", bbox_inches='tight')


#%% Effect of Process conditions: feed %

primary_effect_to_vary = 'feed %'
secondary_effect_to_vary = ['pH','DO']
all_effects_list = [primary_effect_to_vary]+secondary_effect_to_vary
xvar_to_fix = {'Basal medium':'Basal-A', 'Feed medium':'Feed-a'}
suptitle = f'Effect of {primary_effect_to_vary} -- Various {secondary_effect_to_vary} (filter: {xvar_to_fix})'

# filter by xvar_to_fix
df_filt = df.copy()
for xvar, xval in xvar_to_fix.items():
    df_filt = df_filt[df_filt[xvar]==xval].copy()
df_filt = df_filt.sort_values(by=all_effects_list)
  
# get all primary effect groups
primary_effect_group_list = list(set(df_filt[primary_effect_to_vary].tolist()))
primary_effect_group_list.sort()
num_primary_effect_groups = len(primary_effect_group_list)
print(primary_effect_group_list)

# get all secondary effect groups
secondary_effect_group_list__ = list(df_filt[secondary_effect_to_vary].to_records(index=False))
# deduplicate groups
secondary_effect_group_list_ = []
for grp in secondary_effect_group_list__:
    if grp not in secondary_effect_group_list_:
        secondary_effect_group_list_.append(grp)
# filter to remove effects with only one primary effect group
secondary_effect_group_list = []
for secondary_effects in secondary_effect_group_list_:
    df_filt_byseceffect = df_filt.copy()
    for eff_idx, secondary_effect in enumerate(secondary_effects):
        df_filt_byseceffect = df_filt_byseceffect[df_filt_byseceffect[secondary_effect_to_vary[eff_idx]]==secondary_effect].copy()
    if len(list(set(df_filt_byseceffect[primary_effect_to_vary].tolist()))) > 1: 
        secondary_effect_group_list.append(secondary_effects)
num_secondary_effect_groups = len(secondary_effect_group_list)
print(num_secondary_effect_groups, secondary_effect_group_list)

# initialize plot 
fig, ax = plt.subplots(num_secondary_effect_groups,len(var_to_plot), figsize=(num_primary_effect_groups*2*len(var_to_plot), num_secondary_effect_groups*4))

# iterate throguh yvar
for col_idx, yvar in enumerate(var_to_plot): 
    yvar_col = f'{yvar}_14'
    ymin = df_filt[yvar_col].min()*0.9
    ymax = df_filt[yvar_col].max()*1.1
    
    # iterate through secondary effects
    for row_idx, secondary_effects in enumerate(secondary_effect_group_list):
        df_filt_byseceffect = df_filt.copy()
        for eff_idx, secondary_effect in enumerate(secondary_effects):
            df_filt_byseceffect = df_filt_byseceffect[df_filt_byseceffect[secondary_effect_to_vary[eff_idx]]==secondary_effect].copy()
        df_filt_byseceffect.boxplot(column=yvar_col, by=primary_effect_to_vary, ax=ax[row_idx, col_idx], fontsize=12, grid=False, rot=0)
        ax[row_idx, col_idx].set_ylabel(yvar, fontsize=14)
        ax[row_idx, col_idx].set_xlabel('', fontsize=12)
        ax[row_idx, col_idx].set_title(f'{secondary_effect_to_vary}: {secondary_effects}', fontsize=12)
        ax[row_idx, col_idx].set_ylim([ymin, ymax])
    plt.suptitle(suptitle, fontsize=24, y=0.93)
plt.show()
plt.savefig(f"{figure_folder}Boxplots_{primary_effect_to_vary.replace(' ','-')}_{'-'.join(secondary_effect_to_vary)}.png", bbox_inches='tight')
