#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:58:31 2024

@author: charmainechia
"""

# imports
import pandas as pd
import numpy as np
import pickle

def xlsx_to_dict(
        fpath,
        sheet_name,
        skiprows,
        usecols,
        cqa_startcol=5,
        get_inputs=False
        ):
    """
    Description: Extracts CQA data from xlsx sheet into dict
    
    Parameters
    ----------
    fpath : str
        full pathname for xlsx file to be read.
    sheet_name : str
        name of xlsx sheet to be read.
    skiprows : int
        number of rows to ignore at the start of xlsx file.
    usecols : str (e.g. 'B:BD')
        range of columns to use.
    cqa_startcol : int, optional
        index of 1st column in table containing CQA variable data. The default is 5.
    get_inputs : bool, optional
        whether or not to extract input metadata from columns in table before <cqa_startcol>. The default is False.

    Returns
    -------
    d_in : dict
        dict with a nested dicts containing input metadata for each individual experiment.
    d_out : dict
        dict with nested dicts containing CQA data for each individual experiment.
    var_dict : dict
        dict with nested dicts containing list of variables from each xlsx sheet, e.g. input metadata, bioreactor CQAs, AA, MT.

    """
    df = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols)
    var_dict = {}
    
    df_in = df.iloc[:, :cqa_startcol]
    df_in.columns = df_in.iloc[0]
    df_in = df_in.drop(df_in.index[0])
    if get_inputs:
        # get inputs
        df_in = df_in.rename(columns={'feed manner (%, day)': 'feed %', 'pH central':'pH', 'feeding %': 'feed %', 'feeding day':'feed day'})
        if 'feed day' not in df_in.columns.tolist():
            df_in = df_in.rename(columns={df_in.columns[3]: 'feed day'})
        if 'DO' not in df_in:
            df_in['DO'] = 40
        if 'pH' not in df_in:
            df_in['pH'] = 7.0
        df_in = df_in[['Basal medium', 'Feed medium', 'DO', 'pH', 'feed %', 'feed day', 'n']] # reorder columns
        d_in = df_in.to_dict('index')
        var_dict.update({'inputs':df_in.columns.tolist()})
    else: 
        d_in = None
    
    # get output CQAs
    df_out = df.iloc[:, cqa_startcol:]
    out_colnames = list(df_out.columns)
    day_labels = np.array(list(df_out.iloc[0,:]))
    cqa_idx = [i for i, cqa in enumerate(out_colnames) if cqa.find('Unnamed')==-1] 
    cqa_names = [cqa for cqa in out_colnames if cqa.find('Unnamed')==-1]
    var_dict.update({sheet_name:cqa_names})
    
    # iterate through each input dataset
    d_out = {}
    for i in list(df_in.index):
        d_out[i] = {}
        # iterate through CQAs
        for j in range(len(cqa_idx)):
            cqa_name = cqa_names[j]
            col_startidx = cqa_idx[j]
            if j+1<len(cqa_idx):
                col_endidx = cqa_idx[j+1]
                days = day_labels[col_startidx:col_endidx]
                lst = df_out.iloc[i, col_startidx:col_endidx].to_list()
            else: 
                days = day_labels[col_startidx:]
                lst = df_out.iloc[i, col_startidx:].to_list()
            d_out[i][cqa_name] = {
                't': days,
                'y': np.array([float(l) if (isinstance(l,float) or isinstance(l,np.int64)) else np.nan for l in lst])
                }
            
    return d_in, d_out, var_dict
        

def append_to_datadict(d, d_in, d_out):
    """
    Description: Append data from given sheet to general data dict

    Parameters
    ----------
    d : dict
        overall data dict for aggregating the sheet data into, comprising nested dicts for each individual experiment
    d_in : dict
        data dict containing input metadata, comprising nested dicts for each individual experiment
    d_out : dict
        data dict containing CQA data, comprising nested dicts for each individual experiment

    Returns
    -------
    d : dict
        overall data dict with new data appended.

    """
    for i in list(d_out.keys()):
        if i not in d: 
            d[i] = {}
        if d_in is not None:
            d[i].update(d_in[i])
        d[i].update(d_out[i])
    return d

#%%
data_folder = '../ajino-analytics-data/'
rawdata_dict = {
    0: {'fname': '240402_20RP06-19_data for Astar_4conditions_v2.xlsx', 'skiprows':[2,2,2,2,2], 'usecols':['B:BZ', 'B:DQ', 'B:BD', 'B:BD', 'B:AY'], 'cqa_startcol':[5,5,5,5,5], },
    1: {'fname': '240704_22DX05-12_media combination_AJI-Astar_v1.xlsx', 'skiprows':[2,2,2,2,2,2], 'usecols':['B:BH', 'B:FN', 'B:CA', 'B:CA', 'B:BT', 'B:AK'], 'cqa_startcol':[8,8,8,8,8,8], },
    2: {'fname': '240710_22DX05-12_process combination_AJI-Astar_v2.xlsx', 'skiprows':[2,2,2,2,2,2], 'usecols':['B:BF', 'B:FN', 'B:CA', 'B:CA', 'B:BT', 'B:AK'], 'cqa_startcol':[8,8,8,8,8,8], },
    }

dataset_num = 1
dataset_info = rawdata_dict[dataset_num]
fname = dataset_info['fname']
skiprows_list = dataset_info['skiprows']
usecols_list = dataset_info['usecols']
cqa_startcol_list = dataset_info['cqa_startcol']
fpath = f'{data_folder}{fname}'

xl = pd.ExcelFile(fpath)
print(xl.sheet_names)

# initialize dict to aggregate data
d = {}
var_dict_all = {}
# ['Condition', ''Media composition']
# ['VCD, VIA, Titer, metabolites', 'AA', 'VT', 'MT', 'Glyco', 'Nuc, Amine']
#['volume', 'Temp', 'pH', 'DO']

#%% read sheet: 'VCD, VIA, Titer, metabolites'
sheet_name = 'VCD, VIA, Titer, metabolites'
sheetidx = 0
skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=True)
d = append_to_datadict(d, d_in, d_out)
var_dict_all.update(var_dict)

#%% read sheet: 'AA'
sheet_name = 'AA'
sheetidx = 1
skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
d = append_to_datadict(d, d_in, d_out)
var_dict_all.update(var_dict)

#%% read sheet: 'VT'
sheet_name = 'VT'
sheetidx = 2
skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
d = append_to_datadict(d, d_in, d_out)
var_dict_all.update(var_dict)

#%% read sheet: 'MT'
sheet_name = 'MT'
sheetidx = 3
skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
d = append_to_datadict(d, d_in, d_out)
var_dict_all.update(var_dict)

#%% read sheet: 'Glyco'
sheet_name = 'Glyco'
sheetidx = 4
skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
d = append_to_datadict(d, d_in, d_out)
var_dict_all.update(var_dict)

#%% read sheet: 'Nuc, Amine'
sheet_name = 'Nuc, Amine'
sheetidx = 5
skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
d = append_to_datadict(d, d_in, d_out)
var_dict_all.update(var_dict)

#%% add fields
# add experiment label
for i in list(d.keys()):
    d_exp = d[i]
    basal = d_exp['Basal medium'][-1]
    feed = d_exp['Feed medium'][-1]
    feed_amt = int(d_exp['feed %'])
    pH = format(d_exp['pH'],'.1f')
    DO = int(d_exp['DO'])
    n = d_exp['n']
    exp_label = f'{basal}-{feed}-{feed_amt}%_pH{pH}_DO{DO}'
    d[i]['exp_label'] = exp_label
  
# add dataset number to dict keys
d_orig = d.copy()
for i in list(d_orig.keys()):
    d = {f'{dataset_num}.{k}':v for k, v in d_orig.items()}

#%% save files
pkl_fname = f'dataset{dataset_num}.pkl'
with open(f'{data_folder}{pkl_fname}', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('{data_folder}{pkl_fname}', 'rb') as handle:
#     data = pickle.load(handle)

# %% combine all data

data_all = {}

for dataset_num in rawdata_dict:
    with open(f'{data_folder}dataset{dataset_num}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    data_all.update(data)
    
with open(f'{data_folder}DATA.pkl', 'wb') as handle:
    pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)



