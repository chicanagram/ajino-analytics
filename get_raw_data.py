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
from variables import data_folder, sampling_rawdata_dict

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

#%% get excel file
data_all = {}

for dataset_num in [0, 1, 2]:
    print('Dataset:', dataset_num)
    dataset_info = sampling_rawdata_dict[dataset_num]
    fname = dataset_info['fname']
    skiprows_list = dataset_info['skiprows']
    usecols_list = dataset_info['usecols']
    cqa_startcol_list = dataset_info['cqa_startcol']
    fpath = f'{data_folder}{fname}'
    
    xl = pd.ExcelFile(fpath)
    sheet_list = xl.sheet_names
    
    # initialize dict to aggregate data
    d = {}
    var_dict_all = {}
    
    # read sheet: 'VCD, VIA, Titer, metabolites'
    sheet_name = 'VCD, VIA, Titer, metabolites'
    sheetidx = 0
    if sheet_name in sheet_list:
        skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
        d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=True)
        d = append_to_datadict(d, d_in, d_out)
        var_dict_all.update(var_dict)
        print('Obtained VCD, VIA, Titer, metabolites data')
    
    # read sheet: 'AA'
    sheet_name = 'AA'
    sheetidx = 1
    if sheet_name in sheet_list:
        skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
        d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
        d = append_to_datadict(d, d_in, d_out)
        var_dict_all.update(var_dict)
        print('Obtained AA data')
    
    # read sheet: 'VT'
    sheet_name = 'VT'
    sheetidx = 2
    if sheet_name in sheet_list:
        skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
        d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
        d = append_to_datadict(d, d_in, d_out)
        var_dict_all.update(var_dict)
        print('Obtained VT data')
    
    # read sheet: 'MT'
    sheet_name = 'MT'
    sheetidx = 3
    if sheet_name in sheet_list:
        skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
        d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
        d = append_to_datadict(d, d_in, d_out)
        var_dict_all.update(var_dict)
        print('Obtained MT data')
    
    # read sheet: 'Glyco'
    sheet_name = 'Glyco'
    sheetidx = 4
    if sheet_name in sheet_list:
        skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
        d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
        d = append_to_datadict(d, d_in, d_out)
        var_dict_all.update(var_dict)
        print('Obtained Glyco data')
    
    # read sheet: 'Nuc, Amine'
    sheet_name = 'Nuc, Amine'
    sheetidx = 5
    if sheet_name in sheet_list:
        skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
        d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=False)
        d = append_to_datadict(d, d_in, d_out)
        var_dict_all.update(var_dict)
        print('Obtained Nuc, Amine data')
    else: 
        print(f'{sheet_name} not found in xlsx file.')
    
    # Get volume data
    sheet_name = 'volume'
    sheetidx = 6
    # get volume data from sheet
    skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
    exp_idx = list(d.keys())
    vol_arr = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols).iloc[1:, 2:].to_numpy()
    vol_arr_feed = vol_arr[:, [(i-1)*4+2 for i in exp_idx]]
    vol_arr_init = vol_arr[0, [(i-1)*4+1 for i in exp_idx]]
    vol_arr_sum_norm = (np.nansum(vol_arr_feed, axis=0)/vol_arr_init).astype(float)
    # append to dict
    for k in d:
        d[k]['feed vol'] = float(vol_arr_sum_norm[k-1])
    print('Obtained feed volume data')
        
    # add fields
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
    print('Added experimental labels')
      
    # add dataset number to dict keys
    d_orig = d.copy()
    for i in list(d_orig.keys()):
        d = {f'{dataset_num}.{k}':v for k, v in d_orig.items()}

    # save undividual dataset as pkl
    pkl_fname = f'dataset{dataset_num}.pkl'
    with open(f'{data_folder}{pkl_fname}', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved individual dataset to pkl')

    # append to general dict
    data_all.update(d)
    print('Updated general data pkl')

# save all data to single pkl 
with open(f'{data_folder}DATA.pkl', 'wb') as handle:
    pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Saved overall data pkl.')
