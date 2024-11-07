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
from variables import data_folder, sampling_rawdata_dict, overall_glyco_cqas, media_rawdata_dict

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
        if df_in['feed %'].mean() < 1: 
            df_in['feed %'] = df_in['feed %']*100
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

def add_glyco_overall_cqas(d_out, overall_glyco_cqas, printout=False):
    for i_exp, d_exp in d_out.items():
        for glyco_cqa, component_list in overall_glyco_cqas.items():
            t = d_exp[component_list[0]]['t']
            y = np.zeros_like(t)
            for component in component_list:
                y += d_exp[component]['y']
            d_exp.update({glyco_cqa: {'t':t, 'y':y}})
            if printout:
                print(glyco_cqa, t, y)
        d_out.update({i_exp: d_exp})
    return d_out
            

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
    sheets_to_process = [
        'VCD, VIA, Titer, metabolites',
        'AA',
        'VT',
        'MT',
        'Glyco',
        'Nuc, Amine',
        'volume',
        'Media composition'
        ]
    sheets_w_sampling_data = ['VCD, VIA, Titer, metabolites', 'AA', 'VT', 'MT', 'Glyco', 'Nuc, Amine']
    skip_sheets = [] # sheets_w_sampling_data
    
    xl = pd.ExcelFile(fpath)
    sheet_list = xl.sheet_names
    
    # initialize dict to aggregate data
    d = {}
    var_dict_all = {}
    
    # iterate through sheets to agregate raw data
    for sheetidx, sheet_name in enumerate(sheets_to_process):
        print(f'Processing {sheetidx}, {sheet_name}')
        if sheet_name in sheet_list and sheet_name not in skip_sheets:
            
            if sheet_name in sheets_w_sampling_data:
                get_inputs = True if sheetidx==0 else False
                skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
                d_in, d_out, var_dict = xlsx_to_dict(fpath, sheet_name, skiprows=skiprows, usecols=usecols, cqa_startcol=cqa_startcol, get_inputs=get_inputs)
                # calculate mannosylation, fucosylation, glycosylation
                if sheet_name=='Glyco':
                    d_out = add_glyco_overall_cqas(d_out, overall_glyco_cqas, printout=False)
                d = append_to_datadict(d, d_in, d_out)
                var_dict_all.update(var_dict)
                
            elif sheet_name=='volume':
                skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
                exp_idx = list(d.keys())
                vol_arr = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols).iloc[1:, 2:].to_numpy()
                vol_arr_feed = vol_arr[:, [(i-1)*4+2 for i in exp_idx]]
                vol_arr_init = vol_arr[0, [(i-1)*4+1 for i in exp_idx]]
                vol_arr_sum_norm = (np.nansum(vol_arr_feed, axis=0)/vol_arr_init).astype(float)
                # append to dict
                for k in d:
                    d[k]['feed vol'] = float(vol_arr_sum_norm[k-1])
                    
            elif sheet_name=='Media composition':
                skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
                media_df = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols)
                # remove empty rows
                media_df = media_df.dropna(axis=0, how='all')
                # remove rows with redundant headers
                col0 = media_df.iloc[:, 0].tolist()
                idx_to_keep = [i for i, el in enumerate(col0) if (el.find('unit')==-1 and el.find('AA')==-1 and el.find('VT')==-1 and el.find('MT')==-1 and el.find('Sugar')==-1)]
                media_df = media_df.iloc[idx_to_keep,:]
                # reset index
                media_df = media_df.reset_index(drop=True)
                col0 = media_df.iloc[:, 0].tolist()
                # replace non float values with np.nan
                media_df = media_df[media_df.map(np.isreal)]
                media_df.iloc[:,0] = col0
                # update dict
                for k in d:
                    # get Feed and Basal combination for this each sample
                    media_type_dict = {
                        'basal': d[k]['Basal medium'], 
                        'feed':d[k]['Feed medium']
                        }
                    # get basal values in media_df
                    for media_type, media_col_name in media_type_dict.items():
                        nutrient_list = [tuple(el) for el in list(media_df[['unit: mM', media_col_name]].set_index('unit: mM').to_records())]
                        for (nutrient, conc) in nutrient_list:
                            d[k][f'{nutrient}_{media_type}'] = conc
            print(f'Obtained {sheet_name} data')
        else: 
            print(f'{sheet_name} not found in .xlsx file')
        
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

#%% Get averaged media composition

sheet_name = 'Media composition'


for dataset_num, dataset_meta in media_rawdata_dict.items():
    fname, skiprows, usecols, cqa_startcol = dataset_meta['fname'], dataset_meta['skiprows'], dataset_meta['usecols'], dataset_meta['cqa_startcol']
    fpath = f'{data_folder}{fname}'
    df = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols)
    # remove empty rows
    df = df.dropna(axis=0, how='all')
    # remove rows with redundant headers
    col0 = df.iloc[:, 0].tolist()
    idx_to_keep = [i for i, el in enumerate(col0) if (el.find('unit')==-1 and el.find('AA')==-1 and el.find('VT')==-1 and el.find('Amine')==-1 and el.find('Nucleosides')==-1 and el.find('MT')==-1 and el.find('Sugar')==-1)]
    df = df.iloc[idx_to_keep,:]
    # reset index
    df = df.reset_index(drop=True)
    col0 = df.iloc[:, 0].tolist()
    # replace non float values with np.nan
    df = df[df.map(np.isreal)]
    df.iloc[:,0] = col0
    
    if dataset_num == 0: 
        df_merge = df.copy()
    if dataset_num > 0:
        df_merge = df_merge.merge(df, on='unit: mM', how='outer', suffixes=[f'_{dataset_num-1}', f'_{dataset_num}'])

col0 = df_merge.iloc[:, 0].tolist()
colnames = df_merge.columns.tolist()[1:]
colnames_base = []
for c in colnames: 
    if c.find('_')>-1: 
        c = c[:c.find('_')]
    colnames_base.append(c)
colnames_base = list(set(colnames_base))
colnames_base.sort()

## GET DATAFRAME ## 
# initialize df for averaged variables
arr = np.zeros((len(df_merge), len(colnames_base)+1))
arr[:] = np.nan
df_avg = pd.DataFrame(arr, columns=['var']+colnames_base)
df_avg['var'] = col0
for colbase in colnames_base: 
    print(colbase)
    col_list = [c for c in colnames if c.find(colbase)>-1]
    df_col_avg = df_merge[col_list].mean(axis=1)
    # print(df_col_avg)
    df_avg[colbase] = df_col_avg.tolist()

# set var col as index
df_avg = df_avg.set_index('var')

# remove rows that are all zeros
df_avg = df_avg.dropna(axis=0, how = 'all')

# translate dataframe so that nutrient components are the columns, and media labels are the index
df_avg = df_avg.transpose()

# save dataframe to csv
df_avg.to_csv(f'{data_folder}MediaComposition_avg.csv')
