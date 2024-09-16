#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:50:05 2024

@author: charmainechia
"""

import pickle
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plot_utils import data_folder, remove_nandata
import pandas as pd
from variables import var_dict_all, sampling_rawdata_dict, sampling_days, vol_df_exp_cols

def get_feed_volumes(feed_days, df_vol): 
    feed_volumes_dict = {}
    for day in feed_days: 
        vol_prefeed = df_vol[(df_vol.Day==day) & (df_vol.event.str.contains('sampling')) & (~df_vol.event.str.contains('sampling for glucose'))].iloc[0]['vessel volume (mL)']/1000
        vol_feed = df_vol[(df_vol.Day==day) & (df_vol.event=='addition of feed media')].iloc[0]['feed (mL)']/1000
        vol_postfeed = df_vol[(df_vol.Day==day) & (df_vol.event.str.contains('pseudo-sampling'))].iloc[0]['vessel volume (mL)']/1000
        feed_volumes_dict[day] = {'vol_prefeed':vol_prefeed, 'vol_feed':vol_feed, 'vol_postfeed':vol_postfeed}
    return feed_volumes_dict

def adjust_t_sample(t_sample, t_imputed, sampling_days_list):
    t_sample_adj = []
    for i, day in enumerate(t_sample): 
        if day != 10:
            idx = np.where(sampling_days_list==day)[0][0]
            t_sample_adj.append(t_imputed[idx])
        else: 
            t_sample_adj.append(10)
    return np.array(t_sample_adj)


def interpolate_data(t_sample_, y_sample_, k=1, step=1): 
    from scipy.interpolate import splev, splrep
    spl = splrep(t_sample_, y_sample_, k=k)
    t_sample = np.arange(t_sample_[0], t_sample_[-1]+step, step=step)
    y_sample = splev(t_sample, spl)       
    return t_sample, y_sample

def get_arguments_for_optimization(feed_days, sampling_days_list, t_sample, y_sample, t_sample_, y_sample_, vol_df_exp):
# def get_arguments_for_optimization(feed_days, sampling_days_list, t_sample, y_sample, vol_df_exp):
    
    # get points on intermediate days: 7,9,11,13
    feed_days_sampled = [day for day in feed_days if day>=t_sample[0]]
    intermediate_days_sampled = [day+1 for day in feed_days_sampled]
    y_intermediate_days_sampled = np.array([c for day, c in zip(t_sample, y_sample) if day in intermediate_days_sampled])
    final_day_sampled = intermediate_days_sampled[-1]+1
    y_final = y_sample[t_sample==final_day_sampled][0]

    # idxs and values which are not covered in the pre-feed / post-feed / intermediate day set of points, but are non-NaNs in y_imputed
    feed_and_intermediate_days_sampled = list(set(list(feed_days_sampled) + list(intermediate_days_sampled) + [final_day_sampled]))
    days_to_append = [day for day in t_sample if day not in feed_and_intermediate_days_sampled]
    idxs_sampledata_vals_to_append = [(idx, c) for idx, (day, c) in enumerate(zip(t_sample_, y_sample_)) if day in days_to_append]
    
    idxs_imputedata_vals_to_append = []
    for (idx, c) in idxs_sampledata_vals_to_append:
        idx_imputedata = np.where(sampling_days_list==t_sample_[idx])[0][0]
        idxs_imputedata_vals_to_append.append((idx_imputedata, c))

    # get volumes
    df_vol = get_feed_volumes(feed_days, vol_df_exp)
    vol_prefeed_arr = []
    vol_feed_arr = []
    vol_postfeed_arr = []
    for k, day in enumerate(feed_days_sampled):
        vol_prefeed_arr.append(df_vol[day]['vol_prefeed'])
        vol_feed_arr.append(df_vol[day]['vol_feed'])
        vol_postfeed_arr.append(df_vol[day]['vol_postfeed'])
    vol_prefeed_arr = np.array(vol_prefeed_arr)
    vol_feed_arr = np.array(vol_feed_arr)
    vol_postfeed_arr = np.array(vol_postfeed_arr)

    # get trimmed idxs
    nonan_idxs = [i for i, day in enumerate(sampling_days_list) if day in feed_and_intermediate_days_sampled]
    t_imputed_trim = t_imputed[nonan_idxs]
                  
    return y_final, idxs_imputedata_vals_to_append, nonan_idxs, t_imputed_trim, y_intermediate_days_sampled, vol_prefeed_arr, vol_feed_arr, vol_postfeed_arr


def grad(y, t, idx_start):
    if idx_start+1==len(y): 
        return (y[idx_start]-y[-1])/(t[idx_start]-t[-1])
    else:
        return (y[idx_start]-y[idx_start+1])/(t[idx_start]-t[idx_start+1])
                                          
# objective function
def objective(y_prefeed_arr):
    # calculate post-feed concentrations
    y_postfeed_arr = (y_prefeed_arr*vol_prefeed_arr + nutrient_conc_in_feed*vol_feed_arr)/vol_postfeed_arr
    # interpolate vectors to ge sequences of time points
    y_imputed_trim = list(itertools.chain(*zip(list(y_prefeed_arr),list(y_postfeed_arr),list(y_intermediate_days_sampled)))) + [y_final]
    ratio=1.4
    grad_error = 0
    for idx in [1,4,7,10]: 
    # for idx in np.array(list(range(len(pseudo_sampling_days_toadjust_trim))))*3+1: 
        slope1 = ratio*grad(y_imputed_trim,t_imputed_trim, idx)
        slope2 = grad(y_imputed_trim,t_imputed_trim, idx+1)
        grad_error += np.abs(slope1-slope2)
    return grad_error

# get smoothed and integrated VCD for substrate consumption calculations
def get_smoothed_integrated_vcd(t_sample_adj, y_sample_, step=0.01, display_output=False):
    # get log of sample
    log_y_sample_ = np.log(y_sample_)
    # interpolate 
    t_sample, y_sample_0 = interpolate_data(t_sample_adj, y_sample_, k=2, step=step)
    t_sample, log_y_sample_1 = interpolate_data(t_sample_adj, log_y_sample_, k=1, step=step)
    t_sample, log_y_sample_2 = interpolate_data(t_sample_adj, log_y_sample_, k=2, step=step)
    # transform log arrays back
    y_sample_1 = np.exp(log_y_sample_1)
    y_sample_2 = np.exp(log_y_sample_2)
    # get average of curves
    y_sample = np.mean(np.array([y_sample_0, y_sample_1, y_sample_2]), axis=0)

    # get integral of array at each point
    y_integrated = np.cumsum(y_sample)*step    
    
    if display_output:
        # ratio of VCDs on successive initial days
        t_idxs = {}
        for day in [0,1,2,3,4]:
            t_idxs[day] = np.argmin(np.abs(t_sample-day))
        print('1:', *[round(y,3) for y in [y_sample_0[t_idxs[1]]/y_sample_0[t_idxs[0]], y_sample_0[t_idxs[2]]/y_sample_0[t_idxs[1]], y_sample_0[t_idxs[3]]/y_sample_0[t_idxs[2]], y_sample_0[t_idxs[4]]/y_sample_0[t_idxs[3]]]])
        print('2:', *[round(y,3) for y in [y_sample_1[t_idxs[1]]/y_sample_1[t_idxs[0]], y_sample_1[t_idxs[2]]/y_sample_1[t_idxs[1]], y_sample_1[t_idxs[3]]/y_sample_1[t_idxs[2]], y_sample_1[t_idxs[4]]/y_sample_1[t_idxs[3]]]])
        print('3:', *[round(y,3) for y in [y_sample_2[t_idxs[1]]/y_sample_2[t_idxs[0]], y_sample_2[t_idxs[2]]/y_sample_2[t_idxs[1]], y_sample_2[t_idxs[3]]/y_sample_2[t_idxs[2]], y_sample_2[t_idxs[4]]/y_sample_2[t_idxs[3]]]])
        print('AVG:', *[round(y,3) for y in [y_sample[t_idxs[1]]/y_sample[t_idxs[0]], y_sample[t_idxs[2]]/y_sample[t_idxs[1]], y_sample[t_idxs[3]]/y_sample[t_idxs[2]], y_sample[t_idxs[4]]/y_sample[t_idxs[3]]]])
        # smoothed curves
        plt.scatter(t_sample_adj, y_sample_, c='k', s=12)
        plt.plot(t_sample, y_sample_0, '--', alpha=0.7)
        plt.plot(t_sample, y_sample_1, '--', alpha=0.7)
        plt.plot(t_sample, y_sample_2, '--', alpha=0.7)
        plt.plot(t_sample, y_sample)
        plt.legend(['0: sampled data points', '1: y --> y fit with k=2 spline', '2: exp(log(y)) --> log(y) fit with k=1 spline', '3: exp(log(y)) --> log(y) fit with k=2 spline', '4: avg of 1,2,3'], fontsize=8)
        plt.show()
        # integrated curve
        plt.plot(t_sample, y_integrated)
        plt.show()

    return t_sample, y_sample, y_integrated

#%% get data

vol_df_exp_dict = {}
data_all = {}
nutrient_inputs = ['Glucose (g/L)'] + var_dict_all['AA'] + var_dict_all['VT'] + var_dict_all['MT'] + var_dict_all['Nuc, Amine']
# nutrient_inputs = ['Glucose (g/L)'] + var_dict_all['AA'][:5]


# open pickle file
for dataset_num in [0,1,2]:
# for dataset_num in [0]:
    
    # set pickle name
    pkl_fname = f'dataset{dataset_num}'
    
    # get excel parsing metadata
    dataset_info = sampling_rawdata_dict[dataset_num]
    fpath = f'{data_folder}{dataset_info["fname"]}'
    fname = dataset_info['fname']
    skiprows_list = dataset_info['skiprows']
    usecols_list = dataset_info['usecols']
    cqa_startcol_list = dataset_info['cqa_startcol']
    
    with open(f'{data_folder}{pkl_fname}.pkl', 'rb') as handle:
        d = pickle.load(handle)
        exp_idx_list = list(d.keys())

        # load volume sheet
        sheetidx, sheet_name = 6, 'volume'
        skiprows, usecols, cqa_startcol = skiprows_list[sheetidx], usecols_list[sheetidx], cqa_startcol_list[sheetidx]
        vol_df = pd.read_excel(fpath, sheet_name=sheet_name, skiprows=skiprows, usecols=usecols)
        vol_df_cols = list(vol_df.columns)
        
        # iterate through runs
        t_sample_adj = None
        for exp_idx in exp_idx_list:
            print('[EXP_IDX]', exp_idx)
            d_exp = d[exp_idx]
            idx = int(exp_idx[exp_idx.find('.')+1:])
            # find col number in vol_df corresponding to idx
            col_idx = vol_df_cols.index(idx) 
            vol_df_exp = vol_df.iloc[:,[0,1,col_idx, col_idx+1, col_idx+2, col_idx+3]]
            vol_df_exp = vol_df_exp.rename(columns={c: c_renamed for c, c_renamed in zip(vol_df_exp.columns.tolist(), ['Day', 'event', 'time (m)', 'vessel volume (mL)', 'feed (mL)', 'glucose (mL)'])})
            # drop redundant rows
            vol_df_exp = vol_df_exp.iloc[1:, :]
            vol_df_exp = vol_df_exp[(~vol_df_exp['time (m)'].isnull() & ~vol_df_exp['Day'].isnull())]
            vol_df_exp[['Day','time (m)']] = vol_df_exp[['Day','time (m)']].astype(int)
            vol_df_exp[['vessel volume (mL)', 'feed (mL)', 'glucose (mL)']] = vol_df_exp[['vessel volume (mL)', 'feed (mL)', 'glucose (mL)']].astype(float)
            # update 'time (m)' column to remove any 'zeros'
            time_m = vol_df_exp['time (m)'].to_numpy()
            time_zero_idxs_to_replace = np.where(time_m==0)[0][1:]
            time_nonzero_to_replace_with = time_m[time_zero_idxs_to_replace+1]
            time_m[time_zero_idxs_to_replace] = time_nonzero_to_replace_with
            # add precise time in days
            vol_df_exp.insert(2, column='time (d)', value=list(np.round(vol_df_exp['time (m)'].to_numpy().astype(float)/60/24,3)))
            # drop sampling instance on days with two or more sampling instances, both done before the feed
            day_list = vol_df_exp.Day.tolist()
            day_list_unique = list(set(day_list))
            idx_to_drop = []
            for day in day_list_unique:
                vol_df_exp_day = vol_df_exp[(vol_df_exp.Day==day) & (vol_df_exp.event.str.contains('sampling')) & (~vol_df_exp.event.str.contains('sampling for glucose'))]
                # if multiple sampling instances are present, keep only the first instance
                if len(vol_df_exp_day)>1: 
                    idx_to_drop += list(vol_df_exp_day.index)[1:]
            vol_df_exp = vol_df_exp.drop(index=idx_to_drop).reset_index(drop=True)
                        
            # get feed days from metadata
            feed_days = [int(d) for d in d_exp['feed day'].split(',')]
            
            # get sampling days from metadata
            sampling_days_exp = sampling_days[dataset_num]
            sampling_days_all = list(range(15))
            sampling_days_prefeed = list(set([d for d in sampling_days_exp if d not in [0,14]] + feed_days))
            sampling_days_prefeed.sort()
            # get pseudo sampling days (not actual sampling day or feed days)
            pseudo_sampling_days_toremove = [day for day in sampling_days_all if ((day not in sampling_days_exp) and (day not in feed_days))]
            # get pseudo sampling days (not actual sampling days BUT are feed days)
            pseudo_sampling_days_toadjust = [day for day in sampling_days_all if ((day not in sampling_days_exp) and (day in feed_days))]

            # add rows in vol_df_exp for actual sampling (pre-feed), if not included 
            for sampling_day in sampling_days_prefeed: # exclude day 0, 14
                sampling_rows = vol_df_exp[((vol_df_exp.Day==sampling_day) & (vol_df_exp.event.str.contains('sampling')) & (~vol_df_exp.event.str.contains('sampling for glucose')))]
                if len(sampling_rows)==0:
                    # find row number of 1st entry of closest feed day
                    next_feed_day = [feed_day for feed_day in feed_days if feed_day>=sampling_day]
                    if len(next_feed_day)>0:
                        next_feed_day = next_feed_day[0]
                        next_feed_day_idxs = [i for i, day in enumerate(vol_df_exp.Day.tolist()) if day==next_feed_day]
                        insert_row_idx = next_feed_day_idxs[0]
                        last_row_idx = next_feed_day_idxs[-1]
                        next_row = vol_df_exp.iloc[insert_row_idx, :].fillna(0)
                        last_row = vol_df_exp.iloc[last_row_idx, :]
                        if next_feed_day==sampling_day:
                            sampling_time_d, sampling_time_m = next_row['time (d)'], next_row['time (m)']
                        elif next_feed_day > sampling_day: 
                            sampling_time_d, sampling_time_m = last_row['time (d)']-1, last_row['time (m)']-24*60
                        vol_aftersampling = next_row['vessel volume (mL)'] -  next_row['feed (mL)'] -  next_row['glucose (mL)']                                        
                        insert_row = pd.DataFrame({k:v for k, v in zip(vol_df_exp_cols, [sampling_day, 'sampling', float(sampling_time_d), int(sampling_time_m), float(vol_aftersampling), 0, 0])}, index=[insert_row_idx])
                        vol_df_exp = pd.concat([vol_df_exp.iloc[:insert_row_idx], insert_row, vol_df_exp.iloc[insert_row_idx:]]).reset_index(drop=True)
                        # print(f'Added sampling row for feed_day {sampling_day}, {sampling_time_d}')
                    
            # add rows in vol_df_exp for pseudo sampling (post-feed)
            for feed_day in feed_days:
                # if feed_day in sampling_days_exp:
                last_feed_row = vol_df_exp[(vol_df_exp.Day==feed_day)].iloc[-1:]
                last_feed_row = last_feed_row.fillna(0)
                insert_row_idx = int(last_feed_row.index[0]) + 1             
                sampling_time_d, sampling_time_m = last_feed_row.iloc[0]['time (d)'], last_feed_row.iloc[0]['time (m)']
                vol_aftersampling = last_feed_row.iloc[0]['vessel volume (mL)']                       
                insert_row = pd.DataFrame({k:v for k, v in zip(vol_df_exp_cols, [feed_day, 'pseudo-sampling (post-feed)', float(sampling_time_d), int(sampling_time_m), float(vol_aftersampling), 0, 0])}, index=[insert_row_idx])
                vol_df_exp = pd.concat([vol_df_exp.iloc[:insert_row_idx], insert_row, vol_df_exp.iloc[insert_row_idx:]]).reset_index(drop=True)
                # print(f'Added pseudo-sampling row for feed_day {feed_day}, {sampling_time_d}')

            # update vol_df_exp_dict
            vol_df_exp_dict.update({exp_idx: vol_df_exp})
            
            # get sampling times 
            sampling_rows = vol_df_exp[(vol_df_exp.event.str.contains('sampling')) & (~vol_df_exp.event.str.contains('sampling for glucose'))]
            sampling_times = sampling_rows['time (d)'].tolist()
            sampling_days_list = sampling_rows.Day.tolist()
            if 0 not in sampling_days_list: 
                sampling_times = [0] + sampling_times
                sampling_days_list = [0] + sampling_days_list
            if 14 not in sampling_days_list: 
                sampling_times = sampling_times + [14]
                sampling_days_list = sampling_days_list + [14]
            sampling_times = np.array(sampling_times)
            
            sampling_days_list = np.array(sampling_days_list)
            
            # get feed volumes
            feed_volumes_dict = get_feed_volumes(feed_days, vol_df_exp)
            
            ###############################
            # GET SMOOTHED INTEGRATED VCD #
            ###############################
            t_imputed = sampling_times
            t_sample_ = d_exp['VCD (E6 cells/mL)' ]['t']
            vcd_sample_ = d_exp['VCD (E6 cells/mL)' ]['y']
            t_sample_, vcd_sample_ = remove_nandata(t_sample_, vcd_sample_, on_var='xy')
            # # adjust sampling days to account for samping error (Day 9 replaced with Day 10) for selected samples in dataset1
            # if exp_idx in [f'1.{k}' for k in list(range(37,49))]:
            #     sampling_days_list_adj = [int(day) if day != 9 else 10 for day in sampling_days_list]
            # else:
            #     sampling_days_list_adj = sampling_days_list.copy()
            t_sample_adj = adjust_t_sample(t_sample_, t_imputed, sampling_days_list)
            t_smoothed, vcd_smoothed, vcd_integrated = get_smoothed_integrated_vcd(t_sample_adj, vcd_sample_, step=0.001, display_output=False)
            d_exp['VCD (E6 cells/mL)'].update({'t_sample_adj': t_sample_adj, 't_imputed': t_smoothed, 'y_imputed':vcd_smoothed, 'y_integrated': vcd_integrated})
                        
            #############################
            # ITERATE THROUGH NUTRIENTS # 
            #############################
            for nutrient in nutrient_inputs:
                
                if nutrient == 'Glucose (g/L)': nutrient_feed = 'D-glucose'
                else: nutrient_feed = nutrient
                
                if (nutrient not in d_exp) or (f'{nutrient_feed}_feed' not in d_exp) or (math.isnan(d_exp[f'{nutrient_feed}_feed'])):
                    pass
                else: 
                    # get actual sampling data
                    t_sample_ = d_exp[nutrient]['t'].astype(int)
                    y_sample_ = d_exp[nutrient]['y']
                    # interpolate sampled data
                    t_sample, y_sample = interpolate_data(t_sample_, y_sample_)

                    # initialize arrays for full imputed sampling values                  
                    y_imputed = np.zeros((len(t_imputed),)) # for each feed, there should be a measurement before and after
                    y_imputed[:] = np.nan
                    
                    # get adjusted time values based on actual sampling times
                    t_sample_adj = adjust_t_sample(t_sample_, t_imputed, sampling_days_list)
                    d_exp[nutrient]['t_sample_adj'] = t_sample_adj
                    d_exp[nutrient]['y_sample_adj'] = y_sample_
                        
                    nutrient_conc_in_feed = d_exp[f'{nutrient_feed}_feed']
                    # print(nutrient, round(nutrient_conc_in_feed,3), end='  ')
                    
                    # iterate through sample points and update with corresponding values
                    day_log = []
                    for i, (day, t_day) in enumerate(zip(sampling_days_list, t_imputed)):
                        
                        # find match in t_sample
                        if day not in day_log: # new sampling day
                            day_log.append(day)
                            # data present in sampling data
                            if day in t_sample: 
                                idx_match_sampled = np.where(t_sample==day)[0][0]
                                nutrient_conc_sampled = y_sample[idx_match_sampled]
                                y_imputed[i] = nutrient_conc_sampled
                            elif day not in t_sample:
                                continue
                            
                        else: # day was previously sampled -- i.e. this is post-feed pseudo-sample
                            # CALCULATE POST-FEED CONCENTRATIONS #
                            t_prefeed = t_imputed[i-1]
                            t_postfeed = t_day
                            # get pre-feed nutrient concentration
                            nutrient_conc_prefeed = y_imputed[i-1]                             

                            # get volumes (pre, post, feed) ~ units: L
                            vol_prefeed = feed_volumes_dict[day]['vol_prefeed']
                            vol_postfeed = feed_volumes_dict[day]['vol_postfeed']
                            vol_feed = feed_volumes_dict[day]['vol_feed']
                                                
                            # get amount of nutrient before feed (conc*vol) ~ units:mmol
                            nutrient_amt_prefeed = nutrient_conc_prefeed*vol_prefeed
                            
                            # get amt of nutrient fed ~ units: mmol
                            nutrient_amt_feed = nutrient_conc_in_feed*vol_feed
                            
                            if nutrient=='Glucose (g/L)':
                                continue
                                # get glucose bolus concentration (if nutrient == glucose)
                                
                                # get glucose bolus vol (if nutrient == glucose)

                            # get nutrient conc after feed
                            nutrient_amt_postfeed = nutrient_amt_prefeed + nutrient_amt_feed
                            nutrient_conc_postfeed = nutrient_amt_postfeed/vol_postfeed
                                
                            # update imputed array
                            y_imputed[i] = nutrient_conc_postfeed
                            
                        # if len(pseudo_sampling_days_toadjust)>0 and len(feed_days)>5: 
                        #     if day in pseudo_sampling_days_toadjust:
                        #         rel_shift_factor = 0.1
                        #         nutrient_conc_change = nutrient_conc_postfeed - nutrient_conc_prefeed
                        #         y_imputed[i] -= rel_shift_factor*nutrient_conc_change
                        #         y_imputed[i-1] -= rel_shift_factor*nutrient_conc_change   
                        
                    # adjust values if needed by shifting both pre and post feed points down by a certain amount
                    if len(pseudo_sampling_days_toadjust)>0 and len(feed_days)<=5: 
                        
                        # get parameters needed for objective 
                        y_final, idxs_vals_to_append, nonan_idxs, t_imputed_trim, y_intermediate_days_sampled, vol_prefeed_arr, vol_feed_arr, vol_postfeed_arr = get_arguments_for_optimization(feed_days, sampling_days_list, t_sample, y_sample, t_sample_, y_sample_, vol_df_exp)
                        
                        # define the starting point as a random sample from the domain
                        init = y_intermediate_days_sampled + nutrient_conc_in_feed*vol_feed_arr
                        
                        # perform the l-bfgs-b algorithm search
                        result = minimize(objective, init, method='nelder-mead')
                        y_prefeed_arr_solved = result['x']                                
                        
                        # compose final array
                        y_postfeed_arr = (y_prefeed_arr_solved*vol_prefeed_arr + nutrient_conc_in_feed*vol_feed_arr)/vol_postfeed_arr
                        y_imputed_trim = list(itertools.chain(*zip(list(y_prefeed_arr_solved),list(y_postfeed_arr),list(y_intermediate_days_sampled)))) + [y_final]
                        y_imputed = np.zeros_like(t_imputed) 
                        y_imputed[:] = np.nan
                        y_imputed[nonan_idxs] = y_imputed_trim
                        
                        for (idx, c) in idxs_vals_to_append:
                            y_imputed[idx] = c
                        
                        # # summarize the result
                        # print('Status : %s' % result['message'])
                        # print('Total Evaluations: %d' % result['nfev'])
                        # # evaluate solution
                        # evaluation = objective(y_prefeed_arr_solved)
                        # print('Solution: f(%s) = %.5f' % (y_prefeed_arr_solved, evaluation))               
                        

                    # remove data from 'pseudo sampling days'
                    idx_to_keep = np.array([idx for idx, day in enumerate(sampling_days_list) if day not in pseudo_sampling_days_toremove])
                    t_imputed = t_imputed[idx_to_keep]
                    y_imputed = y_imputed[idx_to_keep]
                        
                    if len(y_sample[~np.isnan(y_sample)]) > 0 and not np.array_equal(y_sample[~np.isnan(y_sample)], y_imputed[~np.isnan(y_imputed)]):
                        
                        # print('t_sample:',*t_sample)
                        # print('sampling_days_list:',*sampling_days_list)
                        # print('y_sample:', *np.round(y_sample,3))
                        # print('y_imputed:',*np.round(y_imputed,3))
                        
                        # update dict
                        d_exp[nutrient].update({'t_rounded': sampling_days_list, 't_imputed': t_imputed, 'y_imputed':y_imputed})
                        
                
        
        # update overall datadict
        d.update({exp_idx: d_exp})
    
    with open(f'{data_folder}{pkl_fname}.pkl', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved dataset {dataset_num} to pkl')

    # append to general dict
    data_all.update(d)
    print('Updated general data pkl')

# save all data to single pkl 
with open(f'{data_folder}DATA.pkl', 'wb') as handle:
    pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Saved overall data pkl.')


#%% OPTIMIZATION TESTING

# datadict = data_all.copy()
datadict = d.copy()

exp_idx = '0.1'
var = 'Val' # 'Pro'
t_sample_ = datadict[exp_idx][var]['t']
y_sample_ = datadict[exp_idx][var]['y']
sampling_days_list = datadict[exp_idx][var]['t_rounded']
t_imputed = datadict[exp_idx][var]['t_imputed']
nutrient_conc_in_feed =  datadict[exp_idx][f'{var}_feed']

t_sample, y_sample = interpolate_data(t_sample_,y_sample_)
vol_df_exp = vol_df_exp_dict[exp_idx]

feed_days = [4,6,8,10,12]
t_sample_adj = adjust_t_sample(t_sample_, t_imputed, sampling_days_list)
pseudo_sampling_days_toadjust_trim = [6,8,10,12]

# get parameters needed for objective 
y_final, idxs_vals_to_append, nonan_idxs, t_imputed_trim, y_intermediate_days_sampled, vol_prefeed_arr, vol_feed_arr, vol_postfeed_arr = get_arguments_for_optimization(feed_days, sampling_days_list, t_sample, y_sample, t_sample_, y_sample_, vol_df_exp)
# y_final, idxs_vals_to_append, nonan_idxs, t_imputed_trim, y_intermediate_days_sampled, vol_prefeed_arr, vol_feed_arr, vol_postfeed_arr = get_arguments_for_optimization(feed_days, sampling_days_list, t_sample, y_sample, vol_df_exp)

# define the starting point as a random sample from the domain
init = y_intermediate_days_sampled + nutrient_conc_in_feed*vol_feed_arr

# perform the l-bfgs-b algorithm search
result = minimize(objective, init, method='nelder-mead')
y_prefeed_arr_solved = result['x']              

# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
evaluation = objective(y_prefeed_arr_solved)
print('Solution: f(%s) = %.5f' % (y_prefeed_arr_solved, evaluation))                      

# compose final array
y_postfeed_arr = (y_prefeed_arr_solved*vol_prefeed_arr + nutrient_conc_in_feed*vol_feed_arr)/vol_postfeed_arr
y_imputed_trim = list(itertools.chain(*zip(list(y_prefeed_arr_solved),list(y_postfeed_arr),list(y_intermediate_days_sampled)))) + [y_final]
y_imputed = np.zeros_like(t_imputed) 
y_imputed[:] = np.nan
y_imputed[nonan_idxs] = y_imputed_trim

for (idx, c) in idxs_vals_to_append:
    y_imputed[idx] = c
# plt.plot(t_sample, y_sample)
plt.plot(t_sample_adj, y_sample_)
plt.plot(t_imputed, y_imputed)

# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
evaluation = objective(y_prefeed_arr_solved)
print('Solution: f(%s) = %.5f' % (y_prefeed_arr_solved, evaluation))                                



