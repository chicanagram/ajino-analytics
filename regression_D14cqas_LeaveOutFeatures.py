#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:47:58 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import ast 
import matplotlib.pyplot as plt
from variables import model_params, dict_update, yvar_sublist_sets, sort_list, yvar_list_key
from model_utils import fit_model_with_cv, get_feature_importances, plot_feature_importance_heatmap, plot_feature_importance_barplots, plot_model_metrics, select_subset_of_X, order_features_by_importance
from plot_utils import figure_folder, model_cmap, convert_figidx_to_rowcolidx
from get_datasets import data_folder, get_XYdata_for_featureset

def get_features_to_drop(num_features_to_drop, xvar_list): 
    import itertools
    idx_to_drop_list = []
    idx_to_keep_list = []
    xvar_to_drop_list = []
    xvars_final_list = []
    idxs_all = list(range(len(xvar_list)))
    if num_features_to_drop == 0:
        idx_to_drop_list.append(())
        idx_to_keep_list = [idxs_all]
        xvar_to_drop_list.append([])
        xvars_final_list.append(xvar_list)
    elif num_features_to_drop > 0:
        for i, idx_to_drop in enumerate(itertools.combinations(idxs_all, num_features_to_drop)):
            idx_to_drop_list.append(idx_to_drop)
            idx_to_keep_list.append([idx for idx in idxs_all if idx not in idx_to_drop])
            xvar_to_drop = [xvar_list[k] for k in idx_to_drop]
            xvar_to_drop_list.append(xvar_to_drop)
            xvars_final_list.append([xvar for xvar in xvar_list if xvar not in xvar_to_drop])       
    return idx_to_drop_list, idx_to_keep_list, xvar_to_drop_list, xvars_final_list


#%% Evaluate individual model at a time and get metrics, and feature coefficients / importances 

featureset_list = [(1,0)] # [(0,0), (1,0)]
models_to_eval_list = ['randomforest','plsr', 'lasso'] # ['plsr'] # 
subset_suffix = ''
f = 1
num_features_to_drop_list = [0, 1, 2]

    
# get relevant dataset with chosen features
for (X_featureset_idx, Y_featureset_idx) in featureset_list: 
    # get data
    dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
    Y, X_init, Xscaled_init, yvar_list, xvar_list_init = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, data_folder=data_folder)
    print(f'X dataset size: n={Xscaled_init.shape[0]}, p={Xscaled_init.shape[1]}')

    # initialize variables for storing results
    model_metrics_df = []
    ref_model_metrics = []

    # iterate over number of features to drop
    for num_features_to_drop in num_features_to_drop_list: 
        # get features to drop
        idx_to_drop_list, idx_to_keep_list, xvar_to_drop_list, xvarlist_list = get_features_to_drop(num_features_to_drop, xvar_list_init)
        
        # iterate through model types 
        for model_type in models_to_eval_list: 
            print(model_type)
            
            # iterate through yvar
            for i, yvar in enumerate(yvar_list_key): 
                y = Y[:,i]
                if num_features_to_drop > 0:
                    r2_cv_ref = float(ref_model_metrics_df.loc[(ref_model_metrics_df.model_type==model_type) & (ref_model_metrics_df.yvar==yvar), 'r2_cv'].iloc[0])
                    mae_cv_ref = float(ref_model_metrics_df.loc[(ref_model_metrics_df.model_type==model_type) & (ref_model_metrics_df.yvar==yvar), 'mae_cv'].iloc[0])
            
                for k, (idx_to_drop, idx_to_keep, xvar_to_drop, xvar_list) in enumerate(zip(idx_to_drop_list, idx_to_keep_list, xvar_to_drop_list, xvarlist_list)):
                    # if k>5: break
                    print(f'{k}: Drop {idx_to_drop} {xvar_to_drop}')
                    # drop chosen features from X dataset
                    Xscaled = Xscaled_init[:, np.array(idx_to_keep)]        
                    model_list = model_params[dataset_name][model_type][yvar]
                    print(Xscaled.shape)
                    model_list, metrics = fit_model_with_cv(Xscaled,y, yvar, model_list, plot_predictions=False)
                    # get baseline (for model with no features dropped)
                    if num_features_to_drop==0: 
                        ref_model_metrics.append(metrics)
                        ref_model_metrics_df = pd.DataFrame(ref_model_metrics, columns=['model_type', 'yvar', 'model_params', 'r2', 'r2_cv', 'mae_train', 'mae_norm_train', 'mae_cv', 'mae_norm_cv', 'f', 'xvar_list'])
                        r2_cv_foldchange = 1
                        mae_cv_foldchange = 1
                    else: 
                        # get fold-change in R2 and MAE (CV)
                        r2_cv_foldchange = round(metrics['r2_cv']/r2_cv_ref,3)
                        mae_cv_foldchange = round(metrics['mae_cv']/mae_cv_ref,3)
                    # update metrics dict
                    metrics.update({'f':f, 'num_features_dropped': num_features_to_drop, 
                                    'xvar_idx_dropped': idx_to_drop, 'xvar_dropped': xvar_to_drop, 
                                    'r2_cv_fold-change':r2_cv_foldchange, 
                                    'mae_cv_foldchange': mae_cv_foldchange,
                                    'xvar_list': ', '.join(str(e) for e in xvar_list),                                     
                                    })
                    # append to metrics df
                    model_metrics_df.append(metrics)
                    
    # aggregate all model metrics and save CSV
    model_metrics_df = pd.DataFrame(model_metrics_df)
    model_metrics_df['r2_cv_decrease_vs_ref(%)'] = ((1-model_metrics_df['r2_cv_fold-change'])*100).round(2)
    model_metrics_df['mae_cv_increase_vs_ref(%)'] = ((model_metrics_df['mae_cv_fold-change']-1)*100).round(2)
    model_metrics_df_cols = [
        'model_type', 'yvar', 'model_params', 'f', 
        'r2', 'r2_cv', 'mae_train', 'mae_norm_train', 'mae_cv', 'mae_norm_cv', 
        'num_features_dropped', 'xvar_idx_dropped', 'xvar_dropped', 
        'r2_cv_decrease_vs_ref(%)', 'r2_cv_fold-change', 
        'mae_cv_increase_vs_ref(%)', 'mae_cv_fold-change',
        'xvar_list'
        ]
    model_metrics_df = model_metrics_df[model_metrics_df_cols]
    model_metrics_df = model_metrics_df.sort_values(by=['yvar','r2_cv_fold-change']).reset_index(drop=True)
    model_metrics_df.to_csv(f'{data_folder}model_metrics_LOFV_{dataset_name}{subset_suffix}.csv') 
    
    # save by model type
    for model_type in models_to_eval_list:
        model_metrics_df[model_metrics_df.model_type==model_type].to_csv(f'{data_folder}model_metrics_LOFV_{dataset_name}{subset_suffix}_{model_type}.csv') 
    
    
#%% feature analysis Leave One Out

figtitle = None
savefig = 'feature_importances_LOFV_1feature_barplots'
thres_percent_removeOneFeature = 0
num_xvar_per_legendcol = 17
models_to_eval_list = ['randomforest','plsr']
subset_suffix = ''

fig, ax = plt.subplots(len(models_to_eval_list),4, figsize=(44,8*len(models_to_eval_list)))

for i, model_type in enumerate(models_to_eval_list):
    
    # get data for analysis
    model_metrics_df_bymodel = pd.read_csv(f'{data_folder}model_metrics_LOFV_{dataset_name}{subset_suffix}_{model_type}.csv', index_col=0)
    c = model_cmap[model_type]
    
    for j, yvar in enumerate(yvar_list_key):
        
        model_metrics_df_bymodel_byyvar = model_metrics_df_bymodel[(model_metrics_df_bymodel.model_type==model_type) & (model_metrics_df_bymodel.yvar==yvar)]
        print(model_type, yvar, len(model_metrics_df_bymodel_byyvar))
        # look for all combis that pass threshold for Single Feature Removal
        selected = model_metrics_df_bymodel_byyvar.loc[(model_metrics_df_bymodel_byyvar.num_features_dropped==1) & (model_metrics_df_bymodel_byyvar['r2_cv_decrease_vs_ref(%)']>thres_percent_removeOneFeature), ['xvar_dropped', 'r2_cv_decrease_vs_ref(%)']]
        # plot bar plots of selected features for each model type, each yvar
        xtickpos = np.arange(len(selected))
        xticklabels = [ast.literal_eval(l) for l in selected['xvar_dropped'].tolist()]
        feature_importance = selected['r2_cv_decrease_vs_ref(%)'].to_numpy()
        ax[i][j].bar(xtickpos, feature_importance, color=c, width=0.5)
        if len(selected)>50:
            xtickpos = xtickpos[::2]
        ax[i][j].set_xticks(xtickpos, xtickpos, fontsize=12)
        ax[i][j].set_title(yvar, fontsize=20)
        ax[i][j].set_ylabel('R2 (CV) decrease vs. reference', fontsize=16)
        (xmin, xmax) = ax[i][j].get_xlim()
        (ymin, ymax) = ax[i][j].get_ylim()
        plotwidth = xmax-xmin
        num_legend_cols = int(np.ceil(len(xticklabels)/num_xvar_per_legendcol))
        legend_col_relwidth = 0.22
        for col in range(num_legend_cols):
            idx_start_col = col*num_xvar_per_legendcol
            idx_end_col = min(len(xticklabels), (col+1)*num_xvar_per_legendcol)
            legend = ''
            for k, xvar in enumerate(xticklabels[idx_start_col:idx_end_col]):
                legend += f'{k+col*num_xvar_per_legendcol}: {xvar}\n' 
            xpos_text = round(xmin + plotwidth*((1-legend_col_relwidth/8)-(num_legend_cols-col)*legend_col_relwidth),3)
            ypos_text = ymin+(ymax-ymin)*0.95
            ax[i][j].text(xpos_text, ypos_text, legend, ha='left', va='top', fontsize=12)

        
plt.tight_layout()

if figtitle is not None: 
    ymax = ax.flatten()[0].get_position().ymax
    plt.suptitle(f'{figtitle}', y=ymax*1.06, fontsize=24)    
        
if savefig is not None:
    fig.savefig(f'{figure_folder}{savefig}.png', bbox_inches='tight')

plt.show()


#%% feature analysis: Leave Two Out

figtitle = None
savefig = 'feature_importances_LOFV_1feature_barplots'
thres_percent_removeOneFeature_dict = {'randomforest':2, 'plsr':1.75}
thres_percent_removeTwoFeatures_dict = {'randomforest':2, 'plsr':2}
r2change_foldincrease_dict = {'randomforest':1.4, 'plsr':1.3}
models_to_eval_list = ['randomforest','plsr']
subset_suffix = ''
    
# initialize dataframe to store selected variables
featureselection_res = []

# fig, ax = plt.subplots(len(models_to_eval_list),4, figsize=(44,8*len(models_to_eval_list)))

for i, model_type in enumerate(models_to_eval_list):
    
    # get data for analysis
    model_metrics_df_bymodel = pd.read_csv(f'{data_folder}model_metrics_LOFV_{dataset_name}{subset_suffix}_{model_type}.csv', index_col=0)
    c = model_cmap[model_type]
    
    # get thresholds
    thres_percent_removeOneFeature = thres_percent_removeOneFeature_dict[model_type]
    thres_percent_removeTwoFeatures = thres_percent_removeTwoFeatures_dict[model_type]
    r2change_foldincrease = r2change_foldincrease_dict[model_type]
    
    for j, yvar in enumerate(yvar_list_key):
        
        # get data
        model_metrics_df_bymodel_byyvar = model_metrics_df_bymodel[(model_metrics_df_bymodel.model_type==model_type) & (model_metrics_df_bymodel.yvar==yvar)]
        model_metrics_df_bymodel_byyvar_REF = model_metrics_df_bymodel.loc[(model_metrics_df_bymodel.num_features_dropped==1), ['xvar_dropped', 'r2_cv_decrease_vs_ref(%)']]
        model_metrics_df_bymodel_byyvar_REF['xvar_dropped'] = [ast.literal_eval(xvar)[0] for xvar in model_metrics_df_bymodel_byyvar_REF['xvar_dropped'].tolist()]
        model_metrics_df_bymodel_byyvar_REF = model_metrics_df_bymodel_byyvar_REF.set_index('xvar_dropped', drop=True).to_dict()
        model_metrics_df_bymodel_byyvar_REF = model_metrics_df_bymodel_byyvar_REF['r2_cv_decrease_vs_ref(%)']
        
        # look for all combis that pass threshold for Double Feature Removal
        selected = model_metrics_df_bymodel_byyvar.loc[(model_metrics_df_bymodel_byyvar.num_features_dropped==2) & (model_metrics_df_bymodel_byyvar['r2_cv_decrease_vs_ref(%)']>thres_percent_removeTwoFeatures), ['xvar_dropped', 'r2_cv_decrease_vs_ref(%)']].reset_index(drop=True)
        selected = selected.sort_values(by='r2_cv_decrease_vs_ref(%)', ascending=False)
        # iterate through each row of selected --> compare against ref --> if r2_cv_decrease > than that of ref, retain the example
        idx_featurepairs_to_retain = []
        features_to_retain = []
        for k in range(len(selected)):
            # get xvar_dropped
            r2_cv_decrease = selected.loc[k, 'r2_cv_decrease_vs_ref(%)']
            xvar_dropped = ast.literal_eval(selected.loc[k, 'xvar_dropped'])
            r2_cv_decrease_ref0 = model_metrics_df_bymodel_byyvar_REF[xvar_dropped[0]]
            r2_cv_decrease_ref1 = model_metrics_df_bymodel_byyvar_REF[xvar_dropped[1]]
            if (r2_cv_decrease > r2change_foldincrease*r2_cv_decrease_ref0 and r2_cv_decrease > r2change_foldincrease*r2_cv_decrease_ref1) and (r2_cv_decrease_ref0 > thres_percent_removeOneFeature and r2_cv_decrease_ref1 > thres_percent_removeOneFeature):
                idx_featurepairs_to_retain.append(k)
                features_to_retain += xvar_dropped
                d = {
                    'model_type':model_type, 'yvar':yvar, 
                    'xvar_0':xvar_dropped[0], 'xvar_1':xvar_dropped[1], 
                    'joint_change': r2_cv_decrease, 
                    'xvar_0_change':r2_cv_decrease_ref0, 'xvar_1_change':r2_cv_decrease_ref1
                    }
                featureselection_res.append(d)
                # print(f'{xvar_dropped[0]}, {xvar_dropped[1]}', r2_cv_decrease, r2_cv_decrease_ref0, r2_cv_decrease_ref1)
        
        features_to_retain = sort_list(list(set(features_to_retain)))
        print(model_type, yvar)
        print(f'{len(features_to_retain)} unique features, {len(idx_featurepairs_to_retain)} feature pairs selected')
        for feature in features_to_retain: print(feature)
        
        # filter shortlist
        selected = selected.loc[idx_featurepairs_to_retain,:]    
        print()
        
        
featureselection_res = pd.DataFrame(featureselection_res)
featureselection_res.to_csv(f'{data_folder}featureselection_LOFV_2_{dataset_name}.csv')








 