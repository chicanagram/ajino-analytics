import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_XYdata_for_featureset
from plot_utils import heatmap, annotate_heatmap
from variables import data_folder, figure_folder, var_dict_all, overall_glyco_cqas, sort_list, yvar_list_key, xvar_list_dict_prefilt
from matplotlib import colormaps

# 


# # get data from specific dataset
# X_featureset_idx, Y_featureset_idx = 1, 0
# dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
# dataset_suffix = '' # '_norm_with_val'
# return_XYarr_dict = True
# df = pd.read_csv(data_folder + dataset_name + '.csv', index_col=0).reset_index(drop=True)
# Y, X, Xscaled, yvar_list, xvar_list = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder, return_XYarr_dict=return_XYarr_dict)




mediacomposition_fname = 'MediaComposition_avg.csv'
df = pd.read_csv(f'{data_folder}{mediacomposition_fname}', index_col=0)

# filter for subset of media types
df = df[df.index.isin(['Basal-A', 'Basal-B', 'Basal-C', 'Basal-D', 'Feed-a', 'Feed-d', 'Feed-e', 'Feed-f'])]
media_names = list(df.index)
cmap = [plt.cm.tab10(i) for i in range(len(media_names))]
media_dict = {
    'Basal':[b for b in media_names if b.find('Basal')>-1],
    'Feed':[b for b in media_names if b.find('Feed')>-1],
    }
xvar_list = var_dict_all['media_components'] # df.columns.tolist()

fig, ax = plt.subplots(2,1, figsize=(20,20))

# get normalized values
df_norm = df.copy()
df_norm[:] = np.nan
for i, basal_or_feed in enumerate(['Basal','Feed']):
    media_list = media_dict[basal_or_feed]
    df_basal_or_feed = df[df.index.isin(media_list)]
    for j, xvar in enumerate(xvar_list): 
        y_arr = df_basal_or_feed[xvar].to_numpy()
        num_nonan = len(y_arr[~np.isnan(y_arr)])
        
        if num_nonan > 0:
            y_max = np.nanmax(y_arr)
            y_min = np.nanmin(y_arr)
            if num_nonan==1:
                y_arr = y_arr/y_max
            else:
                y_arr = (y_arr - y_min)/(y_max-y_min)
        df_norm.loc[media_list, xvar] = y_arr
                
# plot normalized values for basal or feed
for i, basal_or_feed in enumerate(['Basal','Feed']):
    media_list = media_dict[basal_or_feed]
    df_basal_or_feed = df[df.index.isin(media_list)]
    for media_type in media_list: 
        x = range(len(xvar_list))
        y = df_norm.loc[media_type,xvar_list].to_numpy()
        ax[i].scatter(x, y, color=cmap[media_names.index(media_type)])
    ax[i].set_xticks(range(len(xvar_list)), labels=xvar_list, rotation=45, ha='right', va='top')
    ax[i].legend(media_list, loc=(1.02,0.5))
    ax[i].set_title(f'{basal_or_feed} media components (min-max scaled)', fontsize=20)
plt.show()
    
            
            
            