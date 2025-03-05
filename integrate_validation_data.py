import pandas as pd
import numpy as np
import pickle
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from variables import data_folder, sampling_rawdata_dict, xvar_list_dict, yvar_list_key

# set parameters
shuffle_data = False

# get validation data (unprocessed)
val_fname = sampling_rawdata_dict['val']
val_data = pd.read_csv(data_folder+val_fname, index_col=0).reset_index(drop=True)
# val_data = val_data[val_data['Exp']=='Val-A'].reset_index(drop=True)

# get xvar list for feature set X1Y0
xvar_list = xvar_list_dict[1]
xvar_base_basal = [x.replace('_basal', '') for x in xvar_list if x.find('_basal')>-1]
xvar_base_feed = [x.replace('_feed', '') for x in xvar_list if x.find('_feed')>-1]

# get media composition for Basal-A, feed-a baseline
media_composition_avg = pd.read_csv(data_folder+'MediaComposition_avg.csv', index_col=0)
x_baseline_basal = media_composition_avg.loc['Basal-A', xvar_base_basal].tolist()
x_baseline_feed = media_composition_avg.loc['Feed-a', xvar_base_feed].tolist()
x_baseline = x_baseline_basal + x_baseline_feed

# append to val experimental data
val_data_X = pd.DataFrame(np.tile(np.array(x_baseline).reshape(1,-1), (len(val_data),1)), columns=[x+'_basal' for x in xvar_base_basal] + [x+'_feed' for x in xvar_base_feed])
val_data = pd.concat([val_data, val_data_X], axis=1)
val_data['feature_val'] = val_data['feature_val'].astype('str')
# val_data = val_data[val_data['Condition']!='CTRL'].dropna(subset=['Condition'])

# get feed vol data
val_data['feed vol'] = 0.24
val_data[xvar_list] = val_data[xvar_list].astype(float)

# update individual values that were changed
for i in list(val_data.index):
    condition = val_data.loc[i,'Condition'].split(', ')
    feature_vals = val_data.loc[i,'feature_val']
    if isinstance(feature_vals, float): 
        feature_vals = [feature_vals]
    else: 
        feature_vals = [float(v) for v in feature_vals.split(', ')]
    print(i, condition, feature_vals)
    for feature, feature_val in zip(condition, feature_vals):
        print(feature, feature_val)
        val_data.at[i,feature] = feature_val
    # update feed vol if needed
    if val_data.loc[i, 'feed %']==4:
        val_data.at[i, 'feed vol'] = 0.149
val_data_tomerge = val_data[xvar_list+yvar_list_key]

# get normalized data from main exp
df_norm = pd.read_csv(data_folder + 'X1Y0_norm.csv').reset_index(drop=True)[xvar_list+yvar_list_key]
df_norm_columns = df_norm.columns.tolist()
val_data_columns = val_data_tomerge.columns.tolist()
df_norm_with_val = pd.concat([df_norm[xvar_list+yvar_list_key], val_data_tomerge[xvar_list+yvar_list_key]], ignore_index=True, axis=0)
df_norm_with_val.to_csv(data_folder + 'X1Y0_norm_with_val.csv')
print(df_norm_with_val.shape)

# get shuffle index
if shuffle_data:
    shuffle_idx = np.arange(len(df_norm_with_val))
    np.random.seed(seed=0)
    np.random.shuffle(shuffle_idx)
else:
    shuffle_idx = np.arange(len(df_norm_with_val))

XYarr_dict = {
    'X': df_norm_with_val[xvar_list].to_numpy()[shuffle_idx,:],
    'Y': df_norm_with_val[yvar_list_key].to_numpy()[shuffle_idx,:],
    'Xscaled': None,
    'xvar_list': xvar_list,
    'yvar_list': yvar_list_key
    }
# pickle dict
with open(data_folder + 'X1Y0_norm_with_val_unshuffled.pkl', 'wb') as f:
    pickle.dump(XYarr_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
#%% plot validation data

fig, ax = plt.subplots(4,2, figsize=(20,20))
plt.subplots_adjust(hspace=0.1, wspace=0.2)

# get axis limits
ax_lim = {}
factor = 0.05
for yvar in yvar_list_key:
     arr = val_data[yvar].to_numpy()
     arr_min = np.nanmin(arr)
     arr_max = np.nanmax(arr)
     arr_range = arr_max-arr_min
     ax_lim[yvar] = (arr_min-arr_range*factor, arr_max+arr_range*factor)
     

# iterate through experiments
for i, exp in enumerate(['Val-A', 'Val-B']): 
    val_data_exp = val_data[val_data['Exp']==exp]
    conditions = val_data_exp['exp_label'].tolist()
    conditions_deduped = []
    # get unique conditions
    for c in conditions: 
        if c not in conditions_deduped:
            conditions_deduped.append(c)
    # iterate througy yvar
    for k, yvar in enumerate(yvar_list_key):
        
        # iterate through conditions
        for j, c in enumerate(conditions_deduped):
            print(c)
            val_data_exp_condition = val_data_exp[val_data_exp['exp_label']==c]
            x = np.array([j,j])
            y = val_data_exp_condition[yvar].to_numpy()
            y_avg = np.mean(y)
            # get control values & plot hline for reference
            if c =='CTRL': 
                ax[k,i].axhline(y=y_avg, linestyle='--')
                y_ref = y.copy()
            # calculate t-test value, if not reference sample
            else: 
                t_stat, p_value = ttest_ind(y_ref, y, equal_var=True)
                # annotate with p-value
                ax[k,i].annotate(round(p_value,3), (j,y_avg))
                print(yvar, c, y, y_ref, round(p_value,3))
            # plot 
            ax[k,i].scatter(x, y)
            ax[k,i].plot(x, y, linestyle='--')
            ax[k,i].set_ylabel(f'{yvar} (normalized)')
            ax[k,i].set_title(f'{yvar} [Exp {exp[-1]}]' )
            if k==len(yvar_list_key)-1:
                ax[k,i].set_xticks(ticks=list(range(len(conditions_deduped))), labels=conditions_deduped, rotation=90)
            else: 
                ax[k,i].set_xticks(ticks=list(range(len(conditions_deduped))), labels=[])
            ax[k,i].set_ylim(ax_lim[yvar])
        print()
plt.show()
            
#%% 
for i, exp in enumerate(['Val-A', 'Val-B']): 
    val_data_exp = val_data[val_data['Exp']==exp]
    plt.scatter(val_data_exp['Titer (mg/L)_14'], val_data_exp['mannosylation_14'])
    plt.xlabel('Titer')
    plt.ylabel('Mannosylation')
    plt.show()
    plt.scatter(val_data_exp['mannosylation_14'], val_data_exp['galactosylation_14'])
    plt.xlabel('Mannosylation')
    plt.ylabel('Galactosylation')
    plt.show()
                
#%% 

df_norm.columns.tolist()












        