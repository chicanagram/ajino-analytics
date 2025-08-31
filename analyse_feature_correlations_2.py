import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt


# Load the uploaded CSV file
data_folder = '../GSK-BTI/Data/'
filename = "DATA_Phase1_ALL.csv"
filepath = data_folder + filename
df = pd.read_csv(filepath)

# Create additional derived variables
## OD_diff_t -- 'growth rate'
for t in [1,2,3,4,5,6,7,8,9,10]:
    # OD_diff
    df[f'OD_diff_{t}'] = df[f'OD_{t}'] - df[f'OD_{t-1}']
    # OD_int 
    df[f'OD_int_{t}'] = 0.5*(df[f'OD_{t}'] + df[f'OD_{t-1}'])
    

## OD_int_all -- area under curve for full time period
# get integrated OD pre drift & post drift
OD_int_predrift_list = []
OD_int_drift_list = []
for i in range(len(df)):
    
    row = df.iloc[i]
    drift_start = row['Drift start']
    OD_int_predrift = row[[f'OD_int_{t}' for t in range(1,drift_start)]].sum()
    OD_int_drift = row[[f'OD_int_{t}' for t in range(drift_start,10)]].sum()
    OD_int_predrift_list.append(round(OD_int_predrift,3))
    OD_int_drift_list.append(round(OD_int_drift,3))
df['OD_int_predrift'] = OD_int_predrift_list
df['OD_int_drift'] = OD_int_drift_list
        
    
#%%

# Define the target variable
target_col = 'PS (LC)'

# Ensure the target column exists
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in the dataset.")

# Identify columns after the target column
target_index = df.columns.get_loc(target_col)
data_cols = df.columns[target_index + 1:]

# Filter to only numeric columns for correlation analysis
numeric_data_cols = df[data_cols].select_dtypes(include=[np.number]).columns

# Extend results to include the number of unique values in the correlated variable
results = []

for col in numeric_data_cols:
    valid_data = df[[target_col, col]].dropna()
    if len(valid_data) > 1:
        try:
            pearson_r, _ = pearsonr(valid_data[target_col], valid_data[col])
            spearman_r, _ = spearmanr(valid_data[target_col], valid_data[col])
        except Exception:
            pearson_r, spearman_r = np.nan, np.nan
        count = len(valid_data)
        num_unique = valid_data[col].nunique()
        
        if col in ['PS (ELISA)', 'OD_int_predrift', 'OD_int_drift'] or len(col.split('_'))==1: 
            col_wtimepoint = col + '_10'
        else:
            col_wtimepoint = col

        results.append({
            'Var_base': '_'.join(col_wtimepoint.split('_')[:-1]),
            'Variable': col_wtimepoint,
            'Timepoint': int(col_wtimepoint.split('_')[-1]),
            'PearsonR': pearson_r,
            'SpearmanR': spearman_r,
            'N': count,
            'Unique Values': num_unique
        })

# Create updated DataFrame
corr_df = pd.DataFrame(results)
corr_df[['PearsonR','SpearmanR']] = corr_df[['PearsonR','SpearmanR']].round(3)
corr_df.to_csv(data_folder + 'Correlations.csv')

# filter data
corr_df_filt = corr_df[corr_df['Unique Values'] >= 4]
corr_df_filt.to_csv(data_folder + 'Correlations_filt.csv')

#%%

# create 2D data grid with time points along x-axis and variable along y-axis
unique_var = corr_df_filt['Variable'].tolist()
unique_var_base = ['PS (ELISA)', 'OD', 'OD_diff', 'OD_int', 'OD_int_predrift', 'OD_int_drift', 'Gluc', 'Glu', 'pH', 'NH4+', 'K+', 'Na+', 'Osm'] # list(set(corr_df_filt['Var_base'].tolist())) # 
unique_timepoints = list(set(corr_df_filt['Timepoint'].tolist()))
unique_timepoints.sort()
arr_init = np.zeros((len(unique_var_base), len(unique_timepoints)))
arr_init[:] = np.nan
df2d_init = pd.DataFrame(arr_init, index=unique_var_base, columns=unique_timepoints)
corr2D = {'PearsonR': df2d_init.copy(), 'SpearmanR': df2d_init.copy()}
for corr_type in ['PearsonR', 'SpearmanR']: 
    for var_base in unique_var_base:
        for timepoint in unique_timepoints:
            var = var_base + '_' + str(timepoint)
            if var in unique_var: 
                corr_val = corr_df_filt.loc[corr_df_filt['Variable']==var, corr_type].iloc[0]
                corr2D[corr_type].loc[var_base, timepoint] = corr_val
    
    corr2D[corr_type] = corr2D[corr_type].round(3)       
    corr2D[corr_type] = corr2D[corr_type].loc[:, [0	,3,5,6,7,10]]     
    corr2D[corr_type].to_csv(data_folder + f'Corr2D_{corr_type}.csv')
    
#%%

# get variables with top 2 correlations for each var_base
top_N = 2
var_with_highest_correlations = {}
corr_eval = corr2D['SpearmanR'].abs() 
for var_base in unique_var_base: 
    print(var_base)
    var_base_absvals = corr_eval.loc[var_base,:].fillna(0).to_numpy()
    top_N_idxs = np.argsort(var_base_absvals)[-top_N:]
    top_N_timepoints = [corr_eval.columns[idx] for idx in top_N_idxs]
    print(var_base_absvals, top_N_idxs, top_N_timepoints)
    # var_with_highest_correlations[var_base] = [f'{var_base}_{timepoint}' for timepoint in top_N_timepoints]
    var_with_highest_correlations[var_base] = [f'{var_base}_{timepoint}' for timepoint in top_N_timepoints if ~np.isnan(corr_eval.loc[var_base,timepoint])]
    print(var_with_highest_correlations[var_base])
    
# plot scatter
nrows, ncols = 5,3
fig, ax = plt.subplots(nrows,ncols, figsize=(4*ncols,3.2*nrows))
axes_wplot = []
for k, var_base in enumerate(unique_var_base): 
    print(k, var_base)
    row_idx = int(np.floor(k/ncols))
    col_idx = k%ncols
    corr_df_filt_byvarbase = corr_df_filt[corr_df_filt['Var_base']==var_base]
    var_to_plot = corr_df_filt_byvarbase['Variable'].tolist()

    var_plotted = []
    corr_plotted = []
    for var in var_to_plot:
        if var in var_with_highest_correlations[var_base]:
            var_plotted.append(var)
            corr_plotted.append(corr_df_filt.loc[corr_df_filt['Variable']==var, 'SpearmanR'].iloc[0])
            if var not in df:
                var = '_'.join(var.split('_')[:-1])
            valid_data = df[[target_col, var]].dropna()
            x = valid_data[var].to_numpy()
            y = valid_data[target_col].to_numpy()
            print(var, 'x:', x)
            # plot
            ax[row_idx,col_idx].scatter(x,y, alpha=0.7, s=56)
            axes_wplot.append(k)
        
    # label axes
    ax[row_idx,col_idx].set_xlabel(var_base, fontsize=14)
    ax[row_idx,col_idx].set_ylabel(target_col, fontsize=14)
    ax[row_idx, col_idx].set_title(f'{target_col} vs {var_base}', fontsize=16)
    ax[row_idx, col_idx].legend([f'{var.split("_")[-1]}: '+r'$corr_{sp}$='+f'{corr}' for var,corr in zip(var_plotted,corr_plotted)], frameon=True, fontsize=14)

# hide empty subplots
for k in range(nrows*ncols):
    if k not in axes_wplot:
        row_idx = int(np.floor(k/ncols))
        col_idx = k%ncols
        ax[row_idx, col_idx].set_axis_off()

plt.tight_layout()
plt.savefig(data_folder + 'ScatterPlot_topCorrelations.png')
plt.show()
        
        
