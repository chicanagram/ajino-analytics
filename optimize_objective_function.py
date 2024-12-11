#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:49:46 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variables import data_folder, yvar_list_key, feature_selections, model_params, model_cmap, process_features
from model_utils import run_trainval_test, plot_model_metrics_all
from utils import sort_list, get_XYdata_for_featureset
from plot_utils import convert_figidx_to_rowcolidx

class OptimizeCQAs:
    def __init__(self, 
                 x0, 
                 res0, 
                 features_selected_idxs_dict,
                 features_opt,
                 idxs_opt, 
                 bounds,
                 XMEAN, 
                 XSTD,
                 SURROGATE_MODELS,
                 yvar_list=yvar_list_key,
                 models_to_evaluate=['randomforest','xgb'],
                 n_initial_points=20,
                 n_calls=200,
                 n_iter=1,
                 titer_wt=2,
                 man5_wt=3,
                 fuc_wt=1,
                 gal_wt=1,
                 csv_fpath=None,
                 optimizer_random_state=None,
                 plot_objective_plot=True
                 ): 
        
        self.ncalls_count = 0
        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.n_iter = n_iter
        self.x0 = x0
        self.x_input_0 = list(self.x0[idxs_opt])
        self.features_opt = features_opt
        self.idxs_opt = idxs_opt
        self.features_selected_idxs_dict = features_selected_idxs_dict
        self.XMEAN = XMEAN
        self.XSTD = XSTD
        self.SURROGATE_MODELS = SURROGATE_MODELS
        self.models_to_evaluate = models_to_evaluate
        self.bounds = bounds
        self.res0 = res0
        self.yvar_list = yvar_list
        self.titer_wt = titer_wt
        self.man5_wt = man5_wt
        self.fuc_wt = fuc_wt
        self.gal_wt = gal_wt
        self.csv = []
        self.csv_columns = ['i', 'obj_fn'] + self.yvar_list + self.features_opt
        self.csv_fpath = csv_fpath
        self.optimizer_random_state = optimizer_random_state
        self.plot_objective_plot = plot_objective_plot
        print('Initial Obj Fn value (based on CQA predicitons from x0):', round(self.f(self.x_input_0),4))          

    def calculate_objective_function(self, res):
        titer_0 = self.res0['Titer (mg/L)_14']
        man5_0 = self.res0['mannosylation_14']
        fuc_0 = self.res0['fucosylation_14']
        gal_0 = self.res0['galactosylation_14']
        titer = res['Titer (mg/L)_14']
        man5 = res['mannosylation_14']
        fuc = res['fucosylation_14']
        gal = res['galactosylation_14']
        # combine
        obj_fn = -self.titer_wt*(titer-titer_0)/titer_0 + self.man5_wt*(man5-man5_0)/man5_0 + self.fuc_wt*(fuc-fuc_0)**2/fuc_0 + self.gal_wt*(gal-gal_0)**2/gal_0
        # update csv
        row_dict = {'i': self.ncalls_count, 'obj_fn': obj_fn, 'Titer (mg/L)_14': titer, 'mannosylation_14': man5, 'fucosylation_14': fuc, 'galactosylation_14': gal}
        row_dict.update({xvar: round(xval,4) for xvar, xval in zip(self.features_opt, self.x)})
        self.csv.append(row_dict)
        return obj_fn

    def f(self, x, print_x=True): 
        # X0, features_selected_dict, yvar_list
        # idxs_opt for variables under consideration
        self.x = x
        res_allmodels = {yvar:[] for yvar in self.yvar_list}
        res = {}
        for i, yvar in enumerate(self.yvar_list): 
            # get features used for model (yvar)
            idxs_features_selected = self.features_selected_idxs_dict[yvar]
            # get original feature values and update features to optimize with new values to test
            x_input = self.x0.copy()
            x_input[self.idxs_opt] = x
            if print_x:
                print('\n', 'Values of features to be optimized:')
                for idx, feature in zip(self.idxs_opt, self.features_opt):
                    print(feature, round(x_input[idx],4))
            # scale data vector
            x_input = x_input[idxs_features_selected]
            x_input = (x_input - self.XMEAN[idxs_features_selected])/self.XSTD[idxs_features_selected]
            # get model for yvar and perform prediction
            for model_type in self.models_to_evaluate:
                model = self.SURROGATE_MODELS[yvar][model_type]
                ypred = float(model.predict(x_input.reshape(1,-1)))
                res_allmodels[yvar].append(ypred)
            # average results, if needed
            res[yvar] = np.mean(np.array(res_allmodels[yvar]))
            print(f'{yvar}: {round(ypred,2)}', end=', ')
            
        # calculate objective function
        obj_fn = self.calculate_objective_function(res)
        self.ncalls_count += 1
        print('\n NCALLS COUNT:', self.ncalls_count)
        print('OBJECTIVE FUNCTION:', round(obj_fn,4))
        print()
        
        return obj_fn
    
    def show_results(self, optimizer_results):
        from skopt.plots import plot_convergence
        from skopt.plots import plot_objective
        
        # print results
        for i, optres in enumerate(optimizer_results):
            print(i)
            print('Best Obj Fun score:')
            _ = self.f(optres.x, print_x=False)
            print('Best parameter vals:')
            for feature, val in zip(self.features_opt, optres.x):
                print(feature, ':', round(val,4))
            print()
        
        # save CSV
        self.csv = pd.DataFrame(self.csv, columns=self.csv_columns)
        if self.csv_fpath is not None:
            self.csv.to_csv(self.csv_fpath, index=False)
        
        # plot convergence
        if self.optimizer_name in ['gp_minimize', 'forest_minimize']:
            plot = plot_convergence(
                (self.optimizer_name, optimizer_results),
                )
            plot.legend(loc="best", prop={'size': 6}, numpoints=1)
            plt.show()
            
            # plot evaluations of optimizer
            if self.plot_objective_plot:
                _ = plot_objective(optimizer_results[0], size=4, n_samples=100)
                plt.show()
        else: 
            self.csv['obj_fn_min'] = np.nan
            for ncall in range(len(self.csv)):
                self.csv.iat[ncall,-1] = self.csv.iloc[:ncall]['obj_fn'].min()
            plt.plot(self.csv['i'], self.csv['obj_fn_min'])
            plt.ylabel('minf(x) after n calls')
            plt.xlabel('Number of calls n')
            plt.title('Convergence plot')
            plt.show()
        


    def run(self, n_iter=1, optimizer='gp_minimize'):
        # Surrogate minimization
        self.optimizer_name = optimizer
        optimizer_results = []
        for n in range(n_iter):
            if optimizer=='gp_minimize':
                from skopt import gp_minimize
                optres = gp_minimize(self.f, self.bounds, x0=self.x_input_0, n_calls=self.n_calls, n_initial_points=self.n_initial_points, random_state=self.optimizer_random_state)
            elif optimizer=='forest_minimize':
                from skopt import forest_minimize
                optres = forest_minimize(self.f, self.bounds, x0=self.x_input_0, n_calls=self.n_calls, n_initial_points=self.n_initial_points, random_state=self.optimizer_random_state)
            elif optimizer=='scipy_minimize':
                from scipy.optimize import minimize
                optres = minimize(self.f, x0=self.x_input_0, bounds=self.bounds)
            elif optimizer=='annealing_minimize':
                from scipy.optimize import dual_annealing
                optres = dual_annealing(self.f, x0=self.x_input_0, bounds=self.bounds, maxiter=self.n_calls)
            optimizer_results.append(optres)
        return optimizer_results
    


#%% RUN MODEL EVALUATION WITH DIFFERENT FEATURE SETS

# get data
X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
featureset_suffix = '_curated' # '_model-opt'
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)

# evaluate models
feature_drop_test_res = []
features_selected_dict = feature_selections[featureset_suffix]
# for yvar in yvar_list_key: 
# for yvar in [yvar_list_key[0],yvar_list_key[3]]: 
for yvar in [yvar_list_key[3]]: 
    print(yvar)
    features_selected_yvar = features_selected_dict[yvar].copy()
    kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS, ypred_train_bymodel, ypred_test_bymodel = run_trainval_test(X, Y, yvar_list=[yvar], xvar_selected={yvar:features_selected_yvar}, xvar_list_all=xvar_list_all, dataset_name_wsuffix=dataset_name_wsuffix, featureset_suffix=featureset_suffix)
    feature_drop_test_res.append({'yvar':yvar, 'feature_dropped':None, 'features_selected': features_selected_yvar, 'r2_test_avg': kfold_metrics_avg.iloc[0]['r2_test_avg']})
    for k, feature in enumerate(features_selected_yvar): 
        features_selected_yvar_MOD = features_selected_dict[yvar].copy()
        features_selected_yvar_MOD.remove(feature)
        features_selected_dict_MOD = {yvar: features_selected_yvar_MOD}    
        print(feature, features_selected_yvar_MOD)
        kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS = run_trainval_test(X, Y, yvar_list=[yvar], xvar_selected=features_selected_dict_MOD, xvar_list_all=xvar_list_all, dataset_name_wsuffix=dataset_name_wsuffix, featureset_suffix=featureset_suffix)
        print(f'Dropped {feature}.  Features selected: {features_selected_yvar_MOD}')
        feature_drop_test_res.append({'yvar':yvar, 'feature_dropped':feature, 'features_selected': features_selected_yvar_MOD, 'r2_test_avg': kfold_metrics_avg.iloc[0]['r2_test_avg']})

feature_drop_test_res = pd.DataFrame(feature_drop_test_res)
print(feature_drop_test_res)

#%% DEFINE OPTIMIZATION INPUTS

starting_point_list = ['best3'] # ['best1', 'best2', 'best3'] # ['best3'] # 
minimizer_list = ['annealing_minimize'] # ['forest_minimize'] # ['gp_minimize', 'forest_minimize']
wts = [3,1,0,0.2, 0.2] # titer_wt, man5_wt, fuc_wt, gal_wt
n_calls = 1500
n_iter = 1
cap_DO_lbnd = False


# get data and scaling constants
df = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
X = df.loc[:, xvar_list_all]
XMEAN = X.mean(axis=0).to_numpy()
XSTD = X.std(axis=0).to_numpy()
df_good = df[(df['Titer (mg/L)_14']>9000) & (df['mannosylation_14']<12)].sort_values(by=['Titer (mg/L)_14'], ascending=False)


starting_point_dict = {
    # Basal-A, Feed-a, pH7, DO40, feed 6%
    'avg-Basal-A-Feed-a': { 
        'x0': df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a')].loc[:, xvar_list_all].mean(axis=0),
        'y0': df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a') & (df['DO']==40) & (df['pH']==7) & (df['feed %']==6)].loc[:, yvar_list_key].mean(axis=0),
        },
    # Basal-D, Feed-a, pH7, DO40, feed 6%
    'avg-Basal-D-Feed-a': { 
        'x0': df[(df['Basal medium']=='Basal-D') & (df['Feed medium']=='Feed-a')].loc[:, xvar_list_all].mean(axis=0),
        'y0': df[(df['Basal medium']=='Basal-D') & (df['Feed medium']=='Feed-a') & (df['DO']==40) & (df['pH']==7) & (df['feed %']==6)].loc[:, yvar_list_key].mean(axis=0)
        },
    # Basal-A, Feed-a, pH7.1, DO60, feed 8%
    'best1': {
        'x0': df_good.iloc[:1,:].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[:1,:].loc[:, yvar_list_key].mean(axis=0)
        },    
    # Basal-A, Feed-a, pH7.0, DO20, feed 6%
    'best2': {
        'x0': df_good.iloc[2:3,:].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[2:3,:].loc[:, yvar_list_key].mean(axis=0)
        },
    # Basal-A, Feed-a, pH7.0, DO20, feed 8%
    'best3': {
        'x0': df_good.iloc[3:,:].loc[:, xvar_list_all].mean(axis=0),
        'y0': df_good.iloc[3:,:].loc[:, yvar_list_key].mean(axis=0)
        }           
    }

# starting point parameters to fix
starting_point_params_to_fix = {
    # 'pH': 7,
    # 'DO': 40,
    # 'feed vol': 0.2335
    }


#%% GET SURROGATE MODEL WITH CHOSEN FEATURE SET

# get data
X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
featureset_suffix =  '_curated2' # '_combi2' # '_compactness-opt'
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
models_to_eval_list = ['randomforest', 'xgb']

yvar_list = yvar_list_key # ['Titer (mg/L)_14']# 
features_selected_dict = feature_selections[featureset_suffix]

# set feature indices to optimize
features_selected_all_ = []
for yvar, features_selected_yvar in features_selected_dict.items():
    features_selected_yvar_ = [feature for feature in features_selected_yvar if feature not in starting_point_params_to_fix]
    features_selected_all_ += features_selected_yvar_
features_selected_all = sort_list(list(set(features_selected_all_))) # sort and deduplicate list
features_opt = features_selected_all.copy()
idxs_opt = np.array([xvar_list_all.index(xvar) for xvar in features_opt])
print('features_opt:', features_opt)
print('idxs_opt:', idxs_opt)


# get results WITHOUT feature selection
kfold_metrics_orig, kfold_metrics_avg_orig, SURROGATE_MODELS_orig, ypred_train_bymodel_orig, ypred_test_bymodel_orig = run_trainval_test(X, Y, yvar_list, xvar_list_all, xvar_list_all, dataset_name_wsuffix, featureset_suffix, models_to_eval_list=models_to_eval_list, model_cmap=model_cmap)

# get results WITH feature selection
kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS, ypred_train_bymodel, ypred_test_bymodel = run_trainval_test(X, Y, yvar_list, features_selected_dict, xvar_list_all, dataset_name_wsuffix, featureset_suffix, models_to_eval_list=models_to_eval_list, model_cmap=model_cmap)

# combine results
model_metrics_df_dict = {0: kfold_metrics_avg_orig, 1: kfold_metrics_avg}
plot_model_metrics_all(model_metrics_df_dict, models_to_eval_list+['ENSEMBLE'], yvar_list, suffix_list=['_train_avg', '_test_avg'], annotate_vals=True)



#%% 
for starting_point in starting_point_list: 
    for minimizer in minimizer_list:
        
        print(minimizer, starting_point)
        # get best conditions in dataset
        x0 = starting_point_dict[starting_point]['x0'].to_numpy().reshape(-1)
        y0 = starting_point_dict[starting_point]['y0'].to_numpy()
        features_selected_idxs_dict = {}
        for yvar in yvar_list_key:    
            features_selected = features_selected_dict[yvar]
            features_selected_idxs_dict[yvar] = [xvar_list_all.index(xvar) for xvar in features_selected]
        print('features_selected_idxs_dict:', features_selected_idxs_dict)
        
        # get bounds for optimization of each feature
        bounds = []
        for feature, feature_idx_to_opt in zip(features_opt, idxs_opt):
            xvar = xvar_list_all[feature_idx_to_opt]
            lbnd = df.loc[:,xvar].min()
            ubnd = df.loc[:,xvar].max()
            if feature=='DO' and cap_DO_lbnd:
                lbnd = 40
            bounds.append((lbnd,ubnd))
        bounds = tuple(bounds)
        
        # get average Y (CQA) values for starting condition
        res0 = {yvar:val for yvar, val in zip(yvar_list_key, y0)}
        OptCQA0 = OptimizeCQAs(x0, res0, features_selected_idxs_dict, features_opt, idxs_opt, bounds, XMEAN, XSTD, SURROGATE_MODELS, 
                             titer_wt=wts[0], man5_wt=wts[1], fuc_wt=wts[2], gal_wt=wts[3], n_calls=1, csv_fpath=None)
        objfn0 = OptCQA0.calculate_objective_function(res0)
        print('Actual Starting CQA values:', res0)
        print('Actual Starting Obj Fn value:', objfn0, '\n')
    
        
        for n in range(n_iter):
            csv_suffix = '_' + minimizer.split('_')[0] + f'_{starting_point}_{n}'
            csv_fpath = f'{data_folder}optimizeCQAs_fs={featureset_suffix[1:]}_wts={"-".join(str(x) for x in wts)}_ncalls={n_calls}{csv_suffix}.csv'
            OptCQA = OptimizeCQAs(x0, res0, features_selected_idxs_dict, features_opt, idxs_opt, bounds, XMEAN, XSTD, SURROGATE_MODELS, 
                                  titer_wt=wts[0], man5_wt=wts[1], fuc_wt=wts[2], gal_wt=wts[3], 
                                  n_calls=n_calls, optimizer_random_state=n, csv_fpath=csv_fpath)
            if minimizer == 'gp_minimize':
                opt_res = OptCQA.run(n_iter=1, optimizer='gp_minimize')
            elif minimizer =='forest_minimize':
                opt_res = OptCQA.run(n_iter=1, optimizer='forest_minimize')
            elif minimizer =='scipy_minimize': # doesn't really work
                opt_res = OptCQA.run(n_iter=1, optimizer='scipy_minimize')
            elif minimizer == 'annealing_minimize':
                opt_res = OptCQA.run(n_iter=1, optimizer='annealing_minimize')
            
            # plot results
            OptCQA.show_results(opt_res)
            
            # plot scatter of Titer vs Mannosylation
            plt.scatter(OptCQA.csv['Titer (mg/L)_14'], OptCQA.csv['mannosylation_14'], color='orange', s=6, alpha=0.5)
            plt.scatter(df['Titer (mg/L)_14'], df['mannosylation_14'], color='b', s=6, alpha=0.5)
            plt.legend(['In-silico exploration', 'Experimental data'])
            plt.xlabel('Titer (mg/L)_14')
            plt.ylabel('mannosylation_14')
            plt.title(f'{minimizer}: wts={", ".join(str(x) for x in wts)}, {starting_point}, iter{n}')
            plt.show()
            
    
#%% get best conditions

# get full experimental data
exp_data_all = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
exp_data = exp_data_all[(exp_data_all['Basal medium']=='Basal-A') & (exp_data_all['Feed medium']=='Feed-a')]
exp_data['exp_or_sim'] = 'exp'
x0 = starting_point_dict['avg-Basal-A-Feed-a']['x0'].to_dict()

sim_results = {
    '_curated2': {'csv_flist': [
        'optimizeCQAs_fs=curated2_wts=3-1-0-0.2-0.2_ncalls=1500_annealing_best3_0.csv'
        ],
        'T_thres':(8800,8800), 'M_thres':(12,8.5)},
    '_combi1': {'csv_flist': [
        'optimizeCQAs_fs=combi1_wts=3-1-0-0.2-0.2_ncalls=1500_annealing_best3_0.csv',
        'optimizeCQAs_fs=combi1_wts=3-2-0-0.2-0.2_ncalls=1500_annealing_best3_0.csv',
        'optimizeCQAs_fs=combi1_wts=3-1-0-0.2-0.2_ncalls=1000_annealing_best2_0.csv',
        'optimizeCQAs_fs=combi1_wts=3-1-0-0.2-0.2_ncalls=1000_annealing_best3_0.csv',
        'optimizeCQAs_fs=combi1_wts=3-1-0-0.2-0.2_ncalls=1500_annealing_best1_0.csv'
        ],
        'T_thres':(8800,8800), 'M_thres':(12,8.5)},
    '_combi2': {'csv_flist': [
        'optimizeCQAs_fs=combi2_wts=3-1-0-0.2-0.2_ncalls=1500_annealing_best1_0.csv',
        'optimizeCQAs_fs=combi2_wts=3-1-0-0.2-0.2_ncalls=1500_annealing_best3_0.csv',
        ],
        'T_thres':(8800,8800), 'M_thres':(12,8.5)},
    }

featureset_suffix = '_curated2' # '_combi1'
csv_fdict = sim_results[featureset_suffix]
titer_thres = csv_fdict['T_thres']
man5_thres = csv_fdict['M_thres']

for i, csv_fname in enumerate(csv_fdict['csv_flist']):
    sim = pd.read_csv(data_folder+csv_fname, index_col=0).drop_duplicates(subset=['obj_fn']+yvar_list_key)
    if i==0:
        sim_all = sim.copy()
    else:
        sim_all = pd.concat([sim_all, sim])
sim_all = sim_all.drop_duplicates(subset=['obj_fn']+yvar_list_key)
sim_all['titer_over_man5'] = sim_all['Titer (mg/L)_14']/sim_all['mannosylation_14']
sim_all.to_csv(data_folder + f'optimizeCQAs_fs={featureset_suffix[1:]}_all.csv')
features_opt = [f for f in sim_all if f in xvar_list_all]
print(features_opt)
sim_all['exp_or_sim'] = 'sim'

# get best conditions
sim_opt = sim_all[(sim_all['Titer (mg/L)_14']>titer_thres[1]) & (sim_all['mannosylation_14']<man5_thres[1])].sort_values(by='titer_over_man5', ascending=False)
print(sim_opt[yvar_list_key + features_opt + ['titer_over_man5']])
print(sim_opt.iloc[0][yvar_list_key+features_opt])

# get scatter plot
plt.scatter(sim_all['Titer (mg/L)_14'], sim_all['mannosylation_14'], color='orange', s=6, alpha=0.5)
plt.scatter(sim_opt['Titer (mg/L)_14'], sim_opt['mannosylation_14'], color='r', s=6, alpha=0.2)
plt.scatter(exp_data['Titer (mg/L)_14'], exp_data['mannosylation_14'], color='b', s=6, alpha=0.5)
plt.legend(['In-silico exploration', 'Experimental data'])
plt.xlabel('Titer (mg/L)_14')
plt.ylabel('mannosylation_14')
plt.title(f'Simulation & experiment data using {featureset_suffix[1:]} featureset')
plt.show()
    
# get violin plots for selected optimized samples
df_all = pd.concat([exp_data[['exp_or_sim']+ yvar_list_key + features_opt], sim_all[['exp_or_sim'] + yvar_list_key + features_opt]], ignore_index=True)

# filter by selected conditions
df_exp_filt = df_all[(df_all['exp_or_sim']=='exp') & (df_all['Titer (mg/L)_14']>titer_thres[0]) & (df_all['mannosylation_14']<man5_thres[0])] 
df_sim_filt = df_all[(df_all['exp_or_sim']=='sim') & (df_all['Titer (mg/L)_14']>titer_thres[1]) & (df_all['mannosylation_14']<man5_thres[1])] 
df_all_filt = pd.concat([df_exp_filt, df_sim_filt], ignore_index=True)
df_all_filt = df_all_filt.drop_duplicates()

# get boxplots
ncols = 6
nrows = int(np.ceil(len(features_opt)/ncols))
fig, ax = plt.subplots(nrows,ncols, figsize=(42,22))
# nrows, ncols = 1,9
# fig, ax = plt.subplots(nrows,ncols, figsize=(42,6))
import seaborn as sns

features_opt_sorted = sort_list([f for f in features_opt if f not in process_features]) + process_features
unused_ax_idxs = [i for i in range(nrows*ncols) if i>=len(features_opt_sorted)]
for i, feature in enumerate(features_opt_sorted):
    row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
    if nrows>1: 
        ax_feature = ax[row_idx,col_idx]
    else: 
        ax_feature = ax[col_idx]
    # df_all_filt.boxplot(feature, by='exp_or_sim', ax=ax[row_idx,col_idx])
    sns.violinplot(x='exp_or_sim', y=feature, data=df_all_filt, ax=ax_feature)
    ax_feature.yaxis.grid(False)
    ax_feature.set_xlabel('exp_or_sim', fontsize=16)
    ax_feature.set_ylabel(feature, fontsize=16)    
for i in unused_ax_idxs:
    row_idx, col_idx = convert_figidx_to_rowcolidx(i, ncols)
    if nrows>1:
        ax[row_idx, col_idx].set_axis_off()
    else:
        ax[col_idx].set_axis_off()
fig.suptitle('Violin plots of feature conditions', y=0.90, fontsize=20)
plt.show()
