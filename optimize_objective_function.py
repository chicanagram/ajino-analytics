#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:49:46 2024

@author: charmainechia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from variables import data_folder, yvar_list_key, features_selected_dict
from model_utils import run_trainval_test, plot_model_metrics_all
from utils import sort_list, get_XYdata_for_featureset

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
                 n_initial_points=20,
                 n_calls=200,
                 n_iter=1,
                 titer_wt=2,
                 man5_wt=3,
                 fuc_wt=1,
                 gal_wt=1,
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
        self.bounds = bounds
        self.res0 = res0
        self.yvar_list = yvar_list
        self.titer_wt = titer_wt
        self.man5_wt = man5_wt
        self.fuc_wt = fuc_wt
        self.gal_wt = gal_wt
        print(self.x_input_0)
        print('Initial Obj Fn value:', self.f(self.x_input_0))          

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
        return obj_fn

    def f(self, x, print_x=True): 
        # X0, features_selected_dict, yvar_list
        # idxs_opt for variables under consideration
        if print_x:
            print('x:', np.round(x, 4))
        res = {}
        for i, yvar in enumerate(self.yvar_list): 
            # get features used for model (yvar)
            idxs_features_selected = self.features_selected_idxs_dict[yvar]
            # get original feature values and update features to optimize with new values to test
            x_input = self.x0.copy()
            x_input[self.idxs_opt] = x
            x_input = x_input[idxs_features_selected]
            # scale data vector
            x_input = (x_input - self.XMEAN[idxs_features_selected])/self.XSTD[idxs_features_selected]
            # get model for yvar and perform prediction
            model = self.SURROGATE_MODELS[yvar]
            ypred = float(model.predict(x_input.reshape(1,-1)))
            res.update({yvar: ypred})
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
        
        # plot convergence
        plot = plot_convergence(
            (self.optimizer_name, optimizer_results),
            )
        plot.legend(loc="best", prop={'size': 6}, numpoints=1)
        plt.show()
        
        # plot evaluations of optimizer
        _ = plot_objective(optimizer_results[0], size=4, n_samples=100)


    def run(self, n_iter=1, optimizer='gp_minimize'):
        from skopt import gp_minimize, forest_minimize
        # Surrogate minimization
        self.optimizer_name = optimizer
        optimizer_results = []
        for n in range(n_iter):
            if optimizer=='gp_minimize':
                optres = gp_minimize(self.f, self.bounds, x0=self.x_input_0, n_calls=self.n_calls, random_state=n, n_initial_points=self.n_initial_points)
            elif optimizer=='forest_minimize':
                optres = forest_minimize(self.f, self.bounds, x0=self.x_input_0, n_calls=self.n_calls, random_state=n, n_initial_points=self.n_initial_points)
            # elif optimizer=='scipy_minimize':
            #     from scipy.optimize import minimize
            #     optres = minimize(self.f, x0=self.x_input_0, bounds=self.bounds)
            optimizer_results.append(optres)
        return optimizer_results
    

class Estimator: 
    def __init__(self):
        return None
    
    def score(self, X): 
        # get x input
        # x0
        # res0
        # features_selected_idxs_dict
        # idxs_opt
        # bounds
        # instantiate Optimizer
        Opt = OptimizeCQAs(x0, res0, features_selected_idxs_dict, idxs_opt, bounds, XMEAN, XSTD, SURROGATE_MODELS)
        opt_res = Opt.run(n_iter=1, optimizer='forest_minimize')
        return opt_res

#%% RUN MODEL EVALUATION WITH DIFFERENT FEATURE SETS

# get data
X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
featureset_suffix = ''
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)

# evaluate models
feature_drop_test_res = []
# for yvar in yvar_list_key: 
# for yvar in [yvar_list_key[0],yvar_list_key[3]]: 
for yvar in [yvar_list_key[3]]: 
    print(yvar)
    features_selected_yvar = features_selected_dict[yvar].copy()
    kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS = run_trainval_test(X, Y, yvar_list=[yvar], xvar_selected={yvar:features_selected_yvar}, xvar_list_all=xvar_list_all, dataset_name_wsuffix=dataset_name_wsuffix, featureset_suffix=featureset_suffix)
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

#%% RUN MODEL WITH SINGLE FEATURE SET

# get data
X_featureset_idx, Y_featureset_idx = 1,0
dataset_name = f'X{X_featureset_idx}Y{Y_featureset_idx}'
dataset_suffix = ''
featureset_suffix = ''
dataset_name_wsuffix = dataset_name + dataset_suffix
Y, X, _, yvar_list_all, xvar_list_all = get_XYdata_for_featureset(X_featureset_idx, Y_featureset_idx, dataset_suffix=dataset_suffix, data_folder=data_folder)
models_to_eval_list = ['randomforest']

yvar_list = yvar_list_key # ['Titer (mg/L)_14']# 

# get results WITHOUT feature selection
kfold_metrics_orig, kfold_metrics_avg_orig, SURROGATE_MODELS_orig = run_trainval_test(X, Y, yvar_list, xvar_list_all, xvar_list_all, dataset_name_wsuffix, featureset_suffix)

# get results WITH feature selection
kfold_metrics, kfold_metrics_avg, SURROGATE_MODELS = run_trainval_test(X, Y, yvar_list, features_selected_dict, xvar_list_all, dataset_name_wsuffix, featureset_suffix)

# combine results
model_metrics_df_dict = {0: kfold_metrics_avg_orig, 1: kfold_metrics_avg}
plot_model_metrics_all(model_metrics_df_dict, models_to_eval_list, yvar_list, suffix_list=['_train_avg', '_test_avg'], annotate_vals=True)


#%% DEFINE OPTIMIZATION INPUTS

# get data and scaling constants
df = pd.read_csv(f'{data_folder}X0Y0.csv', index_col=0)
X = df.loc[:, xvar_list_all]
XMEAN = X.mean(axis=0).to_numpy()
XSTD = X.std(axis=0).to_numpy()

# get average X values for starting condition, e.g. Basal-A, Feed-a, pH 7, DO 40, feed % 6
x0 = df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a') & (df['Feed medium']=='Feed-a')].loc[:, xvar_list_all].mean(axis=0)
x0[['DO', 'pH', 'feed vol']] = [40, 7.0, 0.2335]
x0 = x0.to_numpy().reshape(-1)
features_selected_idxs_dict = {}
for yvar in yvar_list_key:    
    features_selected = features_selected_dict[yvar]
    features_selected_idxs_dict[yvar] = [xvar_list_all.index(xvar) for xvar in features_selected]
print('features_selected_idxs_dict:', features_selected_idxs_dict)
    
# get average Y (CQA) values for starting condition
y0 = df[(df['Basal medium']=='Basal-A') & (df['Feed medium']=='Feed-a') & (df['Feed medium']=='Feed-a') & (df['DO']==40) & (df['pH']==7) & (df['feed %']==6)].loc[:, yvar_list_key].mean(axis=0).to_numpy()
res0 = {yvar:val for yvar, val in zip(yvar_list_key, y0)}
print('Starting CQA values:', res0)

# set feature indices to optimize
features_selected_all = []
for yvar, features_selected_yvar in features_selected_dict.items():
    features_selected_all += features_selected_yvar
features_selected_all = sort_list(list(set(features_selected_all)))
features_opt = features_selected_all.copy()
# features_opt = ['feed vol', 'Riboflavin_feed', 'Riboflavin_basal', 'Pro_feed', 'Folic acid_basal', 'Co_feed', 'Ca_feed', 'DO', 'pH'] 
# features_opt = ['feed vol', 'Riboflavin_feed', 'Riboflavin_basal', 'Folic acid_basal', 'Co_feed', 'Pro_feed', 'DO', 'pH'] 
idxs_opt = np.array([xvar_list_all.index(xvar) for xvar in features_opt])
print('features_opt:', features_opt)
print('idxs_opt:', idxs_opt)

# get bounds for optimization of each feature
bounds = []
for feature_idx_to_opt in idxs_opt:
    xvar = xvar_list_all[feature_idx_to_opt]
    lbnd = df.loc[:,xvar].min()
    ubnd = df.loc[:,xvar].max()
    print(xvar, lbnd, ubnd)
    bounds.append((lbnd,ubnd))
bounds = tuple(bounds)

#%% RUN OPTIMIZATION 
OptGP = OptimizeCQAs(x0, res0, features_selected_idxs_dict, features_opt, idxs_opt, bounds, XMEAN, XSTD, SURROGATE_MODELS, 
                     titer_wt=2, man5_wt=4, fuc_wt=1, gal_wt=1, n_calls=30)
opt_res = OptGP.run(n_iter=1, optimizer='gp_minimize')
# opt_res = OptGP.run(n_iter=1, optimizer='forest_minimize')
# opt_res = OptGP.run(n_iter=100, optimizer='scipy_minimize')

# plot results
OptGP.show_results(opt_res)




