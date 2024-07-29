#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:50:05 2024

@author: charmainechia
"""

import pickle
import numpy as np
from matplotlib.pyplot import cm
from plot_utils import data_folder, figure_folder, plot_data_panel

    
#%% get data

# set dataset name
dataset_name = 'dataset0'
# dataset_name = 'DATA_avg'
pkl_fname = f'{dataset_name}.pkl'

# open pickle file
with open(f'{data_folder}{pkl_fname}', 'rb') as handle:
    datadict = pickle.load(handle)
      
# get number of unique experiment conditions and map to plot colors
exp_list = []
exp_to_color_dict = {}
for k in datadict:
    exp_label = datadict[k]['exp_label']
    if exp_label not in exp_list:
        exp_list.append(exp_label)

color = cm.rainbow(np.linspace(0, 1, len(exp_list)))
for i, c in enumerate(color):
   exp_to_color_dict[exp_list[i]] = c

#%% plot bioreactor CQAs and glycosylation

var_to_plot = [
    # bioreactor CQAs
    'VCD (E6 cells/mL)',
    'Viability (%)',
    'Titer (mg/L)',
    # 'qP (pg/cell/day)',
    'Glucose (g/L)',
    'Lac (g/L)',
    'NH4+ (mM)',
    'Osmolarity (mOsm/kg)',
    # glycosylation
    'Man5',    
    'G0',    
    'G0F',   
    'G1',  
    'G1Fa',   
    'G1Fb',    
    'G2',    
    'G2F',    
    'Other',    
    ]

figsize = (22,20)
nrows = 4
ncols = 4
figtitle = 'VCD, Via, Titer, Metabolite & Glycosylation CQAs'
savefig = f'{figure_folder}{dataset_name}_Bioreactor_Glyco_CQAs.png'

plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, exp_to_color_dict, figtitle, savefig)


#%% plot AA

var_to_plot = [
     'Ala',
     'Arg',
     'Asn',
     'Asp',
     'Gln',
     'Glu',
     'Gly',
     'Gly-Tyr',
     'His',
     'HyPro',
     'Ile',
     'Leu',
     'Lys',
     'Met',
     'Orn',
     'Phe',
     'Pro',
     'Ser',
     'Tau',
     'Thr',
     'Trp',
     'Tyr',
     'Val',
 ]
    
figsize = (28,20)
nrows = 5
ncols = 5
figtitle = 'Amino Acid concentrations'
savefig = f'{figure_folder}{dataset_name}_AA.png'

plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, exp_to_color_dict, figtitle, savefig)

#%% plot VT, Nuc, Amine
var_to_plot = [
        'Choline',
        'Thiamin',
        'Pyridoxal',
        'Pyridoxine',
        'Nicotinamide',
        'Pantothenic  acid',
        'Folic acid',
        'Cyanocobalamin',
        'Riboflavin',
        'Biotin'
        'Ethanolamine', 
        'Putrescine', 
        'Uridine', 
        'Adenosine'
        ]
figsize = (28,16)
nrows = 3
ncols = 5
figtitle = 'Vitamin, Nuc, Amine concentrations'
savefig = f'{figure_folder}{dataset_name}_VT_Nuc_Amine.png'

plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, exp_to_color_dict, figtitle, savefig)

#%% plot MT

var_to_plot = ['Na', 'Mg', 'P', 'K', 'Ca', 'Fe', 'Co', 'Cu', 'Zn', 'Mn']
figsize = (25,16)
nrows = 3
ncols = 4
figtitle = 'Metal ion concentrations'
savefig = f'{figure_folder}{dataset_name}_MT.png'

plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, exp_to_color_dict, figtitle, savefig)