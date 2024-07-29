#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:42:17 2024

@author: charmainechia
"""

def sort_list(lst):
    lst.sort()
    return lst

def dict_update(d, keyval):
    d.update(keyval)
    return d

var_dict_all = {
    'inputs': [
        'Basal medium',
        'Feed medium',
        'DO',
        'pH',
        'feed %',
        'feed day',
        'n'
        ],
    'VCD, VIA, Titer, metabolites': [
        'VCD (E6 cells/mL)',
        'Viability (%)',
        'Titer (mg/L)',
        'Osmolarity (mOsm/kg)',
        'Glucose (g/L)',
        'Lac (g/L)',
        'NH4+ (mM)'
        ],
    'AA': [
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
        'Val'
        ],
    'VT': [
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
        ],
    'MT': [
        'Na', 
        'Mg', 
        'P', 
        'K', 
        'Ca', 
        'Fe', 
        'Co', 
        'Cu', 
        'Zn', 
        'Mn'
        ],
    'Glyco': [
        'Man5', 
        'G0', 
        'G0F', 
        'G1', 
        'G1Fa', 
        'G1Fb', 
        'G2', 
        'G2F', 
        'Other'
        ],
    'Nuc, Amine': [
        'Ethanolamine', 
        'Putrescine', 
        'Uridine', 
        'Adenosine'
        ]
    }

overall_glyco_cqas = {
    'mannosylation': ['Man5'],
    'fucosylation': ['G0F', 'G1Fa', 'G1Fb', 'G2F'],
    'galactosylation': ['G0', 'G0F', 'G1', 'G1Fa', 'G1Fb', 'G2', 'G2F']
    }

#%% X % Y DATASET VARIABLES
xvar_list_all = [
    'Adenosine_0',
    'Ala_0',
    'Arg_0',
    'Asn_0',
    'Asp_0',
    'Biotin_0',
    'Ca_0',
    'Choline_0',
    'Co_0',
    'Cu_0',
    'Cyanocobalamin_0',
    'Ethanolamine_0',
    'Fe_0',
    'Folic acid_0',
    'Gln_0',
    'Glu_0',
    'Glucose (g/L)_0',
    'Gly-Tyr_0',
    'Gly_0',
    'His_0',
    'HyPro_0',
    'Ile_0',
    'K_0',
    'Leu_0',
    'Lys_0',
    'Met_0',
    'Mg_0',
    'Mn_0',
    'Na_0',
    'Nicotinamide_0',
    'Orn_0',
    'P_0',
    'Pantothenic  acid_0',
    'Phe_0',
    'Pro_0',
    'Putrescine_0',
    'Pyridoxal_0',
    'Pyridoxine_0',
    'Riboflavin_0',
    'Ser_0',
    'Tau_0',
    'Thiamin_0',
    'Thr_0',
    'Trp_0',
    'Tyr_0',
    'Uridine_0',
    'Val_0',
    'Zn_0',
    'DO',
    'pH',
    'feed %'
    ]

xvar_list = [
    'Ala_0',
    'Arg_0',
    'Asn_0',
    'Asp_0',
    'Biotin_0',
    'Ca_0',
    'Choline_0',
    'Co_0',
    'Cu_0',
    'Cyanocobalamin_0',
    'Ethanolamine_0',
    'Fe_0',
    'Folic acid_0',
    'Gln_0',
    'Glu_0',
    'Glucose (g/L)_0',
    'Gly_0',
    'His_0',
    'HyPro_0',
    'Ile_0',
    'K_0',
    'Leu_0',
    'Lys_0',
    'Met_0',
    'Mg_0',
    'Mn_0',
    'Na_0',
    'Nicotinamide_0',
    'Orn_0',
    'P_0',
    'Pantothenic  acid_0',
    'Phe_0',
    'Pro_0',
    'Pyridoxal_0',
    'Pyridoxine_0',
    'Riboflavin_0',
    'Ser_0',
    'Tau_0',
    'Thr_0',
    'Trp_0',
    'Tyr_0',
    'Uridine_0',
    'Val_0',
    'Zn_0',
    'DO',
    'pH',
    'feed %'
 ]

yvar_list = [
    'Titer (mg/L)_14',
    'mannosylation_14',
    'fucosylation_14',
    'galactosylation_14',
    'G0_14',
    'G0F_14',
    'G1_14',
    'G1Fa_14',
    'G1Fb_14',
    'G2_14',
    'G2F_14',
    'Other_14'
    ]

yvar_sublist_sets = [
    ['Titer (mg/L)_14', 'mannosylation_14', 'fucosylation_14', 'galactosylation_14'], 
    ['G0_14', 'G0F_14', 'G1_14', 'G1Fa_14'],
    ['G1Fb_14', 'G2_14', 'G2F_14', 'Other_14']
    ]


#%% MODEL PARAMETERS

model_params = {
'randomforest': {
    'Titer (mg/L)_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
    'mannosylation_14': [{'model_type': 'randomforest', 'n_estimators': 120}],
    'fucosylation_14': [{'model_type': 'randomforest', 'n_estimators': 120}],
    'galactosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
    'G0_14': [{'model_type': 'randomforest', 'n_estimators': 80}],
    'G0F_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
    'G1_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
    'G1Fa_14': [{'model_type': 'randomforest', 'n_estimators': 120}],
    'G1Fb_14': [{'model_type': 'randomforest', 'n_estimators': 120}],
    'G2_14': [{'model_type': 'randomforest', 'n_estimators': 80}],
    'G2F_14': [{'model_type': 'randomforest', 'n_estimators': 60}],
    'Other_14': [{'model_type': 'randomforest', 'n_estimators': 80}]    
    },
    
'plsr' : {
    'Titer (mg/L)_14': [{'model_type': 'plsr', 'n_components':5}],
    'mannosylation_14': [{'model_type': 'plsr', 'n_components':4}],
    'fucosylation_14': [{'model_type': 'plsr', 'n_components':10}],
    'galactosylation_14': [{'model_type': 'plsr', 'n_components':10}],
    'G0_14': [{'model_type': 'plsr', 'n_components':10}],
    'G0F_14': [{'model_type': 'plsr', 'n_components':9}],
    'G1_14': [{'model_type': 'plsr', 'n_components':11}],
    'G1Fa_14': [{'model_type': 'plsr', 'n_components':10}],
    'G1Fb_14': [{'model_type': 'plsr', 'n_components':11}],
    'G2_14': [{'model_type': 'plsr', 'n_components':8}],
    'G2F_14': [{'model_type': 'plsr', 'n_components':10}],
    'Other_14': [{'model_type': 'plsr', 'n_components':8}]
    },

'lasso': {
    'Titer (mg/L)_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':50}],
    'mannosylation_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.1}],
    'fucosylation_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.1}],
    'galactosylation_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.1}],
    'G0_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.05}],
    'G0F_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.005}],
    'G1_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.05}],
    'G1Fa_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.05}],
    'G1Fb_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.05}],
    'G2_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.01}],
    'G2F_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.01}],
    'Other_14': [{'model_type': 'lasso', 'max_iter':50000, 'alpha':0.01}]     
    }
}

