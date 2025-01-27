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


# data folder
data_folder = '../ajino-analytics-data/'
# figure fodler
figure_folder = f'{data_folder}figures/'

# colormap 
model_cmap = {'randomforest': 'r', 'plsr': 'b', 'lasso': 'g', 'xgb':'purple', 'ENSEMBLE': 'k'}

sampling_rawdata_dict = {
    0: {'fname': '240402_20RP06-19_data for Astar_4conditions_v2.xlsx', 'skiprows': [2, 2, 2, 2, 2, None, 2, 1], 'usecols': ['B:CA', 'B:DR', 'B:BE', 'B:BE', 'B:AZ', None, 'B:AI', 'A:F'], 'cqa_startcol': [6, 6, 6, 6, 6, None, 2, 1], },
    1: {'fname': '240906_22DX05-12_media combination_AJI-Astar_v2.xlsx', 'skiprows': [2, 2, 2, 2, 2, 2, 2, 2], 'usecols': ['B:BH', 'B:FN', 'B:CA', 'B:CA', 'B:BT', 'B:AK', 'B:GM', 'B:J'], 'cqa_startcol': [8, 8, 8, 8, 8, 8, 2, 1], },
    2: {'fname': '240906_22DX05-12_process combination_AJI-Astar_v3.xlsx', 'skiprows': [2, 2, 2, 2, 2, 2, 2, 2], 'usecols': ['B:BF', 'B:FN', 'B:CA', 'B:CA', 'B:BT', 'B:AK', 'B:GM', 'B:D'], 'cqa_startcol': [8, 8, 8, 8, 8, 8, 2, 1], },
}

media_rawdata_dict = {
    0: {'fname': '240402_20RP06-19_data for Astar_4conditions_v2.xlsx', 'skiprows': 1, 'usecols': 'A:F', 'cqa_startcol': 1, },
    1: {'fname': '240906_22DX05-12_media combination_AJI-Astar_v2.xlsx', 'skiprows': 2, 'usecols': 'B:J', 'cqa_startcol': 1, },
    2: {'fname': '240906_22DX05-12_process combination_AJI-Astar_v3.xlsx', 'skiprows': 2, 'usecols': 'B:D', 'cqa_startcol': 1, },
}

sampling_days = {
    0: [0, 3, 5, 7, 9, 11, 13, 14],
    1: [0, 3, 5, 7, 9, 11, 14],
    2: [0, 3, 5, 7, 9, 11, 14],
}

sampling_days_itpl = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14],
    1: [0, 3, 5, 7, 9, 11, 14],
    2: [0, 3, 5, 7, 9, 11, 14],
}

vol_df_exp_cols = [
    'Day',
    'event',
    'time (d)',
    'time (m)',
    'vessel volume (mL)',
    'feed (mL)',
    'glucose (mL)'
]

overall_glyco_cqas = {
    'mannosylation': ['Man5'],
    'fucosylation': ['G0F', 'G1Fa', 'G1Fb', 'G2F'],
    'galactosylation': ['G1', 'G1Fa', 'G1Fb', 'G2', 'G2F']
}

yvar_sublist_sets = [
    ['Titer (mg/L)_14', 'mannosylation_14',
     'fucosylation_14', 'galactosylation_14'],
    ['G0_14', 'G0F_14', 'G1_14', 'G1Fa_14'],
    ['G1Fb_14', 'G2_14', 'G2F_14', 'Other_14']
]

yvar_list_key = ['Titer (mg/L)_14', 'mannosylation_14',
                 'fucosylation_14', 'galactosylation_14']

process_features = ['DO', 'pH', 'feed vol']

var_dict_all = {
    'inputs': [
        'Basal medium',
        'Feed medium',
        'DO',
        'pH',
        'feed %',
        'feed day',
        'feed vol',
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
    ],
    'media_components': [
        'Ala',
        'Arg',
        'Asn',
        'Asp',
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
        'Choline',
        'Thiamin',
        'Pyridoxal',
        'Pyridoxine',
        'Nicotinamide',
        'Pantothenic  acid',
        'Folic acid',
        'Cyanocobalamin',
        'Riboflavin',
        'Biotin',
        'Ethanolamine',
        'Uridine',
        'Na',
        'Mg',
        'P',
        'K',
        'Ca',
        'Fe',
        'Co',
        'Cu',
        'Zn',
        'Mn',
        'D-glucose'
    ]
}

nutrients_dict_all = {
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
        'Biotin',
        'Choline',
        'Cyanocobalamin',
        'Folic acid',
        'Nicotinamide',
        'Pantothenic  acid',
        'Pyridoxal',
        'Pyridoxine',
        'Riboflavin',
        'Thiamin'
    ],
    'MT': [
        'Ca',
        'Co',
        'Cu',
        'Fe',
        'K',
        'Mg',
        'Mn',
        'Na',
        'P',
        'Zn'
    ],
    'Nuc, Amine': [
        'Adenosine',
        'Ethanolamine',
        'Putrescine',
        'Uridine'],
    'other': [
        'D-glucose',
        'Glucose (g/L)'
    ]
}

nutrients_list_all = [
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
    'Biotin',
    'Choline',
    'Cyanocobalamin',
    'Folic acid',
    'Nicotinamide',
    'Pantothenic  acid',
    'Pyridoxal',
    'Pyridoxine',
    'Riboflavin',
    'Thiamin',
    'Ca',
    'Co',
    'Cu',
    'Fe',
    'K',
    'Mg',
    'Mn',
    'Na',
    'P',
    'Zn',
    'Adenosine',
    'Ethanolamine',
    'Putrescine',
    'Uridine',
    'D-glucose',
]

nutrients_calculated_list = [
    'Arg',
    'Asn',
    'Asp',
    'Glu',
    'Gly-Tyr',
    'His',
    'Ile',
    'Leu',
    'Lys',
    'Met',
    'Phe',
    'Pro',
    'Ser',
    'Tau',
    'Thr',
    'Trp',
    'Val',
    'Choline',
    'Thiamin',
    'Nicotinamide',
    'Pantothenic  acid',
    'Folic acid',
    'Cyanocobalamin',
    'Riboflavin',
    'Biotin',
    'Na',
    'Mg',
    'P',
    'K',
    'Ca',
    'Fe',
    'Co',
    'Cu',
    'Zn',
    'Mn',
    # 'Glucose (g/L)',
    'Pyridoxine',
    'Tyr',
    'Uridine',
    'Ala',
    'Ethanolamine'
]
# %% X % Y DATASET VARIABLES
xvar_list_dict_prefilt = {
    0: [
        'Ala_basal',
        'Arg_basal',
        'Asn_basal',
        'Asp_basal',
        'Glu_basal',
        'Gly_basal',
        'Gly-Tyr_basal',
        'His_basal',
        'HyPro_basal',
        'Ile_basal',
        'Leu_basal',
        'Lys_basal',
        'Met_basal',
        'Orn_basal',
        'Phe_basal',
        'Pro_basal',
        'Ser_basal',
        'Tau_basal',
        'Thr_basal',
        'Trp_basal',
        'Tyr_basal',
        'Val_basal',
        'Choline_basal',
        'Thiamin_basal',
        'Pyridoxal_basal',
        'Pyridoxine_basal',
        'Nicotinamide_basal',
        'Pantothenic  acid_basal',
        'Folic acid_basal',
        'Cyanocobalamin_basal',
        'Riboflavin_basal',
        'Biotin_basal',
        'Ethanolamine_basal',
        'Uridine_basal',
        'Na_basal',
        'Mg_basal',
        'P_basal',
        'K_basal',
        'Ca_basal',
        'Fe_basal',
        'Co_basal',
        'Cu_basal',
        'Zn_basal',
        'Mn_basal',
        'D-glucose_basal',
        'Ala_feed',
        'Arg_feed',
        'Asn_feed',
        'Asp_feed',
        'Glu_feed',
        'Gly_feed',
        'Gly-Tyr_feed',
        'His_feed',
        'HyPro_feed',
        'Ile_feed',
        'Leu_feed',
        'Lys_feed',
        'Met_feed',
        'Orn_feed',
        'Phe_feed',
        'Pro_feed',
        'Ser_feed',
        'Tau_feed',
        'Thr_feed',
        'Trp_feed',
        'Tyr_feed',
        'Val_feed',
        'Choline_feed',
        'Thiamin_feed',
        'Pyridoxal_feed',
        'Pyridoxine_feed',
        'Nicotinamide_feed',
        'Pantothenic  acid_feed',
        'Folic acid_feed',
        'Cyanocobalamin_feed',
        'Riboflavin_feed',
        'Biotin_feed',
        'Ethanolamine_feed',
        'Uridine_feed',
        'Na_feed',
        'Mg_feed',
        'P_feed',
        'K_feed',
        'Ca_feed',
        'Fe_feed',
        'Co_feed',
        'Cu_feed',
        'Zn_feed',
        'Mn_feed',
        'D-glucose_feed',
        'DO',
        'pH',
        'feed %',
        'feed vol'
    ],

    1: [
        'Ala_basal',
        'Arg_basal',
        'Asn_basal',
        'Asp_basal',
        'Glu_basal',
        'Gly_basal',
        'Gly-Tyr_basal',
        'His_basal',
        'HyPro_basal',
        'Ile_basal',
        'Leu_basal',
        'Lys_basal',
        'Met_basal',
        'Orn_basal',
        'Phe_basal',
        'Pro_basal',
        'Ser_basal',
        'Tau_basal',
        'Thr_basal',
        'Trp_basal',
        'Tyr_basal',
        'Val_basal',
        'Choline_basal',
        'Thiamin_basal',
        'Pyridoxal_basal',
        'Pyridoxine_basal',
        'Nicotinamide_basal',
        'Pantothenic  acid_basal',
        'Folic acid_basal',
        'Cyanocobalamin_basal',
        'Riboflavin_basal',
        'Biotin_basal',
        'Ethanolamine_basal',
        'Uridine_basal',
        'Na_basal',
        'Mg_basal',
        'P_basal',
        'K_basal',
        'Ca_basal',
        'Fe_basal',
        'Co_basal',
        'Cu_basal',
        'Zn_basal',
        'Mn_basal',
        'D-glucose_basal',
        'Ala_feed',
        'Arg_feed',
        'Asn_feed',
        'Asp_feed',
        'Glu_feed',
        'Gly_feed',
        'Gly-Tyr_feed',
        'His_feed',
        'HyPro_feed',
        'Ile_feed',
        'Leu_feed',
        'Lys_feed',
        'Met_feed',
        'Orn_feed',
        'Phe_feed',
        'Pro_feed',
        'Ser_feed',
        'Tau_feed',
        'Thr_feed',
        'Trp_feed',
        'Tyr_feed',
        'Val_feed',
        'Choline_feed',
        'Thiamin_feed',
        'Pyridoxal_feed',
        'Pyridoxine_feed',
        'Nicotinamide_feed',
        'Pantothenic  acid_feed',
        'Folic acid_feed',
        'Cyanocobalamin_feed',
        'Riboflavin_feed',
        'Biotin_feed',
        'Ethanolamine_feed',
        'Uridine_feed',
        'Na_feed',
        'Mg_feed',
        'P_feed',
        'K_feed',
        'Ca_feed',
        'Fe_feed',
        'Co_feed',
        'Cu_feed',
        'Zn_feed',
        'Mn_feed',
        'D-glucose_feed',
        'DO',
        'pH',
        'feed vol'
    ],

    2: [
        'Ala_basal',
        'Arg_basal',
        'Asn_basal',
        'Asp_basal',
        'Glu_basal',
        'Gly_basal',
        'Gly-Tyr_basal',
        'His_basal',
        'HyPro_basal',
        'Ile_basal',
        'Leu_basal',
        'Lys_basal',
        'Met_basal',
        'Orn_basal',
        'Phe_basal',
        'Pro_basal',
        'Ser_basal',
        'Tau_basal',
        'Thr_basal',
        'Trp_basal',
        'Tyr_basal',
        'Val_basal',
        'Choline_basal',
        'Thiamin_basal',
        'Pyridoxal_basal',
        'Pyridoxine_basal',
        'Nicotinamide_basal',
        'Pantothenic  acid_basal',
        'Folic acid_basal',
        'Cyanocobalamin_basal',
        'Riboflavin_basal',
        'Biotin_basal',
        'Ethanolamine_basal',
        'Uridine_basal',
        'Na_basal',
        'Mg_basal',
        'P_basal',
        'K_basal',
        'Ca_basal',
        'Fe_basal',
        'Co_basal',
        'Cu_basal',
        'Zn_basal',
        'Mn_basal',
        'D-glucose_basal',
        'Ala_feed',
        'Arg_feed',
        'Asn_feed',
        'Asp_feed',
        'Glu_feed',
        'Gly_feed',
        'Gly-Tyr_feed',
        'His_feed',
        'HyPro_feed',
        'Ile_feed',
        'Leu_feed',
        'Lys_feed',
        'Met_feed',
        'Orn_feed',
        'Phe_feed',
        'Pro_feed',
        'Ser_feed',
        'Tau_feed',
        'Thr_feed',
        'Trp_feed',
        'Tyr_feed',
        'Val_feed',
        'Choline_feed',
        'Thiamin_feed',
        'Pyridoxal_feed',
        'Pyridoxine_feed',
        'Nicotinamide_feed',
        'Pantothenic  acid_feed',
        'Folic acid_feed',
        'Cyanocobalamin_feed',
        'Riboflavin_feed',
        'Biotin_feed',
        'Ethanolamine_feed',
        'Uridine_feed',
        'Na_feed',
        'Mg_feed',
        'P_feed',
        'K_feed',
        'Ca_feed',
        'Fe_feed',
        'Co_feed',
        'Cu_feed',
        'Zn_feed',
        'Mn_feed',
        'D-glucose_feed',
        'DO',
        'pH',
        'feed %'
    ],

    3: [
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
        'feed vol'
    ],

    4: [
        'Ala_NSRC',
        'Arg_NSRC',
        'Asn_NSRC',
        'Asp_NSRC',
        'Gln_NSRC',
        'Glu_NSRC',
        'Gly_NSRC',
        'Gly-Tyr_NSRC',
        'His_NSRC',
        'HyPro_NSRC',
        'Ile_NSRC',
        'Leu_NSRC',
        'Lys_NSRC',
        'Met_NSRC',
        'Orn_NSRC',
        'Phe_NSRC',
        'Pro_NSRC',
        'Ser_NSRC',
        'Tau_NSRC',
        'Thr_NSRC',
        'Trp_NSRC',
        'Tyr_NSRC',
        'Val_NSRC',
        'Choline_NSRC',
        'Thiamin_NSRC',
        'Pyridoxal_NSRC',
        'Pyridoxine_NSRC',
        'Nicotinamide_NSRC',
        'Pantothenic  acid_NSRC',
        'Folic acid_NSRC',
        'Cyanocobalamin_NSRC',
        'Riboflavin_NSRC',
        'Biotin_NSRC',
        'Na_NSRC',
        'Mg_NSRC',
        'P_NSRC',
        'K_NSRC',
        'Ca_NSRC',
        'Fe_NSRC',
        'Co_NSRC',
        'Cu_NSRC',
        'Zn_NSRC',
        'Mn_NSRC',
        'Ethanolamine_NSRC',
        'Putrescine_NSRC',
        'Uridine_NSRC',
        'Adenosine_NSRC',
        'DO',
        'pH',
        'feed vol'
    ],
    5: [f'{MC}_basal' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Ca', 'Pyridoxine', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Zn', 'Tyr', 'Met', ]] +
        [f'{MC}_feed' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Ca', 'Pyridoxine', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Zn', 'Tyr', 'Met', ]] +
        ['DO', 'pH', 'feed vol'],
    # 6: [f'{MC}_basal' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Fe', 'Mn', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Tyr', 'Glu', ]] +
    #     [f'{MC}_feed' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Fe', 'Mn', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Tyr', 'Glu', ]] +
    #     ['DO', 'pH', 'feed vol'],
    6: [f'{MC}_basal' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Choline', 'Met', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Lys', 'Glu', ]] +
        [f'{MC}_feed' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Choline', 'Met', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Lys', 'Glu', ]] +
        ['DO', 'pH', 'feed vol'],
        
    # D7 Nutrient Sampling + D7 VCD + Process Params
    7: [
        'VCD (E6 cells/mL)_7'
        'Glucose (g/L)_7'
        'Ala_7'
        'Arg_7'
        'Asn_7'
        'Asp_7'
        'Gln_7'
        'Glu_7'
        'Gly_7'
        'Gly-Tyr_7'
        'His_7'
        'HyPro_7'
        'Ile_7'
        'Leu_7'
        'Lys_7'
        'Met_7'
        'Orn_7'
        'Phe_7'
        'Pro_7'
        'Ser_7'
        'Tau_7'
        'Thr_7'
        'Trp_7'
        'Tyr_7'
        'Val_7'
        'Choline_7'
        'Thiamin_7'
        'Pyridoxal_7'
        'Pyridoxine_7'
        'Nicotinamide_7'
        'Pantothenic  acid_7'
        'Folic acid_7'
        'Cyanocobalamin_7'
        'Riboflavin_7'
        'Biotin_7'
        'Na_7'
        'Mg_7'
        'P_7'
        'K_7'
        'Ca_7'
        'Fe_7'
        'Co_7'
        'Cu_7'
        'Zn_7'
        'Mn_7'
        'Ethanolamine_7'
        'Putrescine_7'
        'Uridine_7'
        'Adenosine_7'
        'DO'
        'pH'
        'feed vol'
        ],
    
    # D11 Nutrient Sampling + D11 VCD + Process Params
    8: [
        'VCD (E6 cells/mL)_11'
        'Glucose (g/L)_11'
        'Ala_11'
        'Arg_11'
        'Asn_11'
        'Asp_11'
        'Gln_11'
        'Glu_11'
        'Gly_11'
        'Gly-Tyr_11'
        'His_11'
        'HyPro_11'
        'Ile_11'
        'Leu_11'
        'Lys_11'
        'Met_11'
        'Orn_11'
        'Phe_11'
        'Pro_11'
        'Ser_11'
        'Tau_11'
        'Thr_11'
        'Trp_11'
        'Tyr_11'
        'Val_11'
        'Choline_11'
        'Thiamin_11'
        'Pyridoxal_11'
        'Pyridoxine_11'
        'Nicotinamide_11'
        'Pantothenic  acid_11'
        'Folic acid_11'
        'Cyanocobalamin_11'
        'Riboflavin_11'
        'Biotin_11'
        'Na_11'
        'Mg_11'
        'P_11'
        'K_11'
        'Ca_11'
        'Fe_11'
        'Co_11'
        'Cu_11'
        'Zn_11'
        'Mn_11'
        'Ethanolamine_11'
        'Putrescine_11'
        'Uridine_11'
        'Adenosine_11'
        'DO'
        'pH'
        'feed vol'
        ],
    
    # D7+D11 Nutrient Sampling + D7+D11 VCD + Process Params
    9: [
        'VCD (E6 cells/mL)_7'
        'VCD (E6 cells/mL)_11'
        'Glucose (g/L)_7'
        'Glucose (g/L)_11'
        'Ala_7'
        'Ala_11'
        'Arg_7'
        'Arg_11'
        'Asn_7'
        'Asn_11'
        'Asp_7'
        'Asp_11'
        'Gln_7'
        'Gln_11'
        'Glu_7'
        'Glu_11'
        'Gly_7'
        'Gly_11'
        'Gly-Tyr_7'
        'Gly-Tyr_11'
        'His_7'
        'His_11'
        'HyPro_7'
        'HyPro_11'
        'Ile_7'
        'Ile_11'
        'Leu_7'
        'Leu_11'
        'Lys_7'
        'Lys_11'
        'Met_7'
        'Met_11'
        'Orn_7'
        'Orn_11'
        'Phe_7'
        'Phe_11'
        'Pro_7'
        'Pro_11'
        'Ser_7'
        'Ser_11'
        'Tau_7'
        'Tau_11'
        'Thr_7'
        'Thr_11'
        'Trp_7'
        'Trp_11'
        'Tyr_7'
        'Tyr_11'
        'Val_7'
        'Val_11'
        'Choline_7'
        'Choline_11'
        'Thiamin_7'
        'Thiamin_11'
        'Pyridoxal_7'
        'Pyridoxal_11'
        'Pyridoxine_7'
        'Pyridoxine_11'
        'Nicotinamide_7'
        'Nicotinamide_11'
        'Pantothenic  acid_7'
        'Pantothenic  acid_11'
        'Folic acid_7'
        'Folic acid_11'
        'Cyanocobalamin_7'
        'Cyanocobalamin_11'
        'Riboflavin_7'
        'Riboflavin_11'
        'Biotin_7'
        'Biotin_11'
        'Na_7'
        'Na_11'
        'Mg_7'
        'Mg_11'
        'P_7'
        'P_11'
        'K_7'
        'K_11'
        'Ca_7'
        'Ca_11'
        'Fe_7'
        'Fe_11'
        'Co_7'
        'Co_11'
        'Cu_7'
        'Cu_11'
        'Zn_7'
        'Zn_11'
        'Mn_7'
        'Mn_11'
        'Ethanolamine_7'
        'Ethanolamine_11'
        'Putrescine_7'
        'Putrescine_11'
        'Uridine_7'
        'Uridine_11'
        'Adenosine_7'
        'Adenosine_11'
        'DO'
        'pH'
        'feed vol'
        ],


}


xvar_list_dict = {
    1: ['Arg_basal',
        'Asn_basal',
        'Asp_basal',
        'Glu_basal',
        'His_basal',
        'Ile_basal',
        'Leu_basal',
        'Lys_basal',
        'Met_basal',
        'Orn_basal',
        'Phe_basal',
        'Pro_basal',
        'Ser_basal',
        'Thr_basal',
        'Trp_basal',
        'Tyr_basal',
        'Val_basal',
        'Choline_basal',
        'Pyridoxine_basal',
        'Nicotinamide_basal',
        'Pantothenic  acid_basal',
        'Folic acid_basal',
        'Cyanocobalamin_basal',
        'Riboflavin_basal',
        'Biotin_basal',
        'Ethanolamine_basal',
        'Uridine_basal',
        'Na_basal',
        'Mg_basal',
        'P_basal',
        'K_basal',
        'Ca_basal',
        'Fe_basal',
        'Co_basal',
        'Cu_basal',
        'Zn_basal',
        'D-glucose_basal',
        'Arg_feed',
        'Asn_feed',
        'Asp_feed',
        'Glu_feed',
        'Gly-Tyr_feed',
        'His_feed',
        'Ile_feed',
        'Leu_feed',
        'Lys_feed',
        'Met_feed',
        'Phe_feed',
        'Pro_feed',
        'Ser_feed',
        'Thr_feed',
        'Trp_feed',
        'Val_feed',
        'Choline_feed',
        'Pyridoxine_feed',
        'Nicotinamide_feed',
        'Pantothenic  acid_feed',
        'Folic acid_feed',
        'Cyanocobalamin_feed',
        'Riboflavin_feed',
        'Biotin_feed',
        'Uridine_feed',
        'Na_feed',
        'Mg_feed',
        'P_feed',
        'K_feed',
        'Ca_feed',
        'Fe_feed',
        'Co_feed',
        'Cu_feed',
        'Zn_feed',
        'Mn_feed',
        'D-glucose_feed',
        'DO',
        'pH',
        'feed vol'
        ],

    4: [
        'Arg_NSRC',
        'Asn_NSRC',
        'Asp_NSRC',
        'Glu_NSRC',
        'Gly-Tyr_NSRC',
        'His_NSRC',
        'Ile_NSRC',
        'Leu_NSRC',
        'Lys_NSRC',
        'Met_NSRC',
        'Phe_NSRC',
        'Pro_NSRC',
        'Ser_NSRC',
        'Thr_NSRC',
        'Trp_NSRC',
        'Val_NSRC',
        'Choline_NSRC',
        'Pyridoxine_NSRC',
        'Nicotinamide_NSRC',
        'Pantothenic  acid_NSRC',
        'Folic acid_NSRC',
        'Cyanocobalamin_NSRC',
        'Riboflavin_NSRC',
        'Biotin_NSRC',
        'Na_NSRC',
        'Mg_NSRC',
        'P_NSRC',
        'K_NSRC',
        'Ca_NSRC',
        'Fe_NSRC',
        'Co_NSRC',
        'Cu_NSRC',
        'Zn_NSRC',
        'Uridine_NSRC',
        'DO',
        'pH',
        'feed vol'
    ],
    5:  [f'{MC}_basal' for MC in ['Arg', 'Asn', 'Asp', 'Met', 'Folic acid', 'Co', 'Ca', 'Pyridoxine', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Zn', 'Tyr']] +
        [f'{MC}_feed' for MC in ['Arg', 'Asn', 'Asp', 'Met', 'Folic acid', 'Co', 'Ca', 'Pyridoxine', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Zn']] +
        ['DO', 'pH', 'feed vol'],
    # 6: [f'{MC}_basal' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Fe', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Tyr', 'Glu', ]] +
    #     [f'{MC}_feed' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Fe', 'Mn', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Glu', ]] +
    #     ['DO', 'pH', 'feed vol'],
     6: [f'{MC}_basal' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Choline', 'Met', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Lys', 'Glu', ]] +
         [f'{MC}_feed' for MC in ['Arg', 'Asn', 'Asp', 'Folic acid', 'Co', 'Choline', 'Met', 'Mg', 'Ca', 'Zn', 'Ser', 'Thr', 'Pro', 'Uridine', 'Riboflavin', 'Lys', 'Glu', ]] +
         ['DO', 'pH', 'feed vol'],
                
    7: [
        'VCD (E6 cells/mL)_7'
        'Glucose (g/L)_7'
        'Ala_7'
        'Arg_7'
        'Asn_7'
        'Asp_7'
        'Gln_7'
        'Glu_7'
        'Gly_7'
        'Gly-Tyr_7'
        'His_7'
        'HyPro_7'
        'Ile_7'
        'Leu_7'
        'Lys_7'
        'Met_7'
        'Orn_7'
        'Phe_7'
        'Pro_7'
        'Ser_7'
        'Tau_7'
        'Thr_7'
        'Trp_7'
        'Tyr_7'
        'Val_7'
        'Choline_7'
        'Pyridoxal_7'
        'Pyridoxine_7'
        'Nicotinamide_7'
        'Pantothenic  acid_7'
        'Folic acid_7'
        'Cyanocobalamin_7'
        'Riboflavin_7'
        'Biotin_7'
        'Na_7'
        'Mg_7'
        'P_7'
        'K_7'
        'Ca_7'
        'Fe_7'
        'Co_7'
        'Cu_7'
        'Zn_7'
        'Mn_7'
        'DO'
        'pH'
        'feed vol'
        ], 
    
    8: [
        'VCD (E6 cells/mL)_11'
        'Glucose (g/L)_11'
        'Ala_11'
        'Arg_11'
        'Asn_11'
        'Asp_11'
        'Gln_11'
        'Glu_11'
        'Gly_11'
        'Gly-Tyr_11'
        'His_11'
        'HyPro_11'
        'Ile_11'
        'Leu_11'
        'Lys_11'
        'Met_11'
        'Orn_11'
        'Phe_11'
        'Pro_11'
        'Ser_11'
        'Tau_11'
        'Thr_11'
        'Trp_11'
        'Tyr_11'
        'Val_11'
        'Choline_11'
        'Pyridoxal_11'
        'Pyridoxine_11'
        'Nicotinamide_11'
        'Pantothenic  acid_11'
        'Folic acid_11'
        'Cyanocobalamin_11'
        'Riboflavin_11'
        'Biotin_11'
        'Na_11'
        'Mg_11'
        'P_11'
        'K_11'
        'Ca_11'
        'Fe_11'
        'Co_11'
        'Cu_11'
        'Zn_11'
        'Mn_11'
        'DO'
        'pH'
        'feed vol'
        ],
    
    9: [
        'VCD (E6 cells/mL)_7'
        'VCD (E6 cells/mL)_11'
        'Glucose (g/L)_7'
        'Glucose (g/L)_11'
        'Ala_7'
        'Ala_11'
        'Arg_7'
        'Arg_11'
        'Asn_7'
        'Asn_11'
        'Asp_7'
        'Asp_11'
        'Gln_7'
        'Gln_11'
        'Glu_7'
        'Glu_11'
        'Gly_7'
        'Gly_11'
        'Gly-Tyr_7'
        'Gly-Tyr_11'
        'His_7'
        'His_11'
        'HyPro_7'
        'HyPro_11'
        'Ile_7'
        'Ile_11'
        'Leu_7'
        'Leu_11'
        'Lys_7'
        'Lys_11'
        'Met_7'
        'Met_11'
        'Orn_7'
        'Orn_11'
        'Phe_7'
        'Phe_11'
        'Pro_7'
        'Pro_11'
        'Ser_7'
        'Ser_11'
        'Tau_7'
        'Tau_11'
        'Thr_7'
        'Thr_11'
        'Trp_7'
        'Trp_11'
        'Tyr_7'
        'Tyr_11'
        'Val_7'
        'Val_11'
        'Choline_7'
        'Choline_11'
        'Pyridoxal_7'
        'Pyridoxal_11'
        'Pyridoxine_7'
        'Pyridoxine_11'
        'Nicotinamide_7'
        'Nicotinamide_11'
        'Pantothenic  acid_7'
        'Pantothenic  acid_11'
        'Folic acid_7'
        'Folic acid_11'
        'Cyanocobalamin_7'
        'Cyanocobalamin_11'
        'Riboflavin_7'
        'Riboflavin_11'
        'Biotin_7'
        'Biotin_11'
        'Na_7'
        'Na_11'
        'Mg_7'
        'Mg_11'
        'P_7'
        'P_11'
        'K_7'
        'K_11'
        'Ca_7'
        'Ca_11'
        'Fe_7'
        'Fe_11'
        'Co_7'
        'Co_11'
        'Cu_7'
        'Cu_11'
        'Zn_7'
        'Zn_11'
        'Mn_7'
        'Mn_11'
        'DO'
        'pH'
        'feed vol'
        ]
        
}

# %% FEATURE SELECTIONS

feature_selections = {
    '_initial_try': {
    'Titer (mg/L)_14': ['Co_feed', 'Arg_feed', 'Riboflavin_feed', 'Pro_feed', 'His_feed', 'Folic acid_feed', 'Fe_feed', 'Pyridoxine_feed', 'K_feed', 'Nicotinamide_feed', 'Uridine_feed', 'Zn_basal', 'DO', 'pH', 'feed vol'],
    'mannosylation_14': ['Riboflavin_feed', 'Riboflavin_basal', 'Ca_feed', 'Lys_basal', 'Uridine_feed', 'P_feed', 'Asp_basal', 'Thr_feed', 'Folic acid_basal', 'Biotin_basal', 'DO', 'pH', 'feed vol'],
    'fucosylation_14': ['Riboflavin_feed', 'Riboflavin_basal', 'Ca_feed', 'Folic acid_basal', 'Thr_feed', 'Nicotinamide_basal', 'Cu_feed', 'Asn_feed', 'Uridine_feed', 'Lys_basal', 'DO', 'pH', 'feed vol'],
    'galactosylation_14': ['Asp_basal', 'Ser_basal', 'Thr_basal', 'Co_basal', 'Folic acid_basal', 'K_basal', 'Pro_feed', 'Trp_basal', 'Choline_feed', 'Riboflavin_feed', 'DO', 'pH', 'feed vol'],
    },
    
    '_knowledge-opt': {
    'Titer (mg/L)_14': ['Met_feed', 'Pro_feed', 'Mg_feed', 'Tyr_basal', 'Uridine_feed', 'Riboflavin_feed', 'Pro_basal', 'Met_basal', 'Zn_feed', 'DO', 'pH', 'feed vol'],
    'mannosylation_14': ['Riboflavin_feed', 'Riboflavin_basal', 'Uridine_feed', 'Thr_basal', 'Thr_feed', 'Pro_feed', 'Ser_basal', 'Ser_feed', 'Asn_basal', 'DO', 'pH', 'feed vol'],
    'fucosylation_14': ['Riboflavin_basal', 'Riboflavin_feed', 'Thr_basal', 'Folic acid_basal', 'Asn_feed', 'Thr_feed', 'Uridine_feed', 'Ser_feed', 'Pyridoxine_feed', 'DO', 'pH', 'feed vol'],
    'galactosylation_14': ['Ser_basal', 'Thr_basal', 'Folic acid_basal', 'Riboflavin_feed', 'Asn_basal', 'Asn_feed', 'Pyridoxine_basal', 'Thr_feed', 'Riboflavin_basal', 'DO', 'pH', 'feed vol'],
    },
    
    '_model-opt': {
    'Titer (mg/L)_14': ['Co_feed', 'Thr_feed', 'His_feed', 'Riboflavin_feed', 'Fe_feed', 'Tyr_basal', 'Pantothenic  acid_feed', 'Co_basal', 'DO', 'pH', 'feed vol'],
    'mannosylation_14': ['Riboflavin_feed', 'Riboflavin_basal', 'Choline_feed', 'Lys_basal', 'Leu_feed', 'Ca_feed', 'P_feed', 'Asp_basal', 'DO', 'pH', 'feed vol'],
    'fucosylation_14': ['Riboflavin_feed', 'Riboflavin_basal', 'Nicotinamide_basal', 'Tyr_basal', 'Ca_feed', 'Leu_basal', 'Cu_basal', 'K_feed', 'DO', 'pH', 'feed vol'],
    'galactosylation_14': ['Folic acid_basal', 'Asp_basal', 'Co_basal', 'Trp_basal', 'Nicotinamide_basal', 'K_basal', 'Choline_basal', 'Choline_feed', 'DO', 'pH', 'feed vol']
    },
                        
    '_compactness-opt': {
    'Titer (mg/L)_14': ['Asp_basal', 'Asp_feed', 'DO', 'pH', 'feed vol'], #
    'mannosylation_14': ['Riboflavin_feed', 'Riboflavin_basal', 'feed vol', 'DO', 'pH'],
    'fucosylation_14': ['Riboflavin_feed', 'Riboflavin_basal', 'feed vol', 'DO', 'pH'],
    'galactosylation_14': ['Riboflavin_basal', 'Thr_basal', 'Riboflavin_feed', 'DO', 'pH', 'feed vol']
    },
    
    '_curated': {
    'Titer (mg/L)_14': xvar_list_dict[5],
    'mannosylation_14': xvar_list_dict[5],
    'fucosylation_14': xvar_list_dict[5],
    'galactosylation_14': xvar_list_dict[5],
    },
    '_curated2': {
    'Titer (mg/L)_14': xvar_list_dict[6],
    'mannosylation_14': xvar_list_dict[6],
    'fucosylation_14': xvar_list_dict[6],
    'galactosylation_14': xvar_list_dict[6],
    },
    
    '_combi1': {
    'Titer (mg/L)_14': ['Asp_basal', 'Asp_feed', 'Asn_basal', 'Asn_feed', 'Mg_basal', 'Mg_feed', 'Ca_basal', 'Ca_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'DO', 'pH', 'feed vol'],
    'mannosylation_14': ['Asp_basal', 'Asp_feed', 'Asn_basal', 'Asn_feed', 'Mg_basal', 'Mg_feed', 'Ca_basal', 'Ca_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'DO', 'pH', 'feed vol'],
    'fucosylation_14': ['Asp_basal', 'Asp_feed', 'Asn_basal', 'Asn_feed', 'Mg_basal', 'Mg_feed', 'Ca_basal', 'Ca_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'DO', 'pH', 'feed vol'],
    'galactosylation_14':['Asp_basal', 'Asp_feed', 'Asn_basal', 'Asn_feed', 'Mg_basal', 'Mg_feed', 'Ca_basal', 'Ca_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'DO', 'pH', 'feed vol']
    },
        
    '_combi2': {
    'Titer (mg/L)_14': ['Arg_basal', 'Arg_feed', 'Folic acid_basal', 'Folic acid_feed',  'Co_basal', 'Co_feed', 'Zn_basal', 'Zn_feed',  'Thr_basal', 'Thr_feed', 'Pro_basal', 'Pro_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'Tyr_basal', 'DO', 'pH', 'feed vol'],
    'mannosylation_14': ['Arg_basal', 'Arg_feed', 'Folic acid_basal', 'Folic acid_feed',  'Co_basal', 'Co_feed', 'Zn_basal', 'Zn_feed',  'Thr_basal', 'Thr_feed', 'Pro_basal', 'Pro_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'Tyr_basal', 'DO', 'pH', 'feed vol'],
    'fucosylation_14': ['Arg_basal', 'Arg_feed', 'Folic acid_basal', 'Folic acid_feed',  'Co_basal', 'Co_feed', 'Zn_basal', 'Zn_feed',  'Thr_basal', 'Thr_feed', 'Pro_basal', 'Pro_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'Tyr_basal', 'DO', 'pH', 'feed vol'],
    'galactosylation_14': ['Arg_basal', 'Arg_feed', 'Folic acid_basal', 'Folic acid_feed',  'Co_basal', 'Co_feed', 'Zn_basal', 'Zn_feed',  'Thr_basal', 'Thr_feed', 'Pro_basal', 'Pro_feed', 'Uridine_basal', 'Uridine_feed', 'Riboflavin_basal', 'Riboflavin_feed', 'Tyr_basal', 'DO', 'pH', 'feed vol']
    }
    }


#%% DOMAIN KNOWLEDGE 

# ChatGPT
features_to_boost_GLYCO_1 = {
    'Asn':1, 'Gln':0.5, 'Ser':1, 'Thr':1, 'Met':0.5, 'Pro':0.5, 'Gly':0.5, 'Asp':0.5, 'Glu':0.5, 
    'Riboflavin':1, 'Pyridoxine':1, 'Folic acid':0.5, 'Biotin':0.5, 
    'Mn':1, 'Ca':0.5, 'Mg':0.5, 'Zn':0.5, 'Cu':0.5, 
    'Uridine':1, 'Adenosine ':0.5, 'Putrescine':0.5
    }

# Kathy/Ian
features_to_boost_GLYCO_2 = {
    'Gln': 0.9, 'Uridine': 0.9, 'Mn': 0.875, 'Asn': 0.8, 'Fe': 0.8, 'Co': 0.8, 'Riboflavin': 0.75, 'Ca': 0.75, 'Zn': 0.75, 'Glu': 0.7, 
    'Nicotinamide': 0.7, 'Mg': 0.7, 'Cu': 0.675, 'Asp': 0.65, 'Gly': 0.6, 'Orn': 0.6, 'Choline': 0.6, 'Na': 0.6, 'D-glucose': 0.6, 'Pyridoxal': 0.55, 
    'Folic acid': 0.5, 'Ala': 0.4, 'Thr': 0.4, 'Thiamin': 0.4, 'Biotin': 0.4, 'Arg': 0.2
    }

features_to_boost_GLYCO = {
    'Met': 0.25, 'Fe': 0.4, 'Ser': 0.5, 'Zn': 0.625, 'Orn': 0.3, 'Thr': 0.7, 'Folic acid': 0.5, 'Biotin': 0.45, 'Gln': 0.7, 'Pro': 0.25, 
    'Pyridoxal': 0.275, 'Ca': 0.625, 'Putrescine': 0.25, 'D-glucose': 0.3, 'Asp': 0.575, 'Nicotinamide': 0.35, 'Mg': 0.6, 'Cu': 0.5875, 
    'Thiamin': 0.2, 'Mn': 0.9375, 'Riboflavin': 0.875, 'Ala': 0.2, 'Asn': 0.9, 'Adenosine ': 0.25, 'Uridine': 0.95, 'Glu': 0.6, 'Gly': 0.55, 
    'Arg': 0.1, 'Choline': 0.3, 'Co': 0.4, 'Na': 0.3, 'Pyridoxine': 0.5
    }

# ChatGPT
features_to_boost_TITER_1 = {
    'Gln': 1, 'Tyr': 1, 'Met': 1, 'Arg': 1, 'Asn': 1, 'Pro': 1, 'His': 0.5, 'Ser': 0.5, 'Lys': 0.5, 
    'Thiamin': 1, 'Riboflavin': 1, 'Patothenic  acid': 1, 'Pyridoxine': 1, 'Biotin': 1, 'Folic acid': 1, 'Cyanocobalamin': 1, 
    'Fe': 1, 'Zn': 1, 'Cu': 1, 'Mg': 1, 'Ca': 0.5, 'Mn': 0.5, 'Co': 0.5, 'Ni': 0.5, 
    'Adenosine': 1, 'Uridine': 1, 'Putrescine': 0.5, 'Ethanolomine': 0.5
    }

# Kathy/Ian
features_to_boost_TITER_2 = {
    'Tyr': 1.0, 'Asp': 0.8, 'Arg': 0.7, 'Asn': 0.7, 'Glu': 0.7, 'Ser': 0.7, 'His': 0.65, 'Thr': 0.6, 'Pyridoxine': 0.6, 'Zn': 0.6, 'Leu': 0.5, 
    'Lys': 0.5, 'Met': 0.5, 'Phe': 0.5, 'Trp': 0.5, 'Val': 0.5, 'Folic acid': 0.5, 'Na': 0.4, 'Mg': 0.4, 'P': 0.4, 'K': 0.4, 'Ca': 0.4
    }


features_to_boost_TITER = {
    'K': 0.2, 'Met': 0.75, 'Fe': 0.5, 'Ser': 0.6, 'Zn': 0.8, 'Trp': 0.25, 'Thr': 0.3, 'His': 0.575, 'Adenosine': 0.5, 'Biotin': 0.5, 
    'Folic acid': 0.75, 'Gln': 0.5, 'Pro': 0.5, 'Ca': 0.45, 'Putrescine': 0.25, 'Patothenic  acid': 0.5, 'Asp': 0.4, 'Mg': 0.7, 'Cu': 0.5, 
    'Thiamin': 0.5, 'Mn': 0.25, 'Riboflavin': 0.5, 'Phe': 0.25, 'Asn': 0.85, 'Cyanocobalamin': 0.5, 'Uridine': 0.5, 'Tyr': 1.0, 'Glu': 0.35, 
    'Arg': 0.85, 'Lys': 0.5, 'Leu': 0.25, 'Val': 0.25, 'Co': 0.25, 'Na': 0.2, 'Ni': 0.25, 'Pyridoxine': 0.8, 'Ethanolomine': 0.25, 'P': 0.2
    }

features_to_boost_dict = {
    'Titer (mg/L)_14': features_to_boost_TITER_2,
    'mannosylation_14': features_to_boost_GLYCO, 
    'fucosylation_14': features_to_boost_GLYCO, 
    'galactosylation_14': features_to_boost_GLYCO, 
    }


#%% 

default_params = {
    'randomforest': {
        'Titer (mg/L)_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
        'mannosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
        'fucosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
        'galactosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
    },
    'plsr': {
        'Titer (mg/L)_14': [{'model_type': 'plsr', 'n_components': 10}],
        'mannosylation_14': [{'model_type': 'plsr', 'n_components': 9}],
        'fucosylation_14': [{'model_type': 'plsr', 'n_components': 8}],
        'galactosylation_14': [{'model_type': 'plsr', 'n_components': 8}],
    },
    'lasso': {
        'Titer (mg/L)_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
        'mannosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
        'fucosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
        'galactosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.05}],
    },
    'xgb': {
        'Titer (mg/L)_14': [{'model_type': 'xgb', 'n_estimators': 100}],
        'mannosylation_14': [{'model_type': 'xgb', 'n_estimators': 100}],
        'fucosylation_14': [{'model_type': 'xgb', 'n_estimators': 100}],
        'galactosylation_14': [{'model_type': 'xgb', 'n_estimators': 100}],
    },    
 }
model_params = {
    'X1Y0': {
        'randomforest': {
            'Titer (mg/L)_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
            'mannosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
            'fucosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
            'galactosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
        },
        'plsr': {
            'Titer (mg/L)_14': [{'model_type': 'plsr', 'n_components': 10}],
            'mannosylation_14': [{'model_type': 'plsr', 'n_components': 9}],
            'fucosylation_14': [{'model_type': 'plsr', 'n_components': 8}],
            'galactosylation_14': [{'model_type': 'plsr', 'n_components': 8}],
        },
        'lasso': {
            'Titer (mg/L)_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
            'mannosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
            'fucosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
            'galactosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.05}],
        },
        'xgb': {
            'Titer (mg/L)_14': [{'model_type': 'xgb', 'n_estimators': 100}],
            'mannosylation_14': [{'model_type': 'xgb', 'n_estimators': 100}],
            'fucosylation_14': [{'model_type': 'xgb', 'n_estimators': 100}],
            'galactosylation_14': [{'model_type': 'xgb', 'n_estimators': 100}],
        },
    },
    'X4Y0': {
        'randomforest': {
            'Titer (mg/L)_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
            'mannosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
            'fucosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
            'galactosylation_14': [{'model_type': 'randomforest', 'n_estimators': 100}],
        },
        'plsr': {
            'Titer (mg/L)_14': [{'model_type': 'plsr', 'n_components': 11}],
            'mannosylation_14': [{'model_type': 'plsr', 'n_components': 8}],
            'fucosylation_14': [{'model_type': 'plsr', 'n_components': 7}],
            'galactosylation_14': [{'model_type': 'plsr', 'n_components': 9}],
        },
        'lasso': {
            'Titer (mg/L)_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 10}],
            'mannosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.05}],
            'fucosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
            'galactosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.01}],
        }
    },
    'X5Y0': default_params.copy(),
    'X6Y0': {
        'randomforest': {
            'Titer (mg/L)_14': [{'model_type': 'randomforest', 'n_estimators': 300}],
            'mannosylation_14': [{'model_type': 'randomforest', 'n_estimators': 300}],
            'fucosylation_14': [{'model_type': 'randomforest', 'n_estimators': 300}],
            'galactosylation_14': [{'model_type': 'randomforest', 'n_estimators': 300}],
        },
        'plsr': {
            'Titer (mg/L)_14': [{'model_type': 'plsr', 'n_components': 10}],
            'mannosylation_14': [{'model_type': 'plsr', 'n_components': 9}],
            'fucosylation_14': [{'model_type': 'plsr', 'n_components': 8}],
            'galactosylation_14': [{'model_type': 'plsr', 'n_components': 8}],
        },
        'lasso': {
            'Titer (mg/L)_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
            'mannosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
            'fucosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.005}],
            'galactosylation_14': [{'model_type': 'lasso', 'max_iter': 50000, 'alpha': 0.05}],
        },
        'xgb': {
            'Titer (mg/L)_14': [{'model_type': 'xgb', 'n_estimators': 300}],
            'mannosylation_14': [{'model_type': 'xgb', 'n_estimators': 300}],
            'fucosylation_14': [{'model_type': 'xgb', 'n_estimators': 300}],
            'galactosylation_14': [{'model_type': 'xgb', 'n_estimators': 300}],
        },
        },
    
   'X7Y0': default_params.copy(),
   'X8Y0': default_params.copy(),
   'X9Y0': default_params.copy(),
   'X10Y0': default_params.copy(),
   }
   
   
   
   
