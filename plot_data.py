#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:50:05 2024

@author: charmainechia
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# set folder names
data_folder = '../ajino-analytics-data/'
figure_folder = f'{data_folder}figures/'

def convert_figidx_to_rowcolidx(figidx, ncols): 
    """
    Description: Convert a scalar figure index into an x, y position corresponding to its row and column position in the panel.
    
    Parameters
    ----------
    figidx : int
        scalar integer describing position of a given variable to be plotted within the full list.
    ncols : int
        total number of columns in the figure of subplots.

    Returns
    -------
    row_idx : int
        row position of the subplot to be plotted.
    col_idx : int
        column position of the subplot to be plotted.

    """
    row_idx = int(np.floor(figidx/ncols))
    col_idx = np.mod(figidx, ncols)
    return row_idx, col_idx
    

def get_unused_figidx(var_to_plot, nrows, ncols):
    """
    Description: Get list of unused subplot positions in the figure

    Parameters
    ----------
    var_to_plot : list of str
        DESCRIPTION.
    nrows : int
        total number of rows in the figure of subplots.
    ncols : int
        total number of columns in the figure of subplots.

    Returns
    -------
    unused_idxs : list of int
        list of indices of unused positions in the figure of subplots.

    """
    num_vars_to_plot = len(var_to_plot)
    num_subplots = nrows*ncols
    unused_idxs = list(range(num_vars_to_plot, num_subplots))
    return unused_idxs


def remove_nandata(t, y):
    idx_to_keep = np.where(~np.isnan(y))[0]
    y = y[idx_to_keep]
    t = t[idx_to_keep]
    return t, y

def plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, colormapping_dict, figtitle=None, savefig=None):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    legend = []   
    axs_with_content = []
    for k in datadict:
        d_exp = datadict[k]
        
        # get plot color and label
        exp_label = d_exp['exp_label']
        plot_label = f"{exp_label}_{d_exp['n']}"
        color = exp_to_color_dict[exp_label]
        legend.append(plot_label)
        
        # plot variable
        for idx, var in enumerate(var_to_plot):
            i, j = convert_figidx_to_rowcolidx(idx, ncols)
            if var in d_exp:
                t, y = d_exp[var]['t'],d_exp[var]['y']
                t, y = remove_nandata(t,y)
                ax[i][j].plot(t, y, c=color, alpha=0.9)
                ax[i][j].set_ylabel(var)
                if var not in axs_with_content:
                    axs_with_content.append(var)
    
    # remove unused axes
    axs_without_content = [idx for idx, var in enumerate(var_to_plot) if var not in axs_with_content]
    unused_idxs =  get_unused_figidx(var_to_plot, nrows, ncols)
    unused_idxs += axs_without_content
    for idx in unused_idxs:
        i, j = convert_figidx_to_rowcolidx(idx, ncols)
        ax[i][j].set_axis_off()
        
    # plot legend
    fig.legend(legend, bbox_to_anchor=(1.03, 0.8), fontsize=12)
    
    # set title
    ymax = ax.flatten()[0].get_position().ymax
    if figtitle is not None:
        plt.suptitle(figtitle, y=ymax*1.02, fontsize=20)
        
    # save fig
    if savefig is not None: 
        plt.savefig(savefig, bbox_inches='tight')   
        
    # show plot
    plt.show()
    
    
#%% get data

# set dataset name
dataset_name = 'dataset1'
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


#%% plot AA, Nuc, Amine

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
     'Ethanolamine', 
     'Putrescine', 
     'Uridine', 
     'Adenosine'
 ]
    
figsize = (28,20)
nrows = 5
ncols = 6
figtitle = 'Amino Acid, Nuc, Amine concentrations'
savefig = f'{figure_folder}{dataset_name}_AA,Nuc,Amine.png'

plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, exp_to_color_dict, figtitle, savefig)

#%% plot MT

var_to_plot = ['Na', 'Mg', 'P', 'K', 'Ca', 'Fe', 'Co', 'Cu', 'Zn', 'Mn']
figsize = (21,16)
nrows = 3
ncols = 4
figtitle = 'Metal ion concentrations'
savefig = f'{figure_folder}{dataset_name}_MT.png'

plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, exp_to_color_dict, figtitle, savefig)