#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:05:48 2024

@author: charmainechia
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from variables import data_folder, figure_folder



model_cmap = {'randomforest':'r', 'plsr':'b', 'lasso':'g'}


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


def remove_nandata(x, y, on_var='y'):
    if on_var=='y':
        idx_to_keep = np.where(~np.isnan(y))[0]
        y = y[idx_to_keep]
        x = x[idx_to_keep]
    elif on_var=='x':
        idx_to_keep = np.where(~np.isnan(x))[0]
        x = x[idx_to_keep]
        y = y[idx_to_keep]
    elif on_var=='xy':
        idx_to_keep = np.where(~np.isnan(x))[0]
        x = x[idx_to_keep]
        y = y[idx_to_keep]
        idx_to_keep = np.where(~np.isnan(y))[0]
        y = y[idx_to_keep]
        x = x[idx_to_keep]
    return x, y

def plot_data_panel(datadict, var_to_plot, nrows, ncols, figsize, colormapping_dict=None, figtitle=None, savefig=None, plot_suffixes={'':'-'}):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    legend = []   
    axs_with_content = []
    for k in datadict:
        d_exp = datadict[k]
        
        # get plot color and label
        if len(datadict)>1:
            exp_label = d_exp['exp_label']
            plot_label = f"{exp_label}_{d_exp['n']}"
            legend.append(plot_label)
            color = colormapping_dict[exp_label]
        else: 
            color = 'b'
        
        # plot variable
        for idx, var in enumerate(var_to_plot):
            i, j = convert_figidx_to_rowcolidx(idx, ncols)
            for suffix, linestyle in plot_suffixes.items():
                if var in d_exp and f't{suffix}' in d_exp[var]:
                    t, y = d_exp[var][f't{suffix}'],d_exp[var][f'y{suffix}']              
                    t, y = remove_nandata(t,y)
                    ax[i][j].plot(t, y, c=color, alpha=0.9, linestyle=linestyle)
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
    if len(datadict)>1:
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
    
    
    

def annotate_heatmap(array_2D, ax, ndecimals=2):
    for (j,i),label in np.ndenumerate(array_2D):
        if ndecimals==0 and ~np.isnan(label):
            label = int(label)
        else: 
            label = round(label,ndecimals)
        ax.text(i,j,label,ha='center',va='center', color='0.8', fontsize=8, fontweight='bold')


def symlog(data):
    idx_pos = np.where(data>0)
    idx_neg = np.where(data<0)
    data_symlog = np.zeros((data.shape[0],data.shape[1]))
    data_symlog[:] = np.nan
    data_symlog[idx_pos] = np.log(data[idx_pos])
    data_symlog[idx_neg] = -np.log(-data[idx_neg])
    return data_symlog
    
    
def heatmap(array, c='viridis', ax=None, cbar_kw={}, cbarlabel="", datamin=None, datamax=None, logscale_cmap=False, annotate=None, row_labels=None, col_labels=None, show_gridlines=True):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
    cmap = getattr(plt.cm, c)
    
    # get array size and xy labels
    data = array.astype(float)
    ny,nx = data.shape
    
    # get row and column labels
    if row_labels is None:
        row_labels = list(np.arange(ny)+1)
    if col_labels is None:
        col_labels = list(np.arange(nx)+1)

    # get locations of nan values and negative values, replace values so these don't trigger an error
    naninds = np.where(np.isnan(data) == True)
    infinds = np.where(np.isinf(data) == True)
    if len(infinds[0])>0:
        data[infinds] = np.nan
    if len(naninds[0])>0:
        data[naninds] = np.nanmean(data)
    if len(infinds[0])>0:    
        data[infinds] = np.nanmean(data)
    data_cmap = data.copy()

    # get min and max values
    if datamin is None:
        datamin = np.nanmin(data_cmap)
    if datamax is None:
        datamax = np.nanmax(data_cmap)
        
    # get colormap to plot
    if logscale_cmap: # plot on logscale
        data_cmap = symlog(data_cmap) 
        datamin, datamax = np.min(data_cmap), np.max(data_cmap)
        
    # get cmap gradations
    dataint = (datamax-datamin)/100
    norm = plt.Normalize(datamin, datamax+dataint)
    # convert data array into colormap
    colormap = cmap(norm(data_cmap))   
        
    # Set the positions of nan values in colormap to 'lime'
    colormap[naninds[0], naninds[1], :3] = 0,1,0
    colormap[infinds[0], infinds[1], :3] = 1,1,1

    # plot colormap
    im = ax.imshow(colormap, interpolation='nearest')

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.07)
    cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)    
    
    if logscale_cmap == True: 
        cbar_labels = cbar.ax.get_yticks()
        cbar.set_ticks(cbar_labels)
        cbar_labels_unlog = list(np.round(np.exp(np.array(cbar_labels)),2))
        cbar.set_ticklabels(cbar_labels_unlog)
        
    # Turn off gridlines if required
    ax.tick_params(axis='both', which='both', length=0, gridOn=show_gridlines) 
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=8, ha="right")
    ax.set_yticklabels(row_labels, fontsize=8)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90,
             rotation_mode="anchor")
    
    # Annotate
    if annotate is not None:
        if isinstance(annotate, int):
            ndecimals = annotate
        else:
            ndecimals = 3
        annotate_heatmap(array, ax, ndecimals=ndecimals)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # set xticks
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar, ax


