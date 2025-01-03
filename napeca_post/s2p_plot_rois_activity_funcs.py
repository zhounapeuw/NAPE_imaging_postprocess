from ast import AsyncFunctionDef
from cgitb import reset
from inspect import trace
import os
import numpy as np
import random
import utils
import matplotlib.pyplot as plt

def s2p_dir(fdir):
    path_dict={}
    path_dict['s2p_dir'] = os.path.join(fdir, 'suite2p', 'plane0')

    return path_dict

# Creates a dictionary, path_dict, with all of the required path information for the following
# functions including save directory and finding the s2p-output
# Parameters:
#            path_dict - a dictionary with, at this point, only the path of where the data is located
#            tseries_start_end - the time-scale of the roi activity trace that will be outputted
#            rois_to_plot - the number of ROI's whose mask will be outlined
#            output_fig_dir - the path for the output, i.e where the figures will be saved
def define_paths_roi_plots(path_dict, tseries_start_end, rois_to_plot, output_fig_dir):

    path_dict['tseries_start_end'] = tseries_start_end
    path_dict['rois_to_plot'] = rois_to_plot
    path_dict['s2p_F_path'] = os.path.join(path_dict['s2p_dir'], 'F.npy')
    path_dict['s2p_Fneu_path'] = os.path.join(path_dict['s2p_dir'], 'Fneu.npy')
    path_dict['s2p_iscell_path'] = os.path.join(path_dict['s2p_dir'], 'iscell.npy')
    path_dict['s2p_ops_path'] = os.path.join(path_dict['s2p_dir'], 'ops.npy')
    path_dict['s2p_stat_path'] = os.path.join(path_dict['s2p_dir'], 'stat.npy')
    
    path_dict['fig_save_dir'] = output_fig_dir
    return path_dict

# Takes the path information from path_dict and uses it to load and save the files
# they direct towards


def load_s2p_data_roi_plots(path_dict):
    
    s2p_data_dict = {}
    # load s2p data
    s2p_data_dict['F'] = np.load(path_dict['s2p_F_path'], allow_pickle=True)
    s2p_data_dict['Fneu'] = np.load(path_dict['s2p_Fneu_path'], allow_pickle=True)
    s2p_data_dict['iscell'] = np.load(path_dict['s2p_iscell_path'], allow_pickle=True)
    s2p_data_dict['ops'] = np.load(path_dict['s2p_ops_path'], allow_pickle=True).item()
    s2p_data_dict['stat'] = np.load(path_dict['s2p_stat_path'], allow_pickle=True)

    s2p_data_dict['F_npil_corr'] = s2p_data_dict['F'] - s2p_data_dict['ops']['neucoeff'] * s2p_data_dict['Fneu']

    s2p_data_dict['F_npil_corr_dff'] = np.apply_along_axis(utils.calc_dff, 1, s2p_data_dict['F_npil_corr'])

    return s2p_data_dict

#initializes variables for roi plots
def prep_plotting_rois(s2p_data_dict, path_dict, color_all_rois): 
    max_rois_tseries = 10
    plot_vars = {}
    plot_vars['cell_ids'] = np.where( s2p_data_dict['iscell'][:,0] == 1 )[0] # indices of user-curated cells referencing all ROIs detected by s2p
    plot_vars['num_total_rois'] = len(plot_vars['cell_ids'])
    plot_vars['color_all_rois'] = color_all_rois
    
    # determine if only a subset of cells tseries are to be plotted
    if isinstance(path_dict['rois_to_plot'], list): # if user supplied ROIs
        plot_vars['rois_to_tseries'] = path_dict['rois_to_plot']
        plot_vars['num_rois_to_tseries'] = len(plot_vars['rois_to_tseries'])
    elif plot_vars['num_total_rois'] > max_rois_tseries: # if too many cells to visualize tseries, randomly sample from cells
        plot_vars['rois_to_tseries'] = sorted(random.sample(plot_vars['cell_ids'].tolist(), max_rois_tseries))
        plot_vars['num_rois_to_tseries'] = len(plot_vars['rois_to_tseries'])
    else:
        plot_vars['rois_to_tseries'] = plot_vars['cell_ids']
        plot_vars['num_rois_to_tseries'] = plot_vars['num_total_rois'] 
        
    return plot_vars

# initialize templates for contour map
def masks_init(plot_vars, s2p_data_dict):
    
    if plot_vars['color_all_rois']:
        num_rois_to_color = plot_vars['num_total_rois']
    else:    
        num_rois_to_color = plot_vars['num_rois_to_tseries']
    plot_vars['colors_roi'] = plt.cm.viridis(np.linspace(0,1,num_rois_to_color))
    plot_vars['s2p_masks'] = np.empty([plot_vars['num_total_rois'], s2p_data_dict['ops']['Ly'], s2p_data_dict['ops']['Lx']])
    plot_vars['roi_centroids'] = np.empty([plot_vars['num_total_rois'], 2])

    # loop through ROIs and add their spatial footprints to template
    for idx, roi_id in enumerate(plot_vars['cell_ids']):

        zero_template = np.zeros([s2p_data_dict['ops']['Ly'], s2p_data_dict['ops']['Lx']])
        zero_template[ s2p_data_dict['stat'][roi_id]['ypix'], s2p_data_dict['stat'][roi_id]['xpix'] ] = 1
        plot_vars['s2p_masks'][idx,...] = zero_template

        plot_vars['roi_centroids'][idx,...] = [np.min(s2p_data_dict['stat'][roi_id]['ypix']), np.min(s2p_data_dict['stat'][roi_id]['xpix'])]

        if idx == plot_vars['num_total_rois']-1:
            break

    return plot_vars

# plot contours and cell numbers on projection image
def contour_plot(s2p_data_dict, path_dict, plot_vars, show_labels_=True, cmap_scale_ratio=1):
    if 'threshold_scaling_value' in path_dict:
        tsv = path_dict['threshold_scaling_value']
    
    to_plot = s2p_data_dict['ops']['meanImg']

    fig, ax = plt.subplots(1, 1, figsize = (10,10))
    ax.imshow(to_plot, cmap = 'gray', vmin=np.min(to_plot)*(1.0/cmap_scale_ratio), vmax=np.max(to_plot)*(1.0/cmap_scale_ratio))
    ax.axis('off')
    
    idx_color_rois = 0
    for idx, roi_id in enumerate(plot_vars['cell_ids']): 
        if roi_id in plot_vars['rois_to_tseries'] or plot_vars['color_all_rois']:
            this_roi_color = plot_vars['colors_roi'][idx_color_rois]
            idx_color_rois += 1
        else:
            this_roi_color = 'grey'
        ax.contour(plot_vars['s2p_masks'][idx,:,:], colors=[this_roi_color], linewidths=0.5)
        if show_labels_ and roi_id in plot_vars['rois_to_tseries']:
            ax.text(plot_vars['roi_centroids'][idx][1]-1, plot_vars['roi_centroids'][idx][0]-1,  str(roi_id), fontsize=18, weight='bold', color = this_roi_color)

    if 'tsv' in locals():
        save_name_png = os.path.join(path_dict['fig_save_dir'], f'roi_contour_map_{tsv}.png')
        save_name_pdf = os.path.join(path_dict['fig_save_dir'], f'roi_contour_map_{tsv}.pdf')
    else:
        save_name_png = os.path.join(path_dict['fig_save_dir'], 'roi_contour_map.png')
        save_name_pdf = os.path.join(path_dict['fig_save_dir'], 'roi_contour_map.pdf')

    plt.savefig(save_name_png)
    plt.savefig(save_name_pdf)

# initialize variables for plotting time-series
def time_series_plot(s2p_data_dict, path_dict, plot_vars):
    
    if 'threshold_scaling_value' in path_dict:
        tsv = path_dict['threshold_scaling_value']

    fs = s2p_data_dict['ops']['fs']
    num_samps = s2p_data_dict['ops']['nframes']
    total_time = num_samps/fs 
    tvec = np.linspace(0,total_time,num_samps)
        
    # F_npil_corr_dff contains all s2p-detected cells
    trace_data_selected = s2p_data_dict['F_npil_corr_dff'][plot_vars['rois_to_tseries']]
    
    # cut data and tvec to start/end if user defined
    if path_dict['tseries_start_end']:
        sample_start = utils.get_tvec_sample(tvec, path_dict['tseries_start_end'][0])
        sample_end = utils.get_tvec_sample(tvec, path_dict['tseries_start_end'][1])
        tvec = tvec[sample_start:sample_end]
        trace_data_selected = trace_data_selected[:,sample_start:sample_end]
    
    fig, ax = plt.subplots(plot_vars['num_rois_to_tseries'], 1, figsize = (9,2*plot_vars['num_rois_to_tseries']))
    for idx in range(plot_vars['num_rois_to_tseries']):
        
        to_plot = trace_data_selected[idx] 
        
        ax[idx].plot(tvec, np.transpose( to_plot ), color = plot_vars['colors_roi'][idx] )
        
        ax[idx].set_title(f"ROI {plot_vars['rois_to_tseries'][idx]}")
        ax[idx].tick_params(axis='both', which='major', labelsize=13)
        ax[idx].tick_params(axis='both', which='minor', labelsize=13)
        if idx == np.ceil(plot_vars['num_rois_to_tseries']/2-1):
            ax[idx].set_ylabel('Fluorescence Level',fontsize = 20)
            

    plt.setp(ax, xlim=None, ylim=[np.min(trace_data_selected)*1.1, 
                                        np.max(trace_data_selected)*1.1])  

    ax[idx].set_xlabel('Time (s)',fontsize = 20)

    if 'tsv' in locals():
        save_name_png = os.path.join(path_dict['fig_save_dir'], f'roi_ts_{tsv}.png')
        save_name_pdf = os.path.join(path_dict['fig_save_dir'], f'roi_ts_{tsv}.pdf')
    else:
        save_name_png = os.path.join(path_dict['fig_save_dir'], 'roi_ts.png')
        save_name_pdf = os.path.join(path_dict['fig_save_dir'], 'roi_ts.pdf')

    plt.savefig(save_name_png)
    plt.savefig(save_name_pdf)

def heatmap_plot(s2p_data_dict, path_dict, plot_vars):
    # under development
    plt.figure(figsize = (10, 10))
    
    if 'threshold_scaling_value' in path_dict:
        tsv = path_dict['threshold_scaling_value']

    fs = s2p_data_dict['ops']['fs']
    num_samps = s2p_data_dict['ops']['nframes']
    total_time = num_samps/fs 
    tvec = np.linspace(0,total_time,num_samps)
        
    # F_npil_corr_dff contains all s2p-detected cells; cell_ids references those indices
    trace_data_selected = s2p_data_dict['F_npil_corr_dff'][plot_vars['cell_ids']]

    extent_ = [tvec[0], tvec[-1], plot_vars['num_rois'], 0 ]
    
    # cut data and tvec to start/end if user defined
    if path_dict['tseries_start_end']:
        sample_start = utils.get_tvec_sample(tvec, path_dict['tseries_start_end'][0])
        sample_end = utils.get_tvec_sample(tvec, path_dict['tseries_start_end'][1])
        tvec = tvec[sample_start:sample_end]
        trace_data_selected = trace_data_selected[:,sample_start:sample_end]
        
    plt.imshow(trace_data_selected, extent=extent_)