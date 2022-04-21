from ast import AsyncFunctionDef
import os
import numpy as np
import h5py
import tifffile as tiff

import utils

import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = False
#plt.rcParams['text.latex.unicode'] = False

# Creates a dictionary, path_dict, with all of the required path information for the following
# functions including save directory and finding the s2p-output
# Parameters:
#            fdir - the path to the original recording file
#            threshold_scaling_values - the corresponding threshold_scaling value used
#                                       by the automatic script  
#            tseries_start_end - the time-scale of the roi activity trace that will be outputted
#            rois_to_plot - the number of ROI's whose mask will be outlined
#            output_fig_dir - the path for the output, i.e where the figures will be saved
def define_paths_roi_plots(fdir, threshold_scaling_values, tseries_start_end, rois_to_plot, output_fig_dir):
    
    path_dict = {}
    # define paths for loading s2p data
    if threshold_scaling_values == 0:
        path_dict['s2p_dir'] = os.path.join(fdir, 'suite2p', 'plane0')
    else:
        path_dict['s2p_dir'] = os.path.join(fdir, f'threshold_scaling_{threshold_scaling_values}', 'plane0')

    path_dict['threshold_scaling_values'] = threshold_scaling_values
    path_dict['tseries_start_end'] = tseries_start_end
    path_dict['rois_to_plot'] = rois_to_plot
    path_dict['s2p_F_path'] = os.path.join(path_dict['s2p_dir'], 'F.npy')
    path_dict['s2p_Fneu_path'] = os.path.join(path_dict['s2p_dir'], 'Fneu.npy')
    path_dict['s2p_iscell_path'] = os.path.join(path_dict['s2p_dir'], 'iscell.npy')
    path_dict['s2p_ops_path'] = os.path.join(path_dict['s2p_dir'], 'ops.npy')
    path_dict['s2p_stat_path'] = os.path.join(path_dict['s2p_dir'], 'stat.npy')
    
    path_dict['suite2p_dat_dir'] = os.path.join(f'threshold_scaling_{threshold_scaling_values}','plane0')
    path_dict['fig_save_dir'] = output_fig_dir
    # utils.check_exist_dir(path_dict['fig_save_dir'])
    print("this function is working")

    return path_dict

#Takes the path information from path_dict and uses it to load and save the files
#they direct towards
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
def plotting_rois(s2p_data_dict, path_dict):
    plot_vars = {}
    iscell_ids = np.where( s2p_data_dict['iscell'][:,0] == 1 )[0] # indices of user-curated cells referencing all ROIs detected by s2p

    if isinstance(path_dict['rois_to_plot'], int): # if int is supplied, first n user-curated rois included in analysis
        path_dict['rois_to_plot'] = np.arange(path_dict['rois_to_plot'])
    elif path_dict['rois_to_plot'] is None: # if None is supplied, all user-curated rois included in analysis
        path_dict['rois_to_plot'] = np.arange(len(iscell_ids))

    plot_vars['cell_ids'] = iscell_ids[path_dict['rois_to_plot']] # indices of detected cells across all ROIs from suite2p
    plot_vars['num_rois'] = len(path_dict['rois_to_plot'])

    return plot_vars

# initialize templates for contour map
def template_init(plot_vars, s2p_data_dict):

    plot_vars['colors_roi'] = plt.cm.viridis(np.linspace(0,1,plot_vars['num_rois']))
    plot_vars['s2p_masks'] = np.empty([plot_vars['num_rois'], s2p_data_dict['ops']['Ly'], s2p_data_dict['ops']['Lx']])
    plot_vars['roi_centroids'] = np.empty([plot_vars['num_rois'], 2])

    # loop through ROIs and add their spatial footprints to template
    for idx, roi_id in enumerate(plot_vars['cell_ids']):

        zero_template = np.zeros([s2p_data_dict['ops']['Ly'], s2p_data_dict['ops']['Lx']])
        zero_template[ s2p_data_dict['stat'][roi_id]['ypix'], s2p_data_dict['stat'][roi_id]['xpix'] ] = 1
        plot_vars['s2p_masks'][idx,...] = zero_template

        plot_vars['roi_centroids'][idx,...] = [np.min(s2p_data_dict['stat'][roi_id]['ypix']), np.min(s2p_data_dict['stat'][roi_id]['xpix'])]

        if idx == plot_vars['num_rois']-1:
            break

    return plot_vars

# plot contours and cell numbers on projection image
def contour_plot(s2p_data_dict, path_dict, plot_vars):
    threshold_scaling_values = s2p_data_dict['threshold_scaling_values']
    
    to_plot = s2p_data_dict['ops']['meanImg']

    fig, ax = plt.subplots(1, 1, figsize = (10,10))
    ax.imshow(to_plot, cmap = 'gray', vmin=np.min(to_plot)*1.0, vmax=np.max(to_plot)*0.6)
    ax.axis('off')

    for idx, roi_id in enumerate(plot_vars['cell_ids']): 
        ax.contour(plot_vars['s2p_masks'][idx,:,:], colors=[plot_vars['colors_roi'][idx]])
        ax.text(plot_vars['roi_centroids'][idx][1]-1, plot_vars['roi_centroids'][idx][0]-1,  str(idx), fontsize=18, weight='bold', color = plot_vars['colors_roi'][idx]);

    plt.savefig(os.path.join(path_dict['fig_save_dir'], f'roi_contour_map_{threshold_scaling_values}.png'))
    plt.savefig(os.path.join(path_dict['fig_save_dir'], f'roi_contour_map_{threshold_scaling_values}.pdf'))

# initialize variables for plotting time-series
def time_series_plot(s2p_data_dict, path_dict, plot_vars):
    threshold_scaling_values = s2p_data_dict['threshold_scaling_values']

    fs = s2p_data_dict['ops']['fs']
    num_samps = s2p_data_dict['ops']['nframes']
    total_time = num_samps/fs 
    tvec = np.linspace(0,total_time,num_samps)

    # F_npil_corr_dff contains all s2p-detected cells; cell_ids references those indices
    trace_data_selected = s2p_data_dict['F_npil_corr_dff'][plot_vars['cell_ids']]


    fig, ax = plt.subplots(plot_vars['num_rois'], 1, figsize = (9,2*plot_vars['num_rois']))
    for idx in range(plot_vars['num_rois']):
        
        to_plot = trace_data_selected[idx] 
        
        ax[idx].plot(tvec, np.transpose( to_plot ), color = plot_vars['colors_roi'][idx] );
        
        ax[idx].tick_params(axis='both', which='major', labelsize=13)
        ax[idx].tick_params(axis='both', which='minor', labelsize=13)
        if idx == np.ceil(plot_vars['num_rois']/2-1):
            ax[idx].set_ylabel('Fluorescence Level',fontsize = 20)
            
    # Setting the values for all axes.
    if path_dict['tseries_start_end'] is None:
        xlims = [0,tvec[-1]]
    else:
        xlims = path_dict['tseries_start_end']
    plt.setp(ax, xlim=xlims, ylim=[np.min(trace_data_selected)+np.min(trace_data_selected)*0.1, 
                                        np.max(trace_data_selected)+np.max(trace_data_selected)*0.1])  

    ax[idx].set_xlabel('Time (s)',fontsize = 20);

    plt.savefig(os.path.join(path_dict['fig_save_dir'], f'roi_ts.png_{threshold_scaling_values}'))
    plt.savefig(os.path.join(path_dict['fig_save_dir'], f'roi_ts.pdf_{threshold_scaling_values}'))
    