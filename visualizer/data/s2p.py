import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from visualizer import misc
import random
import os 
import pandas as pd

class S2PActivityProcessor:
    def __init__(self, tseries_start_end, show_labels, color_all_rois, rois_to_plot):
        self.tseries_start_end = tseries_start_end
        self.show_labels = show_labels
        self.color_all_rois = color_all_rois
        self.rois_to_plot = rois_to_plot
        self.path_dict = {}
        self.s2p_data_dict = {}
        self.plot_vars = {}

    def define_paths_roi_plots(self, f_contents, fneu_contents, iscell_contents, ops_contents, stat_contents, spks_contents=None):
        self.path_dict['tseries_start_end'] = self.tseries_start_end
        self.path_dict['rois_to_plot'] = self.rois_to_plot
        self.path_dict['s2p_F_path'] = f_contents
        self.path_dict['s2p_Fneu_path'] = fneu_contents
        self.path_dict['s2p_iscell_path'] = iscell_contents
        self.path_dict['s2p_ops_path'] = ops_contents
        self.path_dict['s2p_stat_path'] = stat_contents
        self.path_dict['s2p_spks_path'] = spks_contents

        return self.path_dict

    def load_s2p_data_roi_plots(self):
        self.s2p_data_dict['F'] = np.load(self.path_dict['s2p_F_path'], allow_pickle=True)
        self.s2p_data_dict['Fneu'] = np.load(self.path_dict['s2p_Fneu_path'], allow_pickle=True)
        self.s2p_data_dict['iscell'] = np.load(self.path_dict['s2p_iscell_path'], allow_pickle=True)
        self.s2p_data_dict['ops'] = np.load(self.path_dict['s2p_ops_path'], allow_pickle=True).item()
        self.s2p_data_dict['stat'] = np.load(self.path_dict['s2p_stat_path'], allow_pickle=True)

        self.s2p_data_dict['F_npil_corr'] = self.s2p_data_dict['F'] - self.s2p_data_dict['ops']['neucoeff'] * self.s2p_data_dict['Fneu']
        self.s2p_data_dict['F_npil_corr_dff'] = np.apply_along_axis(misc.calc_dff, 1, self.s2p_data_dict['F_npil_corr'])

    def prep_plotting_rois(self):
        self.plot_vars['cell_ids'] = np.where(self.s2p_data_dict['iscell'][:, 0] == 1)[0]
        self.plot_vars['num_total_rois'] = len(self.plot_vars['cell_ids'])
        self.plot_vars['color_all_rois'] = self.color_all_rois

        if isinstance(self.path_dict['rois_to_plot'], list):
            self.plot_vars['rois_to_tseries'] = self.path_dict['rois_to_plot']
        elif isinstance(self.path_dict['rois_to_plot'], int):
            self.plot_vars['rois_to_tseries'] = [x for x in range(self.path_dict['rois_to_plot'] + 1)]
        else:
            self.plot_vars['rois_to_tseries'] = self.plot_vars['cell_ids']

        self.plot_vars['num_rois_to_tseries'] = len(self.plot_vars['rois_to_tseries'])
    
    def masks_init(self):
        if self.plot_vars['color_all_rois']:
            num_rois_to_color = self.plot_vars['num_total_rois']
        else:    
            num_rois_to_color = self.plot_vars['num_rois_to_tseries']
            
        self.plot_vars['colors_roi_name'] = plt.cm.viridis(np.linspace(0,1,num_rois_to_color))
        self.plot_vars['colors_roi'] = [f'rgb{tuple(np.round(np.array(c[:3]) * 254).astype(int))}' for c in plt.cm.viridis(np.linspace(0, 1, len(self.plot_vars['cell_ids'])))]
        self.plot_vars['s2p_masks'] = np.empty([self.plot_vars['num_total_rois'], self.s2p_data_dict['ops']['Ly'], self.s2p_data_dict['ops']['Lx']])
        self.plot_vars['roi_centroids'] = np.empty([self.plot_vars['num_total_rois'], 2])

        # loop through ROIs and add their spatial footprints to template
        for idx, roi_id in enumerate(self.plot_vars['cell_ids']):

            zero_template = np.zeros([self.s2p_data_dict['ops']['Ly'], self.s2p_data_dict['ops']['Lx']])
            zero_template[ self.s2p_data_dict['stat'][roi_id]['ypix'], self.s2p_data_dict['stat'][roi_id]['xpix'] ] = 1
            self.plot_vars['s2p_masks'][idx,...] = zero_template

            self.plot_vars['roi_centroids'][idx,...] = [np.min(self.s2p_data_dict['stat'][roi_id]['ypix']), np.min(self.s2p_data_dict['stat'][roi_id]['xpix'])]

            if idx == self.plot_vars['num_total_rois']-1:
                break

    def setup_roi_data(self, f_contents, fneu_contents, iscell_contents, ops_contents, stat_contents):
        self.define_paths_roi_plots(f_contents, fneu_contents, iscell_contents, ops_contents, stat_contents)
        self.load_s2p_data_roi_plots()
        self.prep_plotting_rois()
        self.masks_init()
    
    def generate_tsv_and_trace(self):
        fs = self.s2p_data_dict['ops']['fs']
        num_samps = self.s2p_data_dict['ops']['nframes']
        total_time = num_samps / fs
        tvec = np.linspace(0, total_time, num_samps)

        # F_npil_corr_dff contains all s2p-detected cells
        trace_data_selected = self.s2p_data_dict['F_npil_corr_dff'][self.plot_vars['rois_to_tseries']]

        # Cut data and tvec to start/end if user defined
        if self.path_dict['tseries_start_end']:
            sample_start = misc.get_tvec_sample(tvec, self.path_dict['tseries_start_end'][0])
            sample_end = misc.get_tvec_sample(tvec, self.path_dict['tseries_start_end'][1])
            tvec = tvec[sample_start:sample_end]
            trace_data_selected = trace_data_selected[:, sample_start:sample_end]
        
        return tvec, trace_data_selected