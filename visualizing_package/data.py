import os
import numpy as np
import glob
import pickle
import json
import seaborn as sns
import matplotlib.ticker as ticker
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import plotly.graph_objects as go
import util_funcs as utils
import random

class EventTicksProcessor:
    def __init__(self, fname_signal, fname_events, fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization, cond_colors=['steelblue', 'crimson', 'orchid', 'gold']):
        self.signal_to_plot = None
        self.min_max = None
        self.min_max_all = None
        self.tvec = None
        self.event_times = None
        self.conditions = None
        self.cond_colors = cond_colors
        self.fparams = self.define_params(fname_signal, fname_events, fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization)

    def define_params(self, fname_signal, fname_events, fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization):
        # User-defined variables
        fparams = {}
        
        fparams['fname_signal'] = fname_signal
        fparams['fname_events'] = fname_events
        fparams['fs'] = fs
        fparams['opto_blank_frame'] = opto_blank_frame
        fparams['num_rois'] = num_rois
        fparams['selected_conditions'] = selected_conditions
        fparams['flag_normalization'] = flag_normalization

        return fparams

    def calc_dff_percentile(self, activity_vec, perc=25):
        perc_activity = np.percentile(activity_vec, perc)
        return (activity_vec - perc_activity) / perc_activity

    def calc_zscore(self, activity_vec, baseline_samples):
        mean_baseline = np.nanmean(activity_vec[..., baseline_samples])
        std_baseline = np.nanstd(activity_vec[..., baseline_samples])
        return (activity_vec - mean_baseline) / std_baseline

    def load_data(self, signals_content, estimmed_frames=None):
        # Load time-series data
        signals = utils.load_signals(signals_content)
        if self.fparams['opto_blank_frame']:
            try:
                glob_stim_files = glob.glob(estimmed_frames)
                stim_frames = pickle.load(open(glob_stim_files[0], "rb"))
                signals[:, stim_frames['samples']] = None  # Blank out stimmed frames
                flag_stim = True
                print('Detected stim data; replaced stim samples with NaNs')
            except:
                flag_stim = False
                print('Note: No stim preprocessed meta data detected.')

        if self.fparams['flag_normalization'] == 'dff':
            signal_to_plot = np.apply_along_axis(utils.calc_dff, 1, signals)
        elif self.fparams['flag_normalization'] == 'dff_perc':
            signal_to_plot = np.apply_along_axis(self.calc_dff_percentile, 1, signals)
        elif self.fparams['flag_normalization'] == 'zscore':
            signal_to_plot = np.apply_along_axis(self.calc_zscore, 1, signals, np.arange(0, signals.shape[1]))
        else:
            signal_to_plot = signals

        self.signal_to_plot = signal_to_plot

        min_max = [list(min_max_tup) for min_max_tup in zip(np.min(signal_to_plot, axis=1), np.max(signal_to_plot, axis=1))]
        self.min_max = min_max
        self.min_max_all = [np.min(signal_to_plot), np.max(signal_to_plot)]

        if self.fparams['num_rois'] == 'all':
            self.fparams['num_rois'] = signals.shape[0]

        total_session_time = signals.shape[1] / self.fparams['fs']
        tvec = np.round(np.linspace(0, total_session_time, signals.shape[1]), 2)
        self.tvec = tvec

    def load_behav_data(self, fname_events_content):
        if self.fparams['fname_events']:
            glob_event_files = glob.glob(fname_events_content)
            if not glob_event_files:
                print(f'{self.fparams["fname_events"]} not detected. Please check if the path is correct.')
            if 'csv' in glob_event_files[0]:
                event_times = utils.df_to_dict(glob_event_files[0])
            elif 'pkl' in glob_event_files[0]:
                event_times = pickle.load(open(glob_event_files[0], "rb"), fix_imports=True, encoding='latin1')
            event_frames = utils.dict_time_to_samples(event_times, self.fparams['fs'])

            self.event_times = {}
            if self.fparams['selected_conditions']:
                self.conditions = self.fparams['selected_conditions']
            else:
                self.conditions = event_frames.keys()
            for cond in self.conditions:
                self.event_times[cond] = (np.array(event_frames[cond]) / self.fparams['fs']).astype('int')
    
    def load_all_data(self, signals_content, fname_events_content, estimmed_frames=None):
        self.load_data(signals_content, estimmed_frames)
        self.load_behav_data(fname_events_content)

class S2PActivityProcessor:
    def __init__(self, fdir, tseries_start_end, show_labels, color_all_rois, rois_to_plot, output_fig_dir):
        self.fdir = fdir
        self.tseries_start_end = tseries_start_end
        self.show_labels = show_labels
        self.color_all_rois = color_all_rois
        self.rois_to_plot = rois_to_plot
        self.output_fig_dir = output_fig_dir
        self.path_dict = {}
        self.s2p_data_dict = {}
        self.plot_vars = {}

    def define_paths_roi_plots(self, f_contents, fneu_contents, iscell_contents, ops_contents, stat_contents):
        self.path_dict['tseries_start_end'] = self.tseries_start_end
        self.path_dict['rois_to_plot'] = self.rois_to_plot
        self.path_dict['s2p_F_path'] = f_contents
        self.path_dict['s2p_Fneu_path'] = fneu_contents
        self.path_dict['s2p_iscell_path'] = iscell_contents
        self.path_dict['s2p_ops_path'] = ops_contents
        self.path_dict['s2p_stat_path'] = stat_contents

        return self.path_dict

    def load_s2p_data_roi_plots(self):
        self.s2p_data_dict['F'] = np.load(self.path_dict['s2p_F_path'], allow_pickle=True)
        self.s2p_data_dict['Fneu'] = np.load(self.path_dict['s2p_Fneu_path'], allow_pickle=True)
        self.s2p_data_dict['iscell'] = np.load(self.path_dict['s2p_iscell_path'], allow_pickle=True)
        self.s2p_data_dict['ops'] = np.load(self.path_dict['s2p_ops_path'], allow_pickle=True).item()
        self.s2p_data_dict['stat'] = np.load(self.path_dict['s2p_stat_path'], allow_pickle=True)

        self.s2p_data_dict['F_npil_corr'] = self.s2p_data_dict['F'] - self.s2p_data_dict['ops']['neucoeff'] * self.s2p_data_dict['Fneu']
        self.s2p_data_dict['F_npil_corr_dff'] = np.apply_along_axis(utils.calc_dff, 1, self.s2p_data_dict['F_npil_corr'])

    def prep_plotting_rois(self):
        self.plot_vars['cell_ids'] = np.where(self.s2p_data_dict['iscell'][:, 0] == 1)[0]
        self.plot_vars['num_total_rois'] = len(self.plot_vars['cell_ids'])
        self.plot_vars['color_all_rois'] = self.color_all_rois

        max_rois_tseries = 10
        if isinstance(self.path_dict['rois_to_plot'], list):
            self.plot_vars['rois_to_tseries'] = self.path_dict['rois_to_plot']
        elif self.plot_vars['num_total_rois'] > max_rois_tseries:
            self.plot_vars['rois_to_tseries'] = sorted(random.sample(self.plot_vars['cell_ids'].tolist(), max_rois_tseries))
        else:
            self.plot_vars['rois_to_tseries'] = self.plot_vars['cell_ids']

        self.plot_vars['num_rois_to_tseries'] = len(self.plot_vars['rois_to_tseries'])
    
    def masks_init(self):
        if self.plot_vars['color_all_rois']:
            num_rois_to_color = self.plot_vars['num_total_rois']
        else:    
            num_rois_to_color = self.plot_vars['num_rois_to_tseries']
            
        self.plot_vars['colors_roi_name'] = plt.cm.viridis(np.linspace(0,1,num_rois_to_color))
        self.plot_vars['colors_roi'] = [f'rgb{tuple(np.round(np.array(c[:3]) * 254).astype(int))}' for c in plt.cm.viridis(np.linspace(0, 1, self.plot_vars['num_rois_to_tseries']))]
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
            sample_start = utils.get_tvec_sample(tvec, self.path_dict['tseries_start_end'][0])
            sample_end = utils.get_tvec_sample(tvec, self.path_dict['tseries_start_end'][1])
            tvec = tvec[sample_start:sample_end]
            trace_data_selected = trace_data_selected[:, sample_start:sample_end]
        
        return tvec, trace_data_selected