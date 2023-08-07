import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from visualizer import misc
import random
import os 
import pandas as pd

class EventTicksProcessor:
    def __init__(self, fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization, cond_colors=['steelblue', 'crimson', 'orchid', 'gold']):
        self.signal_to_plot = None
        self.min_max = None
        self.min_max_all = None
        self.tvec = None
        self.event_times = None
        self.conditions = None
        self.cond_colors = cond_colors
        self.fparams = self.define_params(fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization)

    def define_params(self, fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization):
        # User-defined variables
        fparams = {}
        
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
        signals = misc.load_signals(signals_content)
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
            signal_to_plot = np.apply_along_axis(misc.calc_dff, 1, signals)
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
        if fname_events_content:
            self.event_times = misc.df_to_dict(fname_events_content)
            event_frames = misc.dict_time_to_samples(self.event_times, self.fparams['fs'])

            self.event_times = {}
            if self.fparams['selected_conditions']:
                self.conditions = self.fparams['selected_conditions'] 
            else:
                self.conditions = event_frames.keys()
            for cond in self.conditions: # convert event samples to time in seconds
                self.event_times[cond] = (np.array(event_frames[cond])/self.fparams['fs']).astype('int')
        self.fname_events_content = fname_events_content
    
    def load_all_data(self, signals_content, fname_events_content, estimmed_frames=None):
        self.load_data(signals_content, estimmed_frames)
        self.load_behav_data(fname_events_content)

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
            sample_start = misc.get_tvec_sample(tvec, self.path_dict['tseries_start_end'][0])
            sample_end = misc.get_tvec_sample(tvec, self.path_dict['tseries_start_end'][1])
            tvec = tvec[sample_start:sample_end]
            trace_data_selected = trace_data_selected[:, sample_start:sample_end]
        
        return tvec, trace_data_selected
    
class EventAnalysisProcessor:
    def __init__(self, fparams):
        self.fparams = fparams
        self.signals_fpath = os.path.join(fparams['fdir'], fparams['fname_signal'])
        self.events_file_path = os.path.join(fparams['fdir'], fparams['fname_events'])
        self.save_dir = os.path.join(fparams['fdir'], 'event_rel_analysis')
        misc.check_exist_dir(self.save_dir)

    def load_data(self):
        signals = misc.load_signals(self.signals_fpath)
        num_rois = signals.shape[0]
        all_nan_rois = np.where(np.apply_along_axis(self.is_all_nans, 1, signals))
        if self.fparams['opto_blank_frame']:
            try:
                glob_stim_files = glob.glob(os.path.join(self.fparams['fdir'], "{}*_stimmed_frames.pkl".format(self.fparams['fname'])))
                stim_frames = pickle.load(open(glob_stim_files[0], "rb"))
                signals[:, stim_frames['samples']] = None
                flag_stim = True
                print('Detected stim data; replaced stim samples with NaNs')
            except:
                flag_stim = False
                print('Note: No stim preprocessed meta data detected.')
        else:
            flag_stim = False

        return signals, num_rois, all_nan_rois, flag_stim

    @staticmethod
    def is_all_nans(vector):
        if isinstance(vector, pd.Series):
            vector = vector.values
        return np.isnan(vector).all()

    def load_event_data(self):
        glob_event_files = glob.glob(self.events_file_path)
        if not glob_event_files:
            print(f'{self.events_file_path} not detected. Please check if path is correct.')
        if 'csv' in glob_event_files[0]:
            event_times = misc.df_to_dict(glob_event_files[0])
        elif any(x in glob_event_files[0] for x in ['pkl', 'pickle']):
            event_times = pickle.load(open(glob_event_files[0], "rb"), fix_imports=True, encoding='latin1')

        event_frames = misc.dict_time_to_samples(event_times, self.fparams['fs'])
        all_conditions = event_frames.keys()
        conditions = [condition for condition in all_conditions if len(event_frames[condition]) > 0]
        conditions.sort()

        return event_frames, conditions

    def extract_trial_data(self, signals, tvec, trial_begEnd_samp, event_frames, conditions, baseline_begEnd_samp):
        data_dict = misc.extract_trial_data(signals, tvec, trial_begEnd_samp, event_frames, conditions,
                                             baseline_start_end_samp=baseline_begEnd_samp, save_dir=self.save_dir)
        return data_dict

    @staticmethod
    def sort_heatmap_peaks(data, tvec, sort_epoch_start_time, sort_epoch_end_time, sort_method='peak_time'):
        sort_epoch_start_samp = misc.find_nearest_idx(tvec, sort_epoch_start_time)[0]
        sort_epoch_end_samp = misc.find_nearest_idx(tvec, sort_epoch_end_time)[0]

        if sort_method == 'peak_time':
            epoch_peak_samp = np.argmax(data[:, sort_epoch_start_samp:sort_epoch_end_samp], axis=1)
            final_sorting = np.argsort(epoch_peak_samp)
        elif sort_method == 'max_value':
            time_max = np.nanmax(data[:, sort_epoch_start_samp:sort_epoch_end_samp], axis=1)
            final_sorting = np.argsort(time_max)[::-1]

        return final_sorting

    def process_data(self):
        signals, num_rois, all_nan_rois, flag_stim = self.load_data()
        event_frames, conditions = self.load_event_data()

        if self.fparams['flag_sort_rois']:
            if not self.fparams['roi_sort_cond']:
                self.fparams['roi_sort_cond'] = conditions[0]

            if self.fparams['roi_sort_cond'] not in event_frames.keys():
                sorted_roi_order = range(num_rois)
                interesting_rois = self.fparams['interesting_rois']
                print('Specified condition to sort by doesn\'t exist! ROIs are in default sorting.')
            else:
                sorted_roi_order = self.sort_heatmap_peaks(self.data_dict[self.fparams['roi_sort_cond']][self.data_trial_avg_key],
                                                           self.tvec, 0, self.trial_start_end_sec[-1],
                                                           self.fparams['user_sort_method'])

                interesting_rois = np.in1d(sorted_roi_order, self.fparams['interesting_rois']).nonzero()[0]
        else:
            sorted_roi_order = range(num_rois)
            interesting_rois = self.fparams['interesting_rois']

        if not all_nan_rois[0].size == 0:
            set_diff_keep_order = lambda main_list, remove_list: [i for i in main_list if i not in remove_list]
            sorted_roi_order = set_diff_keep_order(sorted_roi_order, all_nan_rois)
            interesting_rois = [i for i in self.fparams['interesting_rois'] if i not in all_nan_rois]

        self.data_dict = self.extract_trial_data(signals, self.tvec, self.trial_begEnd_samp, self.event_frames,
                                                 self.conditions, self.baseline_begEnd_samp)
        self.sorted_roi_order = sorted_roi_order
        self.interesting_rois = interesting_rois
        self.flag_stim = flag_stim

    def plot_trial_avg_heatmap(self, data_key, sort_roi_order=None, clims=None):
        """
        Plot the trial-averaged heatmap for the specified data key.

        Parameters:
            data_key (str): The key of the data to plot.
            sort_roi_order (list, optional): The custom order of ROIs for sorting the heatmap. Defaults to None.
            clims (tuple, optional): The color limits for the heatmap. Defaults to None.

        Returns:
            matplotlib.pyplot.figure: The figure object containing the heatmap.
        """
        if sort_roi_order is None:
            sort_roi_order = self.sorted_roi_order

        if clims is None:
            data_in = self.data_dict[data_key]['data']
            clims = self.generate_clims(data_in, self.fparams['norm_type'])

        sorted_data = self.data_dict[data_key]['data'][sort_roi_order, :]
        fig, ax = plt.subplots()
        im = ax.imshow(sorted_data, cmap='viridis', aspect='auto', clim=clims)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ROI')
        ax.set_title(f'{data_key} - Trial Averaged Heatmap')
        plt.colorbar(im, ax=ax)
        plt.show()
        return fig

    def generate_clims(self, data_in, norm_type='zero_center'):
        """
        Generate color limits for data visualization.

        Parameters:
            data_in (numpy.ndarray): Input data.
            norm_type (str, optional): The type of normalization. Can be 'zero_center' or 'abs_max'. Defaults to 'zero_center'.

        Returns:
            tuple: Color limits for the heatmap.
        """
        if norm_type == 'zero_center':
            max_abs = max(np.abs(np.nanmin(data_in)), np.abs(np.nanmax(data_in)))
            clims = (-max_abs, max_abs)
        elif norm_type == 'abs_max':
            clims = (np.nanmin(data_in), np.nanmax(data_in))
        else:
            raise ValueError("Invalid norm_type. Should be either 'zero_center' or 'abs_max'.")
        return clims

    def plot_roi_trial_avg_trace(self, roi_idx):
        """
        Plot the trial-averaged trace for a specific ROI.

        Parameters:
            roi_idx (int): The index of the ROI.

        Returns:
            matplotlib.pyplot.figure: The figure object containing the trace plot.
        """
        fig, ax = plt.subplots()
        for condition in self.conditions:
            trace = np.nanmean(self.data_dict[condition]['data'][roi_idx, :], axis=0)
            ax.plot(self.tvec, trace, label=condition)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Average Signal')
        ax.set_title(f'Trial Averaged Trace - ROI {roi_idx}')
        ax.legend()
        plt.show()
        return fig

    def plot_roi_trial_time_avg_bar(self, roi_idx, sort_method='peak_time', time_window=None):
        """
        Plot the time-averaged bar plot for a specific ROI.

        Parameters:
            roi_idx (int): The index of the ROI.
            sort_method (str, optional): The method used to sort the ROIs. Can be 'peak_time' or 'max_value'. Defaults to 'peak_time'.
            time_window (tuple, optional): The time window (in seconds) for computing the average value. Defaults to None.

        Returns:
            matplotlib.pyplot.figure: The figure object containing the bar plot.
        """
        if time_window is None:
            time_window = (0, self.trial_start_end_sec[-1])

        sort_epoch_start_time, sort_epoch_end_time = time_window
        sort_epoch_start_samp = self.find_nearest_idx(self.tvec, sort_epoch_start_time)[0]
        sort_epoch_end_samp = self.find_nearest_idx(self.tvec, sort_epoch_end_time)[0]

        roi_data = self.data_dict[self.conditions[0]]['data'][roi_idx, sort_epoch_start_samp:sort_epoch_end_samp]

        if sort_method == 'peak_time':
            epoch_peak_samp = np.argmax(roi_data)
            sorted_roi_order = np.argsort(epoch_peak_samp)
        elif sort_method == 'max_value':
            time_max = np.nanmax(roi_data)
            sorted_roi_order = np.argsort(time_max)[::-1]
        else:
            raise ValueError("Invalid sort_method. Should be either 'peak_time' or 'max_value'.")

        fig, ax = plt.subplots()
        x_pos = np.arange(len(self.conditions))
        for idx in sorted_roi_order:
            avg_values = [np.nanmean(self.data_dict[condition]['data'][idx, sort_epoch_start_samp:sort_epoch_end_samp]) for condition in self.conditions]
            ax.bar(x_pos, avg_values, label=f'ROI {idx}')
            x_pos += 0.15
        ax.set_xticks(np.arange(len(self.conditions)) + 0.15 * (len(sorted_roi_order) / 2))
        ax.set_xticklabels(self.conditions)
        ax.set_xlabel('Conditions')
        ax.set_ylabel('Average Signal')
        ax.set_title(f'Time Averaged Bar Plot - ROI {roi_idx}')
        ax.legend()
        plt.show()
        return fig

    def find_nearest_idx(self, array, value):
        """
        Find the nearest index in an array for a given value.

        Parameters:
            array (numpy.ndarray): The input array.
            value (float): The value to find the nearest index for.

        Returns:
            int: The index of the nearest value in the array.
        """
        return np.abs(array - value).argmin()

    def define_params(self, method='single'):
        """
        Define additional parameters based on the chosen method.

        Parameters:
            method (str, optional): The method used for defining additional parameters. Defaults to 'single'.

        Returns:
            dict: A dictionary containing additional parameters.
        """
        # Implementation of the define_params function
        # Depending on the method chosen, return appropriate parameters.
        if method == 'single':
            params = {'param1': value1, 'param2': value2}
        elif method == 'double':
            params = {'param3': value3, 'param4': value4}
        else:
            raise ValueError("Invalid method. Should be either 'single' or 'double'.")
        return params