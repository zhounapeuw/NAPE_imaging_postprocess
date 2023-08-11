import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from visualizer import misc
import random
import os 
import pandas as pd

class BaseGeneralProcesser:
    def __init__(self, signals_content, events_content):
        self.signals_content = signals_content
        self.events_content = events_content
    
    def load_signal_data(self):
        self.signals = misc.load_signals(self.signals_content)
        
        # if opto stim frames were detected in preprocessing, set these frames to be NaN (b/c of stim artifact)
        if self.fparams['opto_blank_frame']:
            try:
                self.glob_stim_files = glob.glob(os.path.join(self.fparams['fdir'], "{}*_stimmed_frames.pkl".format(self.fparams['fname'])))
                self.stim_frames = pickle.load( open( self.glob_stim_files[0], "rb" ) )
                self.signals[:,self.stim_frames['samples']] = None # blank out stimmed frames
                self.flag_stim = True
                print('Detected stim data; replaced stim samples with NaNs')
            except:
                self.flag_stim = False
                print('Note: No stim preprocessed meta data detected.')

    def load_behav_data(self):
        if self.events_content:
            self.event_times = misc.df_to_dict(self.events_content)
            self.event_frames = misc.dict_time_to_samples(self.event_times, self.fparams['fs'])

            self.event_times = {}
            if self.fparams['selected_conditions']:
                self.conditions = self.fparams['selected_conditions'] 
            else:
                self.conditions = self.event_frames.keys()
            for cond in self.conditions: # convert event samples to time in seconds
                self.event_times[cond] = (np.array(self.event_frames[cond])/self.fparams['fs']).astype('int')
    
    def generate_all_data(self):
        self.load_signal_data()
        self.load_behav_data()

class WholeSessionProcessor(BaseGeneralProcesser):
    def __init__(self, fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization, signals_content, events_content, estimmed_frames=None, cond_colors=['steelblue', 'crimson', 'orchid', 'gold']):
        super().__init__(signals_content, events_content)

        self.signal_to_plot = None
        self.min_max = None
        self.min_max_all = None
        self.tvec = None
        self.event_times = None
        self.conditions = None
        self.cond_colors = cond_colors
        self.estimmed_frames = estimmed_frames

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

    def load_signal_data(self):
        super().load_signal_data()

        if self.fparams['flag_normalization'] == 'dff':
            signal_to_plot = np.apply_along_axis(misc.calc_dff, 1, self.signals)
        elif self.fparams['flag_normalization'] == 'dff_perc':
            signal_to_plot = np.apply_along_axis(self.calc_dff_percentile, 1, self.signals)
        elif self.fparams['flag_normalization'] == 'zscore':
            signal_to_plot = np.apply_along_axis(self.calc_zscore, 1, self.signals, np.arange(0, self.signals.shape[1]))
        else:
            signal_to_plot = self.signals

        self.signal_to_plot = signal_to_plot

        min_max = [list(min_max_tup) for min_max_tup in zip(np.min(signal_to_plot, axis=1), np.max(signal_to_plot, axis=1))]
        self.min_max = min_max
        self.min_max_all = [np.min(signal_to_plot), np.max(signal_to_plot)]

        if self.fparams['num_rois'] == 'all':
            self.fparams['num_rois'] = self.signals.shape[0]

        total_session_time = self.signals.shape[1] / self.fparams['fs']
        tvec = np.round(np.linspace(0, total_session_time, self.signals.shape[1]), 2)
        self.tvec = tvec
    
    def generate_all_data(self):
        super().generate_all_data()

class EventRelAnalysisProcessor(BaseGeneralProcesser):
    def __init__(self, fparams, signals_content, events_content):
        super().__init__(signals_content, events_content)

        self.fparams = fparams
    
    def subplot_loc(self, idx, num_rows, num_col):
        if num_rows == 1:
            subplot_index = idx
        else:
            subplot_index = np.unravel_index(idx, (num_rows, int(num_col))) # turn int index to a tuple of array coordinates
        return subplot_index
   
    def generate_reference_samples(self):
        ### create variables that reference samples and times for slicing and plotting the data

        self.trial_start_end_sec = np.array(self.fparams['trial_start_end']) # trial windowing in seconds relative to ttl-onset/trial-onset, in seconds
        self.baseline_start_end_sec = np.array([self.trial_start_end_sec[0], self.fparams['baseline_end']])

        # convert times to samples and get sample vector for the trial 
        self.trial_begEnd_samp = self.trial_start_end_sec*self.fparams['fs'] # turn trial start/end times to samples
        self.trial_svec = np.arange(self.trial_begEnd_samp[0], self.trial_begEnd_samp[1])
        # and for baseline period
        self.baseline_begEnd_samp = self.baseline_start_end_sec*self.fparams['fs']
        self.baseline_svec = (np.arange(self.baseline_begEnd_samp[0], self.baseline_begEnd_samp[1]+1, 1) - self.baseline_begEnd_samp[0]).astype('int')

        # calculate time vector for plot x axes
        self.num_samples_trial = len( self.trial_svec )
        self.tvec = np.round(np.linspace(self.trial_start_end_sec[0], self.trial_start_end_sec[1], self.num_samples_trial+1), 2)

        # find samples and calculations for time 0 for plotting
        self.t0_sample = misc.get_tvec_sample(self.tvec, 0) # grabs the sample index of a given time from a vector of times
        self.event_end_sample = int(np.round(self.t0_sample+self.fparams['event_dur']*self.fparams['fs']))
        self.event_bound_ratio = [(self.t0_sample)/self.num_samples_trial , self.event_end_sample/self.num_samples_trial] # fraction of total samples for event start and end; only used for plotting line indicating event duration
    
    def load_signal_data(self):
        super().load_signal_data()

        self.num_rois = self.signals.shape[0]
        self.all_nan_rois = np.where(np.apply_along_axis(misc.is_all_nans, 1, self.signals))
    
    def get_num_rois(self):
        if self.num_rois:
            return self.num_rois
        return "Not defined yet: run load_signal_data() first"
    
    def trial_preprocessing(self):
        self.data_dict = misc.extract_trial_data(self.signals, self.tvec, self.trial_begEnd_samp, self.event_frames, self.conditions, baseline_start_end_samp = self.baseline_begEnd_samp)
    
    def generate_all_data(self):
        self.generate_reference_samples()
        super().generate_all_data()
        self.trial_preprocessing()