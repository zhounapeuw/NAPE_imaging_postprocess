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

def is_all_nans(vector):
    """
    checks if series or vector contains all nans; returns boolean. Used to identify and exclude all-nan rois
    """
    if isinstance(vector, pd.Series):
        vector = vector.values
    return np.isnan(vector).all()

class EventAnalysisProcessor:
    def __init__(self, fparams, signals_fpath, events_file_path):
        self.fparams = fparams
        self.signals_fpath = signals_fpath
        self.events_file_path = events_file_path
    
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
    
    def grab_data(self):
        self.signals = misc.load_signals(self.signals_fpath)
        
        self.num_rois = self.signals.shape[0]
        self.all_nan_rois = np.where(np.apply_along_axis(is_all_nans, 1, self.signals)) # find rois with activity as all nans
        
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
    
    def load_behavioral_data(self):
        ### load behavioral data and trial info

        self.glob_event_files = glob.glob(self.events_file_path) # look for a file in specified directory
        if not self.glob_event_files:
            print(f'{self.events_file_path} not detected. Please check if path is correct.')
        if 'csv' in self.glob_event_files[0]:
            self.event_times = misc.df_to_dict(self.glob_event_files[0])
        elif any(x in self.glob_event_files[0] for x in ['pkl', 'pickle']):
            self.event_times = pickle.load( open( self.glob_event_files[0], "rb" ), fix_imports=True, encoding='latin1' ) # latin1 b/c original pickle made in python 2
        self.event_frames = misc.dict_time_to_samples(self.event_times, self.fparams['fs'])

        # identify conditions to analyze
        self.all_conditions = self.event_frames.keys()
        self.conditions = [ condition for condition in self.all_conditions if len(self.event_frames[condition]) > 0 ] # keep conditions that have events

        self.conditions.sort()
        if self.fparams['selected_conditions']:
            self.conditions = self.fparams['selected_conditions']

    
    def trial_preprocessing(self):
        """
        MAIN data processing function to extract event-centered data

        extract and save trial data, 
        saved data are in the event_rel_analysis subfolder, a pickle file that contains the extracted trial data
        """
        self.data_dict = misc.extract_trial_data(self.signals, self.tvec, self.trial_begEnd_samp, self.event_frames, self.conditions, baseline_start_end_samp = self.baseline_begEnd_samp)
    
    def generate_all_data(self):
        self.generate_reference_samples()
        self.grab_data()
        self.load_behavioral_data()
        self.trial_preprocessing()