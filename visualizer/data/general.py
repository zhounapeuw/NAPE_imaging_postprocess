import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from visualizer import misc
import os 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, KMeans

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

class EventClusterProcessor(BaseGeneralProcesser):
    def __init__(self, signals_content, events_content, fs, trial_start_end, baseline_end, event_sort_analysis_win, pca_num_pc_method, max_n_clusters, possible_n_nearest_neighbors, selected_conditions, flag_plot_reward_line, second_event_seconds, flag_save_figs, heatmap_cmap_scaling, group_data, group_data_conditions, sortwindow):
        super().__init__(signals_content, events_content)

        self.fs = fs 
        self.trial_start_end = trial_start_end
        self.baseline_end = baseline_end
        self.event_sort_analysis_win = event_sort_analysis_win
        self.pca_num_pc_method = pca_num_pc_method
        self.max_n_clusters = max_n_clusters
        self.possible_n_nearest_neighbors = possible_n_nearest_neighbors
        self.selected_conditions = selected_conditions
        self.flag_plot_reward_line = flag_plot_reward_line
        self.second_event_seconds = second_event_seconds
        self.flag_save_figs = flag_save_figs
        self.heatmap_cmap_scaling = heatmap_cmap_scaling
        self.group_data = group_data
        self.group_data_conditions = group_data_conditions
        self.sortwindow = sortwindow

    def declare_variables(self):
        super().load_signal_data()

        self.trial_start_end_sec = np.array(self.trial_start_end) # trial windowing in seconds relative to ttl-onset/trial-onset, in seconds
        self.baseline_start_end_sec = np.array([self.trial_start_end_sec[0], self.baseline_end])
        self.baseline_begEnd_samp = self.baseline_start_end_sec*self.fs
        self.baseline_svec = (np.arange(self.baseline_begEnd_samp[0], self.baseline_begEnd_samp[1] + 1, 1) - self.baseline_begEnd_samp[0]).astype('int')

        if self.group_data:
            self.conditions = self.group_data_conditions

            if self.selected_conditions:
                self.conditions = self.selected_conditions

            self.num_conditions = len(self.conditions)

            self.populationdata = np.squeeze(np.apply_along_axis(misc.zscore_, -1, self.signals, self.baseline_svec))

            self.num_samples_trial = int(self.populationdata.shape[-1]/len(self.group_data_conditions))
            self.tvec = np.round(np.linspace(self.trial_start_end_sec[0], self.trial_start_end_sec[1], self.num_samples_trial), 2)
        else:
            super().load_behav_data()

            # identify conditions to analyze
            self.all_conditions = self.event_frames.keys()
            self.conditions = [ condition for condition in self.all_conditions if len(self.event_frames[condition]) > 0 ] # keep conditions that have events

            self.conditions.sort()
            if self.selected_conditions:
                self.conditions = self.selected_conditions

            self.num_conditions = len(self.conditions)

            ### define trial timing

            # convert times to samples and get sample vector for the trial 
            self.trial_begEnd_samp = self.trial_start_end_sec*self.fs # turn trial start/end times to samples
            self.trial_svec = np.arange(self.trial_begEnd_samp[0], self.trial_begEnd_samp[1])
            # calculate time vector for plot x axes
            self.num_samples_trial = len( self.trial_svec )
            self.tvec = np.round(np.linspace(self.trial_start_end_sec[0], self.trial_start_end_sec[1], self.num_samples_trial+1), 2)


            """
            MAIN data processing function to extract event-centered data

            extract and save trial data, 
            saved data are in the event_rel_analysis subfolder, a pickle file that contains the extracted trial data
            """
            self.data_dict = misc.extract_trial_data(self.signals, self.tvec, self.trial_begEnd_samp, self.event_frames, self.conditions, baseline_start_end_samp = self.baseline_begEnd_samp)


            #### concatenate data across trial conditions

            # concatenates data across trials in the time axis; populationdata dimentionss are ROI by time (trials are appended)
            self.populationdata = np.concatenate([self.data_dict[condition]['ztrial_avg_data'] for condition in self.conditions], axis=1)
            
            # remove rows with nan values
            self.nan_rows = np.unique(np.where(np.isnan(self.populationdata))[0])
            if self.nan_rows.size != 0:
                self.populationdata = np.delete(self.populationdata, obj=self.nan_rows, axis=0)
                print('Some ROIs contain nan in tseries!')

        self.cmax = np.nanmax(np.abs([np.nanmin(self.populationdata), np.nanmax(self.populationdata)])) # Maximum colormap value. 

        # calculated variables
        self.window_size = int(self.populationdata.shape[1]/self.num_conditions) # Total number of frames in a trial window; needed to split processed concatenated data
        self.sortwindow_frames = [int(np.round(time*self.fs)) for time in self.event_sort_analysis_win] # Sort responses between first lick and 10 seconds.
        self.sortresponse = np.argsort(np.mean(self.populationdata[:,self.sortwindow_frames[0]:self.sortwindow_frames[1]], axis=1))[::-1]
        # sortresponse corresponds to an ordering of the neurons based on their average response in the sortwindow

        self.tvec_convert_dict = {}
        for i in range(len(self.tvec)):
            self.tvec_convert_dict[i] = self.tvec[i] 
    
    def num_pc_explained_var(self, explained_var, explained_var_thresh=90):
        """
        Select pcs for those that capture more than threshold amount of variability in the data
        """
        cum_sum = 0
        for idx, PC_var in enumerate(explained_var):
            cum_sum += PC_var
            if cum_sum > explained_var_thresh:
                return idx+1
    
    def calculate_pca(self):
        load_savedpca_or_dopca = 'dopca'

        # perform PCA across time
        if load_savedpca_or_dopca == 'dopca':
            self.pca = PCA(n_components=min(self.populationdata.shape[0],self.populationdata.shape[1]), whiten=True)
            self.pca.fit(self.populationdata)

        # pca across time
        self.transformed_data = self.pca.transform(self.populationdata)
        # transformed data: each ROI is now a linear combination of the original time-serie

        # grab eigenvectors (pca.components_); linear combination of original axes
        self.pca_vectors = self.pca.components_ 
        print(f'Number of PCs = {self.pca_vectors.shape[0]}')

        # Number of PCs to be kept is defined as the number at which the 
        # scree plot bends. This is done by simply bending the scree plot
        # around the line joining (1, variance explained by first PC) and
        # (num of PCs, variance explained by the last PC) and finding the 
        # number of components just below the minimum of this rotated plot
        self.x = 100*self.pca.explained_variance_ratio_ # eigenvalue ratios
        self.xprime = self.x - (self.x[0] + (self.x[-1]-self.x[0])/(self.x.size-1)*np.arange(self.x.size))

        # define number of PCs
        num_retained_pcs_scree = np.argmin(self.xprime)
        num_retained_pcs_var = self.num_pc_explained_var(self.x, 90)
        if self.pca_num_pc_method == 0:
            self.num_retained_pcs = num_retained_pcs_scree
        elif self.pca_num_pc_method == 1:
            self.num_retained_pcs = num_retained_pcs_var
    
    def calculate_optimum_clusters(self):
        # calculate optimal number of clusters and nearest neighbors using silhouette scores
        self.min_clusters = np.min([self.max_n_clusters+1, int(self.populationdata.shape[0])])
        possible_n_clusters = np.arange(2, self.max_n_clusters+1) #This requires a minimum of 2 clusters.
        # When the data contain no clusters at all, it will be quite visible when inspecting the two obtained clusters, 
        # as the responses of the clusters will be quite similar. This will also be visible when plotting the data in
        # the reduced dimensionality PC space (done below).

        possible_clustering_models = np.array(["Spectral", "Kmeans"])
        silhouette_scores = np.nan*np.ones((possible_n_clusters.size,
                                            self.possible_n_nearest_neighbors.size,
                                            possible_clustering_models.size))

        # loop through iterations of clustering params
        for n_clustersidx, n_clusters in enumerate(possible_n_clusters):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0) #tol=toler_options
            for nnidx, nn in enumerate(self.possible_n_nearest_neighbors):
                spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=nn, random_state=0)
                models = [spectral,kmeans]
                for modelidx,model in enumerate(models):
                    model.fit(self.transformed_data[:,:self.num_retained_pcs])
                    silhouette_scores[n_clustersidx, nnidx, modelidx] = silhouette_score(self.transformed_data[:,:self.num_retained_pcs],
                                                                            model.labels_,
                                                                            metric='cosine')
                    if modelidx == 0:
                        print(f'Done with numclusters = {n_clusters}, num nearest neighbors = {nn}: score = {silhouette_scores[n_clustersidx, nnidx, modelidx]}.3f')
                    else:
                        print(f'Done with numclusters = {n_clusters}, score = {silhouette_scores[n_clustersidx, nnidx, modelidx]}.3f')
        print(silhouette_scores.shape)
        print('Done with model fitting')

        self.silhouette_dict = {}
        self.silhouette_dict['possible_clustering_models'] = possible_clustering_models
        self.silhouette_dict['num_retained_pcs'] = self.num_retained_pcs
        self.silhouette_dict['possible_n_clusters'] = possible_n_clusters
        self.silhouette_dict['possible_n_nearest_neighbors'] = self.possible_n_nearest_neighbors
        self.silhouette_dict['silhouette_scores'] = silhouette_scores
        self.silhouette_dict['shape'] = 'cluster_nn'
    
    def reorder_clusters(self, data, sort_win_frames, rawlabels):
        uniquelabels = list(set(rawlabels))
        responses = np.nan*np.ones((len(uniquelabels),))
        for l, label in enumerate(uniquelabels):
            responses[l] = np.mean(data[rawlabels==label, sort_win_frames[0]:sort_win_frames[1]])
        temp = np.argsort(responses).astype(int)[::-1]
        temp = np.array([np.where(temp==a)[0][0] for a in uniquelabels])
        outputlabels = np.array([temp[a] for a in list(np.digitize(rawlabels, uniquelabels)-1)])
        return outputlabels
    
    def cluster_with_optimal_params(self):
        # Identify optimal parameters from the above parameter space
        temp = np.where(self.silhouette_dict['silhouette_scores']==np.nanmax(self.silhouette_dict['silhouette_scores']))

        n_clusters = self.silhouette_dict['possible_n_clusters'][temp[0][0]]
        n_nearest_neighbors = self.silhouette_dict['possible_n_nearest_neighbors'][temp[1][0]]
        num_retained_pcs = self.silhouette_dict['num_retained_pcs']
        method = self.silhouette_dict['possible_clustering_models'][temp[2][0]]
        print(n_clusters, n_nearest_neighbors, num_retained_pcs, method)

        # Redo clustering with these optimal parameters
        model = None
        if method == 'Spectral':
            model = SpectralClustering(n_clusters=n_clusters,
                                affinity='nearest_neighbors',
                                n_neighbors=n_nearest_neighbors,
                                random_state=0)
        else:
            model = KMeans(n_clusters=n_clusters, random_state=0)


        # model = AgglomerativeClustering(n_clusters=9,
        #                                 affinity='l1',
        #                                 linkage='average')

        model.fit(self.transformed_data[:,:num_retained_pcs])

        temp = silhouette_score(self.transformed_data[:,:num_retained_pcs], model.labels_, metric='cosine')

        print(f'Number of clusters = {len(set(model.labels_))}, average silhouette = {temp}.3f')

        self.newlabels = self.reorder_clusters(self.populationdata, self.sortwindow_frames, model.labels_)

        # Create a new variable containing all unique cluster labels
        self.uniquelabels = list(set(self.newlabels))
        self.numroisincluster = np.nan*np.ones((len(self.uniquelabels),))

        self.colors_for_cluster = plt.cm.viridis(np.linspace(0,1,len(self.uniquelabels)+3))
    
    def generate_all_data(self):
        self.declare_variables()
        self.calculate_pca()
        self.calculate_optimum_clusters()
        self.cluster_with_optimal_params()