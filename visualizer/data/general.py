import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
from visualizer import misc
import os 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering, KMeans
import matplotlib

class BaseGeneralProcesser:
    def __init__(self, signals_content, events_content, file_extension):
        self.signals_content = signals_content
        self.events_content = events_content
        self.file_extension = file_extension
    
    def generate_reference_samples(self):
        ### create variables that reference samples and times for slicing and plotting the data

        self.trial_start_end_sec = np.array(self.fparams['trial_start_end']) # trial windowing in seconds relative to ttl-onset/trial-onset, in seconds
        self.baseline_start_end_sec = np.array([self.trial_start_end_sec[0], self.fparams['baseline_end']])

        #baseline period
        self.baseline_begEnd_samp = self.baseline_start_end_sec*self.fparams['fs']
        self.baseline_svec = (np.arange(self.baseline_begEnd_samp[0], self.baseline_begEnd_samp[1]+1, 1) - self.baseline_begEnd_samp[0]).astype('int')

        # convert times to samples and get sample vector for the trial 
        self.trial_begEnd_samp = self.trial_start_end_sec*self.fparams['fs'] # turn trial start/end times to samples
        self.trial_svec = np.arange(self.trial_begEnd_samp[0], self.trial_begEnd_samp[1])

    def calculate_trial_timing(self):
        # calculate time vector for plot x axes
        self.num_samples_trial = len( self.trial_svec )
        self.tvec = np.round(np.linspace(self.trial_start_end_sec[0], self.trial_start_end_sec[1], self.num_samples_trial+1), 2)
    
    def load_signal_data(self):
        self.signals = misc.load_signals(self.signals_content, self.file_extension[0])
        
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
            self.event_times = misc.df_to_dict(self.events_content, self.file_extension[1])
            self.event_frames = misc.dict_time_to_samples(self.event_times, self.fparams['fs'])

            self.event_times = {}
            if self.fparams['selected_conditions']:
                self.conditions = self.fparams['selected_conditions'] 
            else:
                self.conditions = self.event_frames.keys()
            for cond in self.conditions: # convert event samples to time in seconds
                self.event_times[cond] = (np.array(self.event_frames[cond])/self.fparams['fs']).astype('int')
    
    def trial_preprocessing(self):
        """
        MAIN data processing function to extract event-centered data

        extract and save trial data, 
        saved data are in the event_rel_analysis subfolder, a pickle file that contains the extracted trial data
        """
        
        self.data_dict = misc.extract_trial_data(self.signals, self.tvec, self.trial_begEnd_samp, self.event_frames, self.conditions, baseline_start_end_samp = self.baseline_begEnd_samp)
    
    def generate_all_data(self):
        self.load_signal_data()
        self.load_behav_data()

class WholeSessionProcessor(BaseGeneralProcesser):
    def __init__(self, fs, opto_blank_frame, num_rois, selected_conditions, flag_normalization, signals_content, events_content, estimmed_frames=None, cond_colors=['steelblue', 'crimson', 'orchid', 'gold'], file_extension=[".csv", ".csv"]):
        super().__init__(signals_content, events_content, file_extension)

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
    def __init__(self, fparams, signals_content, events_content, file_extension=[".csv", ".csv"]):
        super().__init__(signals_content, events_content, file_extension)
        self.fparams = fparams
    
    def subplot_loc(self, idx, num_rows, num_col):
        if num_rows == 1:
            subplot_index = idx
        else:
            subplot_index = np.unravel_index(idx, (num_rows, int(num_col))) # turn int index to a tuple of array coordinates
        return subplot_index
   
    def generate_reference_samples(self):
        super().generate_reference_samples()

        self.trial_start_end_sec = np.array(self.fparams['trial_start_end']) # trial windowing in seconds relative to ttl-onset/trial-onset, in seconds
        self.baseline_start_end_sec = np.array([self.trial_start_end_sec[0], self.fparams['baseline_end']])

        super().calculate_trial_timing()

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
    
    def generate_all_data(self):
        self.generate_reference_samples()
        super().generate_all_data()
        super().trial_preprocessing()

class EventClusterProcessor(BaseGeneralProcesser):
    def __init__(self, signals_content, events_content, fs, trial_start_end, baseline_end, event_sort_analysis_win, pca_num_pc_method, max_n_clusters, possible_n_nearest_neighbors, selected_conditions, flag_plot_reward_line, second_event_seconds, heatmap_cmap_scaling, group_data, group_data_conditions, sortwindow, file_extension=[".csv", ".csv"]):
        super().__init__(signals_content, events_content, file_extension)

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
        self.heatmap_cmap_scaling = heatmap_cmap_scaling
        self.group_data = group_data
        self.group_data_conditions = group_data_conditions
        self.sortwindow = sortwindow

        self.define_params()
        self.declare_variables()
    
    def define_params(self):
        self.fparams = {
            "fs": self.fs,
            "opto_blank_frame": False,
            "selected_conditions": self.selected_conditions,
            "trial_start_end": self.trial_start_end,
            "baseline_end": self.baseline_end
        }
        self.fparams['fs'] = self.fs
        self.fparams['opto_blank_frame'] = False
        self.fparams['selected_conditions'] = self.selected_conditions

    def declare_variables(self):
        super().load_signal_data()
        super().generate_reference_samples()

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
            super().calculate_trial_timing()
            super().trial_preprocessing()

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
                    #if modelidx == 0:
                    #    print(f'Done with numclusters = {n_clusters}, num nearest neighbors = {nn}: score = {silhouette_scores[n_clustersidx, nnidx, modelidx]}.3f')
                    #else:
                    #    print(f'Done with numclusters = {n_clusters}, score = {silhouette_scores[n_clustersidx, nnidx, modelidx]}.3f')
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
        self.calculate_pca()
        self.calculate_optimum_clusters()
        self.cluster_with_optimal_params()

class PlotActivityContoursProcesser(BaseGeneralProcesser):
    def __init__(self, signals_content, events_content, simah5_content, sima_mask_content, raw_npilCorr, fs, rois_to_include, analysis_win, activity_name, trial_start_end, baseline_end, selected_conditions, opto_blank_frame, file_extension=[".csv", ".csv"]):
        super().__init__(signals_content, events_content, file_extension)
        self.simah5_content = simah5_content
        self.sima_mask_content = sima_mask_content

        self.analysis_win = analysis_win
        self.activity_name = activity_name
        self.rois_to_include = rois_to_include
        self.raw_npilCorr = raw_npilCorr
        self.fparams = self.define_params(fs, trial_start_end, baseline_end, selected_conditions, opto_blank_frame)
    
    def define_params(self, fs, trial_start_end, baseline_end, selected_conditions, opto_blank_frame):
        fparams = {}

        fparams['fs'] = fs
        fparams['trial_start_end'] = trial_start_end
        fparams['baseline_end'] = baseline_end
        fparams['selected_conditions'] = selected_conditions
        fparams['opto_blank_frame'] = opto_blank_frame

        return fparams
    
    def generate_std_img(self):
        # Using multiprocessing to speed up the processing
        def calculate_std_chunk(chunk):
            return np.std(chunk, axis=0)

        # Split data into chunks for parallel processing
        chunk_size = len(self.sima_data) // 1000  # Determine an appropriate chunk size
        chunks = [self.sima_data[i:i+chunk_size] for i in range(0, len(self.sima_data), chunk_size)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            std_chunk_results = executor.map(calculate_std_chunk, chunks)

        # Combine the results
        self.std_img = np.std(np.array(list(std_chunk_results)), axis=0)

    def identify_conditions(self):
        if self.fparams['selected_conditions']:
            self.conditions = self.fparams['selected_conditions']
        else:
            # identify conditions to analyze
            all_conditions = self.event_frames.keys()
            self.conditions = [ condition for condition in all_conditions if len(self.event_frames[condition]) > 0 ] # keep conditions that have events

            self.conditions.sort()
            
    def load_sima_data(self):
        h5 = h5py.File(self.simah5_content, 'r')
        sima_data = np.squeeze(np.array(h5[list(h5.keys())[0]])).astype('int16')
        h5.close()
        return sima_data
    
    def load_sima_masks(self):
        sima_masks = np.load(self.sima_mask_content)
        return sima_masks
    
    def calculate_std_chunk(self, chunk):
        return np.std(chunk, axis=0)
    
    def generate_binary_array_roi_pixels(self):
        # make binary array of roi pixels for contour plotting
        self.zero_template_manual = np.zeros([self.manual_data_dims[1], self.manual_data_dims[2]])
        self.roi_label_loc_manual = []
        self.roi_signal_sima = np.empty([self.numROI_sima, self.sima_data.shape[0]])

        for iROI in self.rois_to_include:
            # make binary map of ROI pixels
            ypix_roi, xpix_roi = np.where(self.sima_masks[iROI,:,:] == 1)
            if ypix_roi.size == 0:
                self.roi_label_loc_manual.append( [0, 0] )
            else:
                self.zero_template_manual[ ypix_roi, xpix_roi ] = 1*(iROI+2)
                self.roi_label_loc_manual.append( [np.min(ypix_roi), np.min(xpix_roi)] )
                if self.raw_npilCorr == 0:
                    # not npil corr signal
                    self.roi_signal_sima[iROI,:] = np.mean(self.sima_data[:, ypix_roi, xpix_roi  ], axis = 1)
    
    def get_tvec_sample(self, sample_tvec, time):
        sample_index = np.argmin(np.abs(sample_tvec - time))
        return sample_index

    def generate_reference_samples(self):
        super().generate_reference_samples()
        self.num_samps = self.roi_signal_sima.shape[-1]
        self.total_time = self.num_samps/self.fparams['fs']
        self.tvec = np.linspace(0,self.total_time,self.num_samps)
    
    def load_sima_variables(self):
        self.sima_data = self.load_sima_data()
        self.manual_data_dims = self.sima_data.shape
        self.sima_masks = self.load_sima_masks()

        self.numROI_sima = self.sima_masks.shape[0]
        if not self.rois_to_include:
            self.rois_to_include = np.arange(self.numROI_sima)
        self.num_rois_to_include = len(self.rois_to_include)
    
    def generate_all_data(self):
        super().generate_all_data()
        self.load_sima_variables()
        self.generate_std_img()
        self.generate_binary_array_roi_pixels()
        self.generate_reference_samples()
        super().trial_preprocessing()
        self.identify_conditions()

class PlotActivityContoursPlot:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.generate_colors()
    
    def generate_colors(self):
        # # sample the colormaps that you want to use. Use 128 from each so we get 256
        # # colors in total
        colors1 = plt.cm.Greens(np.linspace(1, 0, 128))
        colors2 = plt.cm.Purples(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        self.custom_cmap = mcolors.ListedColormap(colors)

        # Creating lists of rgb values
        self.gpcolors = []
        colors_list = []

        # add the rgb values of a color scheme into the list
        for j in range(self.custom_cmap.N):
            rgba = self.custom_cmap(j)
            colors_list.append(matplotlib.colors.rgb2hex(rgba))

        self.gpcolors.append(colors_list)
    
    def generate_contour_vectors(self, data_dict, analysis_win, name):
        sample_tvec = np.linspace(self.data_processor.fparams['trial_start_end'][0], self.data_processor.fparams['trial_start_end'][1], self.data_processor.data_dict[name]['num_samples'])

        if analysis_win[-1] == None:
            analysis_win[-1] = sample_tvec[-1]
        
        # getting the data corresponding to the activity name from the dictionary
        # CZ tvec needs to be an argument in plot_contour, also correct hardcoding of time selection
        contour_vector = np.mean(data_dict[name]
                                        ['zdata']
                                        [:,:,
                                        misc.get_tvec_sample(sample_tvec, analysis_win[0]):misc.get_tvec_sample(sample_tvec, analysis_win[-1])],
                                        axis=(0,2)) 
        
        # making a copy of the original data vector for coloring purposes
        orig_vector = np.copy(contour_vector)
        
        # normalize the values in the vector so that the colors can be properly indexed
        max = 225 / np.max(contour_vector)
        for i in self.data_processor.rois_to_include:
            contour_vector[i] = abs(contour_vector[i])
            contour_vector[i] *= max
            if (contour_vector[i] >= 255):
                contour_vector[i] = 255
        
        return contour_vector, orig_vector

    def generate_norm_map(self, vector):
        # Get the min and max of the data ignoring outliers
        percentile_min = np.percentile(vector, 7)
        percentile_max = np.percentile(vector, 93)

        # Generate the data normalization structure
        abs_max = max(np.abs(percentile_min), np.abs(percentile_max))
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-abs_max, vmax=abs_max)

        # Creating a ScalarMappable for the colorbar
        sm = ScalarMappable(cmap=self.custom_cmap, norm=norm)
        sm.set_array([])  # Set dummy array for colorbar

        return norm, sm

    def normalize_vector(self, vector):
        """
        A function that accepts a data vector as parameter and normalize
        its values in order to make them meaningful indices for coloring purposes
        """
        for i in self.data_processor.rois_to_include:
            vector[i] = abs(vector[i])
            vector[i] *= 80
            if (vector[i] >= 255):
                vector[i] = 255
    
    def plot_bars(self, axs, roi_idx, event_names, data, error_events, bar_width, x_positions):
        """
        A function that takes a matplotlib axes object, an axes index, an roi index, event names,
        a pandas dataframe, and error events as parameters. It plots each cell's activity values
        corresponding to each event as a barplot and returns the bar container objects.
        """
        barlist = []

        for j, event_name in enumerate(event_names):
            event_data = data[event_name][roi_idx]
            bars = axs.bar(x_positions + j * bar_width, event_data, bar_width, label=event_name)
            barlist.append(bars)
            
        return barlist

    def add_colors(self, bars, vector):
        """
        A function that takes in a matplotlib axes object, tuples of matplotlib containers,
        color indices to the color schemes, event names, data vectors and a boolean
        that indicates whether or not we are handling the first half of the data as parameters.
        It adds colors to each bar in the barplot based on the cell's activity value corresponds to
        the specified events using the color schemes defined above
        """
        
        for i in range(len(bars[0])):
            for j in range(len(self.data_processor.conditions)):
                bars[j][i].set_color(self.gpcolors[0][int(vector[j][i])])
    
    def add_text(self, axs, roi_idx, event_names, x_positions):
        """
        A function that takes a matplotlib axes object, an integer index for color schemes,
        event names and a boolean that indicates whether or not we are handling the
        first half of the data as parameters. It adds texts onto the barplot that indicates
        each cell's activity value;
        """

        for j, event_name in enumerate(event_names):
            for i, v in zip(x_positions, self.data[event_name][roi_idx]):
                if v < 0:
                    v *= -1
                axs.text(i + self.bar_width * j, v + 0.02, str(round(v, 2)),
                    color='black', fontweight='bold', fontsize=12,
                    ha='center', va='bottom')  # Adjust text position here
    
    def split_given_size(self, data, size):
        return np.split(data, np.arange(size,len(data),size))

    def generate_contour_roi_plot(self):
        fig, ax = plt.subplots(1, 1, figsize = (10,10))
        climits = [np.min(self.data_processor.std_img), np.max(self.data_processor.std_img)]
        img = ax.imshow(self.data_processor.std_img, cmap = 'gray', vmin = climits[0]*1, vmax = climits[1]*0.8)

        for i, iroi in enumerate(self.data_processor.rois_to_include):
            cm = plt.contour(self.data_processor.sima_masks[iroi,:,:], colors='g')
            plt.text(self.data_processor.roi_label_loc_manual[i][1] - 5, self.data_processor.roi_label_loc_manual[i][0] - 5,  int(iroi), fontsize=15, color = 'red')
        
        return fig

    def plot_contour_activityname(self, name):
        """
        A function that takes in a name of an event and a index of the color scheme list as
        parameters and uses the corresponding color schemes to plot the event-based activity contour
        plot using the data fetched from the previously loaded in pickle file
        """
        contour_vector, orig_vector = self.generate_contour_vectors(self.data_processor.data_dict, self.data_processor.analysis_win, name)
        norm, sm = self.generate_norm_map(orig_vector)

        # this all below could be placed into a separate function
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        climits = [np.min(self.data_processor.std_img), np.max(self.data_processor.std_img)]
        img = ax.imshow(self.data_processor.std_img, cmap='gray', vmin=climits[0]*1, vmax=climits[1]*0.8)

        #Adding the ROIs onto the image
        for iroi in range(self.data_processor.numROI_sima):
            color = sm.to_rgba(orig_vector[iroi])
            roi_color = color[:3]
            
            # plotting the contours and color them based on each cell's activity value
            cm = plt.contour(self.data_processor.sima_masks[iroi,:,:], colors=[roi_color], norm=norm)
            
            # set the data point text label
            if self.data_processor.activity_name:
                txt = plt.text(self.data_processor.roi_label_loc_manual[iroi][1] - 5, self.data_processor.roi_label_loc_manual[iroi][0] - 5, round(orig_vector[iroi], 2), fontsize=10, color='white')
            else:
                txt = plt.text(self.data_processor.roi_label_loc_manual[iroi][1] - 5, self.data_processor.roi_label_loc_manual[iroi][0] - 5, int(iroi), fontsize=10, color='white')
            txt.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='k')])
        
        # Customizing the plot
        plt.title("Activity: " + name, fontsize=15)
        plt.axis('off')

        # Adding colorbar
        cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # Adjust the position as needed
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label("Activity (Z-Score)")

        return fig
    
    def plot_activity_subplots(self, data_dict, analysis_win, name):
        """
        A function that takes in a name of an event, and a color index as parameter and
        plot each cell's change of activity corresponding to the specified event across a
        certain time frame as subplots
        """
        
        tvec = np.linspace(-2, 8, data_dict[name]['num_samples'])
        trial_avg_data = np.mean(data_dict[name]['zdata'], axis=0)
        min_max = [np.min(trial_avg_data), np.max(trial_avg_data)]

        contour_vector, orig_vector = self.generate_contour_vectors(data_dict, analysis_win, name)
        norm, sm = self.generate_norm_map(orig_vector)
        
        # Set the background color of the plot to black
        plt.style.use('dark_background')

        # Plotting each cell's activity as subplots and color each plot based on the cell's
        # activity value
        (fig, axs) = plt.subplots(nrows=6, ncols=5, figsize=(17, 17))
        fig.suptitle("Activity: " + name + "\n", fontsize=20, color='white')  # Set title color

        counter = 0
        for i in range(6):  # CZ hardcode
            for j in range(5):
                color = sm.to_rgba(orig_vector[counter])  # Get color based on activity value
                roi_color = color[:3]
                
                axs[i, j].plot(tvec, trial_avg_data[counter,:], color=roi_color)
                axs[i, j].set_title("roi " + str(counter), size=20, color='white')  # Set title color
                counter += 1
                axs[i, j].tick_params(axis='both', which='major', labelsize=13, colors='white')  # Set tick label color
                axs[i, j].tick_params(axis='both', which='minor', labelsize=13, colors='white')  # Set tick label color
                axs[i, j].set_ylim(min_max)
                if i == 5 and j == 2:
                    axs[i, j].set_xlabel('Time (s)', size=15, color='white')  # Set label color
                    axs[i, j].set_ylabel('Activity (z-scored)', size=15, color='white')  # Set label color
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].spines['top'].set_visible(False)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the position of the suptitle

        # Set the styles of the plot back to normal
        plt.style.use('default')
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        plt.rcParams["font.family"] = "Arial"

        return fig
    
    def generate_activityname_contours(self):
        graphs = []

        for i in range(len(self.data_processor.conditions)):
            condition_graph = {}
            condition_graph["contour"] = self.plot_contour_activityname(self.data_processor.conditions[i])
            condition_graph["linegraph"] = self.plot_activity_subplots(self.data_processor.data_dict, self.data_processor.analysis_win, self.data_processor.conditions[i])
            graphs.append(condition_graph)
        
        return graphs
    
    def generate_bar_graph(self):
        # Getting the data vector corresponds to the specified event from the dictionary
        contour_vector_events = []
        error_events = []

        self.data = {}
        self.data['rois'] = [('roi' + str(iroi)) for iroi in self.data_processor.rois_to_include]

        for event_name in self.data_processor.conditions:
            contour_vector = np.mean(self.data_processor.data_dict[event_name]
                                      ['zdata']
                                      [:,:,
                                       misc.get_tvec_sample(self.data_processor.tvec, self.data_processor.analysis_win[0]): misc.get_tvec_sample(self.data_processor.tvec, self.data_processor.analysis_win[-1])],
                                      axis=(0,2)) 
            contour_vector_events.append(contour_vector)

            self.data[event_name] = list(contour_vector)
            self.normalize_vector(contour_vector)
            
            error_events.append(np.std(np.mean(self.data_processor.data_dict[event_name]['zdata']
                                                [:,:,misc.get_tvec_sample(self.data_processor.tvec, self.data_processor.analysis_win[0]):misc.get_tvec_sample(self.data_processor.tvec, self.data_processor.analysis_win[-1])], 
                                                axis=2), axis=0) / math.sqrt(self.data_processor.data_dict[event_name]['zdata'].shape[0]))

        # turn the dictionary into a pandas dataframe
        self.data = pd.DataFrame(data=self.data)

        num_rois_per_subplot = 8
        self.bar_width = 0.2

        num_subplots = int(np.ceil(float(self.data_processor.num_rois_to_include)/float(num_rois_per_subplot)))
        subplot_rois = self.split_given_size(np.arange(self.data_processor.num_rois_to_include),num_rois_per_subplot)

        # Plot the barplot for the data and add text
        bargraphs = []
        for i in range(num_subplots):
            fig, ax = plt.subplots(figsize=(15, 6))
            
            x_positions = np.arange(len(subplot_rois[i]))
            bars = self.plot_bars(ax, subplot_rois[i], self.data_processor.conditions, self.data, error_events, self.bar_width, x_positions)
            self.add_text(ax, subplot_rois[i], self.data_processor.conditions, x_positions)
            self.add_colors(bars, contour_vector_events)

            # Decorate the barplot
            placement = float(float(float(self.bar_width) * len(self.data_processor.conditions)) / 2.0)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(self.data['rois'][subplot_rois[i]], fontsize=10)
            ax.set_title("Event Based Activity Barplot (Order: " + str(self.data_processor.conditions) + ")", fontsize=15)
            ax.set_xlabel("ROIS", fontsize=10)
            ax.set_ylabel("Activity", fontsize=10)

            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_ylim(bottom=0)
            
            fig.tight_layout()
            bargraphs.append(fig)
        
        return bargraphs