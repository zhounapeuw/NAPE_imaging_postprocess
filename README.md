# NAPE_imaging_postprocess
 Package for functional imaging postprocessing analysis. The primary goal is to take time-series data from simultaneously-recorded cells/ROIs and event (behavior and/or manipulations) timing to gain an understanding of how individual and groups of cells relate to such external manipulations.
 
 # Anaconda Environment Installation:
 
 1) After cloning/downloading the repository, open Anaconda Prompt (the following processes can also be performed in Anaconda Navigator)
 2) Locate the full path to the repository and in the prompt, execute `cd *path_to_repo*`. Example: `cd C:\Users\napeadmin\Documents\GitHub\NAPE_imaging_postprocess`
 3) Execute `conda env create -n napeca_post -f napeca_post.yml`. This should create a new anaconda environment called napeca_post that contains the relevant packages for running the post-processing scripts
 4) Execute `conda activate napeca_post` to enable the newly created environment
 
 # Modules:
 * __whole_session_event_ticks__: Plot activity traces for each ROI across whole session (Optional - primarily for sanity check/prelim viz purposes) 
 * __plot_activity_contours__: Work-in-progress; Plots the cell ROI contours onto the projection image. ROI contours are color-coded based on the amplitude of the mean event-triggered response for each condition. Bar plots at the end illustrate condition preference.
 * __event_rel_analysis__: Primary event-related analysis script: Plots trial-, roi-, time-resolved event-triggered responses for each and across all ROIs. Also plots average responses in each dimension. 
 * __Example of spectral clustering__: Kmeans/Spectral clustering of ROIs based on event-related responses. __REQUIRES the above script to be run (event_rel_analysis)__

# Suite2p-related Modules:
* __s2p_conversion__: Processes and converts suite2p output signals, and saves a dataframe or np array of neuropil-corrected roi activity traces as both a csv and npy file. Either output files can be used for downstream analyses (eg. event_rel_analysis, s2p_plot_rois_and_activity)
* __s2p_plot_rois_and_activity__: Plot the cell ROI contours from Suite2p onto the session's projection image, and generates lineplots for each ROI's whole-session activity trace

# SIMA-related Modules:
* __sima_plot_ROI_contours__: Plot the cell ROI contours from SIMA/NAPECA onto the session's projection image, and generates lineplots for each ROI's whole-session activity trace

# Example workflows:
#### Ran NAPECA pipeline for preprocessing:
1) Preprocess (motion correct, signal extract, neuropil correct) using NAPECA pipeline
2) If you want to plot ROI contours and associated whole-session activity, run sima_plot_ROI_contours.ipynb
3) The user should create an event CSV that contains trial event types and their timing (see example below) or if using Bruker data, generate a pickle file using the Bruker prepreprocessing code in NAPECA preprocessing repository (https://github.com/zhounapeuw/NAPE_imaging_analysis/blob/master/napeca/prepreprocess/bruker_data_process.ipynb)
4) Run any of the main modules (with the fnames pointing to the NAPECA output; eg. VJ_OFCVTA_7_260_D6_neuropil_corrected_signals_15_50_beta_0.8.npy, and framenumberforevents_VJ_OFCVTA_7_260_D6_trained.pkl. Data from NAPECA are immediately compatible with the main postprocessing modules.
#### Ran Suite2p for preprocessing:
1) Run s2p_conversion.ipynb to convert s2p output signals into either a csv or npy
2) If you want to plot ROI contours and associated whole-session activity, run s2p_plot_rois_and_activity.ipynb
3) The user should create an event CSV that contains trial event types and their timing (see example below)
4) Run any of the main modules

# How should you format your activity/signal data:

1) If you are using Inscopix: reference the csv containing ROI signals 
2) If you preprocessed your 2p data using Suite2p: Run the s2p_conversion.ipynb code first before moving to the main scripts
3) If you are using the NAPECA Preprocessing pipeline: Run the main scripts while referencing the extracted signals or neuropil-corrected signals (npy files)
4) All other data sources: The most straightforward approach is to create CSV files with the dimensions illustrated in the picture below

<img width="400" alt="whole_session_plot" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/napeca_post_signal_csv_format.png">

# How should you format your behavioral data (if not using Bruker prepreprocess script for behavioral event extraction):

1) The code is looking primarily for a dictionary where keys are the trial conditon names and the values are lists containing the event occurrences in samples/frames
2) The most straightforward, general approach is to create a CSV file containing event data in the tidy format illustrated in the picture below

<img width="300" alt="whole_session_plot" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/napeca_post_event_csv_format.png">

# Example Figure Outputs:

### ROI contours plotted on projection image (s2p_plot_rois_and_activity):
<img width="600" alt="roi_contour_map" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/roi_contour_map.png">

### Activity traces for each ROI, colors corresponding to ROIs above (s2p_plot_rois_and_activity):
<img width="600" alt="roi_ts" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/roi_ts.png">

### Activity traces for each ROI across whole session, with event occurrences plotted (whole_session_event_ticks):
<img width="800" alt="whole_session_event" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/whole_session_event.png">

### Event-related responses averaged and across trials (event_rel_analysis):
<img width="800" alt="roi_1_activity" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/roi_1_activity.png">

### Event-related response heatmaps across ROIs (event_rel_analysis):
<img width="800" alt="trial_avg_heatmap" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/trial_avg_heatmap.png">

### Event-related responses averaged across trials and ROIs (event_rel_analysis):
<img width="500" alt="roi_trial_avg_trace" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/roi_trial_avg_trace.png">

### Spectral clustering of ROIs based on event-related response profiles (Example of spectral clustering; to be renamed):
<img width="500" alt="cluster_heatmap" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/cluster_heatmap.png">

### ROI-averaged activity traces after clustering (Example of spectral clustering):
<img width="600" alt="cluster_roiAvg_traces" src="https://github.com/zhounapeuw/NAPE_imaging_postprocess/blob/main/docs/_images/cluster_roiAvg_traces.png">

## Authors

* **Zhe Charles Zhou** - *Initial and ongoing work; Director* - [Zhou NAPE UW](https://github.com/zhounapeuw)
* **Lauren Ran Liao** - *Machine Learning Development* - [Lauren Liao](https://github.com/lr5029)
