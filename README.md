# NAPE_imaging_postprocess
 Package for functional imaging postprocessing analysis. The primary goal is to take time-series data from simultaneously-recorded cells/ROIs and event (behavior and/or manipulations) timing to gain an understanding of how individual and groups of cells relate to such external manipulations.
 
 # Anaconda Environment Installation:
 
 1) After cloning/downloading the repository, open Anaconda Prompt (the following processes can also be performed in Anaconda Navigator)
 2) Locate the full path to the repository and in the prompt, execute `cd *path_to_repo*`. Example: `cd C:\Users\napeadmin\Documents\GitHub\NAPE_imaging_postprocess`
 3) Execute `conda env create -n napeca_post -f napeca_post.yml`. This should create a new anaconda environment called napeca_post that contains the relevant packages for running the post-processing scripts
 4) Execute `conda activate napeca_post` to enable the newly created environment
 
 # Modules:
 * whole_session_event_ticks: Plot activity traces for each ROI across whole session (Optional - primarily for sanity check/prelim viz purposes) 
 * plot_activity_contours: Work-in-progress; Plots the cell ROI contours onto the projection image. ROI contours are color-coded based on the amplitude of the mean event-triggered response for each condition. Bar plots at the end illustrate condition preference.
 * event_rel_analysis: Primary event-related analysis script: Plots trial-, roi-, time-resolved event-triggered responses for each and across all ROIs. Also plots average responses in each dimension. 
 * Example of spectral clustering: Kmeans/Spectral clustering of ROIs based on event-related responses. __REQUIRES the above script to be run (event_rel_analysis) __

# Suite2p-related Modules:
* s2p_conversion: Processes and converts suite2p output signals, and saves a dataframe or np array of neuropil-corrected roi activity traces as both a csv and npy file. Either output files can be used for downstream analyses (eg. event_rel_analysis, s2p_plot_rois_and_activity)
* s2p_plot_rois_and_activity: Plot the cell ROI contours from Suite2p onto the session's projection image, and generates lineplots for each ROI's whole-session activity trace
