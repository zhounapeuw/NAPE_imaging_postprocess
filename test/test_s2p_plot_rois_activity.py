import os
import pytest
import numpy as np
import pandas as pd
from napeca_post import s2p_plot_rois_activity_funcs



# getting our fixtures set up
# initializing the path_dict dictionary 
@pytest.fixture
def path_dict():
    return {}


# defines the directory for which future fixtures will pull from 
@pytest.fixture()
def path_dict_entry(path_dict):
    path_dict['s2p_dir'] = os.path.join(os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6_snippit'), 'suite2p', 'plane0')
    return path_dict


# calls the define_paths_roi_plots function which loads the appropriate paths into a dictionary for subsequent fixtures
@pytest.fixture()
def paths_function_call(path_dict, path_dict_entry):
    path_dict = s2p_plot_rois_activity_funcs.define_paths_roi_plots(path_dict_entry, [0, 10], None, os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6_snippit'))
    return path_dict


# initializing hte s2p_data_dict dictionary 
@pytest.fixture()
def s2p_data_dict():
    return {}


# call the load_s2p_data_roi_plots function to load the npy files to which we reference in the path_dict dictionary
# necessary fixture so we can perform operations on the data
@pytest.fixture()
def load_s2p_data_roi_plots_for_testing(path_dict, paths_function_call, s2p_data_dict):
    s2p_data_dict = s2p_plot_rois_activity_funcs.load_s2p_data_roi_plots(paths_function_call)
    return s2p_data_dict


# initializing plot_vars dictionary
@pytest.fixture()
def plot_vars():
    return {}


# calls the plotting_rois function which tells subsequent plots how many rois to load in
@pytest.fixture()
def creating_plot_vars(path_dict_entry, load_s2p_data_roi_plots_for_testing, s2p_data_dict, plot_vars):
    plot_vars = s2p_plot_rois_activity_funcs.plotting_rois(load_s2p_data_roi_plots_for_testing, path_dict_entry)
    return plot_vars


# initializing trace_data_selected list
@pytest.fixture()
def trace_data_selected():
    return []


# performing calculation using output from all of the previous fixtures to generate an array which is the product of 
# running all of the functions in the s2p_plot_rois_activity_funcs file. This is what will be compared to our ground truth
@pytest.fixture()
def trace_data_selected_init(trace_data_selected, load_s2p_data_roi_plots_for_testing, creating_plot_vars, plot_vars):
    trace_data_selected = load_s2p_data_roi_plots_for_testing['F_npil_corr_dff'][creating_plot_vars['cell_ids']]
    trace_data_selected = trace_data_selected.astype(np.float32)
    return trace_data_selected


# loads in our ground truth for a specific run of our sample data, included in repo, to compare to that which
# was loaded from the previous fixtures
@pytest.fixture()
def ground_truth():
    data = pd.read_csv("trace_data_selected.csv", header = None)
    data = pd.DataFrame.to_numpy(data)
    data = data.astype(np.float32)
    return data


# performs the actual assertion between ground truth and calculted trace_data_selected values
def test_trace_data_selected(trace_data_selected_init, ground_truth):
    # we cannot compare trace_data and ground_truth since they are arrays and 
    # logical operators for comaprison (and, or, etc.) are not defined for them
    # since there are many ways of interpreting it. Thus, we must take the difference of the two arrays, 
    # and apply all() which indicates that we want all of the values to be equal to 0.
    assert (trace_data_selected_init - ground_truth).all() == 0
