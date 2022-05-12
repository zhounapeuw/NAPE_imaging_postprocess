import os
import pytest
import numpy as np
import pandas as pd
from napeca_post import s2p_plot_rois_activity_funcs


# getting our fixtures set up
@pytest.fixture
def path_dict():
    return {}


@pytest.fixture()
def path_dict_entry(path_dict):
    path_dict['s2p_dir'] = os.path.join(os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6_snippit'), 'suite2p', 'plane0')
    return path_dict


@pytest.fixture()
def paths_function_call(path_dict, path_dict_entry):
    path_dict = s2p_plot_rois_activity_funcs.define_paths_roi_plots(path_dict_entry, [0, 10], None, os.path.abspath('./napeca_post/sample_data/VJ_OFCVTA_7_260_D6_snippit'))
    return path_dict


@pytest.fixture()
def s2p_data_dict():
    return {}


@pytest.fixture()
def load_s2p_data_roi_plots_for_testing(path_dict, paths_function_call, s2p_data_dict):
    s2p_data_dict = s2p_plot_rois_activity_funcs.load_s2p_data_roi_plots(paths_function_call)
    return s2p_data_dict


@pytest.fixture()
def plot_vars():
    return {}


@pytest.fixture()
def creating_plot_vars(path_dict_entry, load_s2p_data_roi_plots_for_testing, s2p_data_dict, plot_vars):
    plot_vars = s2p_plot_rois_activity_funcs.plotting_rois(load_s2p_data_roi_plots_for_testing, path_dict_entry)
    return plot_vars


@pytest.fixture()
def trace_data_selected():
    return []


@pytest.fixture()
def trace_data_selected_init(trace_data_selected, load_s2p_data_roi_plots_for_testing, creating_plot_vars, plot_vars):
    trace_data_selected = load_s2p_data_roi_plots_for_testing['F_npil_corr_dff'][creating_plot_vars['cell_ids']]
    trace_data_selected = trace_data_selected.astype(np.float32)
    return trace_data_selected


@pytest.fixture()
def ground_truth():
    data = pd.read_csv("trace_data_selected.csv", header = None)
    data = pd.DataFrame.to_numpy(data)
    data = data.astype(np.float32)
    return data

def test_trace_data_selected(trace_data_selected_init, ground_truth):
    # this part is a bit tricky to understand
    # we cannot simply compare trace_data and ground_truth since they are arrays and 
    # the developers of numpy arrays did not define the logical operators for comaprison (and, or, etc.)
    # since there are many ways of interpreting it. Thus, we must take the difference of the two arrays, apply all()
    # which indicated that we want all of the values to be equal to each other, and compare it to 0.
    assert (trace_data_selected_init - ground_truth).all() == 0
