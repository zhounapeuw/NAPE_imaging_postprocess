import numpy as np
import pickle as pkl
import os
import pandas as pd



# could pass input as a list to make it take threshold_scaling_values sometimes
# could make two different functions and only have one of them be called by the automatic and the other
#called by the local

def define_paths(fdir, fname, threshold_scaling_values):
    path_dict = {}
    # define paths for loading s2p data
    if threshold_scaling_values == 0 :
        path_dict['s2p_dir'] = os.path.join(fdir, 'suite2p', 'plane0')
    else:
        path_dict['s2p_dir'] = os.path.join(fdir, f'threshold_scaling_{threshold_scaling_values}', 'plane0')

    path_dict['s2p_F_path'] = os.path.join(path_dict['s2p_dir'], 'F.npy')
    path_dict['s2p_Fneu_path'] = os.path.join(path_dict['s2p_dir'], 'Fneu.npy')
    path_dict['s2p_iscell_path'] = os.path.join(path_dict['s2p_dir'], 'iscell.npy')
    path_dict['s2p_ops_path'] = os.path.join(path_dict['s2p_dir'], 'ops.npy')

    # define savepaths for converted output data
    path_dict['csv_savepath'] = os.path.join(fdir, f'{fname}_s2p_data_{threshold_scaling_values}.csv')
    path_dict['npy_savepath'] = os.path.join(fdir, f'{fname}_s2p_neuropil_corrected_signals_{threshold_scaling_values}.npy')
    
    return path_dict

def load_s2p_data(path_dict):
    
    s2p_data_dict = {}
    # load s2p data
    s2p_data_dict['F_data'] = np.load(path_dict['s2p_F_path'], allow_pickle=True)
    s2p_data_dict['Fneu_data'] = np.load(path_dict['s2p_Fneu_path'], allow_pickle=True)
    s2p_data_dict['iscell_data'] = np.load(path_dict['s2p_iscell_path'], allow_pickle=True)
    s2p_data_dict['ops_data'] = np.load(path_dict['s2p_ops_path'], allow_pickle=True).item()
    
    return s2p_data_dict

def csv_npy_save(fdir, fname, threshold_scaling_values):
    path_dict = define_paths(fdir, fname, threshold_scaling_values)
    s2p_data_dict = load_s2p_data(path_dict)

    npil_corr_signals = s2p_data_dict['F_data'] -  s2p_data_dict['ops_data']['neucoeff'] * s2p_data_dict['Fneu_data']
    iscell_npil_corr_data = npil_corr_signals[s2p_data_dict['iscell_data'][:,0].astype('bool'),:]
    
    # save cell activity data as a csv with ROIs on y axis and samples on x axis
    np.save(path_dict['npy_savepath'], iscell_npil_corr_data) # this saves the user-curated neuropil corrected signals as an npy file
    pd.DataFrame(data=iscell_npil_corr_data).to_csv(path_dict['csv_savepath'], index=False, header=False) # this saves the same data as a csv file




