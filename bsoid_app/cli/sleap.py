import os.path
import h5py
import joblib
import numpy as np
from tqdm.auto import tqdm
import glob

from bsoid_app.bsoid_utilities.likelihoodprocessing import (
    get_filenamesh5,
    adp_filt_sleap_h5
)


def preprocess_sleap_h5(root_path, data_directories, pose_chosen=None, **kwargs):
    
    data_files = glob.glob(os.path.join(root_path, data_directories[0], '*.h5'))
    file0_df = h5py.File(data_files[0], 'r')
    if pose_chosen is None:
        pose_chosen = list(range(len(file0_df['node_names'][:])))

    raw_input_data, sub_threshold, processed_input_data, input_filenames = preprocess_sleap_all(
        root_path=root_path,
        data_directories=data_directories,
        pose_chosen=pose_chosen,
        progressbar=None,
        **kwargs,
    )

    
    return raw_input_data, sub_threshold, processed_input_data, input_filenames, pose_chosen


def preprocess_sleap(h5_files, pose_chosen, progressbar=None, use_tqdm=False, **kwargs):

    raw_input_data=[]
    sub_threshold=[]
    processed_input_data=[]
    input_filenames=[]

    if progressbar is not None:
        my_bar = progressbar(0)

    if use_tqdm:
        iterator=tqdm(enumerate(h5_files))
    else:
        iterator=enumerate(h5_files)

    for j, filename in iterator:
        file_j_df = h5py.File(filename, 'r')
        file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, pose_chosen, **kwargs)
        raw_input_data.append(file_j_df['tracks'][:][0])
        sub_threshold.append(p_sub_threshold)
        processed_input_data.append(file_j_processed)
        input_filenames.append(filename)
        if progressbar is not None:
            my_bar(round((j + 1) / len(h5_files) * 100))

    return raw_input_data, sub_threshold, processed_input_data, input_filenames

def preprocess_sleap_all(root_path, data_directories, *args, no_files=None, n_jobs=1, **kwargs):
    """
    Args:
        foo

    Return:

        raw_input_data (list): Pose estimation for all files and nodes
        sub_threshold (list): Fraction of missing data for all nodes in each file
        processed_input_data (list): Pose estimation for all files and nodes, with interpolation applied
        input_filenames (list): Source h5 files
    """

    raw_input_data=[]
    sub_threshold=[]
    processed_input_data=[]
    input_filenames=[]

    all_files=[]
    for i, fd in enumerate(data_directories):
        files = get_filenamesh5(root_path, fd)
        if no_files is not None:
            files=files[:no_files]
        all_files.extend(files)



    Output = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
         preprocess_sleap
        )(
            [file], *args, **kwargs
        )
        for i, file in enumerate(all_files)
    )

    for out in Output:
        raw_input_data.extend(out[0])
        sub_threshold.extend(out[1])
        processed_input_data.extend(out[2])
        input_filenames.extend(out[3])


    return raw_input_data, sub_threshold, processed_input_data, input_filenames
