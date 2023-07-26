"""
likelihood processing analysis_utilities
Forward fill low likelihood (x,y)
"""

import glob
import re
import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.mixture import GaussianMixture


def boxcar_center(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg


def convert_int(s):
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def get_filenames(base_path, folder):
    filenames = glob.glob(os.path.join(base_path, folder, '*.h5'))
    sort_nicely(filenames)
    return filenames


def get_filenamesh5(base_path, folder):
    filenames = glob.glob(os.path.join(base_path, folder, '*.h5'))
    sort_nicely(filenames)
    return filenames


def get_filenamesjson(base_path, folder):
    filenames = glob.glob(base_path + folder + '/*.json')
    sort_nicely(filenames)
    return filenames


def import_folders(base_path, folders: list, pose):
    fldrs = []
    filenames = []
    rawdata_li = []
    data_li = []
    perc_rect_li = []
    for i, fd in enumerate(folders):  # Loop through folders
        f = get_filenames(base_path, fd)
        for j, filename in enumerate(f):
            curr_df = pd.read_csv(filename, low_memory=False)
            curr_df_filt, perc_rect = adp_filt(curr_df, pose)
            rawdata_li.append(curr_df)
            perc_rect_li.append(perc_rect)
            data_li.append(curr_df_filt)
        fldrs.append(fd)
        filenames.append(f)
    data = np.array(data_li)
    return fldrs, filenames, data, perc_rect_li


def adp_filt(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    currdf = np.array(currdf[1:])
    for header in pose:
        if currdf[0][header + 1] == "likelihood":
            lIndex.append(header)
        elif currdf[0][header + 1] == "x":
            xIndex.append(header)
        elif currdf[0][header + 1] == "y":
            yIndex.append(header)
    curr_df1 = currdf[:, 1:]
    datax = curr_df1[1:, np.array(xIndex)]
    datay = curr_df1[1:, np.array(yIndex)]
    data_lh = curr_df1[1:, np.array(lIndex)]
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        data_lh_float = data_lh[:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect


def adp_filt_h5(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    headers = np.array(currdf.columns.get_level_values(2)[:])
    for header in pose:
        if headers[header] == "likelihood":
            lIndex.append(header)
        elif headers[header] == "x":
            xIndex.append(header)
        elif headers[header] == "y":
            yIndex.append(header)
    curr_df1 = np.array(currdf)
    datax = curr_df1[:, np.array(xIndex)]
    datay = curr_df1[:, np.array(yIndex)]
    data_lh = curr_df1[:, np.array(lIndex)]
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        data_lh_float = data_lh[:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect

def zero_initialization():
    return np.hstack([[0, 0]])

def nan_initialization():
    return np.hstack([[np.nan, np.nan]])

INIT_METHODS={"zero": zero_initialization, "nan": nan_initialization}


def extract_x_y_score(currdf, pose, frame_numbers=None):
    """
    Extract the x y and score of the body parts in the given order
    """

    # first body part will be used as anchor
    sorted_pose = sorted(pose)
    node_names = [node.decode() for node in currdf["node_names"]]
    node_names = [node_names[i] for i in pose]

    if frame_numbers is None:
        datax = currdf['tracks'][0, 0, sorted_pose]
        datay = currdf['tracks'][0, 1, sorted_pose]
        score = currdf['point_scores'][0, sorted_pose]
    else:
        datax = currdf['tracks'][0, 0, sorted_pose, frame_numbers]
        datay = currdf['tracks'][0, 1, sorted_pose, frame_numbers]
        score = currdf['point_scores'][0, sorted_pose, frame_numbers]

    reorder=[sorted_pose.index(i) for i in pose]
    print(reorder)
    datax=datax[reorder, :]
    datay=datay[reorder, :]
    score=score[reorder, :]
    return datax, datay, score


def apply_score_filter(datax, datay, score, score_filter, pose):

    assert len(score_filter) == len(pose)

    # apply score filters here
    for i, body_part_index in enumerate(pose):
        filter_criteria = score_filter[i]
        if filter_criteria is None:
            threshold=0
        elif filter_criteria == "elbow":
            gmm = GaussianMixture(n_components=2)
            isnan=np.isnan(score[i, :])
            if isnan.all():
                threshold=0
            else:
                bp_score=score[i, ~isnan]
                gmm.fit(bp_score.reshape((-1, 1)))
                means = gmm.means_.flatten()
                stds = np.sqrt(gmm.covariances_).flatten()
                threshold=max(0, means[1]-stds[1])

        elif isinstance(filter_criteria, float):
            threshold=filter_criteria
        
        datax[i, score[i, :] < threshold] = np.nan
        datay[i, score[i, :] < threshold] = np.nan

    return datax, datay



def impute_missing_bodypart(datax, datay, score, body_parts_and_criteria, node_names):
    """
    Handle missing values in pose estimates by applying imputation criteria

    1) interpolate: copy the same value as the body part IN THE LAST observed frame
    2) constant: set to 0,0
    3) body_part: set to the same value as another body part IN THE SAME frame

    Returns:
        currdf_filt: Pose estimate with imputed values of shape timepoints x body_parts*2
        score: Score array with updated scores based on imputed values
    """

    currdf_filt = np.zeros((datax.shape[1], (datax.shape[0]) * 2))

    for body_part, criterion in body_parts_and_criteria:
        pose_indexer=slice(2 * body_part, 2 * body_part + 2)

        if criterion == "interpolate":
            first_not_nan = np.where(np.isnan(datax[body_part, :]) == False)[0]
            if len(first_not_nan) == 0:
                first_not_nan = 0
            else:
                first_not_nan = first_not_nan[0]

            score[body_part, 0] = score[body_part, first_not_nan]
            currdf_filt[0, pose_indexer] = np.hstack([datax[body_part, first_not_nan], datay[body_part, first_not_nan]])

        elif criterion == "constant":
            currdf_filt[0, pose_indexer] = [0, 0]
        elif criterion in node_names:
            ref_body_part = node_names.index(criterion)

            # this line breaks the code if a reference body part is never observed in a whole chunk
            first_not_nan_ref = np.where(np.isnan(datax[ref_body_part, :]) == False)[0][0]
        
            first_not_nan = np.where(np.isnan(datax[body_part, :]) == False)[0]
            if len(first_not_nan) == 0:
                first_not_nan=0
            else:
                first_not_nan=first_not_nan[0]
        
            if first_not_nan_ref < first_not_nan:
                currdf_filt[0, pose_indexer] = np.hstack([datax[ref_body_part, first_not_nan_ref], datay[ref_body_part, first_not_nan_ref]])
                score[body_part, 0] = score[ref_body_part, first_not_nan_ref]
            else:
                currdf_filt[0, pose_indexer] = np.hstack([datax[body_part, first_not_nan], datay[body_part, first_not_nan]])
                score[body_part, 0] = score[body_part, first_not_nan]


        # t+1, t+2, ..., tN
        for t in range(1, datax.shape[1]):         
            if np.isnan(datax[body_part][t]):
                if criterion == "interpolate":
                    currdf_filt[t, pose_indexer] = currdf_filt[t - 1, pose_indexer]
                    score[body_part, t] = score[body_part, t-1]
                elif criterion in node_names:
                    ref_body_part = node_names.index(criterion)
                    currdf_filt[t, pose_indexer] = currdf_filt[t, (2 * ref_body_part):(2 * ref_body_part + 2)]
                    score[body_part, t] = score[ref_body_part, t]
                elif criterion == "constant":
                    currdf_filt[t, pose_indexer] = [0, 0]
            else:
                currdf_filt[t, pose_indexer] = np.hstack([datax[body_part, t], datay[body_part, t]])

    return currdf_filt, score


def adp_filt_sleap_h5(currdf: object, pose, frame_numbers=None, score_filter=None, criteria="interpolate", pb=True):
    """
    Read and postprocess SLEAP pose estimates

    Arguments:
    
        currdf (h5py.File): Handler of a h5py file with at least two datasets:
            1. tracks: shape 1 x 2 x body_parts x timepoints
            2. point_scores: shape 1 x body_parts x timepoints
        
        pose (list): 0-based index of the body parts to be extracted as described in the node_names datasaet of currdf
        frame_numbers (list): 0-based index of the frame numbers to be processed,
            relative to the first frame with a pose estimate (even if a non detection is saved)
        score_filter (list): For every body part in pose, a value among [None, "elbow", X]
            If None, no score-based filtering is applied to the body part
            If elbow, a 2 component mixed Gaussian mixture model is fit to the data
               to select the elbow point in the distribution of scores from which they start to become more frequent
               Such elbow point marks the score threshold which separates predictions likely to be unreliable from those that are reliable
            If a float, this value is used as score threshold, i.e. bypassing the Gaussian mixture model


        criteria (list): For every body part in pose, a keyword among [interpolate, constant]
            If interpolate, the coordinates of the body part are set to the last observed whenever the body part is not detected
            This makes sense for body parts which are almost always visible, or only invisible for brief periods of time,
            because the body part is unlikely to drift much

            If constant, the coordinates of the body part are set to 0, 0.
            This makes sense for body parts which are very seldom visible at all,
            i.e. a non detection is not a pose estimation error, but a true non detection
            We use 0,0 to represent body parts not visible, since it's a coordinarte hardly recheable by any body part

    Returns
        currdf_filt (np.array) Processed pose estimation with shape timepoints x nodes*2
        perc_rect (list): Same length as number of body parts, the ith element contains the count of frames where the ith body part is missing
        interpolated (np.array): timepoints x body_parts. Whether the jth body part was observed with good quality criteria in the ith frame
        score (np.array): body_parts x timepoints. The score achieved by the ith body part in the jth frame.
            If the ith body part is interpolated in the jth frame, then score i,j contains the score of the reference frame j'        

    """
    if score_filter is None:
        score_filter=[None for _ in pose]

    datax, datay, score = extract_x_y_score(currdf=currdf, pose=pose, frame_numbers=frame_numbers)
    datax, datay = apply_score_filter(datax, datay, score=score, score_filter=score_filter, pose=pose)
        
    # set every coordinate to nan if the first body part is not detected
    # i.e. treat as a "must" body part
    datay[:, np.isnan(datax[0, :])]=np.nan
    datax[:, np.isnan(datax[0, :])]=np.nan


    perc_rect = []
    for i in range(len(pose)):
        perc_missing=np.argwhere(np.isnan(datax[i]) == True).shape[0] / datax.shape[1]
        perc_rect.append(perc_missing)


    if isinstance(criteria, str):
        criteria = [criteria for _ in range(datax.shape[0])]

    if pb:
        body_parts_and_criteria=tqdm(zip(range(datax.shape[0]), criteria))
    else:
        body_parts_and_criteria=zip(range(datax.shape[0]), criteria)

    currdf_filt, score=impute_missing_bodypart(datax, datay, score, body_parts_and_criteria, node_names=currdf["node_names"])
    # impute_missing_bodypart leaves no nan, so all the positions where datax was nan have been interpolated
    interpolated = np.isnan(datax[0, :])   

    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect, interpolated, score


def no_filt_sleap_h5(currdf: object, pose):
    datax = currdf['tracks'][0][0][pose]
    datay = currdf['tracks'][0][1][pose]
    pose_ = []
    currdf_nofilt = np.zeros((datax.shape[1], (datax.shape[0]) * 2))
    for x in tqdm(range(datax.shape[0])):
        pose_.append(currdf['node_names'][pose[x]])
        print(pose_)
        for i in range(0, datax.shape[1]):
            currdf_nofilt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[x, i], datay[x, i]])
    currdf_nofilt = np.array(currdf_nofilt)
    header = pd.MultiIndex.from_product([['SLEAP'],
                                         [i for i in pose_],
                                         ['x', 'y']],
                                        names=['algorithm', 'pose', 'coord'])
    df = pd.DataFrame(currdf_nofilt, columns=header)
    return df
