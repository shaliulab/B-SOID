import logging
import numpy as np
from tqdm import tqdm
import itertools
import joblib
import os.path
import scipy.ndimage

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bsoid_app.bsoid_utilities.utils import (
    load_embeddings_,
    load_feats_,
    load_umap_model_,
    window_from_framerate,
)

from .umap_implementations import (
    cpu_umap,
    cuml_umap,
)

def bsoid_extract_xy(dxy_r, weights):
    """
    Extract distance between each pair of body parts

    The distance between 2 body parts is simply the euclidean distance beetwene them,
    frame by frame

    Arguments:
        dxy_r (np.ndarray): Dataset of shape timepoints x pairs x 2
            where the i,j,k element stores the distance between the jth body parts pair along the kth axis on the ith frame
        weights (np.ndarray): 1D array providing weights to comptue a smoothed average oft he angles

    Returns:
        dxy_feat (np.ndarray): Dataset of shape pairs x timepoints
            where the i,j element stores the distance between the body parts of the ith pair at the jth timepoint
    """

    dxy_eu=np.linalg.norm(dxy_r, axis=2)
    dxy_feat = scipy.ndimage.convolve1d(dxy_eu, weights, axis=0, mode="constant", cval=0).T
    return dxy_feat


def bsoid_extract_angular(dxy_r, weights):
    """
    Extract angular speed from distance along each dimension between pairs

    The angular speed of a pair is the angle between two vectors:
        1. the vector produced by connecting each body part at time t
        2. the vector produced by connecting each body part at time t+1

    Arguments:
        dxy_r (np.ndarray): Dataset of shape timepoints x pairs x 2
            where the i,j,k element stores the distance between the jth body parts pair along the kth axis on the ith frame
        weights (np.ndarray): 1D array providing weights to comptue a smoothed average oft he angles

    Returns:
        ang_feat (np.ndarray): Dataset of shape pairs x (timepoints-1)
            where the i,j element stores the angular speed of the ith pair between timepoints j and j+1
    """

    # create two versions of this dataset in 3D
    # both versions have extended the size of their third axis (axis=2)
    # to capture the 3D (volume) instead of 2D (plane)
    # both are missing one timepoint

    # version b is shifted one step forward
    # version a has the original phase   
    b_3d=np.concatenate([
        dxy_r[1:],
        # copy the first dimension (X), set all values to 0 and inject it as the new Z dimension
        dxy_r[1:,:,0][:,:,np.newaxis]*0
    ],axis=2)
    
    a_3d=np.concatenate([
        dxy_r[:-1],
        # copy the first dimension (X), set all values to 0 and inject it as the new Z dimension
        dxy_r[:-1:,:,0][:,:,np.newaxis]*0
    ],axis=2)

    # the 2 datasets are vectors of shape
    # (timepoints-1)xpairsx3
    # cross product of these 2 vectors
    # is a new vector perpendicular to the plane defined by the two input vectors
    # and its direction informs whether the rotation from one member of the pair to the other
    # is clock or counterclock wise
    cross_product=np.cross(b_3d, a_3d)

    magnitude = np.linalg.norm(cross_product, axis=2)
    direction = np.sign(cross_product[:, :, 2])

    ang_rad=np.arctan2(magnitude, np.einsum('ijk,ijk->ij', dxy_r[:-1], dxy_r[1:]))

    # transform from radians (0 -> 2pi) to degrees (0 -> 360)
    to_degs = np.dot(direction, 180) / np.pi
    ang=ang_rad*to_degs

    ang_feat = scipy.ndimage.convolve1d(ang, weights, axis=0, mode="constant", cval=0).T
    return ang_feat
    
def bsoid_extract(dataset, window, stride, my_bar=None):
    """
    Compute spatio-temporal features from the pose estimate

    Args:
        dataset (np.ndarray): Dataset of shape timepoints x (body_parts x 2) i.e 2 (!!) dimensions
        window (int): Size of window in the timeseries which will be used to average the raw estimates
            (i.e. all timepoints in the same window get the same estimate, which will be their mean)
        stride (int): Ratio between the framerate of the pose dataset and 10 FPS (the BSOID working framerate).
            For example, if input dataset has framerate of 150, the stride is 150/10=15       
    
    """

    # NOTE: There should be the same amount of predictions in each window,
    # otherwise the speeds will be biased towards the windows with more predictions
    # i.e. windows with more predictions capture more movement than windows with less predictions
    # even for the exact same behavior 

    # Create a window of weights - in the case of a simple moving average, the weights are all 1
    weights = np.ones(window) / window

    # initialize the singleton features by reshaping the pose dataset
    # to organize like this:
    # timepoints x 2 x body_parts (i.e. 3 dimensions) 
    singleton_feats=np.stack([dataset[:, i:i+2] for i in range(0, dataset.shape[1], 2)], axis=2)

    # initilize the pair features by reshaping the pose dataset
    # to organize like this
    # 2 x timepoints x pairs x 2
    # where:
    # the first 2 refers to either 1 or 2 pair
    # pairs is the number of all possible pairs between two different body parts
    # the second 2 refers to x and y

    pair_feats=np.stack([[
        dataset[:, j:j+2],
        dataset[:, i:i+2]
    ] for i, j in itertools.combinations(range(0, dataset.shape[1], 2), 2)], axis=2)

    # compute displacement of each body part from frame to frame
    # Output has shape (timepoints-1) x body_parts
    disp_r = np.linalg.norm(np.diff(singleton_feats, axis=0), axis=1)


    # Compute a moving average of the displacement (smooth)
    # disp_feat has shape body_parts x (timepoints-1) because it is transposed
    disp_feat = scipy.ndimage.convolve1d(disp_r, weights, axis=0, mode="constant", cval=0).T

    # Compute the difference between each pair of body parts in all frames
    # Output has shape timepoints x pairs x 2
    # (the axis of each pair is dropped in this op)
    dxy_r=np.diff(pair_feats, axis=0)[0]

    # Compute the angular speed of each pair of body parts in all frames
    # Output has shape pairs x (timepoints - 1)
    ang_feat = bsoid_extract_angular(dxy_r, weights)
    
    # Output has shape pairs x timepoints
    dxy_feat=bsoid_extract_xy(dxy_r, weights)


    # Combine all three feature types
    # NOTE The order of this features is locked by the name_features method
    features=np.vstack([
        # distance between body parts in a pair, frame by frame
        dxy_feat[:, 1:],
        # angular speed of two body parts in a pair, between two consecutive frames
        ang_feat,
        # linear speed of each body part in a pair, between two consecutive frames
        disp_feat
    ])

    if my_bar is not None:
        my_bar.update(1)
        
    # because of the diff ops computing features based on speeds
    # the dataset does not have size multiple of the stride
    # so we pad it here with the minimum amount of mock frames
    # so that the size becomes a multiple of the stride

    # this has happened for both time derived features, but also the rest
    # because we cropped the rest so that they could all be vstacked above
    
    # the padding is needed so that the reshape just after it works
    # the padding is done by taking the last value available (-1)
    padding_size=stride - (features.shape[1] % stride)
    padding=[
        features[:,-1].reshape((-1, 1))
        for _ in range(padding_size)
    ]
    
    padding=np.hstack(padding)
    feats_windows=np.concatenate(
        [features, padding],
        axis=1
    ).astype(np.float64)


    # reshape the dataset so that instead of flat timepoints
    # every timepoint within the same window defined by the stride
    # lives in the same dimension of the second axis (axis=1)
    # the third axis (axis=2) loops over all such windows
    feats_windows=feats_windows.reshape((feats_windows.shape[0], stride, -1), order="A")

    # the number of rows of the dxy_feat array tells us
    # how many non time derived features there are
    # all features afterwards are time derived
    first_pair_feature_index = dxy_feat.shape[0]

    # for time derived features, set the last window to be the mean
    # of the actual timepoints without the mock frames
    feats_windows[:first_pair_feature_index, -padding_size:,-1]=feats_windows[
        :first_pair_feature_index,
        (feats_windows.shape[1]-stride):(feats_windows.shape[1]-padding_size),
        -1
    ].mean(axis=1).reshape((-1, 1))
    
    # set the mock frames of time derived features to 0
    feats_windows[first_pair_feature_index:, -padding_size:, -1]=0

    f_integrated=np.concatenate([
        # compute the average distance between all pairs of body parts frame by frame,
        # within each window of size stride 
        feats_windows[:first_pair_feature_index].mean(axis=1),
        # compute the sum of the angular and linear speeds
        feats_windows[first_pair_feature_index:].sum(axis=1)
    ])[:,:-1] # ignore last frame to return the expected timepoints-1
    

    # compute the z score of each feature
    scaler = StandardScaler()
    # transpose because the scaler expects observations x features
    scaler.fit(f_integrated.T)
    scaled_features = scaler.transform(f_integrated.T).T

    return f_integrated, scaled_features
    
class extract:

    def __init__(self, working_dir, prefix, processed_input_data, framerate, fraction=1.0, use_gpu=True):
        self.working_dir = working_dir
        self.prefix = prefix
        self.processed_input_data = processed_input_data
        self.framerate = framerate
        self.train_size = []
        self.features = []
        self.scaled_features = []
        self.sampled_features = []
        self.sampled_embeddings = []
        self.fraction=fraction
        self.stride=self.framerate / 10
        self.use_gpu=use_gpu


    def subsample(self, fraction):

        data_size = 0
        for n in range(len(self.processed_input_data)):
            data_size += len(
                range(
                    round(self.stride),
                    self.processed_input_data[n].shape[0],
                    round(self.stride))
            )

        if fraction == 1.0:
            self.train_size = data_size
        else:
            self.train_size = int(data_size * fraction)

    def compute_features(self, n_jobs=1, **kwargs):
        logging.info('Extracting...')
        try:
            [self.features, self.scaled_features] = load_feats_(self.working_dir, self.prefix)
        except FileNotFoundError:
            [self.features, self.scaled_features] = self.compute_features_parallel(datasets=self.processed_input_data, n_jobs=n_jobs)

        self.save(self.features, self.scaled_features)
        self.learn_embeddings(**kwargs)

    @staticmethod
    def name_features(body_parts):
        pairs = list(itertools.combinations(body_parts, 2))
        pair_names = ["-".join(pair) for pair in pairs]

        pair_distances = [f"distance_{name}" for name in pair_names]
        pair_speeds = [f"angular_{name}" for name in pair_names]
        part_speeds = [f"speed_{name}" for name in body_parts]
        features = pair_distances + pair_speeds + part_speeds
        return features    


    def compute_features_parallel(self, datasets, n_jobs, **kwargs):
        window=window_from_framerate(self.framerate)
        stride=int(self.framerate/10)

        Output = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                bsoid_extract
            )(
                dataset=datasets[n], window=window, stride=stride, **kwargs
            )
            for n, _ in enumerate(datasets)
        )

        features=[]
        scaled_features=[]

        for f, s in  Output:
            features.append(f)
            scaled_features.append(s)

        # NOTE
        # Here we are stacking data of different individuals along the time axis (axis 1 / horizontal in this case)
        features=np.hstack(features)
        scaled_features=np.hstack(scaled_features)
        return features, scaled_features


    def save(self, features, scaled_features):
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_feats.sav'))), 'wb') as filehandle:
            joblib.dump([features, scaled_features], filehandle)

        logging.info(
            'Done extracting features from a total of **{}** training data files. '
            'Now reducing dimensions...'.format(len(self.processed_input_data))
        )

    def learn_embeddings(self, **kwargs):

        input_feats = self.scaled_features.T
        pca = PCA()
        pca.fit(input_feats)
        num_dimensions = (np.argwhere(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0] + 1)
        # the output of this is np.int64. We want a native Python int
        # to ensure downstream UMAP implementations work
        # the CPU one would, but the GPU one complains if num_dimensions is not native int 
        num_dimensions=num_dimensions.item()

        if self.train_size > input_feats.shape[0]:
            self.train_size = input_feats.shape[0]
        np.random.seed(0)
        sampled_input_feats = input_feats[np.random.choice(input_feats.shape[0], self.train_size, replace=False)]
        features_transposed = self.features.T
        np.random.seed(0)
        self.sampled_features = features_transposed[np.random.choice(features_transposed.shape[0],
                                                                     self.train_size, replace=False)]
        logging.info('Randomly sampled **{} minutes**... '.format( self.train_size / 600))

        if self.use_gpu:
            print(f"Running cuml (GPU) UMAP")
            self.model=cuml_umap(data=sampled_input_feats, n_components=num_dimensions, **kwargs)
        else:
            print(f"Running umap-learn (CPU) UMAP")
            self.model=cpu_umap(data=sampled_input_feats, n_components=num_dimensions, **kwargs)
        
        self.sampled_embeddings=self.model.embedding_


        logging.info(
            'Done non-linear embedding of {} instances from **{}** D into **{}** D.'.format(
                *self.sampled_features.shape, self.sampled_embeddings.shape[1]))
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_embeddings.sav'))), 'wb') as f:
            joblib.dump([self.sampled_features, self.sampled_embeddings], f)
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_UMAP_model.sav'))), 'wb') as f:
            joblib.dump(self.model, f)


    def main(self, n_jobs=1, **kwargs):
        try:
            [self.sampled_features, self.sampled_embeddings] = load_embeddings_(self.working_dir, self.prefix)
            self.model = load_umap_model_(self.working_dir, self.prefix)
            logging.info('**_CHECK POINT_**: Done non-linear transformation of **{}** instances '
                        'from **{}** D into **{}** D. Move on to __Identify and '
                        'tweak number of clusters__'.format(*self.sampled_features.shape, self.sampled_embeddings.shape[1]))
        except FileNotFoundError:
            self.subsample(fraction=self.fraction)
            self.compute_features(n_jobs=n_jobs, **kwargs)