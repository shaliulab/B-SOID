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
    window_from_framerate,
)

from .umap_implementations import (
    cpu_umap,
    cuml_umap,
)

def bsoid_extract_xy(dxy_r, weights):

    dxy_eu=np.linalg.norm(dxy_r, axis=2)
    dxy_feat = scipy.ndimage.convolve1d(dxy_eu, weights, axis=0, mode="constant", cval=0).T
    return dxy_feat


def bsoid_extract_angular(dxy_r, weights):

    b_3d=np.concatenate([
        dxy_r[1:],
        dxy_r[1:,:,0][:,:,np.newaxis]*0
    ],axis=2)
    
    a_3d=np.concatenate([
        dxy_r[:-1],
        dxy_r[:-1:,:,0][:,:,np.newaxis]*0
    ],axis=2)

    cross_product=np.cross(b_3d, a_3d)
    
    ang_rad=np.arctan2(
        np.linalg.norm(cross_product, axis=2),
        np.einsum('ijk,ijk->ij', dxy_r[:-1], dxy_r[1:])
    )

    to_degs = np.dot(
        np.sign(cross_product[:, :, 2]), 180
    ) / np.pi

    ang=ang_rad*to_degs
    ang_feat = scipy.ndimage.convolve1d(ang, weights, axis=0, mode="constant", cval=0).T


    return ang_feat
    
def bsoid_extract(dataset, window, stride, my_bar=None):
    """
    Compute spatio-temporal features from the pose estimate

    Args:

        dataset: np.array of shape timepoints x (body_parts x 2)
        window: size of window in the timeseries which will be used to average the raw estimates
        (i.e. all timepoints in the same window get the same estimate, which will be their mean)
        Afterwards, the same window size will be used to aggregate features (window_size x n_features -> 1 x n_features).
            Distance features will be averaged over each window 
            Angular features (within same frame) will be summed over each window
            Displacement features (over time) will be summed over each window
    
    """
    # Create a window of weights - in the case of a simple moving average, the weights are all 1
    weights = np.ones(window) / window

    time_feats=np.stack([dataset[:, i:i+2] for i in range(0, dataset.shape[1], 2)], axis=2)
    space_feats=np.stack([[
        dataset[:, j:j+2],
        dataset[:, i:i+2]
    ] for i, j in itertools.combinations(range(0, dataset.shape[1], 2), 2) ], axis=2)

    disp_r = np.linalg.norm(np.diff(time_feats, axis=0), axis=1)

    # feats x timepoints -1
    disp_feat = scipy.ndimage.convolve1d(disp_r, weights, axis=0, mode="constant", cval=0).T

    dxy_r=np.diff(space_feats, axis=0)[0]
    
    # feats x timepoints -1
    ang_feat = bsoid_extract_angular(dxy_r, weights)
    
    # feats x timepoints   
    dxy_feat=bsoid_extract_xy(dxy_r, weights)


    features=np.vstack((dxy_feat[:, 1:], ang_feat, disp_feat))
    if my_bar is not None:
        my_bar.update(1)
        
    start_pos = dxy_feat.shape[0]

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
    
    feats_windows=feats_windows.reshape((feats_windows.shape[0], stride, -1), order="A")

    feats_windows[:start_pos, -padding_size:,-1]=feats_windows[
        :start_pos,
        (feats_windows.shape[1]-stride):(feats_windows.shape[1]-padding_size),
        -1
    ].mean(axis=1).reshape((-1, 1))
    feats_windows[start_pos:, -padding_size:, -1]=0

    
    f_integrated=np.concatenate([
        feats_windows[:start_pos].mean(axis=1),
        feats_windows[start_pos:].sum(axis=1)
    ])[:,:-1]
    
    scaler = StandardScaler()
    scaler.fit(f_integrated.T)
    scaled_features = scaler.transform(f_integrated.T).T

    # dimensions x timepoints!

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
            self.sampled_embeddings=cuml_umap(data=sampled_input_feats, n_components=num_dimensions, **kwargs)
        else:
            print(f"Running umap-learn (CPU) UMAP")
            self.sampled_embeddings=cpu_umap(data=sampled_input_feats, n_components=num_dimensions, **kwargs)


        logging.info(
            'Done non-linear embedding of {} instances from **{}** D into **{}** D.'.format(
                *self.sampled_features.shape, self.sampled_embeddings.shape[1]))
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_embeddings.sav'))), 'wb') as f:
            joblib.dump([self.sampled_features, self.sampled_embeddings], f)

    def main(self, n_jobs=1, **kwargs):
        try:
            [self.sampled_features, self.sampled_embeddings] = load_embeddings_(self.working_dir, self.prefix)
            logging.info('**_CHECK POINT_**: Done non-linear transformation of **{}** instances '
                        'from **{}** D into **{}** D. Move on to __Identify and '
                        'tweak number of clusters__'.format(*self.sampled_features.shape, self.sampled_embeddings.shape[1]))
        except FileNotFoundError:
            self.subsample(fraction=self.fraction)
            self.compute_features(n_jobs=n_jobs, **kwargs)