import logging
import numpy as np
from tqdm import tqdm
import itertools
import joblib
import os.path
import math
import scipy.ndimage

import umap
from psutil import virtual_memory
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bsoid_app.bsoid_utilities.utils import (
    load_embeddings_,
    load_feats_
)
from bsoid_app.bsoid_utilities.likelihoodprocessing import boxcar_center
from bsoid_app.config import UMAP_PARAMS


class extract:

    def __init__(self, working_dir, prefix, processed_input_data, framerate, fraction=1.0, stride=None):
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
        self.stride=stride


    def subsample(self, fraction):
        if self.stride is None:
            # downsample to 1th of framerate
            self.stride=self.framerate / 10

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

    def compute_features(self, n_jobs=1):
        logging.info('Extracting...')

        try:
            [self.features, self.scaled_features] = load_feats_(self.working_dir, self.prefix)
        except FileNotFoundError:
            self.compute_features_(n_jobs=n_jobs)

        self.save(self.features, self.scaled_features)
        self.learn_embeddings()


    @staticmethod
    def compute_distances_single_data(dataset, window, my_bar=None):
        time_feats=np.stack([dataset[:, i:i+2] for i in range(0, dataset.shape[1], 2)], axis=2)
        space_feats=np.stack([[
            dataset[:, j:j+2],
            dataset[:, i:i+2]
        ] for i, j in itertools.combinations(range(0, dataset.shape[1], 2), 2) ], axis=2)

        disp_r = np.linalg.norm(np.diff(time_feats, axis=0), axis=1)
        dxy_r=np.diff(space_feats, axis=0)[0]

        # Create a window of weights - in the case of a simple moving average, the weights are all 1
        weights = np.ones(window) / window
        # Convolve your data with the window of weights
        disp_feat = scipy.ndimage.convolve1d(disp_r, weights, axis=0, mode="constant", cval=0).T

        dxy_eu=np.linalg.norm(dxy_r, axis=2)
        dxy_feat = scipy.ndimage.convolve1d(dxy_eu, weights, axis=0, mode="constant", cval=0).T

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
        
        features=np.vstack((dxy_feat[:, 1:], ang_feat, disp_feat))
        if my_bar is not None:
            my_bar.update(1)

        return features, dxy_feat.shape[0]

    def compute_distances(self, datasets, n_jobs, **kwargs):
        window = np.int(np.round(0.05 / (1 / self.framerate)) * 2 - 1)

        Output = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                self.compute_distances_single_data
            )(
                datasets[n], window, **kwargs
            )
            for n, _ in enumerate(datasets) 
        )
    
        features=[]
        start_pos=None
        for features_single_data, start_pos in  Output:
            features.append(features_single_data)
        
        return features, start_pos


    def compute_features_(self, n_jobs):
        if n_jobs==1:
            my_bar = tqdm(total=len(self.processed_input_data))
        else:
            my_bar=None
        features, start_pos=self.compute_distances(datasets=self.processed_input_data, n_jobs=n_jobs, my_bar=my_bar)


        for m, _ in enumerate(features):
            feats_windows=np.concatenate([
                features[m], features[m][:,-(start_pos-1):].mean(axis=1).reshape((-1, 1))
            ], axis=1).astype(np.int32)
            feats_windows=feats_windows.reshape((feats_windows.shape[0], self.stride, -1), order="A")

            f_integrated=np.concatenate([
                feats_windows[:start_pos].mean(axis=1),
                feats_windows[start_pos:].sum(axis=1)
            ])[:,:-1]
            
            if m > 0:
                self.features = np.concatenate((self.features, f_integrated), axis=1)
                scaler = StandardScaler()
                scaler.fit(f_integrated.T)
                scaled_f_integrated = scaler.transform(f_integrated.T).T
                self.scaled_features = np.concatenate((self.scaled_features, scaled_f_integrated), axis=1)
            else:
                self.features = f_integrated
                scaler = StandardScaler()
                scaler.fit(f_integrated.T)
                scaled_f_integrated = scaler.transform(f_integrated.T).T
                self.scaled_features = scaled_f_integrated
        
        self.features = np.array(self.features)
        self.scaled_features = np.array(self.scaled_features)
        


    def save(self, features, scaled_features):
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_feats.sav'))), 'wb') as filehandle:
            joblib.dump([features, scaled_features], filehandle)

        logging.info('Done extracting features from a total of **{}** training data files. '
                'Now reducing dimensions...'.format(len(self.processed_input_data)))

    def learn_embeddings(self):
        input_feats = self.scaled_features.T
        pca = PCA()
        pca.fit(self.scaled_features.T)
        num_dimensions = np.argwhere(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0] + 1
        if self.train_size > input_feats.shape[0]:
            self.train_size = input_feats.shape[0]
        np.random.seed(0)
        sampled_input_feats = input_feats[np.random.choice(input_feats.shape[0], self.train_size, replace=False)]
        features_transposed = self.features.T
        np.random.seed(0)
        self.sampled_features = features_transposed[np.random.choice(features_transposed.shape[0],
                                                                     self.train_size, replace=False)]
        logging.info('Randomly sampled **{} minutes**... '.format(self.stride*self.train_size / (60 * self.framerate)))
        mem = virtual_memory()
        available_mb = mem.available >> 20
        logging.info('You have {} MB RAM 🐏 available'.format(available_mb))
        if available_mb > (sampled_input_feats.shape[0] * sampled_input_feats.shape[1] * 32 * 60) / 1024 ** 2 + 64:
            logging.info('RAM 🐏 available is sufficient')
            try:
                learned_embeddings = umap.UMAP(n_neighbors=60, n_components=num_dimensions,
                                               **UMAP_PARAMS).fit(sampled_input_feats)
            except:
                logging.error('Failed on feature embedding. Try again by unchecking sidebar and rerunning extract features.')
        else:
            logging.info(
                'Detecting that you are running low on available memory for this computation, '
                'setting low_memory so will take longer.')
            try:
                learned_embeddings = umap.UMAP(n_neighbors=60, n_components=num_dimensions, low_memory=True,
                                               **UMAP_PARAMS).fit(sampled_input_feats)
            except:
                logging.error('Failed on feature embedding. Try again by unchecking sidebar and rerunning extract features.')
        self.sampled_embeddings = learned_embeddings.embedding_
        logging.info(
            'Done non-linear embedding of {} instances from **{}** D into **{}** D.'.format(
                *self.sampled_features.shape, self.sampled_embeddings.shape[1]))
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_embeddings.sav'))), 'wb') as f:
            joblib.dump([self.sampled_features, self.sampled_embeddings], f)

    def main(self, n_jobs=1):
        try:
            [self.sampled_features, self.sampled_embeddings] = load_embeddings_(self.working_dir, self.prefix)
            logging.info('**_CHECK POINT_**: Done non-linear transformation of **{}** instances '
                        'from **{}** D into **{}** D. Move on to __Identify and '
                        'tweak number of clusters__'.format(*self.sampled_features.shape, self.sampled_embeddings.shape[1]))
        except FileNotFoundError:
            self.subsample(fraction=self.fraction)
            self.compute_features(n_jobs=n_jobs)