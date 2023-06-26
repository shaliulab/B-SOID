import os
import logging

import joblib
import numpy as np

from bsoid_app.cli import visuals
from bsoid_app.bsoid_utilities.utils import load_clusters_
from .hdbdscan_implementations import (
    cpu_hdbscan,
    gpu_hdbscan,
    all_points_membership_vectors
)


class cluster:

    def __init__(self, working_dir, prefix, sampled_embeddings, cluster_range=[], useGPU=True):
        logging.info('IDENTIFY AND TWEAK NUMBER OF CLUSTERS.')
        self.working_dir = working_dir
        self.prefix = prefix
        self.sampled_embeddings = sampled_embeddings
        self.cluster_range = cluster_range
        self.min_cluster_size = []
        self.assignments = []
        self.assign_prob = []
        self.soft_assignments = []
        self.use_gpu=useGPU


    def hierarchy(self, n_jobs=1, cluster_range=None):
        if cluster_range is None:
            cluster_range=self.cluster_range
        
        logging.info('Identifying clusters using {n_jobs} jobs ...')
        max_num_clusters = -np.infty
        self.min_cluster_size = np.linspace(cluster_range[0], cluster_range[1], 25)

        if self.use_gpu:
            n_jobs=1
            hdbscan_call=gpu_hdbscan
        else:
            hdbscan_call=cpu_hdbscan

        hierarchies = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                hdbscan_call
            )(
                c * 0.01, self.sampled_embeddings
            )
            for c in self.min_cluster_size
        )

        for learned_hierarchy in hierarchies:
           num_clusters=len(np.unique(learned_hierarchy.labels_))
           if num_clusters > max_num_clusters:
               retained_hierarchy=learned_hierarchy
               max_num_clusters=num_clusters

        self.assignments = retained_hierarchy.labels_
        self.assign_prob = all_points_membership_vectors(retained_hierarchy)
        self.soft_assignments = np.argmax(self.assign_prob, axis=1)
        logging.info(
            'Done assigning labels for **{}** instances ({} minutes) '
            'in **{}** D space'.format(
                self.assignments.shape,
                round(self.assignments.shape[0] / 600),
                self.sampled_embeddings.shape[1]
            )
        )

    def show_classes(self):
        logging.info(
            'Showing {}% data that were confidently assigned.'.format(
                round(self.assignments[self.assignments >= 0].shape[0] / self.assignments.shape[0] * 100)
            )
        )

        fig1, plt1 = visuals.plot_classes(
            self.sampled_embeddings[self.assignments >= 0],
            self.assignments[self.assignments >= 0]
        )

        plt1.suptitle('HDBSCAN assignment')
        # col1, col2 = st.beta_columns([2, 2])
        # col1.pyplot(fig1)
        return fig1, plt1

    def save(self):
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_clusters.sav'))), 'wb') as f:
            joblib.dump([self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments], f)

    def main(self, n_jobs=1):
        try:
            [self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments] = \
                load_clusters_(self.working_dir, self.prefix)
            logging.info(
                '**_CHECK POINT_**: Done assigning labels for **{}** instances in **{}** D space. Move on to __create '
                'a model__.'.format(self.assignments.shape, self.sampled_embeddings.shape[1]))
            logging.info('Your last saved run range was __{}%__ to __{}%__'.format(self.min_cluster_size[0],
                                                                                  self.min_cluster_size[-1]))
            self.hierarchy(n_jobs=n_jobs)
            self.save()
        except (AttributeError, FileNotFoundError) as e:
            self.hierarchy(n_jobs=n_jobs)
            self.save()



