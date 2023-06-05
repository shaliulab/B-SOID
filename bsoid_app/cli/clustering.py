import os
import logging

import hdbscan
import joblib
import numpy as np
from tqdm import tqdm

from bsoid_app.config import *
from bsoid_app.cli import visuals
from bsoid_app.bsoid_utilities.utils import load_clusters_


class cluster:

    def __init__(self, working_dir, prefix, sampled_embeddings, cluster_range=[]):
        logging.info('IDENTIFY AND TWEAK NUMBER OF CLUSTERS.')
        self.working_dir = working_dir
        self.prefix = prefix
        self.sampled_embeddings = sampled_embeddings
        self.cluster_range = cluster_range
        self.min_cluster_size = []
        self.assignments = []
        self.assign_prob = []
        self.soft_assignments = []

    def hierarchy(self, cluster_range=None):
        if cluster_range is None:
            cluster_range=self.cluster_range
        
        logging.info('Identifying...')
        max_num_clusters = -np.infty
        num_clusters = []
        self.min_cluster_size = np.linspace(cluster_range[0], cluster_range[1], 25)

        for min_c in tqdm(self.min_cluster_size):
            learned_hierarchy = hdbscan.HDBSCAN(
                prediction_data=True, min_cluster_size=int(round(min_c * 0.01 * self.sampled_embeddings.shape[0])),
                **HDBSCAN_PARAMS).fit(self.sampled_embeddings)
            num_clusters.append(len(np.unique(learned_hierarchy.labels_)))
            if num_clusters[-1] > max_num_clusters:
                max_num_clusters = num_clusters[-1]
                retained_hierarchy = learned_hierarchy
        self.assignments = retained_hierarchy.labels_
        self.assign_prob = hdbscan.all_points_membership_vectors(retained_hierarchy)
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

    def main(self):
        try:
            [self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments] = \
                load_clusters_(self.working_dir, self.prefix)
            logging.info(
                '**_CHECK POINT_**: Done assigning labels for **{}** instances in **{}** D space. Move on to __create '
                'a model__.'.format(self.assignments.shape, self.sampled_embeddings.shape[1]))
            logging.info('Your last saved run range was __{}%__ to __{}%__'.format(self.min_cluster_size[0],
                                                                                  self.min_cluster_size[-1]))
            self.hierarchy()
            self.save()
        except (AttributeError, FileNotFoundError) as e:
            self.hierarchy()
            self.save()



