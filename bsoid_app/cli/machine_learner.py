import logging
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.utils import load_classifier_


class protocol:

    def __init__(self, working_dir, prefix, features, sampled_features, assignments):
        logging.info('CREATE A MODEL')
        self.working_dir = working_dir
        self.prefix = prefix
        self.features = features
        self.sampled_features = sampled_features
        self.assignments = assignments
        self.part = 0.2
        self.it = 10
        self.x_test = []
        self.y_test = []
        self.validate_clf = []
        self.clf = []
        self.validate_score = []
        self.predictions = []

    def randomforest(self, n_jobs=1):
        try:
            x = self.sampled_features[self.assignments >= 0, :]
            y = self.assignments[self.assignments >= 0]
            x_train, self.x_test, y_train, self.y_test = train_test_split(x, y.T, test_size=self.part, random_state=42)
            logging.info('Training random forest classifier on randomly partitioned')

            self.validate_clf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
            self.validate_clf.fit(x_train, y_train)
            self.clf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
            self.clf.fit(x, y.T)
            self.predictions = self.clf.predict(self.features.T)
            logging.info(
                'Done training random forest classifier mapping '
                '**{}** features to **{}** assignments.'.format(
                    self.features.T.shape, self.predictions.shape
                )
            )
            self.validate_score = cross_val_score(self.validate_clf, self.x_test, self.y_test, cv=self.it, n_jobs=-1)
            with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_randomforest.sav'))), 'wb') as f:
                joblib.dump([self.x_test, self.y_test, self.validate_clf, self.clf,
                                self.validate_score, self.predictions], f)
        except AttributeError:
            logging.error('Sometimes this takes a bit to update, recheck identify clusters (previous step) '
                        'and rerun this in 30 seconds.')

    def show_confusion_matrix(self):
        fig = visuals.plot_confusion(self.validate_clf, self.x_test, self.y_test)
        logging.info('To improve, either _increase_ minimum cluster size, or include _more data_')
        return fig

    def show_crossval_score(self):
        fig, plt = visuals.plot_accuracy(self.validate_score)
        logging.info('To improve, either _increase_ minimum cluster size, or include _more data_')
        return fig, plt


    def load_classifier(self):
        [self.x_test, self.y_test, self.validate_clf, self.clf, self.validate_score, self.predictions] = load_classifier_(self.working_dir, self.prefix)

    # def main(self):
    #     try:
    #         [self.x_test, self.y_test, self.validate_clf, self.clf, self.validate_score, self.predictions] = \
    #             load_classifier(self.working_dir, self.prefix)
    #         logging.info('**_CHECK POINT_**: Done training random forest classifier '
    #                     'mapping **{}** features to **{}** assignments. Move on to '
    #                     '__Generate video snippets for interpretation__.'.format(
    #                         self.features.shape[0],
    #                         self.predictions.shape[0]
    #                     )
    #                 )
