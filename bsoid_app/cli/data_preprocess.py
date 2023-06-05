import logging
import os
from datetime import date

import h5py
import joblib
import randfacts
import streamlit as st

from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *
from .sleap import preprocess_sleap_h5

DEFAULT_BSOID_DATA=os.environ["BSOID_DATA"]
BSOID_POSE=('SLEAP', 'DeepLabCut', 'OpenPose')
DEFAULT_FRAMERATE=150

class preprocess:

    def __init__(self, prefix, root_path=DEFAULT_BSOID_DATA, software=BSOID_POSE, ftype="h5", framerate=DEFAULT_FRAMERATE, no_subfolders=None, no_files=None):

        self.pose_chosen = []
        self.input_filenames = []
        self.raw_input_data = []
        self.processed_input_data = []
        self.sub_threshold = []
        self.software = software
        self.ftype = ftype
        self.no_files=no_files
        # logging.info('Currently only supporting {} type files'.format(self.ftype))
        # loggin.info('Currently only supporting {} type files'.format(self.ftype))
        
        self.root_path = root_path
        try:
            os.listdir(self.root_path)
        except FileNotFoundError:
            raise Exception('No such directory')

        self.data_directories = []

        logging.info('Your will be training on *{}* data file containing sub-directories.'.format(no_subfolders))
        dirs=(sorted(os.listdir(self.root_path)))
        if no_subfolders is None:
            no_subfolders=len(dirs)

        file_counter=0
        for i in range(no_subfolders):
            d = dirs[i]
            if not d in self.data_directories:
                self.data_directories.append(d)


        logging.info('You have selected **{}** as your _sub-directory(ies)_.'.format(self.data_directories))
        logging.info('Average video frame-rate for xxx.{} pose estimate files.'.format(self.ftype))
        self.framerate = framerate
        logging.info('You have selected **{} frames per second**.'.format(self.framerate))
        self.working_dir = os.path.join(root_path, "output")

        try:
            os.listdir(self.working_dir)
            logging.info('You have selected **{}** for B-SOiD working directory.'.format(self.working_dir))
        except FileNotFoundError:
            logging.error('Cannot access working directory, was there a typo or did you forget to create one?')
        self.prefix=prefix

    def compile_data(self, n_jobs=1):
        logging.info('Identify pose to include in clustering.')
        if self.software == 'DeepLabCut' and self.ftype == 'csv':
            raw_input_data, sub_threshold, processed_input_data, input_filenames = preprocess_dlc_csv()
        elif self.software == 'DeepLabCut' and self.ftype == 'h5':
            raw_input_data, sub_threshold, processed_input_data, input_filenames = preprocess_dlc_h5()

        elif self.software == 'SLEAP' and self.ftype == 'h5':
            logging.info("Preprocessing sleap h5 files")
            raw_input_data, sub_threshold, processed_input_data, input_filenames, self.pose_chosen = preprocess_sleap_h5(
                self.root_path,
                self.data_directories,
                n_jobs=n_jobs,
                no_files=self.no_files
            )

        elif self.software == 'OpenPose' and self.ftype == 'json':
            raw_input_data, sub_threshold, processed_input_data, input_filenames = preprocess_openpose_json()

        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
            joblib.dump(
                [
                    self.root_path, self.data_directories, self.framerate, self.pose_chosen, input_filenames,
                    raw_input_data, np.array(processed_input_data), sub_threshold
                ], f
            )

        logging.info(
            'Processed a total of **{}** .{} files, and compiled into a '
            '**{}** data list.'.format(
                len(processed_input_data), self.ftype,
                np.array(processed_input_data).shape
                )
            )
        return raw_input_data, sub_threshold, processed_input_data, input_filenames


    def show_bar(self):
        visuals.plot_bar(self.sub_threshold)

    def show_data_table(self):
        visuals.show_data_table(self.raw_input_data, self.processed_input_data)
