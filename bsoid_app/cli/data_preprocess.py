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

    def __init__(
            self, prefix, root_path=DEFAULT_BSOID_DATA, software=BSOID_POSE, ftype="h5",
            framerate=DEFAULT_FRAMERATE, pose_chosen=None, subfolders=None, no_files=None
        ):

        self.pose_chosen = []
        self.input_filenames = []
        self.raw_input_data = []
        self.processed_input_data = []
        self.sub_threshold = []
        self.software = software
        self.ftype = ftype
        self.no_files=no_files
        self.pose_chosen=pose_chosen
        
        self.root_path = root_path
        try:
            os.listdir(self.root_path)
        except FileNotFoundError:
            raise Exception('No such directory')

        logging.info('Your will be training on *{}* data file containing sub-directories.'.format(subfolders))
        self.data_directories=subfolders


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

    def compile_data(self, frame_numbers=None, n_jobs=1):
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
                no_files=self.no_files,
                frame_numbers=frame_numbers,
                pose_chosen=self.pose_chosen
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
        
        indexed_data = [
            (
                os.path.basename(input_filenames[i]),
                input_filenames[i],
                raw_input_data[i],
                sub_threshold[i],
                processed_input_data[i]
            )
            for i in range(len(processed_input_data))
        ]
        # maybe not needed, but still good to have,
        # to ensure that the chunks are added in the right order
        # i.e. 2 after 1 and so on
        indexed_data=sorted(indexed_data, key=lambda x: x[0])

        input_filenames=[e[1] for e in indexed_data]
        raw_input_data=[e[2] for e in indexed_data]
        sub_threshold=[e[3] for e in indexed_data]
        processed_input_data=[e[4] for e in indexed_data]

        return raw_input_data, sub_threshold, processed_input_data, input_filenames


    def show_bar(self):
        visuals.plot_bar(self.sub_threshold)

    def show_data_table(self):
        visuals.show_data_table(self.raw_input_data, self.processed_input_data)
