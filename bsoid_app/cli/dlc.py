import glob
import pandas as pd
import numpy as np
import logging


def preprocess_dlc_csv():
    raise NotImplementedError()
    data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.csv')
    file0_df = pd.read_csv(data_files[0], low_memory=False)
    file0_array = np.array(file0_df)
    p = st.multiselect('Identified __pose__ to include:', [*file0_array[0, 1:-1:3]], [*file0_array[0, 1:-1:3]])
    for a in p:
        index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
        if not index in self.pose_chosen:
            self.pose_chosen += index
    self.pose_chosen.sort()
    if st.button("__Preprocess__"):
        funfacts = randfacts.getFact()
        st.info(str.join('', ('Preprocessing... Here is a random fact: ', funfacts)))
        for i, fd in enumerate(self.data_directories):  # Loop through folders
            f = get_filenames(self.root_path, fd)[:3]
            my_bar = st.progress(0)
            for j, filename in enumerate(f):
                file_j_df = pd.read_csv(filename, low_memory=False)
                file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                self.raw_input_data.append(file_j_df)
                self.sub_threshold.append(p_sub_threshold)
                self.processed_input_data.append(file_j_processed)
                self.input_filenames.append(filename)
                my_bar.progress(round((j + 1) / len(f) * 100))
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
            joblib.dump(
                [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                    self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
            )
        st.info('Processed a total of **{}** .{} files, and compiled into a '
                '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                            np.array(self.processed_input_data).shape))
        st.balloons()
        
def preprocess_dlc_h5():
    raise NotImplementedError()
    data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.h5')
    file0_df = pd.read_hdf(data_files[0], low_memory=False)
    p = st.multiselect('Identified __pose__ to include:',
                        [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])],
                        [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])])
    for a in p:
        index = [i for i, s in enumerate(np.array(file0_df.columns.get_level_values(1))) if a in s]
        if not index in self.pose_chosen:
            self.pose_chosen += index
    self.pose_chosen.sort()
    if st.button("__Preprocess__"):
        funfacts = randfacts.getFact()
        st.info(str.join('', ('Preprocessing... Here is a random fact: ', funfacts)))
        for i, fd in enumerate(self.data_directories):
            f = get_filenamesh5(self.root_path, fd)
            my_bar = st.progress(0)
            for j, filename in enumerate(f):
                file_j_df = pd.read_hdf(filename, low_memory=False)
                file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen)
                self.raw_input_data.append(file_j_df)
                self.sub_threshold.append(p_sub_threshold)
                self.processed_input_data.append(file_j_processed)
                self.input_filenames.append(filename)
                my_bar.progress(round((j + 1) / len(f) * 100))
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
            joblib.dump(
                [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                    self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
            )
        st.info('Processed a total of **{}** .{} files, and compiled into a '
                '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                            np.array(self.processed_input_data).shape))
        st.balloons()