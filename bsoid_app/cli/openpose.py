import logging
import glob
import os.path


def preprocess_openpose_json():
    raise NotImplementedError()

    data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.json')
    file0_df = read_json_single(data_files[0])
    file0_array = np.array(file0_df)
    p = ([*file0_array[0, 1:-1:3]], [*file0_array[0, 1:-1:3]])
    for a in p:
        index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
        if not index in self.pose_chosen:
            self.pose_chosen += index
    self.pose_chosen.sort()

    logging.info(str.join('', ('Preprocessing... Here is a random fact: ', funfacts)))
    for i, fd in enumerate(self.data_directories):
        f = get_filenamesjson(self.root_path, fd)
        json2csv_multi(f)
        filename = f[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
        file_j_df = pd.read_csv(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')),
                                low_memory=False)
        file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
        self.raw_input_data.append(file_j_df)
        self.sub_threshold.append(p_sub_threshold)
        self.processed_input_data.append(file_j_processed)
        self.input_filenames.append(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')))
    with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
        joblib.dump(
            [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
        )
    logging.info('Processed a total of **{}** .{} files, and compiled into a '
            '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                        np.array(self.processed_input_data).shape))
