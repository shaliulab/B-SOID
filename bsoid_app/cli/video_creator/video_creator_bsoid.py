import logging
import base64

import ffmpeg
import h5py
import math

from bsoid_app.bsoid_utilities.bsoid_classification import bsoid_predict
from bsoid_app.cli.extract_features import bsoid_extract
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *
from bsoid_app.bsoid_utilities.videoprocessing import *
from bsoid_app.bsoid_utilities.utils import (
    window_from_framerate,
)

def selected_file(d_file):
    return d_file


def selected_vid(vid_file):
    return vid_file


class creator:

    def __init__(
        self, root_path, data_directories, processed_input_data,
        pose_chosen, working_dir, prefix, framerate, clf, input_filenames,
        filetype="h5", min_time=200,
    ):
        self.root_path = root_path
        self.data_directories = data_directories
        self.processed_input_data = processed_input_data
        self.pose_chosen = pose_chosen
        self.working_dir = working_dir
        self.prefix = prefix
        self.framerate = framerate
        self.clf = clf
        self.input_filenames = input_filenames

        self.vid_file = []
        self.frame_dir = []
        self.filetype = []
        self.width = []
        self.height = []
        self.bit_rate = []
        self.num_frames = []
        self.avg_frame_rate = []
        self.shortvid_dir = []
        self.min_frames = []
        self.number_examples = None
        self.out_fps = []
        self.file_j_processed = []
        self.filetype = input_filenames[0].split(".")[-1]
        self.min_time = min_time#st.number_input('Enter minimum time for bout in ms:', value=200)

    def setup(self, d_file, vid_file, playback_speed = 1, number_examples=10):
        self.d_file=d_file
        self.vid_file=vid_file
        self.file_directory=os.path.dirname(d_file).replace(self.root_path, "")
        self.number_examples = number_examples

        self.vid_file = selected_vid(vid_file)
        if self.filetype == 'csv' or self.filetype == 'h5':
            logging.info('You have selected **{}** matching **{}**.'.format(self.vid_file, self.d_file))
            csvname = os.path.basename(self.d_file).rpartition('.')[0]
        else:
            logging.info(
                'You have selected **{}** matching **{}** json directory.'.format(self.vid_file, self.file_directory))
            csvname = os.path.basename(self.file_directory)
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/pngs')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/pngs', '/', csvname)))
        except FileExistsError:
            pass

        self.frame_dir = str.join('', (self.root_path, self.file_directory, '/pngs', '/', csvname))
        logging.info('Created {} as your **video frames** directory.'.format(self.frame_dir, self.vid_file))
        probe = ffmpeg.probe(self.vid_file)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.num_frames = int(video_info['nb_frames'])
        self.bit_rate = int(video_info['bit_rate'])
        self.avg_frame_rate = round(
            int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/mp4s')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/mp4s', '/', csvname)))
        except FileExistsError:
            pass
        self.shortvid_dir = str.join('', (self.root_path, self.file_directory, '/mp4s', '/', csvname))
        logging.info('Created {} as your **behavioral snippets** directory.'.format(self.shortvid_dir, self.vid_file))
        self.min_frames = round(float(self.min_time) * 0.001 * float(self.framerate))
        logging.info('Entered **{} ms** as minimum duration per bout, '
                    'which is equivalent to **{} frames**.'.format(self.min_time, self.min_frames))
        logging.info(
            'Your will obtain a maximum of **{}** non-repeated output examples per group.'.format(self.number_examples))
        
        self.out_fps = int(float(playback_speed) * float(self.framerate))
        logging.info('Playback at **{} x speed** (rounded to {} FPS).'.format(playback_speed, self.out_fps))

    def frame_extraction(self):
        logging.info(
            'Start frame extraction for {} frames '
            'at {} frames per second'.format(self.num_frames, self.avg_frame_rate)
        )

        logging.info('Extracting frames from the video... ')
        try:
            (ffmpeg.input(self.vid_file)
                .filter('fps', fps=self.avg_frame_rate)
                .output(str.join('', (self.frame_dir, '/frame%01d.png')), video_bitrate=self.bit_rate,
                        s=str.join('', (str(int(self.width * 0.5)), 'x', str(int(self.height * 0.5)))),
                        sws_flags='bilinear', start_number=0)
                .run(capture_stdout=True, capture_stderr=True))
            logging.info('Done extracting **{}** frames from video **{}**.'.format(self.num_frames, self.vid_file))
        except ffmpeg.Error as e:
            logging.error('stdout:', e.stdout.decode('utf8'))
            logging.error('stderr:', e.stderr.decode('utf8'))
        logging.info('Done extracting {} frames from {}'.format(self.num_frames, self.vid_file))

    def create_videos(self):

        if True:
            try:
                for file_name in glob.glob(self.shortvid_dir + "/*"):
                    os.remove(file_name)
            except:
                pass
        if True:
            if self.filetype == 'csv' or self.filetype == 'json':
                file_j_df = pd.read_csv(
                    self.d_file,
                    low_memory=False)
                file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
            elif self.filetype == 'h5':
                try:
                    file_j_df = pd.read_hdf(self.d_file, low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen)
                except:
                    logging.info('Detecting a SLEAP .h5 file...')
                    file_j_df = h5py.File(self.d_file, 'r')
                    file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
            
            self.file_j_processed = [file_j_processed]
            labels_fs = []
            fs_labels = []
            logging.info('Predicting labels... ')

            for dataset in tqdm(self.file_j_processed):
                labels._fs.append(
                    self.inference(self.clf, dataset, framerate=self.framerate)
                )
    
            logging.info('Frameshifted arrangement of labels... ')
            for k, _ in enumerate(labels_fs):
                labels_fs2 = []
                for l in range(math.floor(self.framerate / 10)):
                    labels_fs2.append(labels_fs[k][l])
                fs_labels.append(np.array(labels_fs2).flatten('F'))
            logging.info('Done frameshift-predicting **{}**.'.format(self.d_file))
            create_labeled_vid(fs_labels[0], int(self.min_frames), int(self.number_examples), int(self.out_fps),
                                self.frame_dir, self.shortvid_dir)

            logging.info('**_CHECK POINT_**: Done generating video snippets. Move on to '
                        '__Predict old/new files using a model__.')

    @staticmethod
    def inference(model, dataset, framerate):
        window = window_from_framerate(framerate)
        feats_new, _ = bsoid_extract(dataset=dataset, window=window)

        labels = bsoid_predict(feats_new, model)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        return labels_pad.astype(int)


    def show_snippets(self):
        video_bytes = []
        grp_names = []
        files = []
        for file in os.listdir(self.shortvid_dir):
            files.append(file)
        sort_nicely(files)
        logging.info('Creating gifs from mp4s...')
        for file in files:
            if file.endswith('0.mp4'):
                try:
                    example_vid_file = open(os.path.join(
                        str.join('', (self.shortvid_dir, '/', file.partition('.')[0], '.gif'))), 'rb')
                except FileNotFoundError:
                    convert2gif(str.join('', (self.shortvid_dir, '/', file)), TargetFormat.GIF)
                    example_vid_file = open(os.path.join(
                        str.join('', (self.shortvid_dir, '/', file.partition('.')[0], '.gif'))), 'rb')
                contents = example_vid_file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                video_bytes.append(data_url)
                grp_names.append('{}'.format(file.partition('.')[0]))
        
        return video_bytes, grp_names
        

    def main(self):
        self.setup()
        self.create_videos()
        if st.checkbox(
            "Show a collage of example group? "
            "This could take some time for gifs conversions.".format(
                self.shortvid_dir
            ),
        False, key='vs'):
            self.show_snippets()

