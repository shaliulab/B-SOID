import logging
import base64

print(__name__)

import ffmpeg
import h5py
import math
import shutil

import joblib

from bsoid_app.bsoid_utilities.bsoid_classification import bsoid_predict
from bsoid_app.cli.extract_features import bsoid_extract
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *
from bsoid_app.bsoid_utilities.videoprocessing import *
from bsoid_app.bsoid_utilities.utils import (
    window_from_framerate,
)
from bsoid_app.cli.utils import make_datasets
from imgstore.stores.utils.mixins.extract import _extract_store_metadata

OUTPUT_FOLDER=os.path.join(os.environ["BSOID_DATA"], "output")

def selected_file(d_file):
    return d_file


def selected_vid(vid_file):
    return vid_file


def inference(model, dataset, framerate):
    window = window_from_framerate(framerate)

    feats_new, _ = bsoid_extract(dataset=dataset, window=window, stride=int(framerate/10))
    labels = bsoid_predict([feats_new], model)
    
    for m in range(0, len(labels)):
        labels[m] = labels[m][::-1]
    labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
    for n, l in enumerate(labels):
        labels_pad[n][0:len(l)] = l
        labels_pad[n] = labels_pad[n][::-1]
        if n > 0:
            labels_pad[n][0:n] = labels_pad[n - 1][0:n]
    return labels_pad.astype(int)
    
def ffmpeg_pipeline(vid_file, avg_frame_rate, bit_rate, size, destination, num_frames, start_number=0):

    if bit_rate is None:
        probe = ffmpeg.probe(vid_file)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        bit_rate=int(video_info['bit_rate'])

    try:
        (ffmpeg.input(vid_file)
            .filter('fps', fps=avg_frame_rate)
            .output(destination, video_bitrate=bit_rate,
                    s=size,
                    sws_flags='bilinear', start_number=start_number)
            .run(capture_stdout=True, capture_stderr=True))
        logging.info('Done extracting **{}** frames from video **{}**.'.format(num_frames, vid_file))
    except ffmpeg.Error as e:
        logging.error('stdout:', e.stdout.decode('utf8'))
        logging.error('stderr:', e.stderr.decode('utf8'))
    logging.info('Done extracting {} frames from {}'.format(num_frames, vid_file))


def probe_video(vid_file):
    if os.path.basename(vid_file) == "metadata.yaml":
        metadata = _extract_store_metadata(vid_file)

        height, width = metadata["imgshape"]
        num_frames =  metadata["chunksize"]
        bit_rate = None
        avg_frame_rate = metadata["framerate"]

    else: 
        probe = ffmpeg.probe(vid_file)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        
        width = int(video_info['width'])
        height = int(video_info['height'])
        num_frames = int(video_info['nb_frames'])
        bit_rate = int(video_info['bit_rate'])
        avg_frame_rate = round(
            int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2])
        )

    return width, height, num_frames, bit_rate, avg_frame_rate


class creator:

    def __init__(
        self, root_path,
        pose_chosen, working_dir, prefix, framerate, clf,
        filetype="h5", min_time=200
    ):
        self.root_path = root_path
        self.pose_chosen = pose_chosen
        self.working_dir = working_dir
        self.prefix = prefix
        self.framerate = framerate
        self.clf = clf

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
        self.filetype = filetype
        self.min_time = min_time#st.number_input('Enter minimum time for bout in ms:', value=200)


    def setup(self, identity, experiment, chunks, **kwargs):

        self.d_file = os.path.join(os.environ["BSOID_DATA"], f"{experiment}__{str(identity).zfill(2)}", f"{experiment}__{str(identity).zfill(2)}.h5")
        assert os.path.exists(self.d_file)
        tokens = experiment.split("_")
        self.vid_file = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], tokens[0], tokens[1], "_".join(tokens[2:4]), "flyhostel", "single_animal", str(identity).zfill(3), "metadata.yaml")


        self.frame_dirs=[]
        all_attrs=[]
       
        Output = [
            self._setup(
                root_path=self.root_path,
                d_file=self.d_file, vid_file=self.vid_file,
                framerate=self.framerate, min_time=self.min_time,
                chunks=chunks,
                **kwargs
            )
        ]

        for frame_dir, attrs in Output:
            self.frame_dirs.append(frame_dir)
            all_attrs.append(attrs)



        pngs_dir=os.path.join(self.root_path,  "pngs")
        self.csvname=f"{experiment}_{identity}"
        self.frame_dir=os.path.join(pngs_dir,  self.csvname)

        # if os.path.exists(self.frame_dir):
        #     shutil.rmtree(self.frame_dir)
        
        os.makedirs(self.frame_dir, exist_ok=True)
        for folder in self.frame_dirs:
            files=glob.glob(os.path.join(folder, "*"))
            for file in files:
                dest_file=os.path.join(self.frame_dir, os.path.basename(file))

                if not os.path.exists(dest_file):
                    shutil.move(file, dest_file)
        logging.info(f'Created {self.frame_dir} as your **video frames** directory.')

        return all_attrs


    def create_folders(self, root_path, file_directory, csvname):
        pngs_dir=os.path.join(root_path, file_directory, 'pngs')
        self.frame_dir=os.path.join(pngs_dir,  csvname)
        
        os.makedirs(pngs_dir, exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)


    def _setup(self, root_path, d_file, vid_file, framerate, min_time, playback_speed = 1, number_examples=10, **kwargs):

        file_directory=os.path.dirname(d_file).replace(root_path, "")

        logging.info('You have selected **{}** matching **{}**.'.format(vid_file, d_file))
        csvname = os.path.basename(d_file).rpartition('.')[0]

        self.create_folders(root_path, file_directory, csvname)
        
        width, height, num_frames, bit_rate, avg_frame_rate = probe_video(vid_file)


        min_frames = round(float(min_time) * 0.001 * float(framerate))
        logging.info('Entered **{} ms** as minimum duration per bout, '
                    'which is equivalent to **{} frames**.'.format(min_time, min_frames))
        logging.info(
            'Your will obtain a maximum of **{}** non-repeated output examples per group.'.format(number_examples))
        
        out_fps = int(float(playback_speed) * float(framerate))
        logging.info('Playback at **{} x speed** (rounded to {} FPS).'.format(playback_speed, out_fps))
        self.frame_extraction(self.frame_dir, vid_file, width, height, bit_rate, avg_frame_rate, num_frames, **kwargs)

        attrs={
            "width": width, "height": height,
            "num_frames": num_frames,
            "bit_rate": bit_rate, "avg_frame_rate": avg_frame_rate,
            "out_fps": out_fps, "min_frames": min_frames,
            "number_examples": number_examples
        }

        return self.frame_dir, attrs

    @staticmethod
    def frame_extraction(frame_dir, vid_file, width, height, bit_rate, avg_frame_rate, num_frames, n_jobs=1, chunks=None):
        logging.info(
            'Start frame extraction for {} frames '
            'at {} frames per second'.format(num_frames, avg_frame_rate)
        )

        logging.info('Extracting frames from the video... ')
        destination=os.path.join(frame_dir, f"frame_{os.path.basename(vid_file)}_" + "%05d.png")
        logging.info(f"Saving extracted frames to {destination}")
        size=f"{str(int(width * 0.5))}x{str(int(height * 0.5))}"

        if os.path.basename(vid_file) == "metadata.yaml":
            assert chunks is not None
            chunk_files=[os.path.join(os.path.dirname(vid_file), f"{str(chunk).zfill(6)}.mp4") for chunk in chunks]
            for i, chunk_file in enumerate(chunk_files):
                ffmpeg_pipeline(chunk_file, avg_frame_rate, bit_rate, size, destination, num_frames, start_number=i*45000)


            # joblib.Parallel(n_jobs=n_jobs)(
            #     joblib.delayed(ffmpeg_pipeline)(
            #         vid_file, avg_frame_rate, bit_rate, size, destination, num_frames, start_number=i*45000

            #     )
            #     for i, vid_file in enumerate(vid_files)
            # )             

        else:
            ffmpeg_pipeline(vid_file, avg_frame_rate, bit_rate, size, destination, num_frames, start_number=0)

    def create_videos(
            self, processed_input_data, input_filenames,
            out_fps, number_examples, min_frames,
            **kwargs):

        mp4s_dir=os.path.join(self.root_path,  "mp4s")
        self.shortvid_dir=os.path.join(mp4s_dir,  self.csvname)
        os.makedirs(mp4s_dir, exist_ok=True)
        
        if os.path.exists(self.shortvid_dir):
            shutil.rmtree(self.shortvid_dir)
        os.makedirs(self.shortvid_dir, exist_ok=True)
        
        # logging.info('Created {} as your **behavioral snippets** directory.'.format(self.shortvid_dir, self.vid_file))
        datasets=[processed_input_data]
        self.out_fps = out_fps
        self.min_frames = min_frames
        self.number_examples = number_examples
        
        self.create_videos_(datasets)


    def create_videos_(self, datasets):
        labels_fs = []
        fs_labels = []
        for dataset in datasets:
            logging.info(f"Inferring dataset {dataset.shape}")
            labels_fs.append(
                inference(self.clf, dataset, framerate=self.framerate).T
            )

        fs_labels = labels_fs

        # logging.info('Frameshifted arrangement of labels... ')
        # for k, _ in enumerate(labels_fs):
        #     labels_fs2 = []
        #     for l in range(math.floor(self.framerate / 10)):
        #         labels_fs2.append(labels_fs[k][l])
        #     fs_labels.append(np.array(labels_fs2).flatten('F'))


        logging.info('Done frameshift-predicting **{}**.'.format(self.d_file))

        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_video.sav'))), 'wb') as f:
            joblib.dump([fs_labels[0], int(self.min_frames), int(self.number_examples), int(self.out_fps),
                            self.frame_dir, self.shortvid_dir], f)

        create_labeled_vid(fs_labels[0], int(self.min_frames), int(self.number_examples), int(self.out_fps),
                            self.frame_dir, self.shortvid_dir)

        logging.info('**_CHECK POINT_**: Done generating video snippets. Move on to '
                    '__Predict old/new files using a model__.')


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

