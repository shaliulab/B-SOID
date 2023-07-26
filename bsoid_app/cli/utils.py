import sqlite3
import os.path
import itertools
import glob
import logging
from typing import Iterator, Tuple

import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

logger = logging.getLogger(__name__)

def make_concatenation_pattern(dbfile, local_identities, experiment, chunks):

    tokens = []
    for chunk in chunks:
        for local_identity in local_identities:
            tokens.append((dbfile, chunk, local_identity))

    connections={}
    concatenation=[]
    for i, (handler, chunk, local_identity) in enumerate(tokens):
        if handler not in connections:
            connections[handler]=sqlite3.connect(handler)
        cur=connections[handler].cursor()
        args=(local_identity, chunk)
        # print(args)
        cur.execute("SELECT identity FROM CONCATENATION WHERE local_identity = ? AND chunk = ?", args)
        identity=cur.fetchall()[0][0]


        experiment=os.path.splitext(os.path.basename(handler))[0]

        concatenation.append((experiment, chunk, local_identity, identity))

    concatenation=pd.DataFrame.from_records(concatenation)
    concatenation.columns=["experiment", "chunk","local_identity", "identity"]
    concatenation=concatenation.loc[concatenation["experiment"]==experiment]                
    dfiles=[]
    vid_files=[]
    for i, row in concatenation.iterrows():
        data_directory = row['experiment'] + "_" + str(row['local_identity']).zfill(3)
        filename=str(row['chunk']).zfill(6) + ".mp4.predictions.h5"
        dfiles.append(os.path.join(os.environ["BSOID_DATA"], data_directory, filename))

        flyhostel_id, number_of_animals, date_, time_ = row["experiment"].split("_")

        vid_file=os.path.join(
            os.environ["FLYHOSTEL_VIDEOS"], flyhostel_id, number_of_animals,
            f"{date_}_{time_}", "flyhostel", "single_animal", str(row['local_identity']).zfill(3),
            str(row["chunk"]).zfill(6) + ".mp4"
        )
        vid_files.append(vid_file)
    concatenation["dfile"] = dfiles
    concatenation["vid_file"] = vid_files
    concatenation.sort_values(["identity","chunk"], inplace=True)
    concatenation.to_csv(os.path.join(os.environ["BSOID_DATA"], "output", "concatenation-overlap.csv"))

    return concatenation

def make_datasets(concatenation, identities, processed_input_data, input_filenames):
    datasets=[]
    # assumes processed_input_data is sorted by id and time
    for identity in identities:
        target_files=concatenation.loc[(concatenation["identity"]==identity), "dfile"].tolist()
        positions=np.where([f in target_files for f in input_filenames])[0]
        dataset=[processed_input_data[i] for i in positions]
        dataset=np.vstack(dataset)
        df=pd.DataFrame(dataset.copy())
        df.fillna(method="backfill",inplace=True)
        df.fillna(method="ffill", inplace=True)
        datasets.append(df.values)

    return datasets


def get_video_files(experiment, animal):
    data_directories=sorted(os.listdir(os.environ["BSOID_DATA"]))
    data_directories=[directory for directory in data_directories if directory.startswith(experiment.replace("/","_"))]

    pattern=os.path.join(os.environ["BSOID_DATA"],  data_directories[animal], "*h5")
    
    with h5py.File(glob.glob(pattern)[0], "r") as file:
        files=[e.decode() for e in file["files"][:]]
    return files

def create_animation(experiment, datasets, animal, chunksize, skip=15000, number_of_frames=None, **kwargs):
    """
    A handy wrapper around create_animation_ where the inputs are more high level
    """

    files = get_video_files(experiment, animal)
        
    video_files=list(itertools.chain(*[[".".join(file.split(".")[:-2]), ] * chunksize for file in files]))[::skip]
    
    pose_estimates=datasets[animal][::skip]
    frame_numbers=np.arange(pose_estimates.shape[0])
    frame_indices=frame_numbers % chunksize   
    origin=list(zip(video_files, frame_indices))
    if number_of_frames is None:
        return create_animation_(pose_estimates=pose_estimates, origin=origin, limits=[(0, 100), (0,100)], **kwargs)
    else:
        return create_animation_(pose_estimates=pose_estimates[:number_of_frames], origin=origin[:number_of_frames], limits=[(0, 100), (0,100)], **kwargs)



def create_animation_(pose_estimates, pairs, labels, limits=None, origin=None, filename='pose.mp4'):
    """
    Project the processed POSE on the original video sequences

    This is useful to control for potential artifacts introduced in the postprocessing of pose estimator output

    Arguments:

        pose_estimates (np.ndarray): pose estimate with time (frames) on the vertical axis and body parts in the y axis
            Each body part is represented by n columns, where n is the dimensionality of the pose estimate.
            Only 2D estimates are supported i.e. n is always 2
            This means a pose of two body parts contains 4 columns: x,y,x,y of the first and then second body parts
        pairs (list): Edges of the pose, represented as a tuple of two indices, containing the 0-based index of the body parts to be connected
            For example, in a pose of 4 body parts, [(1,3)] will connect only the second and fourth body parts
        labels (list): Name of the body parts
        limits (list): Limits of the X and Y axis, as a list of two tuples of length 2
        origin (list): If provided, a list of length equal to the number of frames in the pose estimates.
            Each element should be a tuple with the path to the video file containing the original frame, and the frame index in the video
        filename (str): Output video
    """
    num_frames, num_coords = pose_estimates.shape
    num_parts = num_coords // 2


    if origin is not None:
        assert len(origin) == pose_estimates.shape[0], f"Length of origin is not the same as rows in pose_estimates"

    
    # Create a figure and an axis
    fig, ax = plt.subplots()


    # Create a scatter plot, a placeholder for lines, and a placeholder for text
    scatter = ax.scatter([], [], color='red')
    lines = [ax.plot([], [], color='blue')[0] for _ in pairs]
    texts = [ax.text(0, 0, '', fontsize=8) for _ in range(num_parts)]

    # Set the limits of the plot
    if limits is None:
        x_min, y_min = np.min(pose_estimates[:, ::2]), np.min(pose_estimates[:, 1::2])
        x_max, y_max = np.max(pose_estimates[:, ::2]), np.max(pose_estimates[:, 1::2])
    else:
        x_min, y_min = limits[0]
        x_max, y_max = limits[1]
        
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    last_video=None
    cap=None
    frames=[]

    def animate(i):

        nonlocal last_video
        nonlocal cap
        nonlocal frames
        
        # Get the coordinates of the body parts
        x_coords = pose_estimates[i, ::2]
        y_coords = pose_estimates[i, 1::2]
        
        if origin is not None:
            video, frame_index = origin[i]
            if last_video is None:
                last_video=video
                logger.debug(f"Opening {video}")
                cap = cv2.VideoCapture(video)
            elif video != last_video:
                if cap is not None:
                    cap.release()

                logger.debug(f"Opening {video}")
                cap = cv2.VideoCapture(video)
                last_video = video

            cap.set(1, frame_index)
            logger.debug(f"Reading frame {frame_index}")
            ret, frame = cap.read()
            frames.append(frame)
            if not ret:
                logger.warning(f"Problem with video {video} and frame {frame_index}")
            else:    
                ax.set_xlim(0, frame.shape[1])
                ax.set_ylim(frame.shape[0], 0)
                ax.imshow(frame)

        # Update the scatter plot
        scatter.set_offsets(np.c_[x_coords, y_coords])

        # Update the lines
        for pair, line in zip(pairs, lines):
            x1, y1 = x_coords[pair[0]], y_coords[pair[0]]
            x2, y2 = x_coords[pair[1]], y_coords[pair[1]]
            line.set_data([x1, x2], [y1, y2])

        # Update the text
        for j, text in enumerate(texts):
            text.set_position((x_coords[j], y_coords[j]))
            text.set_text(labels[j])

        return [scatter] + lines + texts

    # Create an animation
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100, blit=True)
    
    # Save the animation
    anim.save(filename, writer='ffmpeg')

    if cap is not None:
        cap.release()

    # cv2.imwrite("frames.png", np.hstack(frames))
    

# Example usage:
# pose_estimates = np.random.rand(100, 2*10)  # 100 frames, 10 body parts
# pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]  # Connect parts 0-1, 1-2, 2-3, 3-4
# labels = ['part' + str(i) for i in range(10)]  # Labels for the body parts
# create_animation(pose_estimates, pairs, labels)


def ignore_features_derived_from_missing_body_part(datasets, features, feature_names, body_parts, stride=15):
    """
    Set value of computed features to 0
    whenever any of the body parts in which it relies 
    has a value of 0 in the pose
    (which means it's actually missing, not that it is really at 0)
    """
    
    for body_part_index, body_part in body_parts:
        
        # true in position i if the body part is missing in frame i
        missing_part = np.array(list(itertools.chain(*[
            (datasets[i][
                ::stride,
                (body_part_index*2):(body_part_index*2)+2
            ] == 0).all(axis=1) [:-1]
            for i in range(len(datasets))
        ])))
    
        # true if feature j is based on the body part
        derived_features = np.array([body_part in feat for feat in feature_names])
    
        assert len(derived_features) ==  features.shape[0]
        assert len(missing_part) == features.shape[1]
        print(f"{missing_part.mean()*100} % of frames are missing {body_part}")    
        features[np.ix_(derived_features, missing_part)]=0

    return features


def run_length_encode(data: str) -> Iterator[Tuple[str, int]]:
    """Returns run length encoded Tuples for string"""
    # A memory efficient (lazy) and pythonic solution using generators
    return ((x, sum(1 for _ in y)) for x, y in itertools.groupby(data))


def interpolate_between_contiguous_predictions(datasets, ref_parts, body_parts, stride):
    """
    Interpolate body parts only if contiguous predictions found it
    i.e. it is missing only because SLEAP skipped it, not because it predicted it is not there 
    """
    
    for i in range(len(datasets)):
        dataset=datasets[i]
        for body_part in body_parts:
            body_part_index = ref_parts.index(body_part)
            
            x=dataset[:,body_part_index].copy()
            x[x!=0]=1
            decoded = "".join([str(e) for e in np.int32(x)])
            
            current_frame=0
            for value, count in run_length_encode(decoded):
                if value == '0' and count == (stride-1):
                    contiguous_prediction=dataset[current_frame-1,body_part_index:(body_part_index+2)] 
                    dataset[current_frame:(current_frame+stride-1), body_part_index:(body_part_index+2)]=contiguous_prediction  
                current_frame+=count

    return datasets


def make_data_egocentric(datasets):
    # center pose around the thorax
    egocentric_data=[]
    for data in datasets:
        egocentric_data.append(data.copy())
        ego_data=egocentric_data[-1][:,(ego_part*2):(ego_part*2+2)].copy()
        for body_part_index in data.shape[1] // 2:
            body_part_data = egocentric_data[-1][:, (body_part_index*2):(body_part_index*2+2)]
            detected_frames = (body_part_data!=0).all(axis=1)
            print(detected_frames.mean())
            body_part_data[detected_frames, :]-=ego_data[detected_frames, :]
            egocentric_data[-1][:, (body_part_index*2):(body_part_index*2+2)]=body_part_data

    return egocentric_data


def filter_data(datasets, animals, chunks, chunksize, first_chunk=50):
    """
    Keep data for only specific chunks and animals
    """

    filtered_data=[
        d[
            np.concatenate([
                np.arange(0,chunksize) + chunksize*(chunk-first_chunk) for chunk in chunks
            ])
        ]
        for d in [datasets[i] for i in animals]
    ]
    
    return filtered_data


def read_animal_speed(sqlite3_file, identity, chunks, chunksize=45000, first_chunk=50):
    
    placeholders = ', '.join('?' for _ in chunks)
    
    with sqlite3.connect(sqlite3_file) as conn:
        cur=conn.cursor()
        cmd=f"""
            SELECT R0.x, R0.y, R0.frame_number
            FROM ROI_0 AS R0
                INNER JOIN STORE_INDEX AS IDX ON R0.frame_number = IDX.frame_number AND IDX.chunk IN ({placeholders}) AND IDX.half_second = 1
                INNER JOIN IDENTITY AS ID ON R0.frame_number = ID.frame_number AND R0.in_frame_index = ID.in_frame_index AND ID.identity = {identity};
        """
        # print(cmd)
        
        cur.execute(cmd, tuple([*chunks]))
        records=cur.fetchall()
    data=pd.DataFrame.from_records(records, columns=["x", "y", "frame_number"])
        
    diff=np.diff(data[["x","y"]].values, axis=0)
    speed=np.sqrt((diff**2).sum(axis=1))
    data["speed"]=0

    assert len(speed) == data.shape[0]-1
    data.loc[1:, "speed"]=speed
    data["bsoid_fn"]=data["frame_number"] - chunksize*first_chunk    
    return data


def compute_speed(sqlite3_file, animals, chunks, centroid_chunksize, centroid_framerate, framerate):
    scaler = StandardScaler()
    expected_size = len(chunks)*centroid_chunksize
    all_centroid_data=[]
    for animal in animals:
        d=read_animal_speed(sqlite3_file, identity=animal+1, chunks=chunks.tolist())
        d["animal"]=animal
        if len(d) < expected_size:
            logging.warning(f"Centroid data for animal {animal} is missing {expected_size-len(d)} datapoints!")
            last_row = d.tail(1)
            for _ in range(expected_size-len(d)):
                d=pd.concat([d, pd.DataFrame({
                    "x": 0, "y": 0,
                    "frame_number": last_row["frame_number"]+framerate//centroid_framerate,
                    "speed": 0,
                    "bsoid_fn": last_row["bsoid_fn"]+framerate//centroid_framerate,
                    "animal": animal,
            })])
        d["scaled_speed"] = scaler.fit_transform(d["speed"].values.reshape(-1, 1)).flatten()
        all_centroid_data.append(d)
    centroid_data=pd.concat(all_centroid_data)

    return centroid_data