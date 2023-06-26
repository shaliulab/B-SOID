import sqlite3
import os.path
import pandas as pd
import numpy as np


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