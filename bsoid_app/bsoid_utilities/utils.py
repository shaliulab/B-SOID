import os.path
import numpy as np
import joblib

def load_data_(path, name):
    with open(os.path.join(path, str.join('', (name, '_data.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


def load_feats_(path, name):

    with open(os.path.join(path, str.join('', (name, '_feats.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]

def load_embeddings_(path, name):
    with open(os.path.join(path, str.join('', (name, '_embeddings.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]

def load_umap_model_(path, name):
    with open(os.path.join(path, str.join('', (name, '_UMAP_model.sav'))), 'rb') as fr:
        model = joblib.load(fr)
    return model

def load_clusters_(path, name):
    with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]

def load_classifier_(path, name):
    with open(os.path.join(path, str.join('', (name, '_randomforest.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]

def window_from_framerate(framerate):
    # window=np.int(np.round(0.1 * framerate)  - 1) #  150 / 10  - 1 = 14
    window=np.int(np.round(0.05 / (1 / framerate)) * 2 - 1)
    return window