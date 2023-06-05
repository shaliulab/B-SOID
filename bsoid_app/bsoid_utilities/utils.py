import os.path
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

def load_clusters_(path, name):
    with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]

def load_classifier_(path, name):
    with open(os.path.join(path, str.join('', (name, '_randomforest.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]
