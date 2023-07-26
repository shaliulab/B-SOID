import codetiming
import time
import logging
from bsoid_app.config import UMAP_PARAMS

# umap-learn
from psutil import virtual_memory
import umap
def cpu_umap(data, **kwargs):
    mem = virtual_memory()
    available_mb = mem.available >> 20
    logging.info('You have {} MB RAM 🐏 available'.format(available_mb))
    umap_params=UMAP_PARAMS.copy()
    umap_params.update(kwargs)

    if available_mb > (data.shape[0] * data.shape[1] * 32 * 60) / 1024 ** 2 + 64:
        logging.info('RAM 🐏 available is sufficient')
        try:
            print("Fitting data to UMAP")
            with codetiming.Timer():
                model = umap.UMAP(**umap_params)
                model.fit(data)
        except:
            logging.error('Failed on feature embedding. Try again by unchecking sidebar and rerunning extract features.')
    else:
        umap_params.update({"low_memory": True})
        logging.info(
            'Detecting that you are running low on available memory for this computation, '
            'setting low_memory so will take longer.')
        try:
            print("Fitting data to UMAP")
            with codetiming.Timer():
                model = umap.UMAP(*umap_params)
                model.fit(data)
        except:
            logging.error('Failed on feature embedding. Try again by unchecking sidebar and rerunning extract features.')

    # output has:
    #    same number of rows as input
    #    columns given by num_dimensions
    #    dtype float 32
    return model

# # gpumap
# import gpumap
def gpu_umap(data, **kwargs):
    raise NotImplementedError()
#     umap_params=UMAP_PARAMS.copy()
#     umap_params.update(kwargs)
#     embedding = gpumap.GPUMAP(**umap_params).fit_transform(data)

#     return embedding


# cuml
from cuml.manifold.umap import UMAP as cuUMAP

def cuml_umap(data, **kwargs):
    umap_params=UMAP_PARAMS.copy()
    umap_params.update(kwargs)
    print("Fitting data to UMAP")
    with codetiming.Timer():
        model = cuUMAP(**umap_params, hash_input=False)
        model.fit(data)

    return model
    