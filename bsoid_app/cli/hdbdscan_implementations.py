from bsoid_app.config import HDBSCAN_PARAMS
import codetiming
import hdbscan

def all_points_membership_vectors(*args, **kwargs):
    return hdbscan.all_points_membership_vectors(*args, **kwargs)

def cpu_hdbscan(c, embeddings):
    min_cluster_size=int(round(c * embeddings.shape[0]))

    print("CPU HDBSCAN")
    with codetiming.Timer():
        learned_hierarchy = hdbscan.HDBSCAN(
            prediction_data=True,
            min_cluster_size=min_cluster_size,
            **HDBSCAN_PARAMS
        ).fit(embeddings)
    
    return learned_hierarchy

from cuml.cluster.hdbscan import HDBSCAN as GPUHDBSCAN
def gpu_hdbscan(c, embeddings):
    min_cluster_size=int(round(c * embeddings.shape[0]))

    print(f"GPU HDBSCAN. Min cluster size = {min_cluster_size}")
    with codetiming.Timer():
        learned_hierarchy = GPUHDBSCAN(
            prediction_data=True,
            min_cluster_size=min_cluster_size,
            **HDBSCAN_PARAMS
        ).fit(embeddings)
    
    return learned_hierarchy