import numpy as np
import pytest
import sys
import shutil
import os
import logging
sys.path.append("/home/vibflysleep/opt/B-SOID/")
from unittest.mock import patch
import codetiming
from bsoid_app.cli.extract_features import extract
from bsoid_app.bsoid_utilities.utils import window_from_framerate



from bsoid_app import extract_features


@patch('streamlit.checkbox', return_value=True)
@patch('streamlit.button', return_value=True)
@patch('streamlit.info', new=logging.warning)
def test_bsoid_extract(mock_checkbox, mock_button):
    # First we'll create some fake data for our test
    timepoints=1000
    dimensions=6
    datasets = [np.random.rand(timepoints, dimensions) for _ in range(1)]
    framerate=50
    window = window_from_framerate(framerate)
    assert window != 0
    
    
    # Now we'll call the function with our test data
    os.makedirs("./dev_test", exist_ok=True)
    with codetiming.Timer():
        extractor_dev=extract(working_dir="./dev_test", prefix="test", processed_input_data=datasets, framerate=framerate)
        # features_dev, scaled_features_dev=extractor_dev.compute_features_parallel(datasets=datasets, n_jobs=2)
        extractor_dev.main(1)

    print("dev done")
    
    
    # And finally we'll assert some conditions that should be true
    # assert features.shape == (dimensions*10, timepoints-1)
    # assert scaled_features.shape == (dimensions*10, timepoints-1)
    
    # You can add more assertions based on your knowledge of the function.
    os.makedirs("./original_test", exist_ok=True)
    with codetiming.Timer():
        extractor=extract_features.extract(working_dir="./original_test", prefix="test", processed_input_data=datasets, framerate=framerate)
        extractor.main()

    features_dev=extractor_dev.features
    scaled_features_dev=extractor_dev.scaled_features
    features=extractor.features
    scaled_features=extractor.scaled_features


    assert features_dev.shape==features.shape
    assert scaled_features_dev.shape==scaled_features.shape

    diff=features_dev[:10, :10] - features[:10, :10]
    maxx=np.round(
        np.stack([
            features_dev[:10, :10],
            features[:10, :10]
        ], axis=2)
    ).max(axis=2)

    rel_diff=np.round(diff) / maxx

    rel_diff[maxx==0]=0

    print(rel_diff)



    print(features_dev[:10, :10])
    print(np.round(features[:10, :10]))
    
    shutil.rmtree("./dev_test")
    shutil.rmtree("./original_test")




    
if __name__ == "__main__":
    test_bsoid_extract()
