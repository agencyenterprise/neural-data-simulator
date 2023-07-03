"""Example of a custom script module."""
import numpy as np

from neural_data_simulator.samples import Samples


def run_post_transformation(data: Samples) -> Samples:
    """This function is called by the custom postprocessor."""
    data.data = np.clip(data.data, 0, None)
    return data
