import numpy as np

"""Example of a custom EncoderModel."""
from neural_data_simulator.core.models import EncoderModel
from neural_data_simulator.core.samples import Samples
from read_xdf import read_xdf

class AffectModel(EncoderModel):
    """ExampleModel implementation."""
    file_path = '/Users/aepinilla/.nds/sample_data/PJGHY.xdf'
    trials = read_xdf(file_path)
    training_trial = trials[4]

    def encode(self, X: Samples) -> Samples:
        """Perform the encode and return the result."""
        return Samples(
            timestamps=X.timestamps, data=np.random.rand(X.timestamps.shape[0], 190)
        )

def create_model() -> EncoderModel:
    """Instantiate the AffectModel."""
    return AffectModel()