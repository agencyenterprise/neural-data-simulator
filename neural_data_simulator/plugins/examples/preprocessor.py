"""Example of a custom Preprocessor implementation."""
from neural_data_simulator.encoder import Processor
from neural_data_simulator.samples import Samples


class PassThroughPreprocessor(Processor):
    """Custom Preprocessor implementation."""

    def execute(self, data: Samples) -> Samples:
        """Process the data and return the transformation."""
        return data


def create_preprocessor() -> Processor:
    """Instantiate the Preprocessor."""
    return PassThroughPreprocessor()
