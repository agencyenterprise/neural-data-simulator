"""Example of a custom Preprocessor implementation."""
from nds.encoder import Processor
from nds.samples import Samples


class PassThroughPreprocessor(Processor):
    """Custom Preprocessor implementation."""

    def execute(self, data: Samples) -> Samples:
        """Process the data and return the transformation."""
        return data


def create_preprocessor() -> Processor:
    """Instantiate the Preprocessor."""
    return PassThroughPreprocessor()
