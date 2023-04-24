"""Example of a custom Postprocessor implementation."""
import numpy as np

from nds.encoder import Processor
from nds.samples import Samples


class PassThroughPostprocessor(Processor):
    """Custom Postprocessor implementation."""

    def execute(self, data: Samples) -> Samples:
        """Process the data and return the transformation."""
        data.data = np.clip(data.data, 0, None)
        return data


def create_postprocessor() -> Processor:
    """Instantiate the Postprocessor."""
    return PassThroughPostprocessor()
