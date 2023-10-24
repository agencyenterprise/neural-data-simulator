"""Example of a custom Postprocessor implementation."""

# This is how a custom script can be imported from the same directory as this one
import custom_script  # type: ignore

from neural_data_simulator.core.encoder import Processor
from neural_data_simulator.core.samples import Samples


class PassThroughPostprocessor(Processor):
    """Custom Postprocessor implementation."""

    def execute(self, data: Samples) -> Samples:
        """Process the data and return the transformation."""
        return custom_script.run_post_transformation(data)


def create_postprocessor() -> Processor:
    """Instantiate the Postprocessor."""
    return PassThroughPostprocessor()
