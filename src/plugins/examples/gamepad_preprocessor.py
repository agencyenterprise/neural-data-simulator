"""Preprocessor for LSL Gamepad app for velocity tuning model."""
from neural_data_simulator.encoder import Processor
from neural_data_simulator.samples import Samples


class GamepadPreprocessor(Processor):
    """Custom Preprocessor implementation."""

    def execute(self, data: Samples) -> Samples:
        """Process the data and return the transformation."""
        # Velocity model is expecting 2-channel data in the range of -15:+15.
        # The LSL gamepad ranges from -1:1 and is 6 channel
        # (left x,y, right x,y, left trig, right trig)
        # The gamepad has a substantial deadspace, prohibiting low-velocity movements.
        # So we do a transformation that expands the lower-velocity representation.
        # data.data = 15 * np.tanh(2 * data.data[:, :2])  # <-- Made the problem worse!
        data.data = 15 * data.data[:, :2] ** 3
        return data


def create_preprocessor() -> Processor:
    """Instantiate the Preprocessor."""
    return GamepadPreprocessor()
