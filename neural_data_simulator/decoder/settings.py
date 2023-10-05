"""Schema for Decoder settings."""

from pydantic import BaseModel

from neural_data_simulator.settings import LSLInputModel
from neural_data_simulator.settings import LSLOutputModel


class DecoderSettings(BaseModel):
    """Decoder settings."""

    class Input(BaseModel):
        """Decoder input settings."""

        lsl: LSLInputModel

    class Output(BaseModel):
        """Decoder output settings."""

        sampling_rate: float
        n_channels: int
        lsl: LSLOutputModel

    input: Input
    output: Output
    model_file: str
    spike_threshold: float
