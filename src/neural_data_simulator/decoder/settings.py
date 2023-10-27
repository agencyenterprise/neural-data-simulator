"""Schema for Decoder settings."""

from pydantic import BaseModel
from pydantic import Extra

from neural_data_simulator.core.settings import LSLInputModel
from neural_data_simulator.core.settings import LSLOutputModel


class DecoderSettings(BaseModel, extra=Extra.forbid):
    """Decoder settings."""

    class Input(BaseModel, extra=Extra.forbid):
        """Decoder input settings."""

        lsl: LSLInputModel

    class Output(BaseModel, extra=Extra.forbid):
        """Decoder output settings."""

        sampling_rate: float
        n_channels: int
        lsl: LSLOutputModel

    input: Input
    output: Output
    model_file: str
    spike_threshold: float
