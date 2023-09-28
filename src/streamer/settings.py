"""Settings schema for the streamer."""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic import validator
from pydantic_yaml import YamlStrEnum

from neural_data_simulator.settings import LSLChannelFormatType
from neural_data_simulator.settings import LSLOutputModel


class StreamerInputType(YamlStrEnum):
    """Possible types for the streamer input."""

    NPZ = "npz"
    Blackrock = "blackrock"


class LSLSimplifiedOutputModel(BaseModel):
    """Settings for all LSL outlets."""

    class _Instrument(BaseModel):
        manufacturer: str
        model: str
        id: int

    channel_format: LSLChannelFormatType
    instrument: _Instrument


class Streamer(BaseModel):
    """Settings specific to the streamer."""

    class NPZ(BaseModel):
        """Settings for streaming from a numpy archive file (.npz)."""

        class Output(BaseModel):
            """Settings for outputting to LSL."""

            sampling_rate: float
            n_channels: int
            lsl: LSLOutputModel

        class Input(BaseModel):
            """Settings for reading in from a .npz file."""

            file: Path
            timestamps_array_name: str
            data_array_name: str

        output: Output
        input: Input

    class Blackrock(BaseModel):
        """Settings for streaming from Blackrock Neurotech files."""

        class Output(BaseModel):
            """Settings for outputting to LSL."""

            lsl: LSLSimplifiedOutputModel

        class Input(BaseModel):
            """Settings for reading in from a Blackrock Neurotech file."""

            file: Path

        output: Output
        input: Input

    blackrock: Optional[Blackrock]
    npz: Optional[NPZ]
    input_type: StreamerInputType
    lsl_chunk_frequency: float
    stream_indefinitely: bool

    @validator("input_type")
    def _config_is_set_for_input_type(cls, v, values):
        if v == StreamerInputType.Blackrock.value and values.get("blackrock") is None:
            raise ValueError(
                "blackrock fields need to be configured"
                " for a Blackrock Neurotech file"
            )
        if v == StreamerInputType.NPZ.value and values.get("npz") is None:
            raise ValueError("npz fields need to be configured for an npz file")
        return v
