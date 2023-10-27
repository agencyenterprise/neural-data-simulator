"""Settings schema for the streamer."""
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic import Extra
from pydantic import validator

from neural_data_simulator.core.settings import LSLChannelFormatType
from neural_data_simulator.core.settings import LSLOutputModel


class StreamerInputType(str, Enum):
    """Possible types for the streamer input."""

    NPZ = "npz"
    Blackrock = "blackrock"


class LSLSimplifiedOutputModel(BaseModel, extra=Extra.forbid):
    """Settings for all LSL outlets."""

    class _Instrument(BaseModel, extra=Extra.forbid):
        manufacturer: str
        model: str
        id: int

    channel_format: LSLChannelFormatType
    instrument: _Instrument


class Streamer(BaseModel, extra=Extra.forbid):
    """Settings specific to the streamer."""

    class NPZ(BaseModel, extra=Extra.forbid):
        """Settings for streaming from a numpy archive file (.npz)."""

        class Output(BaseModel, extra=Extra.forbid):
            """Settings for outputting to LSL."""

            sampling_rate: float
            n_channels: int
            lsl: LSLOutputModel

        class Input(BaseModel, extra=Extra.forbid):
            """Settings for reading in from a .npz file."""

            file: Path
            timestamps_array_name: str
            data_array_name: str

        output: Output
        input: Input

    class Blackrock(BaseModel, extra=Extra.forbid):
        """Settings for streaming from Blackrock Neurotech files."""

        class Output(BaseModel, extra=Extra.forbid):
            """Settings for outputting to LSL."""

            lsl: LSLSimplifiedOutputModel

        class Input(BaseModel, extra=Extra.forbid):
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
        if v == StreamerInputType.Blackrock and values.get("blackrock") is None:
            raise ValueError(
                "blackrock fields need to be configured"
                " for a Blackrock Neurotech file"
            )
        if v == StreamerInputType.NPZ and values.get("npz") is None:
            raise ValueError("npz fields need to be configured for an npz file")
        return v
