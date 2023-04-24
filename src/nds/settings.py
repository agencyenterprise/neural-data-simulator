"""Models for parsing and validating the contents of `settings.yaml`."""
from typing import Dict, Optional

from pydantic import BaseModel
from pydantic import Json
from pydantic import validator
from pydantic_yaml import VersionedYamlModel
from pydantic_yaml import YamlStrEnum


class LogLevel(YamlStrEnum):
    """Possible log levels."""

    _DEBUG = "DEBUG"
    _INFO = "INFO"
    _ERROR = "ERROR"
    _WARNING = "WARNING"
    _CRITICAL = "CRITICAL"


class EncoderEndpointType(YamlStrEnum):
    """Possible types for the encoder input or output."""

    FILE = "file"
    LSL = "LSL"


class EphysGeneratorEndpointType(YamlStrEnum):
    """Possible types of input for the ephys generator."""

    TESTING = "testing"
    LSL = "LSL"


class EncoderModelType(YamlStrEnum):
    """Possible types of input for the encoder model."""

    PLUGIN = "plugin"
    VELOCITY_TUNING_CURVES = "velocity_tuning_curves"


class LSLChannelFormatType(YamlStrEnum):
    """Possible values for the LSL channel format."""

    _FLOAT32 = "float32"
    _DOUBLE64 = "double64"
    _INT8 = "int8"
    _INT16 = "int16"
    _INT32 = "int32"
    _INT64 = "int64"


class TimerModel(BaseModel):
    """Settings for the timer implementation."""

    max_cpu_buffer: float
    loop_time: float


class LSLInputModel(BaseModel):
    """Settings for all LSL inlets."""

    connection_timeout: float
    stream_name: str


class LSLOutputModel(BaseModel):
    """Settings for all LSL outlets."""

    class _Instrument(BaseModel):
        manufacturer: str
        model: str
        id: int

    channel_format: LSLChannelFormatType
    stream_name: str
    stream_type: str
    source_id: str
    instrument: _Instrument


class EncoderSettings(BaseModel):
    """Settings for the encoder."""

    class Input(BaseModel):
        """Settings for the encoder input."""

        class File(BaseModel):
            """Settings for the encoder input type file."""

            path: str
            sampling_rate: float
            timestamps_array_name: str
            data_array_name: str

        type: EncoderEndpointType
        file: Optional[File]
        lsl: Optional[LSLInputModel]

    class Output(BaseModel):
        """Settings for the encoder output."""

        n_channels: int
        type: EncoderEndpointType
        file: Optional[str]
        lsl: Optional[LSLOutputModel]

    model: str
    preprocessor: Optional[str]
    postprocessor: Optional[str]
    model_weights_file: Optional[str]
    input: Input
    output: Output

    @validator("model")
    def _model_entry_point_must_be_a_python_file(cls, v):
        if v is not None and v.endswith(".py"):
            return v
        raise ValueError("The model entry point must be a Python file")

    @validator("preprocessor", "postprocessor")
    def _plugin_entry_point_must_be_a_python_file(cls, v):
        if v is not None and v.endswith(".py"):
            return v
        raise ValueError("The plugin entry point must be a Python file")

    @validator("input", "output")
    def _file_type_must_have_a_file_object(cls, value):
        if value.type == EncoderEndpointType.FILE and not value.file:
            raise ValueError("File type must have a file object")
        return value

    @validator("input", "output")
    def _lsl_type_must_have_a_lsl_object(cls, value):
        if value.type == EncoderEndpointType.LSL and not value.lsl:
            raise ValueError("LSL type must have a lsl object")
        return value


class EphysGeneratorSettings(BaseModel):
    """Settings for the spike generator."""

    class Waveforms(BaseModel):
        """Settings for the spike waveform prototypes."""

        n_samples: int
        prototypes: Dict[int, Json[list[float]]]
        unit_prototype_mapping: Dict[str, int]

        @validator("prototypes")
        def _prototypes_have_the_same_length(cls, v):
            prototype_lengths = set([len(prototype) for prototype in v.values()])
            if len(prototype_lengths) > 1:
                raise ValueError("prototypes do not share the same length.")
            return v

        @validator("unit_prototype_mapping")
        def _unit_prototype_mapping_needs_to_point_to_a_valid_prototype(cls, v, values):
            mapped_waveforms = list(v.values())
            prototypes = list(values["prototypes"].keys())
            if not set(mapped_waveforms).issubset(prototypes):
                raise ValueError("Mapped prototype was not configured.")
            return v

        @validator("unit_prototype_mapping")
        def _unit_prototype_mapping_needs_a_default_value(cls, v):
            if "default" not in v:
                raise ValueError("Mapped prototype doesn't have a default value.")
            return v

    class Input(BaseModel):
        """Settings for the ephys generator input."""

        class Testing(BaseModel):
            """Settings for the ephys generator input type testing."""

            n_channels: int

        type: EphysGeneratorEndpointType

        lsl: Optional[LSLInputModel]
        testing: Optional[Testing]

    class Output(BaseModel):
        """Settings for the ephys generator output."""

        class Raw(BaseModel):
            """Settings for the ephys generator output type raw."""

            lsl: LSLOutputModel

        class LFP(BaseModel):
            """Settings for the ephys generator output type LFP."""

            data_frequency: float
            filter_cutoff: float
            filter_order: int
            lsl: LSLOutputModel

        class SpikeEvents(BaseModel):
            """Settings for the ephys generator output type spike events."""

            lsl: LSLOutputModel

        raw: Raw
        lfp: LFP
        spike_events: SpikeEvents

    class Noise(BaseModel):
        """Settings for the ephys generator noise."""

        beta: float
        standard_deviation: float
        fmin: float
        samples: int

    waveforms: Waveforms
    input: Input
    output: Output
    noise: Noise

    resolution: float
    random_seed: Optional[int]
    raw_data_frequency: float
    n_units_per_channel: int
    refractory_time: float
    lsl_chunk_frequency: float

    @validator("input")
    def _input_lsl_type_must_have_a_lsl_object(cls, input_value):
        if input_value.type == EphysGeneratorEndpointType.LSL and not input_value.lsl:
            raise ValueError("LSL type must have a lsl object")
        return input_value

    @validator("input")
    def _input_testing_type_must_have_a_testing_object(cls, input_value):
        if (
            input_value.type == EphysGeneratorEndpointType.TESTING
            and not input_value.testing
        ):
            raise ValueError("Testing type must have a testing object")
        return input_value


class Settings(VersionedYamlModel):
    """All settings for the NDS main package."""

    log_level: LogLevel
    timer: TimerModel
    encoder: EncoderSettings
    ephys_generator: EphysGeneratorSettings
