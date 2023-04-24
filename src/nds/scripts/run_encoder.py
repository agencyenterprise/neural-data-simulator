r"""Script that starts the encoder.

The encoder default configuration is located in `NDS_HOME/settings.yaml`
(see :mod:`nds.scripts.post_install_config`). The script can use different
config file specified via the `\--settings-path` argument.

The config file has an `encoder` section where the settings for the model,
input and output can be adjusted. By default, the encoder expects to read
data from an LSL stream and output to an LSL outlet. In absence of the
input stream, the encoder will not be able to start.
"""
import argparse
import importlib.machinery
import importlib.util
import logging
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable, cast, Optional, Union

import numpy as np

from nds import encoder
from nds import inputs
from nds import models
from nds import outputs
from nds import runner
from nds import timing
from nds.outputs import LSLOutputDevice
from nds.outputs import StreamConfig
from nds.samples import Samples
from nds.scripts.errors import InvalidPluginError
from nds.settings import EncoderEndpointType
from nds.settings import EncoderSettings
from nds.settings import LSLOutputModel
from nds.settings import Settings
from nds.util.runtime import configure_logger
from nds.util.runtime import get_abs_path
from nds.util.runtime import unwrap
from nds.util.settings_loader import get_script_settings

logger = logging.getLogger(__name__)


def _setup_npz_input(
    behavior_file: str, timestamps_array_name: str, data_array_name: str
) -> inputs.SamplesInput:
    """Set up the NPZ file input.

    Load data from npz file and create a samples dataclass with the entirety
    of the data.

    Args:
        behavior_file: The npz file containing the behavior data.
        timestamps_array_name: The key for the array containing the timestamps.
        data_array_name: The key for the array containing the samples.

    Returns:
        Samples input with behavior data loaded in.
    """
    data = np.load(get_abs_path(behavior_file))
    all_samples = Samples(
        timestamps=data[timestamps_array_name], data=data[data_array_name]
    )

    data_input = inputs.SamplesInput(all_samples)
    return data_input


def _setup_LSL_input(stream_name: str, connection_timeout: float) -> inputs.LSLInput:
    """Set up LSL input to read data from the behavior stream.

    Args:
        stream_name: LSL stream name.
        connection_timeout: Maximum time for attempting a connection to the
          LSL input stream.

    Returns:
        LSL stream input that can be used to read data from.
    """
    data_input = inputs.LSLInput(stream_name, connection_timeout)
    return data_input


def _load_module(module_path: str, module_name: str) -> ModuleType:
    """Load an external module and return it."""
    module_path = get_abs_path(module_path)
    module_dir_path = Path(module_path).parent
    sys.path.append(str(module_dir_path.parent.absolute()))

    loader = importlib.machinery.SourceFileLoader(module_name, module_path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec:
        plugin_module = importlib.util.module_from_spec(spec)
        loader.exec_module(plugin_module)
        return plugin_module
    raise Exception(f"Couldn't load module from '{module_path}'")


def _load_plugin_model(module_path: str) -> models.EncoderModel:
    """Instantiate the custom encoder model.

    Load the module defined in module_path, return the
    encoder model instantiated by the module exposed `create_model`
    function
    """
    plugin_module = _load_module(module_path, "model")

    try:
        model = plugin_module.create_model()
    except AttributeError:
        raise InvalidPluginError(
            "Plugin module does not implement the create_model() function."
        )

    if not isinstance(model, models.EncoderModel):
        raise InvalidPluginError("Custom model is not implementing 'EncoderModel'.")
    return model


def _setup_preprocessor(
    encoder_settings: EncoderSettings,
) -> Optional[encoder.Processor]:
    """Instantiate the custom preprocessor when it is set."""
    if encoder_settings.preprocessor:
        plugin_module = _load_module(encoder_settings.preprocessor, "preprocessor")
        try:
            preprocessor = plugin_module.create_preprocessor()
        except AttributeError:
            raise InvalidPluginError(
                "Plugin module does not implement the create_preprocessor() function."
            )

        if not isinstance(preprocessor, encoder.Processor):
            raise InvalidPluginError(
                "Custom preprocessor is not implementing 'Processor'."
            )
        return preprocessor
    return None


def _setup_postprocessor(
    encoder_settings: EncoderSettings,
) -> Optional[encoder.Processor]:
    """Instantiate the custom postprocessor when it is set."""
    if encoder_settings.postprocessor:
        plugin_module = _load_module(encoder_settings.postprocessor, "postprocessor")
        try:
            postprocessor = plugin_module.create_postprocessor()
        except AttributeError:
            raise InvalidPluginError(
                "Plugin module does not implement the create_postprocessor() function."
            )

        if not isinstance(postprocessor, encoder.Processor):
            raise InvalidPluginError(
                "Custom postprocessor is not implementing 'Processor'."
            )
        return postprocessor
    return None


def _setup_model(encoder_settings: EncoderSettings) -> models.EncoderModel:
    """Instantiate the model to be used in the encoder.

    Args:
        encoder_settings: The encoder settings with the model path.

    Returns:
        An instance of an :class:`nds.models.EncoderModel`.
    """
    return _load_plugin_model(encoder_settings.model)


def _setup_file_output(output_file: str, channel_count: int) -> outputs.FileOutput:
    """Set up the file output.

    It will save data into a CSV file. Note: currently, the output file
    has no header.

    Args:
        output_file: The absolute or relative path of the output file.

    Returns:
        File output that can be used to save data.
    """
    data_output = outputs.FileOutput(
        file_name=get_abs_path(output_file), channel_count=channel_count
    )
    return data_output


def _setup_LSL_output(
    sampling_rate: Union[float, Callable],
    n_channels: int,
    lsl_output_settings: LSLOutputModel,
) -> outputs.LSLOutputDevice:
    """Set up the output that will make the data available via an LSL stream.

    Args:
        sampling_rate: The expected data sampling rate.
        n_channels: The number of output channels.
        lsl_output_settings: LSL output settings.

    Returns:
        LSL output that can be used to stream data.
    """
    stream_config = StreamConfig.from_lsl_settings(
        lsl_output_settings, sampling_rate, n_channels
    )
    data_output = LSLOutputDevice(stream_config)
    return data_output


def run():
    """Load the configuration and start the encoder."""
    parser = argparse.ArgumentParser(description="Run encoder.")
    parser.add_argument(
        "--settings-path",
        type=Path,
        help="Path to the settings.yaml file.",
    )
    settings = cast(
        Settings,
        get_script_settings(
            parser.parse_args().settings_path, "settings.yaml", Settings
        ),
    )
    configure_logger("nds-encoder", settings.log_level)

    if settings.encoder.input.type == EncoderEndpointType.FILE:
        input_file = unwrap(settings.encoder.input.file)
        data_input = _setup_npz_input(
            input_file.path,
            input_file.timestamps_array_name,
            input_file.data_array_name,
        )
        sampling_rate = input_file.sampling_rate
    elif settings.encoder.input.type == EncoderEndpointType.LSL:
        lsl_input_settings = unwrap(settings.encoder.input.lsl)
        data_input = _setup_LSL_input(
            lsl_input_settings.stream_name, lsl_input_settings.connection_timeout
        )
        sampling_rate = lambda: data_input.get_info().sample_rate
    else:
        raise ValueError(f"Unexpected input type {settings.encoder.input.type}")

    model = _setup_model(settings.encoder)
    preprocessor = _setup_preprocessor(settings.encoder)
    postprocessor = _setup_postprocessor(settings.encoder)

    if settings.encoder.output.type == EncoderEndpointType.FILE:
        output_file = unwrap(settings.encoder.output.file)
        data_output = _setup_file_output(
            output_file, settings.encoder.output.n_channels
        )
    elif settings.encoder.output.type == EncoderEndpointType.LSL:
        lsl_output_settings = unwrap(settings.encoder.output.lsl)
        data_output = _setup_LSL_output(
            sampling_rate,
            settings.encoder.output.n_channels,
            lsl_output_settings,
        )
    else:
        raise ValueError(f"Unexpected output type {settings.encoder.output.type}")

    sim = encoder.Encoder(
        input_=data_input,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        model=model,
        output=data_output,
    )

    timer_settings = settings.timer
    timer = timing.get_timer(timer_settings.loop_time, timer_settings.max_cpu_buffer)

    try:
        runner.run(sim, timer)
    except KeyboardInterrupt:
        logger.info("CTRL+C received. Exiting...")


if __name__ == "__main__":
    run()
