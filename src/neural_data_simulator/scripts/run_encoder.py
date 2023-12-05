r"""Script that starts the encoder.

The encoder default configuration is located in `NDS_HOME/settings.yaml`
(see :mod:`neural_data_simulator.scripts.post_install_config`). The script can use
different config file specified via the `\--settings-path` argument.

The config file has an `encoder` section where the settings for the model,
input and output can be adjusted. By default, the encoder expects to read
data from an LSL stream and output to an LSL outlet. In absence of the
input stream, the encoder will not be able to start.
"""
import argparse
import logging
from pathlib import Path
from typing import Callable, cast, Optional, Union

import numpy as np
from rich.pretty import pprint
import yaml

from neural_data_simulator.core import encoder
from neural_data_simulator.core import models
from neural_data_simulator.core import runner
from neural_data_simulator.core import timing
from neural_data_simulator.core.inputs import api as inputs
from neural_data_simulator.core.inputs.lsl_input import LSLInput
from neural_data_simulator.core.inputs.samples_input import SamplesInput
from neural_data_simulator.core.outputs import api as outputs
from neural_data_simulator.core.outputs.lsl_output import LSLOutputDevice
from neural_data_simulator.core.samples import Samples
from neural_data_simulator.core.settings import EncoderEndpointType
from neural_data_simulator.core.settings import EncoderSettings
from neural_data_simulator.core.settings import Settings
from neural_data_simulator.scripts.errors import InvalidPluginError
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import get_configs_dir
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.runtime import load_module
from neural_data_simulator.util.runtime import unwrap
from neural_data_simulator.util.settings_loader import check_config_override_str
from neural_data_simulator.util.settings_loader import load_settings

SCRIPT_NAME = "nds-encoder"
logger = logging.getLogger(__name__)


def _setup_npz_input(
    behavior_file: str, timestamps_array_name: str, data_array_name: str
) -> SamplesInput:
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

    data_input = SamplesInput(all_samples)
    return data_input


def _setup_LSL_input(stream_name: str, connection_timeout: float) -> LSLInput:
    """Set up LSL input to read data from the behavior stream.

    Args:
        stream_name: LSL stream name.
        connection_timeout: Maximum time for attempting a connection to the
          LSL input stream.

    Returns:
        LSL stream input that can be used to read data from.
    """
    data_input = LSLInput(stream_name, connection_timeout)
    return data_input


def _setup_data_input(
    input_settings: EncoderSettings.Input,
) -> tuple[inputs.Input, Union[Callable, float]]:
    """Set up the input to read data from.

    Args:
        input_settings: Encoder input settings.

    Returns:
        data_input: The input that can be used to read data from.
        sampling_rate: The sampling rate of the input.
    """
    if input_settings.type == EncoderEndpointType.FILE:
        input_file = unwrap(input_settings.file)
        data_input = _setup_npz_input(
            input_file.path,
            input_file.timestamps_array_name,
            input_file.data_array_name,
        )
        sampling_rate = input_file.sampling_rate
    elif input_settings.type == EncoderEndpointType.LSL:
        lsl_input_settings = unwrap(input_settings.lsl)
        data_input = _setup_LSL_input(
            lsl_input_settings.stream_name, lsl_input_settings.connection_timeout
        )
        sampling_rate = lambda: data_input.get_info().sample_rate
    else:
        raise ValueError(f"Unexpected input type {input_settings.type}")

    return data_input, sampling_rate


def _load_plugin_model(module_path: str) -> models.EncoderModel:
    """Instantiate the custom encoder model.

    Load the module defined in module_path, return the
    encoder model instantiated by the module exposed `create_model`
    function
    """
    plugin_module = load_module(module_path, "model")

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
        plugin_module = load_module(encoder_settings.preprocessor, "preprocessor")
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
        plugin_module = load_module(encoder_settings.postprocessor, "postprocessor")
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
        An instance of an :class:`neural_data_simulator.models.EncoderModel`.
    """
    return _load_plugin_model(encoder_settings.model)


def _setup_data_output(
    output_settings: EncoderSettings.Output,
    sampling_rate: Union[float, Callable],
) -> outputs.Output:
    """Set up the output that will make the data available via an LSL stream.

    Args:
        output_settings: output module settings.
        sampling_rate: The expected data sampling rate.

    Returns:
        output data sink
    """
    if output_settings.type == EncoderEndpointType.FILE:
        output_file = unwrap(output_settings.file)
        data_output = outputs.FileOutput(
            file_name=get_abs_path(output_file),
            channel_count=output_settings.n_channels,
        )
    elif output_settings.type == EncoderEndpointType.LSL:
        lsl_output_settings = unwrap(output_settings.lsl)
        data_output = LSLOutputDevice.from_lsl_settings(
            lsl_settings=lsl_output_settings,
            sampling_rate=sampling_rate,
            n_channels=output_settings.n_channels,
        )
    else:
        raise ValueError(f"Unexpected output type {output_settings.type}")

    return data_output


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert behavioral input into simulated neural activity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=Path(get_configs_dir()).joinpath("settings.yaml"),
        help="Path to the settings.yaml file.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        nargs="*",
        type=check_config_override_str,
        help=(
            "Specify settings overrides as key-value pairs, separated by spaces. "
            "For example: -o log_level=DEBUG encoder.output.n_channels=20"
        ),
    )
    parser.add_argument(
        "--print-settings-only",
        "-p",
        action="store_true",
        help="Parse/print the settings and exit.",
    )
    args = parser.parse_args()
    return args


def run():
    """Load the configuration and start the encoder."""
    initialize_logger(SCRIPT_NAME)
    args = _parse_args()
    settings: Settings = cast(
        Settings,
        load_settings(
            args.settings_path,
            settings_parser=Settings,
            override_dotlist=args.overrides,
        ),
    )
    if args.print_settings_only:
        pprint(settings)
        return

    configure_logger(SCRIPT_NAME, settings.log_level)
    logger.debug(f"run_encoder settings:\n{yaml.dump(settings.dict())}")

    data_input, sampling_rate = _setup_data_input(settings.encoder.input)
    model = _setup_model(settings.encoder)
    preprocessor = _setup_preprocessor(settings.encoder)
    postprocessor = _setup_postprocessor(settings.encoder)

    output_settings = settings.encoder.output
    data_output = _setup_data_output(output_settings, sampling_rate)

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
