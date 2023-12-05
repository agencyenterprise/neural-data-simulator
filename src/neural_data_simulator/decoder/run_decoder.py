"""Script that configures and starts the example :class:`Decoder`."""

import argparse
import logging
from pathlib import Path
from typing import cast

from pydantic import Extra
from pydantic_yaml import VersionedYamlModel
from rich.pretty import pprint
import yaml

from neural_data_simulator.core import timing
from neural_data_simulator.core.inputs import api as inputs
from neural_data_simulator.core.inputs.lsl_input import LSLInput
from neural_data_simulator.core.outputs import api as outputs
from neural_data_simulator.core.outputs.lsl_output import LSLOutputDevice
from neural_data_simulator.core.settings import LogLevel
from neural_data_simulator.core.settings import TimerModel
from neural_data_simulator.decoder.decoders import Decoder
from neural_data_simulator.decoder.decoders import PersistedFileDecoderModel
from neural_data_simulator.decoder.settings import DecoderSettings
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import get_configs_dir
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.runtime import open_connection
from neural_data_simulator.util.settings_loader import check_config_override_str
from neural_data_simulator.util.settings_loader import load_settings

SCRIPT_NAME = "nds-decoder"
logger = logging.getLogger(__name__)


class _Settings(VersionedYamlModel):
    """Decoder app settings."""

    log_level: LogLevel
    decoder: DecoderSettings
    timer: TimerModel

    class Config:
        extra = Extra.forbid


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Decode behavior from input neural data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=Path(get_configs_dir()).joinpath("settings_decoder.yaml"),
        help="Path to the settings_decoder.yaml file.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        nargs="*",
        type=check_config_override_str,
        help=(
            "Specify settings overrides as key-value pairs, separated by spaces. "
            "For example: -o log_level=DEBUG decoder.spike_threshold=-210"
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


def _read_decode_send(
    data_input: inputs.Input, dec: Decoder, data_output: outputs.Output
) -> None:
    """Read input data, decode it, and send to the output stream.

    Args:
        data_input: Input data source.
        dec: Decoder.
        data_output: Output data sink.
    """
    samples = data_input.read()
    if not samples.empty:
        decoded_samples = dec.decode(samples)
        if not decoded_samples.empty:
            data_output.send(decoded_samples)


def run():
    """Run the decoder loop."""
    initialize_logger(SCRIPT_NAME)
    args = _parse_args()
    settings: _Settings = cast(
        _Settings,
        load_settings(
            args.settings_path,
            settings_parser=_Settings,
            override_dotlist=args.overrides,
        ),
    )
    if args.print_settings_only:
        pprint(settings)
        return

    configure_logger(SCRIPT_NAME, settings.log_level)
    logger.debug(f"run_decoder settings:\n{yaml.dump(settings.dict())}")

    # Set up timer
    timer_settings = settings.timer
    timer = timing.get_timer(timer_settings.loop_time, timer_settings.max_cpu_buffer)

    # Create LSL input and output objects
    output_settings = settings.decoder.output
    data_output = LSLOutputDevice.from_lsl_settings(
        lsl_settings=output_settings.lsl,
        sampling_rate=output_settings.sampling_rate,
        n_channels=output_settings.n_channels,
    )
    lsl_input_settings = settings.decoder.input.lsl
    data_input = LSLInput(
        stream_name=lsl_input_settings.stream_name,
        connection_timeout=lsl_input_settings.connection_timeout,
    )
    logger.debug(f"Querying info from LSL stream: {lsl_input_settings.stream_name}")

    # Set up decoder
    decoder_model = PersistedFileDecoderModel(get_abs_path(settings.decoder.model_file))
    dec = Decoder(
        model=decoder_model,
        input_sample_rate=data_input.get_info().sample_rate,
        output_sample_rate=1.0 / timer_settings.loop_time,
        n_channels=data_input.get_info().channel_count,
        threshold=settings.decoder.spike_threshold,
    )

    logger.debug("Attempting to open LSL connections...")
    try:
        with open_connection(data_output), open_connection(data_input):
            timer.start()
            # Run the decoder periodically
            while True:
                _read_decode_send(
                    data_input=data_input, dec=dec, data_output=data_output
                )
                timer.wait()
    except KeyboardInterrupt:
        logger.info("CTRL+C received. Exiting...")


if __name__ == "__main__":
    run()
