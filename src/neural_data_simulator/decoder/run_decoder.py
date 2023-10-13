"""Script that configures and starts the example :class:`Decoder`."""

import argparse
import logging
from pathlib import Path
from typing import cast

from pydantic_yaml import VersionedYamlModel

from neural_data_simulator.core import inputs
from neural_data_simulator.core import outputs
from neural_data_simulator.core import timing
from neural_data_simulator.core.settings import LogLevel
from neural_data_simulator.core.settings import TimerModel
from neural_data_simulator.decoder.decoders import Decoder
from neural_data_simulator.decoder.decoders import PersistedFileDecoderModel
from neural_data_simulator.decoder.settings import DecoderSettings
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.runtime import open_connection
from neural_data_simulator.util.settings_loader import get_script_settings

SCRIPT_NAME = "nds-decoder"
logger = logging.getLogger(__name__)


class _Settings(VersionedYamlModel):
    """Decoder app settings."""

    log_level: LogLevel
    decoder: DecoderSettings
    timer: TimerModel


def _parse_args_settings_path() -> Path:
    """Parse command-line arguments for the settings path."""
    parser = argparse.ArgumentParser(description="Run decoder.")
    parser.add_argument(
        "--settings-path",
        type=Path,
        help="Path to the settings_decoder.yaml file.",
    )
    args = parser.parse_args()
    return args.settings_path


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

    settings = cast(
        _Settings,
        get_script_settings(
            _parse_args_settings_path(), "settings_decoder.yaml", _Settings
        ),
    )
    configure_logger(SCRIPT_NAME, settings.log_level)

    # Set up timer
    timer_settings = settings.timer
    timer = timing.get_timer(timer_settings.loop_time, timer_settings.max_cpu_buffer)

    # Create LSL input and output objects
    output_settings = settings.decoder.output
    data_output = outputs.LSLOutputDevice.from_lsl_settings(
        lsl_settings=output_settings.lsl,
        sampling_rate=output_settings.sampling_rate,
        n_channels=output_settings.n_channels,
    )
    lsl_input_settings = settings.decoder.input.lsl
    data_input = inputs.LSLInput(
        stream_name=lsl_input_settings.stream_name,
        connection_timeout=lsl_input_settings.connection_timeout,
    )

    # Set up decoder
    decoder_model = PersistedFileDecoderModel(get_abs_path(settings.decoder.model_file))
    dec = Decoder(
        model=decoder_model,
        input_sample_rate=data_input.get_info().sample_rate,
        output_sample_rate=1.0 / timer_settings.loop_time,
        n_channels=data_input.get_info().channel_count,
        threshold=settings.decoder.spike_threshold,
    )

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
