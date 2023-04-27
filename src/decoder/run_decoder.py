"""Script that configures and starts the example :class:`Decoder`."""

import argparse
import logging
from pathlib import Path
from typing import cast

from decoder.decoders import Decoder
from decoder.decoders import PersistedFileDecoderModel
from pydantic import BaseModel
from pydantic_yaml import VersionedYamlModel

from neural_data_simulator import inputs
from neural_data_simulator import outputs
from neural_data_simulator import timing
from neural_data_simulator.outputs import StreamConfig
from neural_data_simulator.settings import LogLevel
from neural_data_simulator.settings import LSLInputModel
from neural_data_simulator.settings import LSLOutputModel
from neural_data_simulator.settings import TimerModel
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import open_connection
from neural_data_simulator.util.settings_loader import get_script_settings

logger = logging.getLogger(__name__)


class _Settings(VersionedYamlModel):
    """Decoder settings."""

    class Decoder(BaseModel):
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

    log_level: LogLevel
    decoder: Decoder
    timer: TimerModel


def _setup_LSL_input(stream_name: str, connection_timeout: float) -> inputs.LSLInput:
    """Prepare LSL input to read the raw data stream.

    Args:
        stream_name: LSL stream name.
        connection_timeout: Maximum time for attempting
        a connection to the LSL input stream.

    Returns:
        The LSL stream input that can be used to read data from.
    """
    data_input = inputs.LSLInput(stream_name, connection_timeout)
    return data_input


def _setup_LSL_output(
    output_settings: _Settings.Decoder.Output,
) -> outputs.LSLOutputDevice:
    """Prepare output that will make the data available via an LSL stream.

    Args:
        output_settings: Decoder output settings.

    Returns:
        An LSL output stream that can be used by the decoder to publish data.
    """
    lsl_settings = output_settings.lsl
    stream_config = StreamConfig.from_lsl_settings(
        lsl_settings, output_settings.sampling_rate, output_settings.n_channels
    )
    lsl_output = outputs.LSLOutputDevice(stream_config)
    return lsl_output


def _setup_decoder(
    model_file_path: str,
    input_sample_rate: float,
    output_sample_rate: float,
    n_channels: int,
    spike_threshold: float,
) -> Decoder:
    """Initialize the decoder.

    Args:
        model_file_path: Path to the model file.
        input_sample_rate: Input sample rate.
        output_sample_rate: Output sample rate.
        n_channels: Number of channels.
        spike_threshold: Spike threshold.
    """
    model = PersistedFileDecoderModel(model_file_path)
    decoder = Decoder(
        model, input_sample_rate, output_sample_rate, n_channels, spike_threshold
    )

    return decoder


def run():
    """Run the decoder loop."""
    parser = argparse.ArgumentParser(description="Run decoder.")
    parser.add_argument(
        "--settings-path",
        type=Path,
        help="Path to the settings_decoder.yaml file.",
    )

    settings = cast(
        _Settings,
        get_script_settings(
            parser.parse_args().settings_path, "settings_decoder.yaml", _Settings
        ),
    )
    configure_logger("nds-decoder", settings.log_level)

    timer_settings = settings.timer
    timer = timing.get_timer(timer_settings.loop_time, timer_settings.max_cpu_buffer)

    data_output = _setup_LSL_output(settings.decoder.output)
    lsl_input_settings = settings.decoder.input.lsl
    data_input = _setup_LSL_input(
        lsl_input_settings.stream_name, lsl_input_settings.connection_timeout
    )
    input_sample_rate = data_input.get_info().sample_rate
    n_channels = data_input.get_info().channel_count
    output_sample_rate = 1.0 / timer_settings.loop_time

    dec = _setup_decoder(
        get_abs_path(settings.decoder.model_file),
        input_sample_rate,
        output_sample_rate,
        n_channels,
        settings.decoder.spike_threshold,
    )

    try:
        with open_connection(data_output), open_connection(data_input):
            timer.start()
            while True:
                samples = data_input.read()
                if not samples.empty:
                    samples = dec.decode(samples)
                    if not samples.empty:
                        data_output.send(samples)
                timer.wait()
    except KeyboardInterrupt:
        logger.info("CTRL+C received. Exiting...")


if __name__ == "__main__":
    run()
