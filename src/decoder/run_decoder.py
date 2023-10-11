"""Script that configures and starts the example :class:`Decoder`."""

import logging

from decoder.decoders import Decoder
from decoder.decoders import PersistedFileDecoderModel
from decoder.settings import DecoderSettings
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel

from neural_data_simulator import inputs
from neural_data_simulator import outputs
from neural_data_simulator import timing
from neural_data_simulator.outputs import StreamConfig
from neural_data_simulator.settings import LogLevel
from neural_data_simulator.settings import TimerModel
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.runtime import NDS_HOME
from neural_data_simulator.util.runtime import open_connection

SCRIPT_NAME = "nds-decoder"
logger = logging.getLogger(__name__)


class _Settings(BaseModel):
    """Decoder app settings."""

    log_level: LogLevel
    decoder: DecoderSettings
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
    output_settings: DecoderSettings.Output,
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


@hydra.main(config_path=NDS_HOME, config_name="settings_decoder", version_base="1.3")
def run_with_config(cfg: DictConfig):
    """Run the decoder loop."""
    initialize_logger(SCRIPT_NAME)
    # Validate Hydra config with Pydantic
    cfg = OmegaConf.to_object(cfg)
    settings = _Settings(**cfg)

    configure_logger(SCRIPT_NAME, settings.log_level)
    logger.debug("run_decoder configuration:\n" + OmegaConf.to_yaml(cfg))

    timer_settings = settings.timer
    timer = timing.get_timer(timer_settings.loop_time, timer_settings.max_cpu_buffer)

    data_output = _setup_LSL_output(settings.decoder.output)
    lsl_input_settings = settings.decoder.input.lsl
    data_input = _setup_LSL_input(
        lsl_input_settings.stream_name, lsl_input_settings.connection_timeout
    )
    logger.debug(f"Querying info from LSL stream: {lsl_input_settings.stream_name}")
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

    logger.debug("Attempting to open LSL connections...")
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


def run():
    """Run the script, with an informative error if config is not found."""
    try:
        run_with_config()
    except hydra.errors.MissingConfigException as exc:
        raise FileNotFoundError(
            "Run 'nds_post_install_config' to copy the default settings files."
        ) from exc


if __name__ == "__main__":
    run()
