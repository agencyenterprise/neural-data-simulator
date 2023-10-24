"""Script that configures and starts the example :class:`Decoder`."""

import logging

import hydra
import hydra.errors
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel

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
from neural_data_simulator.util.runtime import NDS_HOME
from neural_data_simulator.util.runtime import open_connection

SCRIPT_NAME = "nds-decoder"
logger = logging.getLogger(__name__)


class _Settings(BaseModel):
    """Decoder app settings."""

    log_level: LogLevel
    decoder: DecoderSettings
    timer: TimerModel


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


@hydra.main(config_path=NDS_HOME, config_name="settings_decoder", version_base="1.3")
def run_with_config(cfg: DictConfig):
    """Run the decoder loop."""
    initialize_logger(SCRIPT_NAME)
    # Validate Hydra config with Pydantic
    cfg_resolved = OmegaConf.to_object(cfg)
    settings = _Settings.parse_obj(cfg_resolved)

    configure_logger(SCRIPT_NAME, settings.log_level)
    logger.debug("run_decoder configuration:\n" + OmegaConf.to_yaml(cfg))

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
