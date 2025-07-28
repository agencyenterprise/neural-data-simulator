r"""Script that starts the ephys generator.

(see :mod:`neural_data_simulator.scripts.post_install_config`). The script can use
different config file specified via the `\--settings-path` argument.

The config file has an `ephys_generator` section where the settings for the input,
output, noise, etc. can be adjusted. By default, the ephys generator expects to read
data from an LSL stream and output to an LSL outlet. In absence of the
input stream, the ephys generator will not be able to start.
"""

import argparse
import logging
from pathlib import Path
from typing import cast, Optional

import numpy as np
from rich.pretty import pprint
import yaml

from neural_data_simulator.core.ephys_generator import ContinuousData
from neural_data_simulator.core.ephys_generator import NoiseData
from neural_data_simulator.core.ephys_generator import ProcessOutput
from neural_data_simulator.core.ephys_generator import Spikes
from neural_data_simulator.core.ephys_generator import Waveforms
from neural_data_simulator.core.health_checker import HealthChecker
from neural_data_simulator.core.inputs.api import SpikeRateInput
from neural_data_simulator.core.inputs.lsl_input import LSLInput
from neural_data_simulator.core.inputs.lsl_input import LSLSpikeRateInputAdapter
from neural_data_simulator.core.inputs.testing_input import SpikeRateTestingInput
from neural_data_simulator.core.outputs.lsl_output import LSLOutputDevice
from neural_data_simulator.core.outputs.lsl_output import StreamConfig
from neural_data_simulator.core.settings import EphysGeneratorEndpointType
from neural_data_simulator.core.settings import EphysGeneratorSettings
from neural_data_simulator.core.settings import Settings
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import get_configs_dir
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.runtime import unwrap
from neural_data_simulator.util.settings_loader import check_config_override_str
from neural_data_simulator.util.settings_loader import load_settings

SCRIPT_NAME = "nds-ephys-generator"
logger = logging.getLogger(__name__)


def _setup_test_input(n_channels: int, n_units_per_channel: int) -> SpikeRateInput:
    """Set up a spike rate input for testing.

    The testing input will result in generating spikes from constant
    spike rates.

    Args:
        n_channels: Number of input channels.
        n_units_per_channel: Number of neurons captured by each electrode.

    Returns:
        A type of input that can be consumed by the ephys generator process.
    """
    n_units = n_units_per_channel * n_channels
    spike_rate_input = SpikeRateTestingInput(n_channels, n_units)
    return spike_rate_input


def _setup_LSL_input(
    stream_name: str, connection_timeout: float
) -> LSLSpikeRateInputAdapter:
    """Set up an LSL stream as spike rate input.

    Args:
        stream_name: LSL stream name.
        connection_timeout: Maximum time for attempting a connection
          to the LSL input stream.

    Returns:
        LSL stream input that can be used to read data from.
    """
    lsl_inlet = LSLInput(stream_name, connection_timeout)
    spike_rate_input = LSLSpikeRateInputAdapter(lsl_inlet)
    return spike_rate_input


def _setup_LSL_output(config: StreamConfig) -> LSLOutputDevice:
    """Set up output that will make the data available via an LSL stream.

    Args:
        config: the output stream configuration.

    Returns:
        An LSL output stream that can be used by the ephys generator
          to publish data.
    """
    lsl_output = LSLOutputDevice(config)
    lsl_output.connect()
    return lsl_output


def _get_process_output_params(
    ephys_generator_settings: EphysGeneratorSettings,
) -> ProcessOutput.Params:
    return ProcessOutput.Params(
        ephys_generator_settings.n_units_per_channel,
        ephys_generator_settings.lsl_chunk_frequency,
        ephys_generator_settings.raw_data_frequency,
        ephys_generator_settings.resolution,
    )


def _get_continuous_data_params(
    ephys_generator_settings: EphysGeneratorSettings,
) -> ContinuousData.Params:
    return ContinuousData.Params(
        ephys_generator_settings.raw_data_frequency,
        ephys_generator_settings.n_units_per_channel,
        ephys_generator_settings.waveforms.n_samples,
        ephys_generator_settings.output.lfp.data_frequency,
        ephys_generator_settings.output.lfp.filter_cutoff,
        ephys_generator_settings.output.lfp.filter_order,
    )


def _get_spikes_params(
    ephys_generator_settings: EphysGeneratorSettings,
) -> Spikes.Params:
    return Spikes.Params(
        ephys_generator_settings.raw_data_frequency,
        ephys_generator_settings.n_units_per_channel,
        ephys_generator_settings.refractory_time,
    )


def _get_waveforms_params(
    ephys_generator_settings: EphysGeneratorSettings,
) -> Waveforms.Params:
    waveforms_settings = ephys_generator_settings.waveforms
    return Waveforms.Params(
        waveforms_settings.prototypes,
        waveforms_settings.unit_prototype_mapping,
        waveforms_settings.n_samples,
    )


def _set_random_seed(random_seed: Optional[int]):
    if random_seed:
        logger.info(f"Using random seed '{random_seed}'")
        np.random.seed(random_seed)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate electrophysiology data from firing rates.",
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
            "For example: -o log_level=DEBUG ephys_generator.input.type=testing"
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
    """Load the configuration and start the ephys generator."""
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

    _set_random_seed(settings.ephys_generator.random_seed)
    configure_logger(SCRIPT_NAME, settings.log_level)
    logger.debug(f"run_ephys_generator settings:\n{yaml.dump(settings.dict())}")

    if settings.ephys_generator.input.type == EphysGeneratorEndpointType.LSL:
        lsl_input_settings = unwrap(settings.ephys_generator.input.lsl)
        spike_rate_input = _setup_LSL_input(
            lsl_input_settings.stream_name, lsl_input_settings.connection_timeout
        )
        spike_rate_input.connect()
    elif settings.ephys_generator.input.type == EphysGeneratorEndpointType.TESTING:
        n_channels = unwrap(settings.ephys_generator.input.testing).n_channels
        spike_rate_input = _setup_test_input(
            n_channels, settings.ephys_generator.n_units_per_channel
        )
    else:
        raise ValueError(f"Unexpected input type {settings.ephys_generator.input.type}")

    continuous_data_output = _setup_LSL_output(
        StreamConfig.from_lsl_settings(
            settings.ephys_generator.output.raw.lsl,
            settings.ephys_generator.raw_data_frequency,
            spike_rate_input.channel_count,
        )
    )

    lfp_output = _setup_LSL_output(
        StreamConfig.from_lsl_settings(
            settings.ephys_generator.output.lfp.lsl,
            settings.ephys_generator.output.lfp.data_frequency,
            spike_rate_input.channel_count,
        )
    )

    spike_events_config = StreamConfig.from_lsl_settings(
        settings.ephys_generator.output.spike_events.lsl,
        0,  # irregular sampling rate
        settings.ephys_generator.waveforms.n_samples,
    )
    spike_events_config.channel_labels = [
        "channels",
        "units",
    ] + spike_events_config.channel_labels

    spike_events_output = _setup_LSL_output(spike_events_config)

    process_output_params = _get_process_output_params(settings.ephys_generator)

    noise_settings = settings.ephys_generator.noise
    noise_data = NoiseData(
        spike_rate_input.channel_count,
        noise_settings.beta,
        noise_settings.standard_deviation,
        noise_settings.fmin,
        noise_settings.samples,
        settings.ephys_generator.random_seed,
    )

    continuous_data = ContinuousData(
        noise_data,
        spike_rate_input.channel_count,
        _get_continuous_data_params(settings.ephys_generator),
    )

    n_units = (
        spike_rate_input.channel_count * settings.ephys_generator.n_units_per_channel
    )
    waveforms = Waveforms(_get_waveforms_params(settings.ephys_generator), n_units)

    spikes = Spikes(
        spike_rate_input.channel_count,
        waveforms,
        _get_spikes_params(settings.ephys_generator),
    )

    outputs = ProcessOutput.Outputs(
        continuous_data_output, lfp_output, spike_events_output
    )

    optimal_num_samples_per_iteration = int(
        settings.ephys_generator.raw_data_frequency
        / settings.ephys_generator.lsl_chunk_frequency
    )
    health_checker = HealthChecker(
        int(settings.ephys_generator.lsl_chunk_frequency),
        optimal_num_samples_per_iteration,
    )
    po = ProcessOutput(
        continuous_data,
        spikes,
        spike_rate_input,
        outputs,
        process_output_params,
        health_checker,
    )
    try:
        po.start()
    except KeyboardInterrupt:
        logger.info("CTRL+C received. Exiting...")
    finally:
        po.stop()
        continuous_data_output.disconnect()
        lfp_output.disconnect()
        spike_events_output.disconnect()
        del spike_rate_input


if __name__ == "__main__":
    run()
