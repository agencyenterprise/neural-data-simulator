r"""Script that starts the streamer.

The streamer default configuration is located in `NDS_HOME/settings_streamer.yaml`
(see :mod:`neural_data_simulator.scripts.post_install_config`). The script can use
different config file specified via the `\--settings-path` argument.

Upon start, the streamer expects to read data from a file and output to an LSL
outlet. By default, a sample behavior data file will be downloaded by the
:mod:`neural_data_simulator.scripts.post_install_config` script, so the streamer should
be able to run without any additional configuration. If the input file cannot be found,
the streamer will not be able to start.
"""
import argparse
import contextlib
import logging
from pathlib import Path
from typing import cast, Dict, Iterator, List, Optional, Tuple

from neo.rawio.blackrockrawio import BlackrockRawIO
import numpy as np
from pydantic import Extra
from pydantic_yaml import VersionedYamlModel
from rich.pretty import pprint
import yaml

from neural_data_simulator.core.outputs.lsl_output import LSLOutputDevice
from neural_data_simulator.core.outputs.lsl_output import StreamConfig
from neural_data_simulator.core.samples import Samples
from neural_data_simulator.core.settings import LogLevel
from neural_data_simulator.streamer import settings
from neural_data_simulator.streamer import streamers
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import get_configs_dir
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.runtime import unwrap
from neural_data_simulator.util.settings_loader import check_config_override_str
from neural_data_simulator.util.settings_loader import load_settings

SCRIPT_NAME = "nds-streamer"
logger = logging.getLogger(__name__)


class _Settings(VersionedYamlModel):
    """Pydantic base settings for running the streamer."""

    log_level: LogLevel
    streamer: settings.Streamer

    class Config:
        extra = Extra.forbid


class StreamGroup:
    """Utility class for managing a list of streams."""

    def __init__(self, streams_configs: List[StreamConfig]):
        """Create a new instance."""
        self.streams_configs = streams_configs
        self.lsl_outputs: List[LSLOutputDevice] = []

    def connect(self):
        """Connect all streams."""
        for stream_config in self.streams_configs:
            lsl_output = LSLOutputDevice(stream_config)
            lsl_output.connect()
            self.lsl_outputs.append(lsl_output)

    def disconnect(self):
        """Disconnect all streams."""
        for lsl_output in self.lsl_outputs:
            lsl_output.disconnect()

    @contextlib.contextmanager
    def open_connection(self) -> Iterator[None]:
        """Open a managed connection.

        The connection is released after it is consumed.
        """
        try:
            self.connect()
            yield
        finally:
            self.disconnect()


def load_blackrock_file(
    filepath: Path, output_settings: settings.LSLSimplifiedOutputModel
) -> Tuple[List[StreamConfig], List[Samples]]:
    """Parse streams from a Blackrock Neurotech file."""
    neo_io = BlackrockRawIO(filename=filepath)
    neo_io.parse_header()

    all_stream_configs, all_samples = _get_analog_streams(neo_io, output_settings)
    stream_config, samples = _get_spikes_stream(neo_io, output_settings)
    if stream_config is not None and samples is not None:
        all_stream_configs.append(stream_config)
        all_samples.append(samples)

    return all_stream_configs, all_samples


def _get_analog_streams(
    neo_io: BlackrockRawIO, output_settings: settings.LSLSimplifiedOutputModel
) -> Tuple[List[StreamConfig], List[Samples]]:
    samples = []
    stream_configs = []
    # Build stream info for each sampling group
    for stream_ix in range(neo_io.signal_streams_count()):
        stream_id = neo_io.header["signal_streams"][stream_ix]["id"]
        channels = neo_io.header["signal_channels"]
        channels = channels[channels["stream_id"] == stream_id]
        fs = channels[0]["sampling_rate"]
        analog_signals = neo_io.get_analogsignal_chunk(
            stream_index=stream_ix
        )  # Samples x channels

        proc_times = round(
            neo_io.get_signal_t_start(
                block_index=0, seg_index=0, stream_index=stream_ix
            )
            * fs
        ) + np.arange(analog_signals.shape[0], dtype=np.int64)
        analog_timestamps = proc_times / fs
        analog_timestamps = analog_timestamps - analog_timestamps[0]
        samples.append(Samples(analog_timestamps, analog_signals))

        stream_config = _get_regular_stream_config(
            stream_id, fs, channels, output_settings
        )
        stream_configs.append(stream_config)
    return stream_configs, samples


def _get_regular_stream_config(
    stream_id: str,
    sample_rate: float,
    channels: Dict,
    output_settings: settings.LSLSimplifiedOutputModel,
) -> StreamConfig:
    stream_config = StreamConfig(
        name=f"Blackrock-Group{stream_id}-Inst{output_settings.instrument.id}",
        type="Blackrock SMP",
        source_id=f"playback-SMP{stream_id}-Inst{output_settings.instrument.id}",
        acquisition={
            "manufacturer": output_settings.instrument.manufacturer,
            "model": output_settings.instrument.model,
            "instrument_id": output_settings.instrument.id,
        },
        sample_rate=sample_rate,
        channel_format=output_settings.channel_format,
        channel_labels=channels["name"],
    )
    return stream_config


def _get_spikes_stream(
    neo_io: BlackrockRawIO, output_settings: settings.LSLSimplifiedOutputModel
) -> Tuple[Optional[StreamConfig], Optional[Samples]]:
    n_spike_channels = neo_io.spike_channels_count()
    if n_spike_channels == 0:
        return None, None

    ch_ids = np.zeros((0,), dtype=int)
    un_ids = np.zeros((0,), dtype=int)
    proc_times = np.zeros((0,), dtype=np.int64)
    spike_waveforms = None
    for glbl_u_idx in range(n_spike_channels):
        ch_id, un_id = neo_io.internal_unit_ids[glbl_u_idx]
        ch_ids = np.hstack((ch_ids, [ch_id] * neo_io.spike_count(0, 0, glbl_u_idx)))
        un_ids = np.hstack((un_ids, [un_id] * neo_io.spike_count(0, 0, glbl_u_idx)))
        proc_times = np.hstack(
            (
                proc_times,
                neo_io.get_spike_timestamps(0, 0, glbl_u_idx).astype(np.int64),
            )
        )
        new_waveforms = neo_io.get_spike_raw_waveforms(
            block_index=0, seg_index=0, spike_channel_index=glbl_u_idx
        )
        if spike_waveforms is None:
            spike_waveforms = np.zeros((0, new_waveforms.shape[-1]), dtype=np.int16)
        spike_waveforms = np.vstack((spike_waveforms, new_waveforms[:, 0, :]))

    if spike_waveforms is None:
        return None, None

    re_ix = np.argsort(proc_times)
    ch_ids = ch_ids[re_ix]
    un_ids = un_ids[re_ix]
    proc_times = proc_times[re_ix]
    spike_waveforms = spike_waveforms[re_ix]
    spike_times = neo_io.rescale_spike_timestamp(proc_times, dtype="float64")
    spike_times = spike_times - spike_times[0]
    samples = Samples(
        spike_times, np.hstack((ch_ids[:, None], un_ids[:, None], spike_waveforms))
    )

    stream_config = _get_irregular_stream_config(
        spike_waveforms.shape[1], output_settings
    )
    return stream_config, samples


def _get_irregular_stream_config(
    n_waveforms: int, output_settings: settings.LSLSimplifiedOutputModel
) -> StreamConfig:
    wf_ch_labels = ["wf_" + str(-10 + _) for _ in range(n_waveforms)]
    stream_config = StreamConfig(
        name=f"Blackrock-SPK-Inst{output_settings.instrument.id}",
        type="Blackrock SPK",
        source_id=f"playback-SPK-Inst{output_settings.instrument.id}",
        acquisition={
            "manufacturer": output_settings.instrument.manufacturer,
            "model": output_settings.instrument.model,
            "instrument_id": output_settings.instrument.id,
        },
        sample_rate=0.0,
        channel_format=output_settings.channel_format,
        channel_labels=["ch_id", "unit_id"] + wf_ch_labels,
    )
    return stream_config


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Stream file neurodata to LSL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=Path(get_configs_dir()).joinpath("settings_streamer.yaml"),
        help="Path to the settings_streamer.yaml file.",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        nargs="*",
        type=check_config_override_str,
        help=(
            "Specify settings overrides as key-value pairs, separated by spaces. "
            "For example: -o log_level=DEBUG streamer.lsl_chunk_frequency=50"
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
    """Load the configuration and start the streamer."""
    initialize_logger(SCRIPT_NAME)
    args = _parse_args()
    run_settings: _Settings = cast(
        _Settings,
        load_settings(
            args.settings_path,
            settings_parser=_Settings,
            override_dotlist=args.overrides,
        ),
    )
    if args.print_settings_only:
        pprint(run_settings)
        return

    configure_logger(SCRIPT_NAME, run_settings.log_level)
    logger.debug(f"run_decoder settings:\n{yaml.dump(run_settings.dict())}")

    if run_settings.streamer.input_type == settings.StreamerInputType.NPZ:
        input_settings = unwrap(run_settings.streamer.npz).input
        samples = [
            Samples.load_from_npz(
                get_abs_path(input_settings.file),
                timestamps_array_name=input_settings.timestamps_array_name,
                data_array_name=input_settings.data_array_name,
            )
        ]
        output_settings = unwrap(run_settings.streamer.npz).output
        lsl_settings = output_settings.lsl
        stream_config = StreamConfig.from_lsl_settings(
            lsl_settings, output_settings.sampling_rate, output_settings.n_channels
        )
        stream_group = StreamGroup([stream_config])
    elif run_settings.streamer.input_type == settings.StreamerInputType.Blackrock:
        input_settings = unwrap(run_settings.streamer.blackrock).input
        output_settings = unwrap(run_settings.streamer.blackrock).output
        lsl_settings = output_settings.lsl
        stream_configs, samples = load_blackrock_file(
            Path(get_abs_path(input_settings.file)), lsl_settings
        )
        stream_group = StreamGroup(stream_configs)
    else:
        raise ValueError(f"Unsupported input type: {run_settings.streamer.input_type}")

    with stream_group.open_connection():
        streamer = streamers.LSLStreamer(
            stream_group.lsl_outputs,
            samples,
            run_settings.streamer.lsl_chunk_frequency,
            run_settings.streamer.stream_indefinitely,
        )

        try:
            streamer.stream()
        except KeyboardInterrupt:
            logger.info("CTRL+C received. Exiting...")


if __name__ == "__main__":
    run()
