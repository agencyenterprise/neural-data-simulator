"""Test for run_streamer.py."""

import argparse
import os.path
from unittest.mock import Mock

import hydra
import numpy as np
import pytest

from neural_data_simulator import streamer
from neural_data_simulator.streamer import run_streamer
from neural_data_simulator.streamer import settings


class BlackrockRawIOFake:
    """Fake BlackrockRawIO."""

    def __init__(self, *args, **kwargs):
        """Fake init method."""
        self.header = {
            "signal_streams": {0: {"id": "3"}, 1: {"id": "6"}},
            "signal_channels": np.array(
                [
                    ("chan1", "1", 2000.0, "int16", "uV", 0.25, 0.0, "3"),
                    ("chan1", "2", 2000.0, "int16", "uV", 0.25, 0.0, "3"),
                    ("chan1", "1", 2000.0, "int16", "uV", 0.25, 0.0, "6"),
                    ("chan1", "2", 2000.0, "int16", "uV", 0.25, 0.0, "6"),
                ],
                dtype=[
                    ("name", "U16"),
                    ("id", "U16"),
                    ("sampling_rate", "f8"),
                    ("dtype", "U16"),
                    ("units", "U16"),
                    ("gain", "f8"),
                    ("offset", "f8"),
                    ("stream_id", "U16"),
                ],
            ),
        }
        self.internal_unit_ids = [(1, 0), (2, 0), (1, 1), (2, 1)]

    def parse_header(self):
        """Fake parse_header method."""
        pass

    def signal_streams_count(self):
        """Fake signal_streams_count method."""
        return 2

    def get_analogsignal_chunk(self, stream_index):
        """Fake get_analogsignal_chunk method."""
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    def get_signal_t_start(self, block_index, seg_index, stream_index=None):
        """Fake get_signal_t_start method."""
        return 0.0

    def spike_channels_count(self):
        """Fake spike_channels_count method."""
        return 1

    def spike_count(self, block_index=0, seg_index=0, spike_channel_index=0):
        """Fake spike_count method."""
        return 1

    def get_spike_timestamps(
        self,
        block_index=0,
        seg_index=0,
        spike_channel_index=0,
        t_start=None,
        t_stop=None,
    ):
        """Fake get_spike_timestamps method."""
        return np.array([1.0])

    def get_spike_raw_waveforms(
        self,
        block_index=0,
        seg_index=0,
        spike_channel_index=0,
        t_start=None,
        t_stop=None,
    ):
        """Fake get_spike_raw_waveforms method."""
        return np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])

    def rescale_spike_timestamp(self, spike_timestamps, dtype):
        """Fake rescale_spike_timestamp method."""
        return spike_timestamps


@pytest.fixture(autouse=True)
def fake_blackrockrawio(monkeypatch):
    """Override the nnds.scripts.run_streamer.BlackrockRawIO with a fake."""
    monkeypatch.setattr(streamer.run_streamer, "BlackrockRawIO", BlackrockRawIOFake)


@pytest.fixture(autouse=True)
def fake_parse_args(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    """Fake command line arguments passed to the script."""
    parse_args_result = argparse.Namespace(settings_path=None)

    def parse_args(self, args=None, namespace=None):
        return parse_args_result

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", parse_args)
    return parse_args_result


@pytest.fixture(autouse=True)
def mock_numpy_load(monkeypatch: pytest.MonkeyPatch):
    """Mock numpy load."""
    numpy_load_mock = Mock()
    monkeypatch.setattr("numpy.load", numpy_load_mock)
    numpy_load_mock.side_effect = [
        {"timestamps_train": np.array([1.0]), "vel_train": np.array([[2.5, 3.5]])}
    ]
    return numpy_load_mock


@pytest.fixture(autouse=True)
def mock_path_exists(monkeypatch: pytest.MonkeyPatch):
    """Mock the os function that checks if a path exists."""
    path_exists_mock = Mock()
    path_exists_mock.return_value = True
    monkeypatch.setattr("os.path.exists", path_exists_mock)
    return path_exists_mock


@pytest.fixture(autouse=True)
def default_hydra_config() -> run_streamer._Settings:
    """Mock get_script_settings to return the default settings."""
    package_dir = os.path.dirname(streamer.__file__)
    with hydra.initialize_config_dir(
        config_dir=os.path.join(package_dir, "config"), version_base="1.3"
    ):
        cfg = hydra.compose("settings_streamer.yaml")
    cfg.streamer.stream_indefinitely = False

    return cfg


class TestRunStreamer:
    """Test execution of the run_streamer script."""

    def test_run_streamer(self, default_hydra_config):
        """Test run with default config."""
        run_streamer.run_with_config(default_hydra_config)

    def test_run_blackrock_streamer(self, default_hydra_config):
        """Test run with blackrock config."""
        default_hydra_config.streamer.input_type = settings.StreamerInputType.Blackrock
        run_streamer.run_with_config(default_hydra_config)
