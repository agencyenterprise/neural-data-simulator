"""Test for run_ephys_generator.py."""

import argparse
import os
from pathlib import Path
from unittest.mock import call
from unittest.mock import Mock

import pylsl
import pytest

import neural_data_simulator
from neural_data_simulator.core.ephys_generator import ContinuousData
from neural_data_simulator.core.ephys_generator import Spikes
from neural_data_simulator.core.settings import Settings
from neural_data_simulator.scripts import run_ephys_generator
from neural_data_simulator.util import settings_loader


@pytest.fixture(autouse=True)
def fake_parse_args(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    """Fake command line arguments passed to the script."""
    parse_args_result = argparse.Namespace(settings_path=None)

    def parse_args(self, args=None, namespace=None):
        return parse_args_result

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", parse_args)
    return parse_args_result


@pytest.fixture(autouse=True)
def mock_get_script_settings(monkeypatch: pytest.MonkeyPatch):
    """Mock get_script_settings to return the default settings."""
    default_settings: Settings = settings_loader.get_script_settings(
        Path(f"{os.path.dirname(neural_data_simulator.__file__)}/config/settings.yaml"),
        "settings.yaml",
        Settings,
    )
    get_script_settings_mock = Mock()
    get_script_settings_mock.return_value = default_settings
    monkeypatch.setattr(
        "neural_data_simulator.scripts.run_ephys_generator.get_script_settings",
        get_script_settings_mock,
    )
    return get_script_settings_mock


@pytest.fixture(autouse=True)
def mock_process_output(monkeypatch):
    """Mock the process that runs the ephys generator."""
    process_output_mock = Mock()
    monkeypatch.setattr(
        "neural_data_simulator.scripts.run_ephys_generator.ProcessOutput",
        process_output_mock,
    )
    return process_output_mock


@pytest.fixture(scope="class", autouse=True)
def fake_spike_rates_outlet():
    """Set up a fake spike rates outlet."""
    stream_info = pylsl.StreamInfo(
        name="NDS-SpikeRates",
        type="behavior",
        channel_count=2,
        nominal_srate=50,
        channel_format="int16",
        source_id="a-test-fake",
    )
    return pylsl.stream_outlet(stream_info)


class TestRunEphysGenerator:
    """Test execution of the run_ephys_generator script."""

    def test_run_ephys_generator(self, mock_process_output):
        """Test run with default config."""
        run_ephys_generator.run()

        [
            po_params,
            po_lsl_out,
            po_init,
            po_start,
            po_stop,
        ] = mock_process_output.mock_calls

        assert len(po_lsl_out.args) == 3
        assert po_params == call.Params(1, 1000.0, 30000.0, 0.25)
        assert po_init.args[0]._params == ContinuousData.Params(
            raw_data_frequency=30000.0,
            n_units_per_channel=1,
            n_samples_waveform=48,
            lfp_data_frequency=1000.0,
            lfp_filter_cutoff=300.0,
            lfp_filter_order=4,
        )
        assert po_init.args[1].spike_times._params == Spikes.Params(
            raw_data_frequency=30000.0,
            n_units_per_channel=1,
            refractory_time=0.001,
        )
        assert po_start == call().start()
        assert po_stop == call().stop()
