"""Test for run_ephys_generator.py."""

import os
from unittest.mock import call
from unittest.mock import Mock

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pylsl
import pytest

import neural_data_simulator
from neural_data_simulator.core.ephys_generator import ContinuousData
from neural_data_simulator.core.ephys_generator import Spikes
from neural_data_simulator.core.settings import Settings
from neural_data_simulator.scripts import run_ephys_generator


@pytest.fixture
def default_hydra_config() -> DictConfig:
    """Hydra-loader to return the default settings."""
    package_dir = os.path.dirname(neural_data_simulator.__file__)
    with hydra.initialize_config_dir(
        config_dir=os.path.join(package_dir, "config"), version_base="1.3"
    ):
        cfg = hydra.compose("settings.yaml")

    return cfg


@pytest.fixture
def default_settings(default_hydra_config: DictConfig) -> Settings:
    """Convert hydra-loaded config to Settings object."""
    cfg_resolved = OmegaConf.to_object(default_hydra_config)
    settings = Settings.parse_obj(cfg_resolved)

    return settings


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

    def test_run_ephys_generator(self, default_hydra_config, mock_process_output):
        """Test run with default config."""
        run_ephys_generator.run_with_config(default_hydra_config)

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
