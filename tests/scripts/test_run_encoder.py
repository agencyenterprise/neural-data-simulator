"""Test for run_encoder.py."""

import os
from unittest.mock import Mock

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytest

import neural_data_simulator
from neural_data_simulator import inputs
from neural_data_simulator import models
from neural_data_simulator import outputs
from neural_data_simulator.encoder import Encoder
from neural_data_simulator.encoder import Processor
from neural_data_simulator.scripts import run_encoder
from neural_data_simulator.settings import EncoderEndpointType
from neural_data_simulator.settings import EncoderSettings
from neural_data_simulator.settings import Settings

velocity_tuning_curves = {
    "b0": np.array([1]),
    "m": np.array([2]),
    "pd": np.array([3]),
    "bs": np.array([4]),
}


@pytest.fixture(autouse=True)
def mock_numpy_load(monkeypatch: pytest.MonkeyPatch):
    """Mock numpy load."""
    numpy_load_mock = Mock()
    monkeypatch.setattr("numpy.load", numpy_load_mock)
    numpy_load_mock.side_effect = [velocity_tuning_curves]
    return numpy_load_mock


@pytest.fixture(autouse=True)
def mock_load_module(monkeypatch: pytest.MonkeyPatch):
    """Mock load_module to avoid the actual loading on plugins."""
    mock = Mock()
    monkeypatch.setattr("neural_data_simulator.scripts.run_encoder.load_module", mock)
    return mock


@pytest.fixture
def default_hydra_config() -> DictConfig:
    """Mock hydra-loader to return the default settings."""
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
def fake_runner(monkeypatch):
    """Fake the runner that iterates the encoder."""

    class runner_fake(object):
        encoder = None

        @staticmethod
        def run(encoder: Encoder, *args, **kwargs):
            runner_fake.encoder = encoder

    monkeypatch.setattr("neural_data_simulator.scripts.run_encoder.runner", runner_fake)
    return runner_fake


class TestRunEncoder:
    """Test execution of the run_encoder script."""

    def _get_encoder_settings(self, default_settings) -> EncoderSettings:
        return default_settings.encoder

    def test_run_encoder(self, default_hydra_config, fake_runner):
        """Test run with default config."""
        run_encoder.run_with_config(default_hydra_config)

        encoder = fake_runner.encoder
        assert isinstance(encoder.input, inputs.LSLInput)
        assert isinstance(encoder.output, outputs.LSLOutputDevice)
        assert encoder.preprocessor is None
        assert isinstance(encoder.postprocessor, Processor)
        assert isinstance(encoder.model, models.EncoderModel)

    def test_run_encoder_with_file_input_and_file_output(
        self, fake_runner, default_hydra_config, mock_numpy_load
    ):
        """Test run with input from file and output to file."""
        hydra_config = default_hydra_config
        hydra_config.encoder.input.type = EncoderEndpointType.FILE
        hydra_config.encoder.output.type = EncoderEndpointType.FILE

        mock_numpy_load.side_effect = [
            # Input samples
            {"timestamps_train": np.array([1.0]), "vel_train": np.array([2.5])},
            velocity_tuning_curves,
        ]

        run_encoder.run_with_config(hydra_config)

        encoder = fake_runner.encoder
        assert isinstance(encoder.input, inputs.SamplesInput)
        assert isinstance(encoder.output, outputs.FileOutput)
        assert encoder.preprocessor is None
        assert isinstance(encoder.postprocessor, Processor)
        assert isinstance(encoder.model, models.EncoderModel)

    def test_run_encoder_with_preprocessor_and_postprocessor(
        self, fake_runner, default_hydra_config
    ):
        """Test run with a configured preprocessor and postprocessor."""
        hydra_config = default_hydra_config

        hydra_config.encoder.preprocessor = "path_a.py"
        hydra_config.encoder.postprocessor = "path_b.py"

        run_encoder.run_with_config(hydra_config)

        encoder = fake_runner.encoder
        assert encoder.preprocessor is not None
        assert encoder.postprocessor is not None
