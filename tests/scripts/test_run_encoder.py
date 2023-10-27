"""Test for run_encoder.py."""

import argparse
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import neural_data_simulator
from neural_data_simulator.core import inputs
from neural_data_simulator.core import models
from neural_data_simulator.core.encoder import Encoder
from neural_data_simulator.core.encoder import Processor
from neural_data_simulator.core.outputs.lsl_output import LSLOutputDevice
from neural_data_simulator.core.outputs import api as outputs
from neural_data_simulator.core.settings import EncoderEndpointType
from neural_data_simulator.core.settings import EncoderSettings
from neural_data_simulator.core.settings import Settings
from neural_data_simulator.scripts import run_encoder
from neural_data_simulator.util import settings_loader

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
def mock_importlib(monkeypatch: pytest.MonkeyPatch):
    """Mock importlib to avoid the actual loading on plugins."""
    machinery_mock = Mock()
    monkeypatch.setattr("importlib.machinery", machinery_mock)
    util_mock = Mock()
    monkeypatch.setattr("importlib.util", util_mock)
    return machinery_mock, util_mock


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
        "neural_data_simulator.scripts.run_encoder.get_script_settings",
        get_script_settings_mock,
    )
    return get_script_settings_mock


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

    def _get_encoder_settings(self, mock_get_script_settings) -> EncoderSettings:
        return mock_get_script_settings.return_value.encoder

    def test_run_encoder(self, fake_runner):
        """Test run with default config."""
        run_encoder.run()

        encoder = fake_runner.encoder
        assert isinstance(encoder.input, inputs.LSLInput)
        assert isinstance(encoder.output, LSLOutputDevice)
        assert encoder.preprocessor is None
        assert isinstance(encoder.postprocessor, Processor)
        assert isinstance(encoder.model, models.EncoderModel)

    def test_run_encoder_with_file_input_and_file_output(
        self, fake_runner, mock_get_script_settings, mock_numpy_load
    ):
        """Test run with input from file and output to file."""
        encoder_settings = self._get_encoder_settings(mock_get_script_settings)
        encoder_settings.input.type = EncoderEndpointType.FILE
        encoder_settings.output.type = EncoderEndpointType.FILE

        mock_numpy_load.side_effect = [
            # Input samples
            {"timestamps_train": np.array([1.0]), "vel_train": np.array([2.5])},
            velocity_tuning_curves,
        ]

        run_encoder.run()

        encoder = fake_runner.encoder
        assert isinstance(encoder.input, inputs.SamplesInput)
        assert isinstance(encoder.output, outputs.FileOutput)
        assert encoder.preprocessor is None
        assert isinstance(encoder.postprocessor, Processor)
        assert isinstance(encoder.model, models.EncoderModel)

    def test_run_encoder_with_preprocessor_and_postprocessor(
        self, fake_runner, mock_get_script_settings
    ):
        """Test run with a configured preprocessor and postprocessor."""
        encoder_settings = self._get_encoder_settings(mock_get_script_settings)
        encoder_settings.preprocessor = "path_a"
        encoder_settings.postprocessor = "path_b"

        run_encoder.run()

        encoder = fake_runner.encoder
        assert encoder.preprocessor is not None
        assert encoder.postprocessor is not None
