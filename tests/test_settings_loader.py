"""Test settings_loader.py module."""
import os.path
from pathlib import Path
from unittest.mock import call
from unittest.mock import mock_open
from unittest.mock import patch

from pydantic_yaml import VersionedYamlModel
import pytest

from neural_data_simulator.settings import Settings
from neural_data_simulator.util import settings_loader
from neural_data_simulator.util.runtime import NDS_HOME


class _SettingsModel(VersionedYamlModel):
    number: int


@patch("builtins.open", new_callable=mock_open, read_data="version: 1.0.0\nnumber: 1")
class TestSettingsLoader:
    """Test settings_loader utility."""

    def test_settings_loader_returns_parsed_settings_from_given_file(self, mock_open):
        """Test calling the settings_loader with a known settings_file."""
        settings_file = Path("some/path/on/disk")
        settings = settings_loader.get_script_settings(
            settings_file, "test_settings.yaml", _SettingsModel
        )
        assert settings.number == 1
        assert mock_open.call_count == 1
        assert mock_open.mock_calls[0] == call(
            os.path.join(NDS_HOME, "some", "path", "on", "disk")
        )

    @patch("pathlib.Path.exists")
    def test_settings_loader_returns_parsed_settings_from_default_file(
        self, mock_path_exists, mock_open
    ):
        """Test settings_loader returns settings from default file."""
        settings_file = None
        mock_path_exists.return_value = True
        settings = settings_loader.get_script_settings(
            settings_file, "test_settings.yaml", _SettingsModel
        )
        assert settings.number == 1
        assert mock_open.call_count == 1
        assert mock_open.mock_calls[0] == call(
            os.path.join(NDS_HOME, "test_settings.yaml")
        )

    @patch("pathlib.Path.exists")
    def test_settings_loader_raises_exception_when_default_settings_file_is_not_found(
        self, mock_path_exists, mock_open
    ):
        """Test settings_loader returns settings from default file."""
        settings_file = None
        mock_path_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            settings_loader.get_script_settings(
                settings_file, "test_settings.yaml", _SettingsModel
            )

    def test_settings_loader_raises_value_error_for_wrong_version(self, mock_open):
        """Test calling the settings_loader with an unexpected settings version."""
        settings_file = Path("some/path/on/disk")
        with pytest.raises(ValueError):
            # Settings model expects version 1.0.1 and we are passing 1.0.0
            settings_loader.get_script_settings(
                settings_file, "test_settings.yaml", Settings
            )
