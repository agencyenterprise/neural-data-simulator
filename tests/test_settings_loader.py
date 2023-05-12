"""Test settings_loader.py module."""
import os.path
from pathlib import Path
import sys
import types
from unittest.mock import call
from unittest.mock import mock_open
from unittest.mock import patch

from pydantic_yaml import VersionedYamlModel
import pytest

from neural_data_simulator.errors import SettingsMigrationError
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
    def test_raises_exception_when_default_settings_file_is_not_found(
        self, mock_path_exists, mock_open
    ):
        """Test settings_loader returns settings from default file."""
        settings_file = None
        mock_path_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            settings_loader.get_script_settings(
                settings_file, "test_settings.yaml", _SettingsModel
            )

    def test_raises_migration_error_for_nonexisting_migration(self, mock_open):
        """Test calling the settings_loader with an unmigrated settings file."""
        settings_file = Path("some/path/on/disk")
        with pytest.raises(SettingsMigrationError) as e:
            # Settings model expects a newer version than 1.0.0
            # but no migration is found for the file `test_settings.yaml`
            settings_loader.get_script_settings(
                settings_file, "test_settings.yaml", Settings
            )
        assert str(e.value) == str(
            "No migrations found for this settings file. " "Unable to migrate settings."
        )

    def test_raises_migration_error_for_nonexisting_module(self, mock_open):
        """Test settings_loader with a settings file and an unknown migration module."""
        settings_file = Path("some/path/on/disk")
        with pytest.raises(SettingsMigrationError) as e:
            # Settings model expects a newer version than 1.0.0
            # but the provided settings module is not found
            settings_loader.get_script_settings(
                settings_file, "test_settings.yaml", Settings, module_name="nonexisting"
            )
        assert str(e.value) == "Migration module not found. Unable to migrate settings."

    def test_raises_migration_error_for_wrong_existing_module(self, mock_open):
        """Test settings_loader with a settings file and an bad migration module.

        The migration module exists but does not have a MIGRATIONS attribute.
        """
        settings_file = Path("some/path/on/disk")
        with pytest.raises(SettingsMigrationError) as e:
            # Settings model expects a newer version than 1.0.0
            # but the provided settings module does not have a MIGRATIONS attribute
            migrations_module = types.ModuleType("migrations", "The migrations module")
            sys.modules["tests.config.migrations"] = migrations_module
            settings_loader.get_script_settings(
                settings_file, "test_settings.yaml", Settings, module_name="tests"
            )
        assert str(e.value) == str(
            "Migration module does not have a MIGRATIONS attribute. "
            "Unable to migrate settings."
        )
