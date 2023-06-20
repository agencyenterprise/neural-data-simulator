"""Test settings_loader.py module."""
import inspect
import os.path
import sys
import types
from unittest.mock import call
from unittest.mock import mock_open
from unittest.mock import patch

from pydantic import validator
from pydantic_yaml import VersionedYamlModel
import pytest

from neural_data_simulator.errors import SettingsMigrationError
from neural_data_simulator.errors import UnexpectedSettingsVersion
from neural_data_simulator.util import settings_loader
from neural_data_simulator.util.runtime import NDS_HOME


@pytest.fixture(autouse=True)
def test_settings_path(tmp_path):
    """Create a test_settings.yaml file in a temporary path."""
    test_settings_yaml = os.path.join(tmp_path, "test_settings.yaml")
    with open(test_settings_yaml, "w") as f:
        f.write("version: 1.0.0\nnumber: 1")
    return test_settings_yaml


class _SettingsModel(VersionedYamlModel):
    number: int


class _UpgradedSettingsModel(VersionedYamlModel):
    """All settings for the NDS main package."""

    added_in_1_0_1: int
    added_in_1_0_2: int

    @validator("version")
    def _expected_settings_version(cls, v):
        expected_version = "1.0.2"
        if v != expected_version:
            raise UnexpectedSettingsVersion(v, expected_version)
        return v


class TestSettingsLoader:
    """Test settings_loader utility."""

    def test_settings_loader_returns_parsed_settings_from_given_file(
        self, test_settings_path
    ):
        """Test calling the settings_loader with a known settings_file."""
        settings = settings_loader.get_script_settings(
            test_settings_path, "test_settings.yaml", _SettingsModel
        )
        assert settings.number == 1

    @patch(
        "builtins.open", new_callable=mock_open, read_data="version: 1.0.0\nnumber: 1"
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
        self, mock_path_exists
    ):
        """Test settings_loader raises FileNotFoundError."""
        settings_file = None
        mock_path_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            settings_loader.get_script_settings(
                settings_file, "test_settings.yaml", _SettingsModel
            )

    def test_raises_migration_error_for_nonexisting_migration(self, test_settings_path):
        """Test calling the settings_loader with an unmigrated settings file."""
        with pytest.raises(SettingsMigrationError) as e:
            # Settings model expects a newer version than 1.0.0
            # but no migration is found for the file `test_settings.yaml`
            settings_loader.get_script_settings(
                test_settings_path, "test_settings.yaml", _UpgradedSettingsModel
            )
        assert str(e.value) == str(
            "No migrations found for this settings file. " "Unable to migrate settings."
        )

    def test_raises_migration_error_for_nonexisting_module(self, test_settings_path):
        """Test settings_loader with a settings file and an unknown migration module."""
        with pytest.raises(SettingsMigrationError) as e:
            # Settings model expects a newer version than 1.0.0
            # but the provided settings module is not found
            settings_loader.get_script_settings(
                test_settings_path,
                "test_settings.yaml",
                _UpgradedSettingsModel,
                module_name="nonexisting",
            )
        assert str(e.value) == "Migration module not found. Unable to migrate settings."

    def test_raises_migration_error_for_wrong_existing_module(self, test_settings_path):
        """Test settings_loader with a settings file and an bad migration module.

        The migration module exists but does not have a MIGRATIONS attribute.
        """
        # Use a unique name just in case we run tests in parallel
        module_name = inspect.currentframe().f_code.co_name
        migrations_module = types.ModuleType("migrations", "The migrations module")
        sys.modules[f"{module_name}.config.migrations"] = migrations_module

        with pytest.raises(SettingsMigrationError) as e:
            # Settings model expects a newer version than 1.0.0
            # but the provided settings module does not have a MIGRATIONS attribute
            settings_loader.get_script_settings(
                test_settings_path,
                "test_settings.yaml",
                _UpgradedSettingsModel,
                module_name,
            )
        assert str(e.value) == str(
            "Migration module does not have a MIGRATIONS attribute. "
            "Unable to migrate settings."
        )

    def test_apply_migrations(self, test_settings_path):
        """Test settings_loader with a settings file and an bad migration module.

        The migration module exists but does not have a MIGRATIONS attribute.
        """
        # Use a unique name just in case we run tests in parallel
        module_name = inspect.currentframe().f_code.co_name
        migrations_module = types.ModuleType("migrations", "The migrations module")
        sys.modules[f"{module_name}.config.migrations"] = migrations_module

        def _migrate_100_to_101(settings):
            assert settings["version"] == "1.0.0"
            settings["version"] = "1.0.1"
            settings["added_in_1_0_1"] = 1
            return settings

        def _migrate_101_to_102(settings):
            assert settings["version"] == "1.0.1"
            settings["version"] = "1.0.2"
            settings["added_in_1_0_2"] = 2
            return settings

        migration_100_to_101 = types.ModuleType("migration", "A migration module")
        migration_100_to_101.apply_migration = _migrate_100_to_101

        migration_101_to_102 = types.ModuleType("migration", "A migration module")
        migration_101_to_102.apply_migration = _migrate_101_to_102

        migrations_module.MIGRATIONS = {
            "test_settings.yaml": {
                "1.0.0": migration_100_to_101,
                "1.0.1": migration_101_to_102,
            }
        }

        settings = settings_loader.get_script_settings(
            test_settings_path,
            "test_settings.yaml",
            _UpgradedSettingsModel,
            module_name,
        )
        assert settings.added_in_1_0_1 == 1
        assert settings.added_in_1_0_2 == 2
