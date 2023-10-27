"""Test settings_loader.py module."""
import argparse
from pathlib import Path
from unittest.mock import call
from unittest.mock import mock_open
from unittest.mock import patch

import pydantic
from pydantic import Extra
from pydantic_yaml import VersionedYamlModel
import pytest

from neural_data_simulator.util import settings_loader


class _SettingsModel(VersionedYamlModel):
    number: int

    class Config:
        extra = Extra.forbid


class TestLoadSettings:
    """Test settings_loader.load_settings utility."""

    @patch(
        "builtins.open", new_callable=mock_open, read_data="version: 1.0.0\nnumber: 1"
    )
    def test_settings_loader_returns_parsed_settings(self, mock_open):
        """Test calling the settings_loader with a known settings_file."""
        settings_file = Path("some/path/on/disk")
        settings = settings_loader.load_settings(
            settings_file, settings_parser=_SettingsModel
        )
        assert settings.number == 1
        assert mock_open.call_count == 1
        assert mock_open.mock_calls[0] == call(settings_file, "r")

    @patch("builtins.open", new_callable=mock_open)
    def test_settings_loader_raises_exception_when_settings_file_is_not_found(
        self, mock_open
    ):
        """Test settings_loader returns settings from default file."""
        mock_open.side_effect = FileNotFoundError
        with pytest.raises(FileNotFoundError):
            settings_loader.load_settings("missing_file", _SettingsModel)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="version: 1.0.0\nnumber: not_an_int",
    )
    def test_settings_loader_raises_exception_when_parsing_fails(self, mock_open):
        """Test settings_loader validates input types."""
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            settings_loader.load_settings("some/path/on/disk", _SettingsModel)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="version: 1.0.0\nnumber: 1\nextra_field: 1",
    )
    def test_settings_loader_raises_exception_when_extra_yaml_field(self, mock_open):
        """Test calling the settings_loader with a known settings_file and overrides."""
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            settings_loader.load_settings("some/path/on/disk", _SettingsModel)

    @patch(
        "builtins.open", new_callable=mock_open, read_data="version: 1.0.0\nnumber: 1"
    )
    def test_settings_loader_overrides(self, mock_open):
        """Test calling the settings_loader with a known settings_file and overrides."""
        settings_file = Path("some/path/on/disk")
        settings = settings_loader.load_settings(
            settings_file,
            settings_parser=_SettingsModel,
            override_dotlist=["number=3"],
        )
        assert settings.number == 3

    @patch(
        "builtins.open", new_callable=mock_open, read_data="version: 1.0.0\nnumber: 1"
    )
    def test_settings_loader_raises_exception_when_invalid_override_key(
        self, mock_open
    ):
        """Test calling the settings_loader with a known settings_file and overrides."""
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            settings_loader.load_settings(
                "some/path/on/disk", _SettingsModel, override_dotlist=["extra_field=1"]
            )


class TestCheckConfigOverrideStr:
    """Test settings_loader.check_config_override_str."""

    def test_check_config_override_valid(self):
        """Test valid config override."""
        assert settings_loader.check_config_override_str("key=value") == "key=value"
        assert (
            settings_loader.check_config_override_str("key.subkey=value")
            == "key.subkey=value"
        )

    def test_check_config_override_str_invalid(self):
        """Test invalid config override."""
        with pytest.raises(argparse.ArgumentTypeError):
            settings_loader.check_config_override_str("=value")
        with pytest.raises(argparse.ArgumentTypeError):
            settings_loader.check_config_override_str("key..subkey=value")
