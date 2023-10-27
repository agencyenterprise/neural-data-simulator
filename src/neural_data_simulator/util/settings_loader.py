"""Parse settings from files."""
import argparse
import logging
import os
from typing import Optional, Type

from omegaconf import OmegaConf
from pydantic import BaseModel
import pydantic.error_wrappers
import yaml

from neural_data_simulator.util.runtime import get_configs_dir

logger = logging.getLogger(__name__)


def load_settings(
    settings_file: os.PathLike,
    settings_parser: Type[BaseModel],
    override_dotlist: Optional[list[str]] = None,
):
    """Load settings from a YAML file and parse them into a Pydantic model.

    Args:
        settings_file: Path to the YAML file containing the settings.
        settings_parser: Pydantic model to parse the settings into.
        override_dotlist: Optional list of dot-separated key-value pairs to override
            the settings loaded from the file.

    Returns:
        An instance of the `settings_parser` model containing the parsed settings.

    Raises:
        FileNotFoundError: If the `settings_file` is not found.
        pydantic.error_wrappers.ValidationError: If the settings file/overrides do not
            match the schema
    """
    try:
        with open(settings_file, "r") as f:
            settings_dict = yaml.safe_load(f)
    except FileNotFoundError as file_error:
        raise FileNotFoundError(
            f"Settings file not found: {settings_file}.\n"
            "\tRun 'nds_post_install_config' to copy the default settings files\n"
            f"\tto: {get_configs_dir()}\n"
            "\tAlternatively, you can specify the path to the settings file via the\n"
            "\t'--settings-path' argument."
        ) from file_error

    # Validate settings file
    try:
        settings = settings_parser.parse_obj(settings_dict)
    except pydantic.error_wrappers.ValidationError:
        logger.error("Settings file does not match expected schema.")
        raise

    if override_dotlist:
        override_conf = OmegaConf.from_dotlist(override_dotlist)
        merged_conf = OmegaConf.merge(settings_dict, override_conf)
        settings_dict = OmegaConf.to_object(merged_conf)
        # Re-validate merged config
        # If validation is slow, we can use `pydantic-partial` to only
        # re-validate the dot-list overrides.
        try:
            settings = settings_parser.parse_obj(settings_dict)
        except pydantic.error_wrappers.ValidationError:
            logger.error("Overrides list does not match expected schema.")
            raise

    return settings


def check_config_override_str(value: str) -> str:
    """Custom argparse type to check for individual dot-list arguments.

    Args:
        value (str): The value to check.

    Raises:
        argparse.ArgumentTypeError: If the key or subkeys are empty
            Note: this is a pre-checker for OmegaConf.from_dotlist,
            which actually handles a wide variety of str formats.

    Returns:
        str: The value if it is in the expected format.
    """
    parts = value.split("=")
    key = parts[0]

    key_parts = key.split(".")
    if (not key_parts) or ("" in key_parts):
        raise argparse.ArgumentTypeError(
            f"Invalid config-override: {value}\n"
            "\tExpected format: `key=value` or `key.subkey=value`"
        )
    return value
