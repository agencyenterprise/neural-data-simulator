"""Parse settings from files."""
import errno
from importlib import import_module
import logging
from pathlib import Path
import shutil
from typing import Dict, Optional, Protocol, Type

from pydantic_yaml import VersionedYamlModel
from ruamel.yaml import YAML

from neural_data_simulator.errors import SettingsMigrationError
from neural_data_simulator.errors import UnexpectedSettingsVersion
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import NDS_HOME

logger = logging.getLogger(__name__)


class Migration(Protocol):
    """Protocol for a settings migration.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def apply_migration(self, data: Dict) -> Dict:
        """Apply the migration to the settings data.

        Args:
            data: The settings data.

        Returns:
            The migrated settings data.
        """
        ...


def get_script_settings(
    settings_file: Optional[Path],
    filename: str,
    settings_parser: Type[VersionedYamlModel],
    module_name: str = "neural_data_simulator",
):
    """Load script settings from the filepath."""
    if settings_file is None:
        default_settings_file = Path(f"{NDS_HOME}/{filename}")
        if not default_settings_file.exists():
            raise FileNotFoundError(
                errno.ENOENT,
                "Default settings file not found. Run 'nds_post_install_config' "
                "to copy the default settings files.\n"
                "Alternatively, you can specify the path to the settings file via the "
                "'--settings-path' argument.",
                str(default_settings_file),
            )

        settings_file = default_settings_file
    settings_path = get_abs_path(str(settings_file))

    try:
        settings = _parse_settings(settings_path, settings_parser)
    except UnexpectedSettingsVersion as e:
        logger.info(
            f"Settings out of date. Expected version: {e.expected_version}, "
            f"found version: {e.expected_version}."
        )

        try:
            module = import_module(f"{module_name}.config.migrations")
            available_migrations = module.MIGRATIONS[filename]
        except ModuleNotFoundError:
            raise SettingsMigrationError(
                "Migration module not found. Unable to migrate settings."
            )
        except AttributeError:
            raise SettingsMigrationError(
                "Migration module does not have a MIGRATIONS attribute. "
                "Unable to migrate settings."
            )
        except KeyError:
            raise SettingsMigrationError(
                "No migrations found for this settings file. "
                "Unable to migrate settings."
            )

        migrate_settings(settings_path, e.current_version, available_migrations)
        settings = _parse_settings(settings_path, settings_parser)

    logger.info(f"Using settings from '{settings_file}'")

    return settings


def _parse_settings(settings_path: str, settings_parser: Type[VersionedYamlModel]):
    with open(settings_path) as f:
        return settings_parser.parse_raw(f.read(), proto="yaml")


def migrate_settings(
    settings_path: str, from_version: str, available_migrations: Dict[str, Migration]
):
    """Migrate the settings file to the latest available version.

        Creates a backup of the settings file before migrating the file
        through each version.

    Args:
        settings_path: The path to the settings file.
        from_version: The version to migrate from.
        available_migrations: A dictionary of available migrations. Keys are the
            version to migrate to and values are the `Migration` objects.
    """
    shutil.copy(settings_path, settings_path + f"{from_version}.bak")
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    with open(settings_path, "r+") as fp:
        fp.seek(0)
        data = yaml.load(fp)
        assert data["version"] == from_version

        newer_versions = [
            key for key in available_migrations.keys() if key >= from_version
        ]
        logger.info(f"Applying migrations: {newer_versions} to {settings_path}")
        for version in newer_versions:
            data = available_migrations[version].apply_migration(data)

        fp.seek(0)
        yaml.dump(data, fp)
        fp.truncate()
