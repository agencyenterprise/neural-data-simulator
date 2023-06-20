"""Script errors."""


class InvalidPluginError(Exception):
    """Plugin cannot be used."""

    pass


class SettingsMigrationError(Exception):
    """Cannot migrate settings."""

    pass


class UnexpectedSettingsVersion(Exception):
    """Loaded settings version is different than the expected version."""

    def __init__(self, current_version, expected_version):
        """Initialize the exception.

        Args:
            current_version: The loaded settings version.
            expected_version: The expected settings version.
        """
        self.current_version = current_version
        self.expected_version = expected_version
        self.message = (
            f"Loaded settings version {current_version} is "
            f"different than the expected version {expected_version}."
        )
        super().__init__(self.message)
