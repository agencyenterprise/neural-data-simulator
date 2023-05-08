"""Script errors."""


class InvalidPluginError(Exception):
    """Plugin cannot be used."""

    pass


class UnexpectedSettingsVersion(Exception):
    """Loaded settings version is different than the expected version."""

    def __init__(self, loaded_version, expected_version):
        """Initialize the exception.

        Args:
            loaded_version: The loaded settings version.
            expected_version: The expected settings version.
        """
        self.loaded_version = loaded_version
        self.expected_version = expected_version
        self.message = (
            f"Loaded settings version {loaded_version} is "
            "different than the expected version {expected_version}."
        )
        super().__init__(self.message)
