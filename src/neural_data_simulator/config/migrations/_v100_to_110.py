from typing import Dict


def apply_migration(data: Dict) -> Dict:
    """Apply the migration to the settings data.

    Args:
        data: The settings data.

    Returns:
        The migrated settings data.
    """
    assert data["version"] == "1.0.0"

    noise = data["ephys_generator"]["noise"]
    noise["type"] = "gaussian"
    noise["file"] = {"path": "psd.npz", "psd_array_name": "PSD"}
    noise["gaussian"] = {"beta": noise["beta"], "fmin": noise["fmin"]}

    data["version"] = "1.1.0"
    return data
