"""Utilities for handling data in the desired NDS format."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import os

from numpy import ndarray
import numpy as np


@dataclass
class Samples:
    """Unified collection of timestamps and data points."""

    timestamps: ndarray
    """Timestamps for each data sample. Each row corresponds to a data sample."""

    data: ndarray
    """Array of data samples. Each row corresponds to a data sample, while each column
    corresponds to a dimension of the data sample."""

    def __post_init__(self):
        """Execute validation checks on the data."""
        _validate_inputs(self.timestamps, self.data)

    def __len__(self):
        """Return the number of data points."""
        return next(iter(np.shape(self.timestamps)), 0)

    @property
    def empty(self) -> bool:
        """Check if the samples are empty.

        Returns:
            True if there are no data points.
        """
        return self.__len__() == 0

    def __eq__(self, o: object) -> bool:
        """Compare data and timestamps of two samples objects."""
        if not isinstance(o, Samples):
            return False
        return np.array_equal(self.data, o.data) and np.array_equal(
            self.timestamps, o.timestamps
        )

    @classmethod
    def empty_samples(cls) -> Samples:
        """Create an empty samples instance.

        Returns:
            Samples instance with empty timestamps and data arrays.
        """
        return Samples(np.array([]), np.array([]))

    @classmethod
    def load_from_npz(
        cls,
        filepath: str,
        timestamps_array_name: str = "timestamps",
        data_array_name: str = "data",
    ) -> Samples:
        """Load the timestamps and data from the file into a new samples instance.

        Args:
            filepath: `.npz` file path with the timestamps and data
            timestamps_array_name: Name of the timestamp array defined
              when creating the file (see `np.savez` documentation for details). The
              loaded array should be in the shape of (Nx1), N = number of samples.
              Defaults to "timestamps"
            data_array_name: Name of the data array defined when creating the file
              (see `np.savez` documentation for details). The loaded array should be
              in the shape of (NxM), N = number of samples and M is the number of
              channels. Defaults to "data".
        """
        file_data = cls._load_file_data(filepath)
        return Samples(
            timestamps=file_data[timestamps_array_name], data=file_data[data_array_name]
        )

    @classmethod
    def _check_file_exists(cls, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    @classmethod
    def _load_file_data(cls, filepath: str) -> ndarray:
        cls._check_file_exists(filepath)
        return np.load(filepath)


def _validate_inputs(timestamps: ndarray, data: ndarray):
    _validate_timestamps_shape(timestamps)
    _validate_data_shape(data)
    _validate_shapes_match(timestamps, data)


def _validate_timestamps_shape(timestamps: ndarray):
    if (len(timestamps.shape) == 2 and timestamps.shape[1] > 1) or (
        len(timestamps.shape) > 2
    ):
        raise ValueError(
            "Timestamps should be of shape (N, 1) or (N,) where N is the number of"
            " samples"
        )


def _validate_data_shape(data: ndarray):
    if (len(data.shape) == 2 and data.shape[1] == 0) or (len(data.shape) > 2):
        raise ValueError(
            "data should be of shape (N, C) where N is the number of data points and C"
            " is the number of channels"
        )


def _validate_shapes_match(timestamps: ndarray, data: ndarray):
    """Execute validation checks on the data."""
    data_points = next(iter(np.shape(data)), 0)
    time_points = next(iter(np.shape(timestamps)), 0)
    if data_points != time_points:
        raise ValueError(
            f"Number of data points ({data_points}) does not match the"
            f" number of timestamps ({time_points})"
        )
