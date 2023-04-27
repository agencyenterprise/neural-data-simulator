"""Tests for Samples.py module."""
import numpy as np
import pytest

from neural_data_simulator.samples import Samples


class TestSamples:
    """Tests for Samples class."""

    def test_instantiate_valid_samples(self):
        """Test creating samples with valid data and timestamps."""
        Samples(timestamps=np.array([0, 1]), data=np.array([[1, 2, 3], [4, 5, 6]]))

    def test_validate_samples_with_no_timestamps(self):
        """Test that an exception is raised there is data but no timestamps."""
        with pytest.raises(ValueError):
            Samples(timestamps=np.array([]), data=np.arange(10).reshape(-1, 2))

    def test_validate_samples_with_no_data(self):
        """Test an exception is raised when timestamps are given, but data is empty."""
        with pytest.raises(ValueError):
            Samples(timestamps=np.arange(10), data=np.array([]))

    def test_samples_from_empty_array(self):
        """Test that empty samples is created from empty arrays."""
        s = Samples(np.array([]), np.array([]))
        assert s.empty

    def test_samples_timestamp_with_too_many_dims(self):
        """Test create samples with timestamps that have more than 1 dimension.

        Expect that an exception is raised when the timestamps has more than 1
        dimension.
        """
        with pytest.raises(ValueError):
            Samples(
                timestamps=np.arange(10).reshape(-1, 2),
                data=np.arange(10).reshape(-1, 2),
            )

    def test_samples_data_with_too_many_dims(self):
        """Test that an exception is raised if the data has more than 2 dimensions."""
        with pytest.raises(ValueError):
            Samples(
                timestamps=np.arange(5).reshape(-1, 1),
                data=np.arange(20).reshape(-1, 2),
            )

    def test_samples_with_mismatched_shapes(self):
        """Test create samples with mismatched timestamps and data array sizes.

        Expect that an exception is raised if the timestamps and data have mismatched
        sizes.
        """
        with pytest.raises(ValueError):
            Samples(
                timestamps=np.arange(5).reshape(-1, 1),
                data=np.arange(12).reshape(-1, 2),
            )
