"""Test the buffer.py module."""
import numpy as np
import pytest

from nds.util.buffer import RingBuffer


class TestRingBuffer:
    """Test the RingBuffer class."""

    def test_read_from_buffer(self):
        """Test if the data stored in the buffer can be read back."""
        buffer = RingBuffer(max_samples=100, n_channels=6)
        buffer.add(np.arange(12).reshape(-1, 6))
        assert np.array_equal(buffer[:, :3], [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]])
        assert np.array_equal(buffer[:, 0], [0.0, 6.0])
        buffer[:, 0] = -1
        assert np.array_equal(buffer[:, :2], [[-1.0, 1.0], [-1.0, 7.0]])
        assert np.array_equal(
            buffer.read(2),
            [[-1.0, 1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
        )
        assert buffer.read(1).size == 0

    def test_raises_overflow_error_when_full(self):
        """Test if exception is raised when adding more samples to a full buffer."""
        buffer = RingBuffer(max_samples=2, n_channels=1)
        assert not buffer.is_full
        buffer.add(np.zeros((2, 1)))
        assert buffer.is_full

        with pytest.raises(OverflowError):
            buffer.add(np.zeros((1, 1)))
