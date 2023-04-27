"""A type of ring buffer implementation."""
from numpy import ndarray
import numpy as np


class RingBuffer:
    """A buffer with a fixed size.

    When it fills up, adding another element raises an overflow error.
    """

    def __init__(self, *, max_samples: int, n_channels: int):
        """Create a new buffer instance with a given size."""
        self._buffer = np.empty((max_samples, n_channels))
        self._buffer_size = max_samples
        self._read_idx = 0
        self._write_idx = 0
        self._channel_indices = np.arange(n_channels)[None, :]

    def __getitem__(self, key):
        """Get a slice of the buffer."""
        return self._view_all()[key]

    def __setitem__(self, key, value):
        """Edit a slice of the buffer."""
        to_edit = self.read_all()
        to_edit[key] = value
        self.add(to_edit)

    def __len__(self):
        """Return the number of samples in the buffer."""
        if self._write_idx >= self._read_idx:
            return self._write_idx - self._read_idx
        else:
            return self._write_idx + (self._buffer_size - self._read_idx)

    def __array__(self) -> ndarray:
        """Return the buffer as a numpy array."""
        return self._view_all()

    @property
    def shape(self):
        """Tuple of buffer dimensions."""
        return len(self), self._buffer.shape[1]

    @property
    def is_full(self) -> bool:
        """Check if the buffer is full.

        Returns true if there is no more space in the buffer.
        """
        return len(self) == self._buffer_size

    @property
    def dtype(self):
        """Return the type of data that is being buffered."""
        return self._buffer.dtype

    def __repr__(self):
        """Return a string representation of the buffer."""
        return "<RingBuffer of {!r}>".format(np.asarray(self))

    def add(self, values: ndarray):
        """Append to the buffer.

        Args:
            values: An array object.
        """
        n_samples_to_append = values.shape[0]
        if n_samples_to_append > self._remaining_size():
            raise OverflowError
        else:
            new_write_idx = n_samples_to_append + self._write_idx
            writing_idxs = np.ravel_multi_index(
                (
                    np.arange(self._write_idx, new_write_idx)[:, None],
                    self._channel_indices,
                ),
                self._buffer.shape,
                mode="wrap",
            )
            self._buffer.flat[writing_idxs] = values.flat
            if new_write_idx > self._buffer_size:
                self._write_idx = new_write_idx - self._buffer_size
            else:
                self._write_idx = new_write_idx

    def read_all(self) -> ndarray:
        """Consume the entire content of buffer.

        Returns:
            An array object.
        """
        return self.read(len(self))

    def _rewind(self, n):
        self._read_idx -= n
        if self._read_idx < 0:
            self._read_idx += self._buffer_size

    def read(self, n: int) -> ndarray:
        """Consume n entries from the buffer.

        Args:
            n: Number of entries.

        Returns:
            An array object.
        """
        data = self._view(n)
        self._read_idx += n
        if self._read_idx > self._buffer_size:
            self._read_idx = self._read_idx - self._buffer_size
        return data

    def _view_all(self):
        return self._view(len(self))

    def _view(self, n):
        if n > len(self):
            n = len(self)
        data = self._buffer.take(
            np.arange(self._read_idx, self._read_idx + n), axis=0, mode="wrap"
        )
        return data

    def _remaining_size(self):
        return self._buffer_size - len(self)
