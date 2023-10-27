"""A collection of outputs that can be used by NDS."""
import abc
import logging
from typing import Any, IO, Optional

from numpy import ndarray
import numpy as np

from neural_data_simulator.core.samples import Samples


class Output(abc.ABC):
    """Represents an abstract output that can be used to send samples."""

    @property
    @abc.abstractmethod
    def channel_count(self) -> int:
        """Return the number of channels."""
        pass

    def wait_for_consumers(self, timeout: int) -> bool:
        """Wait for consumers to connect until the timeout expires.

        Args:
            timeout: Timeout in seconds.

        Returns:
            True if consumers are connected, False otherwise.
        """
        return True

    def has_consumers(self) -> bool:
        """Return whether there are consumers connected to the output."""
        return True

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect to output."""
        pass

    def disconnect(self) -> None:
        """Disconnect from output. The default implementation does nothing."""
        pass

    @abc.abstractmethod
    def _send(self, samples: Samples) -> None:
        """Send samples to output.

        Args:
            samples: Samples to output.
        """
        pass

    @abc.abstractmethod
    def _send_array(self, data: ndarray, timestamps: Optional[ndarray]) -> None:
        """Send data to output.

        Args:
            data: Data to output.
            timestamps: Timestamps to output.
        """
        pass

    def _validate_data_shape(self, data: ndarray) -> None:
        """Validate the shape of the samples."""
        if not len(data):
            return

        if (len(data.shape) == 1 and self.channel_count != data.shape[0]) or (
            len(data.shape) == 2 and self.channel_count != data.shape[1]
        ):
            raise ValueError(
                f"Output expects data with {self.channel_count} channels,"
                + f" received data with {data.shape[-1]} channels"
            )

    def send(self, samples: Samples) -> Samples:
        """Push samples to output and return the data unchanged.

        Args:
            samples: Samples to output.

        Returns:
            The input samples unchanged.
        """
        self._validate_data_shape(samples.data)
        self._send(samples)
        return samples

    def send_array(self, data: ndarray, timestamps: Optional[ndarray]):
        """Push array data to output.

        Args:
            data: Data to output.
            timestamps: Timestamps to output.
        """
        self._validate_data_shape(data)
        self._send_array(data, timestamps)


class ConsoleOutput(Output):
    """Represents an output device that prints to the terminal."""

    def __init__(self, channel_count: int):
        """Initialize the ConsoleOutput class."""
        self.logger = logging.getLogger(__name__)
        self._channel_count = channel_count

    @property
    def channel_count(self) -> int:
        """The number of channels.

        Returns:
            Number of channels of the output.
        """
        return self._channel_count

    def _send(self, samples: Samples) -> None:
        """Send data to console without index or header.

        Args:
            samples: :class:`neural_data_simulator.core.samples.Samples` dataclass with
              timestamps and data.
        """
        self._send_array(samples.data, samples.timestamps)

    def _send_array(self, data: ndarray, timestamps: Optional[ndarray]) -> None:
        """Send data to console without index or header.

        Args:
            data: Data to output.
            timestamps: Timestamps to output.
        """
        if timestamps is None:
            timestamps = np.full(data.shape[0], np.nan)
        timestamps_and_data = np.column_stack((timestamps, data))
        print(np.array2string(timestamps_and_data))

    def connect(self) -> None:
        """Connect to the device within a context.

        The default implementation does nothing.
        """
        pass


class FileOutput(Output):
    """Represents an output device that writes to a file."""

    def __init__(self, channel_count: int, file_name: str = "output.csv"):
        """Initialize FileOutput class.

        Args:
            channel_count: Number of channels for this output.
            file_name: File path to write the samples via the
                `send` method. Defaults to "output.csv".
        """
        self.logger = logging.getLogger(__name__)
        self._channel_count = channel_count
        self.file: Optional[IO[Any]] = None
        self.file_name = file_name

    @property
    def channel_count(self) -> int:
        """The number of channels.

        Returns:
            Number of channels of the output.
        """
        return self._channel_count

    def _send(self, samples: Samples) -> None:
        """Write the samples into the file.

        Args:
            samples: :class:`neural_data_simulator.core.samples.Samples` dataclass.
        """
        self._send_array(samples.data, samples.timestamps)

    def _send_array(self, data: ndarray, timestamps: Optional[ndarray]) -> None:
        """Send data to file without index or header.

        Args:
            data: Data to output.
            timestamps: Timestamps to output.
        """
        if self.file is not None:
            if timestamps is None:
                timestamps = np.full(data.shape[0], np.nan)
            timestamps_and_data = np.column_stack((timestamps, data))
            np.savetxt(self.file, timestamps_and_data, delimiter=",", fmt="%f")

    def connect(self) -> None:
        """Open the output file."""
        self.logger.info(f"Opening output file {self.file_name}")
        self.file = open(self.file_name, "w")

    def disconnect(self) -> None:
        """Close the output file."""
        if self.file is not None:
            self.file.close()
