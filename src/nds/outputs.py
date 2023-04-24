"""A collection of outputs that can be used by NDS."""
import abc
from dataclasses import dataclass
import logging
from typing import Any, Callable, IO, List, Optional, Union

from numpy import ndarray
import numpy as np
import pylsl

from nds.samples import Samples
from nds.settings import LSLOutputModel


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

    def _validate_data_shape(self, data: ndarray) -> None:
        """Validate the shape of the samples."""
        if not len(data):
            return

        if (len(data.shape) == 1 and self.channel_count != data.shape[0]) or (
            len(data.shape) == 2 and self.channel_count != data.shape[1]
        ):
            raise ValueError(
                f"Output expects data with {self.channel_count} channels,"
                + f" received data with {data.shape[1]} channels"
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
        """Send data to a file without index or header.

        Args:
            samples: :class:`nds.samples.Samples` dataclass with timestamps and data.
        """
        timestamps_and_data = np.column_stack((samples.timestamps, samples.data))
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
            samples: :class:`nds.samples.Samples` dataclass.
        """
        if self.file is not None:
            timestamps_and_data = np.column_stack((samples.timestamps, samples.data))
            np.savetxt(self.file, timestamps_and_data, delimiter=",", fmt="%f")

    def connect(self) -> None:
        """Open the output file."""
        self.logger.info(f"Opening output file {self.file_name}")
        self.file = open(self.file_name, "w")

    def disconnect(self) -> None:
        """Close the output file."""
        if self.file is not None:
            self.file.close()


@dataclass
class StreamConfig:
    """Parameters of an LSL stream."""

    name: str
    """LSL stream name."""

    type: str
    """LSL stream type."""

    source_id: str
    """LSL source id."""

    acquisition: dict
    """Information regarding the acquisition device."""

    sample_rate: Union[float, Callable[[], float]]
    """Sampling rate in Hz."""

    channel_format: str
    """Stream data type, for example `float32` or `int32`."""

    channel_labels: List[str]
    """Channel labels. The number of labels must match the number
    of channels."""

    @classmethod
    def from_lsl_settings(
        cls,
        lsl_settings: LSLOutputModel,
        sampling_rate: Union[float, Callable],
        n_channels: int,
    ):
        """Create a StreamConfig from an :class:`nds.settings.LSLOutputModel`.

        Args:
            lsl_settings: :class:`nds.settings.LSLOutputModel` instance.
            sampling_rate: Sampling rate in Hz.
            n_channels: Number of channels.
        """
        acquisition = {
            "manufacturer": lsl_settings.instrument.manufacturer,
            "model": lsl_settings.instrument.model,
            "instrument_id": lsl_settings.instrument.id,
        }
        channel_labels = [str(i) for i in range(n_channels)]
        return StreamConfig(
            lsl_settings.stream_name,
            lsl_settings.stream_type,
            lsl_settings.source_id,
            acquisition,
            sampling_rate,
            lsl_settings.channel_format,
            channel_labels,
        )


class LSLOutputDevice(Output):
    """An output device that can be used to stream data via LSL."""

    def __init__(self, stream_config: StreamConfig):
        """Initialize the LSL Output Device from a :class:`nds.outputs.StreamConfig`.

        Args:
            stream_config: :class:`nds.outputs.StreamConfig` instance.
        """
        self.logger = logging.getLogger(__name__)
        self._stream_config = stream_config
        self._outlet: Optional[pylsl.StreamOutlet] = None
        self._stream_info: Optional[pylsl.StreamInfo] = None
        self._stream_configured = False

    @property
    def _dtype(self):
        """Return the numpy data type of the stream."""
        channel_format = self._stream_config.channel_format
        if channel_format == "float32":
            return np.float32
        elif channel_format == "double64":
            return np.longdouble
        elif channel_format == "int8":
            return np.int8
        elif channel_format == "int16":
            return np.int16
        elif channel_format == "int32":
            return np.int32
        elif channel_format == "int64":
            return np.int64
        else:
            raise ValueError(f"Unsupported channel format: {channel_format}")

    @property
    def channel_count(self) -> int:
        """The number of channels.

        Returns:
            Number of channels of the output.
        """
        return len(self._stream_config.channel_labels)

    @property
    def sample_rate(self) -> Union[float, Callable[[], float]]:
        """Sample rate of the stream.

        Returns:
            The sample rate in Hz.
        """
        return self._stream_config.sample_rate

    @property
    def name(self) -> str:
        """The name of the stream.

        Returns:
            The configured name of the output stream.
        """
        return self._stream_config.name

    def _check_connection(self):
        if self._outlet is None:
            raise ConnectionError(
                "LSL StreamOutlet is not connected, ensure you run connect before send."
            )

    def _send(self, samples: Samples):
        """Push the data to the LSL outlet.

        Args:
            samples: :class:`nds.samples.Samples` dataclass with timestamps and data.

        Raises:
            ValueError: LSL StreamOutlet is not connected. `connect` should be called
              before `send`.
        """
        for timestamp, data_point in zip(samples.timestamps, samples.data):
            self.send_as_sample(data_point, timestamp)

    def send_as_chunk(self, data: ndarray, timestamp: Optional[float] = None):
        """Send a list of data points to the LSL outlet together with an optional timestamp.

        Args:
            data: An array of data points.
            timestamp: An optional timestamp corresponding to the data points.

        Raises:
            ValueError: LSL StreamOutlet is not connected. `connect` should be called
              before `send`.
            ValueError: There was nothing to send because the data array is empty.
        """
        self._check_data(data)
        self._check_connection()
        assert self._outlet is not None
        # cast data to expected channel format
        data_out = data.astype(self._dtype)
        if timestamp:
            self._outlet.push_chunk(data_out, timestamp)
        else:
            self._outlet.push_chunk(data_out)

    def send_as_sample(self, data: ndarray, timestamp: Optional[float] = None):
        """Send a single sample with the corresponding timestamp.

        A sample consisting of a data point per channel will be pushed to the LSL
        outlet together with an optional timestamp.

        Args:
            data: A single data point as an array of 1 value per channel.
            timestamp: An optional timestamp corresponding to the data point.

        Raises:
            ValueError: LSL StreamOutlet is not connected. `connect` should be called
              before `send`.
            ValueError: There was nothing to send because the data array is empty.
        """
        self._check_data(data)
        self._check_connection()
        assert self._outlet is not None
        # cast data to expected channel format
        data_out = data.astype(self._dtype)
        if timestamp:
            self._outlet.push_sample(data_out, timestamp)
        else:
            self._outlet.push_sample(data_out)

    def _check_data(self, data: ndarray):
        if len(data) == 0:
            self.logger.debug("No data to output")
            raise ValueError("No data data to output")
        self._validate_data_shape(data)

    @staticmethod
    def _get_open_stream_names() -> List[str]:
        stream_infos = pylsl.resolve_streams()
        return [lsl_stream_info.name() for lsl_stream_info in stream_infos]

    def connect(self):
        """Connect to the LSL stream."""
        self.logger.info("Initializing LSL output stream...")

        self._stream_info = LSLOutputDevice._get_info_from_config(self._stream_config)

        outlet_buffer_time_in_s = 1
        self._outlet = pylsl.StreamOutlet(
            self._stream_info, max_buffered=outlet_buffer_time_in_s
        )
        self.logger.info(f"Created LSL output stream: '{self._stream_info.name()}'")
        self._stream_configured = True

    def disconnect(self):
        """Forget the connection to the LSL stream."""
        self.logger.info("Destroying LSL output stream...")
        del self._outlet
        self._outlet = None
        self._stream_info = None
        self._stream_configured = False

    def has_consumers(self) -> bool:
        """Check if there are consumers connected to the stream.

        Return:
            True if there are consumers, False if there aren't any.
        """
        if self._outlet:
            return self._outlet.have_consumers()
        return False

    def wait_for_consumers(self, timeout: int) -> bool:
        """Wait for consumers to connect until the timeout expires.

        Args:
            timeout: Timeout in seconds.

        Returns:
            True if consumers are connected, False otherwise.
        """
        if self._outlet:
            return self._outlet.wait_for_consumers(timeout=timeout)
        return False

    @staticmethod
    def _build_xml_from_dict(node: pylsl.XMLElement, dict_info: dict):
        """Build XML recursively."""
        for k, v in dict_info.items():
            if isinstance(v, dict):
                new_node = node.append_child(k)
                LSLOutputDevice._build_xml_from_dict(new_node, v)
            else:
                node.append_child_value(k, str(v))

    @staticmethod
    def _get_info_from_config(config: StreamConfig) -> pylsl.StreamInfo:
        if callable(config.sample_rate):
            sample_rate = config.sample_rate()
        else:
            sample_rate = config.sample_rate
        out_info = pylsl.stream_info(
            name=config.name,
            type=config.type,
            channel_count=len(config.channel_labels),
            nominal_srate=sample_rate,
            channel_format=config.channel_format,
            source_id=config.source_id,
        )
        channels_xml = out_info.desc().append_child("channels")
        for _, ch_label in enumerate(config.channel_labels):
            chan = channels_xml.append_child("channel")
            chan.append_child_value("label", ch_label)
        acq = out_info.desc().append_child("acquisition")
        LSLOutputDevice._build_xml_from_dict(acq, config.acquisition)
        return out_info
