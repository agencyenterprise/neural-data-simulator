"""LSL output device."""
from dataclasses import dataclass
import logging
from typing import Callable, List, Optional, Union

from numpy import ndarray
import numpy as np
import pylsl

from neural_data_simulator.core.outputs.api import Output
from neural_data_simulator.core.samples import Samples
from neural_data_simulator.core.settings import LSLOutputModel


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
        """Create a StreamConfig from an LSLOutputModel.

        Args:
            lsl_settings: :class:`neural_data_simulator.core.settings.LSLOutputModel`
              instance.
            sampling_rate: Sampling rate in Hz.
            n_channels: Number of channels.
        """
        acquisition = {
            "manufacturer": lsl_settings.instrument.manufacturer,
            "model": lsl_settings.instrument.model,
            "instrument_id": lsl_settings.instrument.id,
        }
        if lsl_settings.channel_labels is not None:
            channel_labels = lsl_settings.channel_labels
            if len(channel_labels) != n_channels:
                raise ValueError(
                    f"Number of channel labels ({len(channel_labels)}) does not match "
                    + f"number of channels ({n_channels})"
                )
        else:
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
        """Initialize the LSL Output Device from a StreamConfig.

        Args:
            stream_config: :class:`neural_data_simulator.outputs.StreamConfig` instance.
        """
        self.logger = logging.getLogger(__name__)
        self._stream_config = stream_config
        self._outlet: Optional[pylsl.StreamOutlet] = None
        self._stream_info: Optional[pylsl.StreamInfo] = None
        self._stream_configured = False

    @classmethod
    def from_lsl_settings(
        cls,
        lsl_settings: LSLOutputModel,
        sampling_rate: Union[float, Callable],
        n_channels: int,
    ):
        """Initialize from :class:`neural_data_simulator.core.settings.LSLOutputModel`.

        Args:
            lsl_settings: :class:`neural_data_simulator.core.settings.LSLOutputModel`
              instance.
            sampling_rate: Sampling rate in Hz.
            n_channels: Number of channels.
        """
        stream_config = StreamConfig.from_lsl_settings(
            lsl_settings=lsl_settings,
            sampling_rate=sampling_rate,
            n_channels=n_channels,
        )
        return LSLOutputDevice(stream_config)

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
            samples: :class:`neural_data_simulator.core.samples.Samples` dataclass with
              timestamps and data.

        Raises:
            ValueError: LSL StreamOutlet is not connected. `connect` should be called
              before `send`.
        """
        self._send_array(samples.data, samples.timestamps)

    def _send_array(
        self, data: ndarray, timestamps: Optional[Union[float, ndarray]] = None
    ):
        """Send a list of data points to the LSL outlet.

        Args:
            data: An array of data points.
            timestamps: timestamp(s) corresponding to the data points.

        Raises:
            ValueError: LSL StreamOutlet is not connected. `connect` should be called
              before `send`.
            ValueError: There was nothing to send because the data array is empty.
        """
        self._check_connection()
        assert self._outlet is not None
        if len(data) == 0:
            return
        # cast data to expected channel format
        data_out = data.astype(self._dtype)
        if timestamps is not None:
            self._outlet.push_chunk(data_out, timestamps)  # type: ignore[arg-type]
        else:
            self._outlet.push_chunk(data_out)

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
            channel_format=config.channel_format,  # type: ignore[arg-type]
            source_id=config.source_id,
        )
        channels_xml = out_info.desc().append_child("channels")
        for _, ch_label in enumerate(config.channel_labels):
            chan = channels_xml.append_child("channel")
            chan.append_child_value("label", ch_label)
        acq = out_info.desc().append_child("acquisition")
        LSLOutputDevice._build_xml_from_dict(acq, config.acquisition)
        return out_info
