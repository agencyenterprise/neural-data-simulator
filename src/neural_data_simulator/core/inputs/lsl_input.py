"""Input implementation for LSL streams."""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
import logging
import math
from typing import List, Optional

import numpy as np
import pylsl

from neural_data_simulator.core.inputs.api import Input
from neural_data_simulator.core.inputs.api import SpikeRateInput
from neural_data_simulator.core.samples import Samples


@dataclass
class StreamInfo:
    """Selected advertised properties of an LSL stream."""

    name: str = field(init=False)
    """Name of the LSL stream."""

    sample_rate: float = field(init=False)
    """Advertised sample rate of the LSL stream."""

    channel_count: int = field(init=False)
    """Number of channels in the LSL stream."""

    lsl_stream_info: InitVar[pylsl.stream_info]
    """pylsl stream info object."""

    def __post_init__(self, lsl_stream_info: pylsl.stream_info):
        """Actually init members from stream info."""
        self.name = lsl_stream_info.name()
        self.sample_rate = lsl_stream_info.nominal_srate()
        self.channel_count = lsl_stream_info.channel_count()


LSL_DTYPES = [
    [],
    np.float32,
    np.float64,
    None,
    np.int32,
    np.int16,
    np.int8,
    np.int64,
]


class LSLInput(Input):
    """Represents an LSL Inlet stream for behavior data."""

    def __init__(
        self,
        stream_name: str,
        connection_timeout: float = 60.0,
        resolve_streams_wait_time: float = 1.0,
    ):
        """Initialize LSLInput class.

        Args:
            stream_name: Name of the LSL stream to retrieve data from.
            connection_timeout: Maximum time for attempting a connection
                to an LSL input stream.
            resolve_streams_wait_time: Maximum waiting time to get the list
                of available streams. Should be bigger than 0.5 to ensure all streams
                are returned.
        """
        self._stream_name = stream_name
        self._stream_info: Optional[pylsl.StreamInfo] = None
        self._inlet: Optional[pylsl.StreamInlet] = None
        self._connection_timeout = connection_timeout
        self._resolve_streams_wait_time = resolve_streams_wait_time
        self._logger = logging.getLogger(__name__)

    def get_info(self) -> StreamInfo:
        """Get information about the LSL stream.

        If the stream is not connected, it will try to resolve the stream
        and return the information.

        Returns:
            LSL stream properties.

        Raises:
            ValueError: If the stream is not found.
        """
        if self._inlet is not None:
            return StreamInfo(self._inlet.info())
        results = pylsl.resolve_stream("name", self._stream_name)
        if (results is not None) and (len(results) > 0):
            return StreamInfo(results[0])
        raise ValueError("Stream not found")

    def _check_connection(self):
        if self._inlet is None:
            raise ConnectionError(
                "LSL StreamInlet is not connected, ensure you run connect before read."
            )

    def read(self) -> Samples:
        """Read available data from the inlet as a samples.

        Returns:
            :class:`neural_data_simulator.samples.Samples` dataclass with timestamps and
            data read from the LSL StreamInlet. If no data is available, an empty
            Samples is returned.

        Raises:
            ValueError: LSL StreamInlet is not connected. `connect` should be called
              before `read`.
        """
        self._check_connection()
        assert self._inlet is not None

        _, timestamps = self._inlet.pull_chunk(
            timeout=0.0, max_samples=self.buffer.shape[0], dest_obj=self.buffer
        )
        if timestamps:
            data = self.buffer[: len(timestamps), :]
            return Samples(np.array(timestamps), np.array(data))
        return Samples.empty_samples()

    def set_connection_timeout(self, timeout: float) -> None:
        """Set the maximum time that the inlet search for the desired LSL stream.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            ValueError: if timeout equals or less than 0.
        """
        if timeout > 0:
            self._connection_timeout = timeout
        else:
            raise ValueError(f"Timeout must be greater than 0, received {timeout}")

    def connect(self):
        """Connect to the LSL Inlet stream."""
        self._logger.info("Connecting to device...")
        self._stream_info = self._get_stream()
        # could verify # of channels and sampling rate if it exists in a config file
        self._logger.info(f"Found stream: '{self._stream_name}'")
        self._inlet = LSLInput._create_inlet(self._stream_info)
        self._logger.info(f"Connected to LSL input stream: '{self._stream_name}'")

        bufsize = (
            200 * math.ceil(self._stream_info.nominal_srate()),
            self._stream_info.channel_count(),
        )
        self.buffer = np.empty(
            bufsize, dtype=LSL_DTYPES[self._stream_info.channel_format()]
        )

    def disconnect(self):
        """Disconnect from the LSL Inlet stream."""
        self._logger.info("Disconnecting LSL input stream...")
        del self._inlet
        self._inlet = None
        self._stream_info = None

    def _get_stream(self) -> pylsl.StreamInfo:
        """Resolve the first LSL source stream with the provided stream name.

        Returns:
            pylsl.StreamInfo object of the first stream with the provided name.
        """
        stream_infos = pylsl.resolve_byprop(
            "name", self._stream_name, timeout=self._connection_timeout
        )
        if len(stream_infos) > 0:
            available_streams = pylsl.resolve_streams(
                wait_time=self._resolve_streams_wait_time
            )
            streams_with_target_name = [
                stream
                for stream in available_streams
                if stream.name() == self._stream_name
            ]
            if len(streams_with_target_name) > 1:
                raise Exception(f"Multiple streams with same name {self._stream_name}")
            return stream_infos[0]
        else:
            available_stream_names = [
                info.name for info in self._get_available_streams()
            ]
            raise ConnectionError(
                f"Did not find a {self._stream_name} LSL stream.\n"
                f"The available streams are: {available_stream_names}"
            )

    def _get_available_streams(self) -> List[StreamInfo]:
        stream_infos = pylsl.resolve_streams(wait_time=self._resolve_streams_wait_time)
        return [StreamInfo(lsl_stream_info) for lsl_stream_info in stream_infos]

    @staticmethod
    def _create_inlet(stream_info: pylsl.StreamInfo) -> pylsl.StreamInlet:
        """Create and return an LSL StreamInlet given a StreamInfo."""
        return pylsl.StreamInlet(
            info=stream_info,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
        )


class LSLSpikeRateInputAdapter(SpikeRateInput):
    """Reads spike rates from an LSL input."""

    @property
    def channel_count(self) -> int:
        """Get the LSL stream channel count.

        Returns:
            The input channel count.
        """
        return self.lsl_input.get_info().channel_count

    def __init__(self, lsl_input: LSLInput):
        """Create an adapter for a given LSL input.

        Args:
            lsl_input: The LSL input to adapt.
        """
        self.lsl_input = lsl_input

    def __del__(self):
        """Disconnect from the LSL input stream."""
        self.lsl_input.disconnect()

    def connect(self):
        """Connect to the LSL input stream."""
        self.lsl_input.connect()

    def read(self) -> Optional[np.ndarray]:
        """Connect to the LSL input stream.

        Returns:
            An array of spike rates with shape (n_units,) or None if no samples
            are available.
        """
        samples = self.lsl_input.read()
        if len(samples) > 0:
            return np.array(samples.data[-1])
        return None
