"""This module contains classes that are used to generate spikes from spike rates."""
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Protocol, Tuple

import colorednoise
from numpy import ndarray
import numpy as np
import pylsl

from nds.filters import LowpassFilter
from nds.health_checker import HealthChecker
from nds.inputs import LSLInput
from nds.outputs import LSLOutputDevice
from nds.timing import Timer
from nds.util.buffer import RingBuffer


class SpikeRateInput(Protocol):
    """An abstract input that can be used to read spike rates.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    @property
    def channel_count(self) -> int:
        """Get the number of input channels.

        Returns:
            The input channel count.
        """
        ...

    def read(self) -> Optional[ndarray]:
        """Read spike rates, one per channel.

        Returns:
            An array of spike rates with shape (n_units,) or None if no samples
            are available.
        """
        ...


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

    def read(self) -> Optional[ndarray]:
        """Connect to the LSL input stream.

        Returns:
            An array of spike rates with shape (n_units,) or None if no samples
            are available.
        """
        samples = self.lsl_input.read()
        if len(samples) > 0:
            return np.array(samples.data[-1])
        return None


class SpikeRateTestingInput(SpikeRateInput):
    """A constant spike rate input that can be used for testing.

    Generates spike rates so that spikes are more likely to happen
    on channels of a higher order and less likely on channels of a lower order.
    The spike rate for a channel is always constant.
    """

    def __init__(self, n_channels: int, n_units: int):
        """Create a testing spike rate input.

        Args:
            n_channels: The number of input channels.
            n_units: The total number of units, which should be a multiple
                of the number of channels.
        """
        self.n_channels = n_channels
        self.n_units = n_units

    @property
    def channel_count(self) -> int:
        """Get the number of input channels.

        Returns:
            The input channel count.
        """
        return self.n_channels

    def read(self) -> Optional[ndarray]:
        """Read spike rates, one per channel.

        Returns:
            The array of testing spike rates with shape (n_units,).
            For example, if `n_channels = 50` and `n_units_per_channel = 1`, the
            spike rates will be constant and equal to:

            `[ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34.
            36. 38. 40. 42. 44. 46. 48. 50. 52. 54. 56. 58. 60. 62. 64. 66. 68. 70.
            72. 74. 76. 78. 80. 82. 84. 86. 88. 90. 92. 94. 96. 98.]`
        """
        rates = np.arange(self.n_units) * 100 / self.n_units
        return rates.astype(int)


@dataclass
class SpikeEvents:
    """Spike events that are generated during an execution of :class:`ProcessOutput`.

    During each execution `n_samples` of data are processed and sent to output.
    Depending on the current spike rates, `n_spike_events >= 0` may be generated
    for all units (n_units) and inserted into the output data.
    """

    time_idx: ndarray
    """An array of time indices of spike events with shape `(n_spike_events,)`.
    The time index is the index of the sample in the output data,
    from 0 to `n_samples - 1`. For example, `time_idx` can be `[6 3 5]`
    in case of 3 spike events. This means that the first spike event
    corresponds to the 6th sample, the second at to the 3rd sample,
    and the third to the 5th sample.
    """

    unit: ndarray
    """An array of the same size as `time_idx` that contains the unit
    number that spiked for the corresponding `time_idx` entries.
    For example, `unit` can be `[0 1 1]` in case of 3 spike events. This means that the
    first spike event corresponds to the unit 0, the second and the third to the unit 1.
    """

    waveform: ndarray
    """The spike waveforms with shape `(n_samples_waveform, n_samples)`,
    where `n_samples_waveform` is configurable.
    The values are the amplitudes of the spike waveforms in counts.
    """

    class SpikeEvent(NamedTuple):
        """A single spike event."""

        time_idx: int
        """The time index of the spike event."""

        unit: int
        """The unit that spiked."""

        waveform: ndarray
        """The spike waveform."""

    def __iter__(self):
        """Return an iterator over the spike events."""
        self._current = 0
        return self

    def __next__(self):
        """Return the next spike event."""
        if self._current >= len(self.time_idx):
            raise StopIteration
        result = self.SpikeEvent(
            self.time_idx[self._current],
            self.unit[self._current],
            self.waveform[:, self._current],
        )
        self._current += 1
        return result

    def __len__(self):
        """Return the number of spike events."""
        return len(self.time_idx)

    def __getitem__(self, key):
        """Return a subset of the spike events."""
        return SpikeEvents(self.time_idx[key], self.unit[key], self.waveform[:, key])


class NoiseData:
    """Multi-channel colored noise generator."""

    def __init__(
        self,
        n_channels: int,
        beta: float,
        standard_deviation: float,
        fmin: float,
        samples: int,
        random_seed: Optional[int],
    ) -> None:
        """Initialize the noise generator.

        Gaussian distributed noise will be pre-generated on all channels.

        Args:
            n_channels: The number of channels.
            beta: The power-spectrum of the generated noise is proportional to
                (1 / f)**beta.
            standard_deviation: The desired standard deviation of the noise.
            fmin: Low-frequency cutoff. `fmin` is normalized frequency and the range is
                between `1/samples` and `0.5`, which corresponds to the Nyquist
                frequency.
            samples: The number of samples to generate per channel.
            random_seed: The random seed to use. Use a fixed seed
                for reproducible results.
        """
        noise_data = []
        for _ in range(n_channels):
            noise_data.append(
                colorednoise.powerlaw_psd_gaussian(
                    beta, samples, fmin=fmin, random_state=random_seed
                )
            )
        self._n_channels = n_channels
        self._noise_data = np.array(noise_data).T * standard_deviation
        self._current_index = 0
        self._total_samples = samples

    def get_slice(self, shape: Tuple[int, int]) -> ndarray:
        """Get the next random noise slice of a given size.

        After computing the current noise slice, the window is advanced by `n_samples`
        and next time this function is called the upcoming window is returned.

        Args:
            shape: The noise slice array.

        Returns:
            An array with shape (n_samples, n_channels).
        """
        (n_samples, _) = shape
        slice_rows = [
            idx % self._total_samples
            for idx in range(self._current_index, self._current_index + n_samples)
        ]
        noise_slice = self._noise_data[slice_rows, :]
        self._current_index = (self._current_index + n_samples) % self._total_samples
        return noise_slice


class ContinuousData:
    """Generator of the electrophysiology raw data."""

    @dataclass
    class Params:
        """Initialization parameters for the :class:`ContinuousData` class."""

        raw_data_frequency: float
        """The electrophysiology raw data output sample rate in Hz."""

        n_units_per_channel: int
        """The number of units per channel."""

        n_samples_waveform: int
        """The number of samples in a spike waveform."""

        lfp_data_frequency: float
        """The LFP data sample rate in Hz."""

        lfp_filter_cutoff: float
        """The LFP filter cutoff frequency in Hz."""

        lfp_filter_order: int
        """The LFP filter order."""

        @property
        def lfp_downsample_rate(self) -> float:
            """Get the LFP downsample rate."""
            return int(self.raw_data_frequency / self.lfp_data_frequency)

    def __init__(self, noise_data: NoiseData, n_channels: int, params: Params):
        """Initialize the ContinuousData class.

        Args:
            noise_data: The background noise generator.
            n_channels: The number of input/output channels.
            params: The initialization parameters.
        """
        self._params = params
        self.n_units = n_channels * self._params.n_units_per_channel
        self.n_channels = n_channels
        self.spike_to_channel_idxs = self._get_spike_to_channel_idxs()
        self.lfp_filter = self._get_lfp_data_filter()
        self.lfp_current_index = 0

        self._spikes_buffer = RingBuffer(
            max_samples=int(self._params.raw_data_frequency * 10),
            n_channels=n_channels,
        )
        self._spikes_buffer.add(np.zeros((self._params.n_samples_waveform, n_channels)))
        self._noise_data = noise_data

    def get_continuous_data(self, n_samples: int, spike_events: SpikeEvents) -> ndarray:
        """Get continuous data from spike events and background noise.

        Args:
            n_samples: The number of continuous data samples to return.
            spike_events: The spike events to use for generating the data.

        Returns:
            The synthesized continuous data with combined spikes and noise
            as a (n_samples, n_units) array.
        """
        spikes = self._spike_events_to_continuous(
            spike_events, n_samples + self._params.n_samples_waveform, self.n_units
        )
        combined_spikes = self._combine_same_channel_units(spikes)
        combined_spikes[: self._params.n_samples_waveform] += self._spikes_buffer.read(
            self._params.n_samples_waveform
        )
        self._spikes_buffer.add(combined_spikes)
        data = self._spikes_buffer.read(n_samples)  # computed spikes
        data += self._noise_data.get_slice((n_samples, self.n_channels))  # add noise
        return data

    def get_lfp_data(self, data: ndarray) -> ndarray:
        """Filter and downsample raw data to get LFP data.

        Args:
            data: The raw continuous data with shape (n_samples, n_units).

        Returns:
            The filtered and downsampled data with shape (n_filtered_samples, n_units).
        """
        data_filtered = self.lfp_filter.execute(data)
        n_samples = data_filtered.shape[0]
        downsample_indexes = np.arange(
            self.lfp_current_index,
            n_samples,
            self._params.lfp_downsample_rate,
            dtype="int32",
        )
        if len(downsample_indexes):
            self.lfp_current_index = (
                downsample_indexes[-1] + self._params.lfp_downsample_rate - n_samples
            )
        d = data_filtered[downsample_indexes, :]
        return d

    def _combine_same_channel_units(self, spikes: ndarray):
        return np.add.reduceat(spikes, self.spike_to_channel_idxs, axis=1)

    def _get_lfp_data_filter(self) -> LowpassFilter:
        return LowpassFilter(
            name="lfp_filter",
            filter_order=self._params.lfp_filter_order,
            critical_frequency=self._params.lfp_filter_cutoff,
            sample_rate=self._params.raw_data_frequency,
            num_channels=self.n_channels,
            enabled=True,
        )

    def _get_spike_to_channel_idxs(self) -> ndarray:
        unit_to_channel_map = np.arange(self.n_channels).repeat(
            self._params.n_units_per_channel
        )
        _, counts = np.unique(unit_to_channel_map, return_counts=True)
        return np.hstack(([0], np.cumsum(counts[:-1])))

    def _spike_events_to_continuous(
        self, spike_events: SpikeEvents, n_samples: int, n_units: int
    ) -> ndarray:
        data = np.zeros((n_samples, n_units))
        time_positions = (
            spike_events.time_idx[:, None]
            + np.arange(self._params.n_samples_waveform)[None, :]
        )
        unit_positions = np.repeat(
            spike_events.unit, self._params.n_samples_waveform
        ).reshape(-1, self._params.n_samples_waveform)
        data[time_positions, unit_positions] = spike_events.waveform.T
        return data


class Waveforms:
    """Spike waveforms loader."""

    @dataclass
    class Params:
        """Initialization parameters for the :class:`Waveforms` class."""

        prototypes_definitions: Dict[int, List[float]]
        """The waveform prototypes definitions. The keys are the prototype IDs
        and the values are the waveform definitions.
        """

        unit_prototype_mapping: Dict[str, int]
        """The unit prototype mapping. The keys are the unit numbers and the values
        are the prototype IDs. The "default" key is used for all units that are not
        explicitly defined. Unit numbers are 0-based.
        """

        n_samples: int
        """The number of samples in the waveform."""

        @property
        def prototypes(self) -> ndarray:
            """The waveform prototypes."""
            return np.array(list(self.prototypes_definitions.values()))

        @property
        def prototypes_ids(self) -> list[int]:
            """The waveform prototypes IDs."""
            return list(self.prototypes_definitions.keys())

    def __init__(self, params: Params, n_units: int):
        """Initialize Waveforms class.

        Args:
            params: The waveforms parameters.
            n_units: The number of units.
        """
        self.params = params
        self.n_units = n_units
        self.waveforms = self._load_waveforms()

    def get_spike_waveforms(self, units: ndarray) -> ndarray:
        """Get the spike waveforms for a given list of units.

        Args:
            units: The units array to get the waveforms for.

        Returns:
            The waveform samples for the given units.
        """
        return self.waveforms[:, units]

    def _get_unit_waveform_ids(self) -> ndarray:
        params = self.params
        default = params.unit_prototype_mapping.pop("default")
        waveform_ids = np.repeat(params.prototypes_ids.index(default), self.n_units)
        for unit, prototype_id in params.unit_prototype_mapping.items():
            waveform_ids[int(unit)] = params.prototypes_ids.index(prototype_id)
        return waveform_ids

    def _load_waveforms(self):
        params = self.params
        unit_waveform_ids = self._get_unit_waveform_ids()
        unit_waveforms = np.zeros((params.n_samples, self.n_units))
        samples_to_copy = min(params.n_samples, params.prototypes.shape[1])
        unit_waveforms[:samples_to_copy, :] = params.prototypes[unit_waveform_ids, :].T[
            :samples_to_copy, :
        ]
        unit_waveforms *= np.random.uniform(0.5, 1.0, size=self.n_units)
        return unit_waveforms


class Spikes:
    """Spike generator.

    This class generates random spike events according to unit spike rates
    within a given time interval determined by a number of samples and a known
    raw data sample rate. It also ensures that the spikes are within
    the refractory period of each other taking into account the occurrence of
    the spikes generated in the previous time interval.
    """

    @dataclass
    class Params:
        """Initialization parameters for the :class:`Spikes` class."""

        raw_data_frequency: float
        """The electrophysiology raw data output sample rate in Hz."""

        n_units_per_channel: int
        """The number of units per channel."""

        refractory_time: float
        """The refractory time in seconds."""

        n_samples_waveform: int
        """The number of samples in the spike waveform."""

    def __init__(self, n_channels: int, waveforms: Waveforms, params: Params):
        """Initialize Spikes class.

        Args:
            n_channels: The number of channels. This value together with the configured
                number of units per channel determines the total number of units
                for which spikes are generated.
            waveforms: The :class:`Waveforms` instance with spike waveform prototypes.
            params: The spike generator parameters.
        """
        self._params = params
        self.waveforms = waveforms
        self.n_units = n_channels * self._params.n_units_per_channel

        self.n_refractory_samples = int(
            self._params.raw_data_frequency * self._params.refractory_time
        )

        self.spikes_buffer = RingBuffer(
            max_samples=int(self._params.raw_data_frequency) * 10,
            n_channels=self.n_units,
        )
        self.spikes_buffer.add(np.zeros((self.n_refractory_samples, self.n_units)))

    def _get_spikes(self, rates: ndarray, n_samples: int) -> ndarray:
        return np.random.random(
            (n_samples, self.n_units)
        ) < self._get_spike_chance_for_rate(rates)

    def _add_last_chunk_refractory_period(self, spikes: ndarray) -> ndarray:
        self.spikes_buffer.add(spikes)
        spikes = self.spikes_buffer.read_all()
        return spikes

    def _remove_spikes_in_refractory_period(
        self, spikes: ndarray, units: ndarray, time_idx: ndarray
    ) -> Tuple[ndarray, ndarray, ndarray]:
        # Iterative process to remove spikes within refractory period.
        # We can easily check if two spikes are within refractory period,
        # but it's harder to do the same when there are three spikes
        # and you remove the one in the middle if the two remaining ones
        # are still within that period.
        # This iterative process solves that problem.
        for _ in range(5):
            # 5 is just an arbitrary max number of steps in this iterative process
            in_refractory = (
                np.where(
                    np.logical_and(
                        np.diff(units) == 0,
                        np.diff(time_idx) < self.n_refractory_samples,
                    )
                )[0]
                + 1
            )
            in_refractory = np.delete(
                in_refractory, np.where(np.diff(in_refractory) == 1)[0] + 1
            )
            if len(in_refractory) > 0:
                spikes[time_idx[in_refractory], units[in_refractory]] = 0
                units = np.delete(units, in_refractory)
                time_idx = np.delete(time_idx, in_refractory)
            else:
                break

        return spikes, units, time_idx

    def _correct_timeidx(
        self, units: ndarray, time_idx: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """Correct time_idx to account for data from previous chunk."""
        time_idx -= self.n_refractory_samples
        units = np.delete(units, np.where(time_idx < 0)[0])
        time_idx = np.delete(time_idx, np.where(time_idx < 0)[0])
        return units, time_idx

    def generate_spikes(self, rates: ndarray, n_samples: int) -> SpikeEvents:
        """Generate spikes for the given rates and a given number of samples.

        Spike times are calculated using the spike rate input for each channel.
        When there are multiple units in a channel, the rate is divided equally
        across the units.

        First, the chances of a spike for each sample in each unit is calculated from
        the rates (spikes/sec). A random draw is performed for each of n_samples for
        each unit and a spike is assigned or not depending on the spike chance.
        An iterative process is then performed to remove spikes within the
        configured refractory period.

        Args:
            rates: The spike rates array with shape (n_units,).
                Each element in the array represents the spike rate in
                spikes per second for the corresponding unit.
            n_samples: The number of samples to output.

        Returns:
            The generated spikes as :class:`SpikeEvents`.
        """
        spikes = self._get_spikes(rates, n_samples)
        spikes = self._add_last_chunk_refractory_period(spikes)
        units, time_idx = np.where(spikes.T)
        spikes, units, time_idx = self._remove_spikes_in_refractory_period(
            spikes, units, time_idx
        )
        units, time_idx = self._correct_timeidx(units, time_idx)

        self.spikes_buffer.add(spikes[-self.n_refractory_samples :, :])
        waveforms = self.waveforms.get_spike_waveforms(units)

        return SpikeEvents(time_idx, units, waveforms)

    def _get_spike_chance_for_rate(self, spk_per_sec: ndarray) -> ndarray:
        return spk_per_sec / self._params.raw_data_frequency


class ProcessOutput:
    """Process that reads spike rates and outputs spiking data.

    The process can be started by calling the :meth:`start` method.
    The output streams are the raw continuous data stream,
    the LFP data stream and spike events stream.
    """

    @dataclass
    class Params:
        """Initialization parameters for the :class:`ProcessOutput` class."""

        n_units_per_channel: int
        """The number of units in each channel."""

        lsl_chunk_frequency: float
        """The frequency at which to stream data to the LSL outlets in Hz."""

        raw_data_frequency: float
        """The electrophysiology raw data output sample rate in Hz."""

        resolution: float
        """The unit resolution in `uV` per count."""

        @property
        def lsl_chunk_interval(self) -> float:
            """The interval at which to stream data to the LSL outlets."""
            return 1.0 / self.lsl_chunk_frequency

    @dataclass
    class LSLOutputs:
        """Possible LSL output streams."""

        raw: Optional[LSLOutputDevice] = None
        """The raw continuous data stream."""

        lfp: Optional[LSLOutputDevice] = None
        """The LFP data stream."""

        spike_events: Optional[LSLOutputDevice] = None
        """The spike events stream."""

    def __init__(
        self,
        continuous_data: ContinuousData,
        spikes: Spikes,
        input_: SpikeRateInput,
        outputs: LSLOutputs,
        params: Params,
        health_checker: HealthChecker,
    ):
        """Initialize the ProcessOutput class.

        Args:
            continuous_data: The continuous data generator.
            spikes: The spikes generator.
            input_: The spike rates input.
            outputs: The LSL output streams.
            params: The initialization parameters.
            health_checker: The health monitor.
        """
        self._input = input_
        self._outputs = outputs
        self._params = params
        self._health_checker = health_checker

        self._resolution = params.resolution
        self.spikes = spikes
        self.continuous_data = continuous_data
        self._last_output_time: Optional[float] = None

        n_units = self._input.channel_count * self._params.n_units_per_channel
        self.rates = np.zeros(n_units).astype(int)

        self._timer = Timer(self._params.lsl_chunk_interval)
        self._should_stop = False

    def start(self):
        """Start the process.

        The process keeps iterating until the :meth:`stop` method is called.
        On each execution, spike rates are read from the input and used to
        generate spike events. For each spike event, the corresponding waveform
        is selected and combined with random noise to obtain raw continuous data.
        The raw continuous data is then filtered to obtain the LFP data.
        Spike events, raw continuous data and LFP data are then streamed via the
        configured LSL outlets.
        """
        self._timer.start()
        while not self._should_stop:
            self._execute()
            self._timer.wait()

    def stop(self):
        """Stop the process."""
        self._should_stop = True

    def _execute(self):
        self._update_spike_rates()
        if self._last_output_time is None:
            self._last_output_time = pylsl.local_clock()
            return

        time_now = pylsl.local_clock()
        time_elapsed = time_now - self._last_output_time
        self._last_output_time = time_now

        n_samples = np.rint(self._params.raw_data_frequency * time_elapsed).astype(int)
        self._health_checker.record_processed_samples(n_samples)

        spike_events = self.spikes.generate_spikes(self.rates, n_samples)
        continuous_data = self.continuous_data.get_continuous_data(
            n_samples, spike_events
        )

        self._stream_continuous_data(continuous_data)
        self._stream_lfp(continuous_data)
        self._stream_spike_events(spike_events, n_samples, time_elapsed)

    def _stream_continuous_data(self, continuous_data: ndarray):
        if self._outputs.raw is not None:
            self._outputs.raw.send_as_chunk(continuous_data / self._resolution)

    def _stream_spike_events(
        self, spike_events: SpikeEvents, n_samples: int, time_elapsed: float
    ):
        if self._last_output_time is None:
            raise ValueError("Last output time is not set.")

        if self._outputs.spike_events is not None and len(spike_events) > 0:
            channels = spike_events.unit.reshape(len(spike_events), 1)
            units = np.zeros((len(spike_events), 1))
            data_to_stream = np.hstack(
                (
                    channels,
                    units,
                    spike_events.waveform.T / self._resolution,
                )
            )

            time_interval_per_sample = time_elapsed / n_samples
            spike_lsl_times = (
                self._last_output_time
                + spike_events.time_idx * time_interval_per_sample
            )

            for i, data in enumerate(data_to_stream):
                self._outputs.spike_events.send_as_sample(
                    data=data, timestamp=spike_lsl_times[i]
                )

    def _stream_lfp(self, continuous_data: ndarray):
        if self._outputs.lfp is not None:
            lfp = self.continuous_data.get_lfp_data(continuous_data)
            if len(lfp) > 0:
                self._outputs.lfp.send_as_chunk(lfp / self._resolution)

    def _update_spike_rates(self):
        rates = self._input.read()  # update spike rates
        if rates is not None:
            self.rates = np.repeat(
                rates / self._params.n_units_per_channel,
                self._params.n_units_per_channel,
            )
