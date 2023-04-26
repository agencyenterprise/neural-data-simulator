"""Tests for the ephys_generator.py module."""
from copy import deepcopy
from unittest import mock

import numpy as np
import pytest

from neural_data_simulator.ephys_generator import ContinuousData
from neural_data_simulator.ephys_generator import RingBuffer
from neural_data_simulator.ephys_generator import Spikes
from neural_data_simulator.ephys_generator import Waveforms


@pytest.fixture
def spike_generator_params():
    """Get a generic set of parameters for instantiating Spikes."""
    return Spikes.Params(
        raw_data_frequency=30_000,
        n_units_per_channel=1,
        refractory_time=0.001,
        n_samples_waveform=48,
    )


@pytest.fixture
def waveform_params():
    """Get a generic set of parameters for instantiating Waveforms."""
    return Waveforms.Params(
        prototypes_definitions={1: [10, 0, -10]},
        unit_prototype_mapping={"default": 1},
        n_samples=48,
    )


@pytest.fixture
def continuous_data_params():
    """Get a generic set of parameters for instantiating ContinuousData."""
    return ContinuousData.Params(
        raw_data_frequency=30_000,
        n_units_per_channel=1,
        n_samples_waveform=48,
        lfp_data_frequency=1000,
        lfp_filter_cutoff=300,
        lfp_filter_order=4,
    )


@pytest.fixture
def spike_rates():
    """Get a generic set of spike rates for testing.

    Currently the returned rates are not being used.
    """
    rates = np.array([])
    return rates.astype(int)


@pytest.fixture
def mock_noise_data():
    """Mock the noise_data module."""

    def get_slice_stub(shape):
        """Return a slice of constant noise data."""
        return np.full(shape, 10.0)

    noise_data = mock.Mock()
    noise_data.get_slice.side_effect = get_slice_stub
    return noise_data


@pytest.fixture(autouse=True)
def mock_random_uniform(monkeypatch: pytest.MonkeyPatch):
    """Mock numpy random uniform to return fixed value."""

    def constant_random(*args, **kwargs):
        return 1

    monkeypatch.setattr("numpy.random.uniform", constant_random)


class TestWaveforms:
    """Tests for ephys generator waveforms."""

    def test_get_spike_waveforms(self, waveform_params):
        """Test that get_spike_waveforms returns correct waveforms."""
        waveform_params.n_samples = 4

        waveforms = Waveforms(waveform_params, n_units=2)

        # spike waveform for a single unit
        spike_waveforms = waveforms.get_spike_waveforms(np.array([1]))
        assert np.array_equal(spike_waveforms, [[10.0], [0.0], [-10.0], [0.0]])

        # spike waveform for two units
        spike_waveforms = waveforms.get_spike_waveforms(np.array([0, 1]))
        assert np.array_equal(
            spike_waveforms, [[10.0, 10.0], [0.0, 0.0], [-10.0, -10.0], [0.0, 0.0]]
        )

    def test_get_spike_waveforms_mapped_prototype(self):
        """Test that mapped prototypes are returned when mapped."""
        waveform_params = Waveforms.Params(
            prototypes_definitions={1: [10, 0, -10], 2: [20, 0, -20]},
            unit_prototype_mapping={"default": 1, 1: 2},
            n_samples=4,
        )

        waveforms = Waveforms(waveform_params, n_units=2)

        # getting default spike waveform
        spike_waveforms = waveforms.get_spike_waveforms(np.array([0]))
        assert np.array_equal(spike_waveforms, [[10.0], [0.0], [-10.0], [0.0]])

        # getting mapped waveform
        spike_waveforms = waveforms.get_spike_waveforms(np.array([1]))
        assert np.array_equal(spike_waveforms, [[20.0], [0.0], [-20.0], [0.0]])


class TestEphysGenerator:
    """Tests for the ephys generator process."""

    @classmethod
    def setup_class(cls):
        """Initialize before running the tests in this class."""
        cls.n_channels = 3
        cls.n_samples = 10

    def test_generate_spikes_returns_all_possible_spikes(
        self, spike_generator_params, spike_rates, waveform_params
    ):
        """Test that all possible spikes are returned if the rates are 1."""
        waveforms = Waveforms(waveform_params, self.n_channels)
        spike_generator = Spikes(self.n_channels, waveforms, spike_generator_params)

        with mock.patch.object(
            # spike chance is 100%
            Spikes,
            "_get_spikes",
            lambda obj, rates, samples: np.ones(samples),
        ):
            spikes = spike_generator.generate_spikes(spike_rates, self.n_samples)
            assert np.array_equal(
                spikes.time_idx, [0, 6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 6, 7, 8, 9]
            )
            # the consecutive spikes happen because of the number of cycles in the
            # iterative algorithm that removes spikes during the refractory period
            assert np.array_equal(
                spikes.unit, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
            )

    def test_generate_spikes_takes_into_account_refractory_period(
        self, spike_generator_params, spike_rates, waveform_params
    ):
        """Test that spikes in the refractory period are removed."""
        n_channels = 1
        n_samples = 10
        params = deepcopy(spike_generator_params)
        params.raw_data_frequency = 1
        params.refractory_time = 2
        waveforms = Waveforms(waveform_params, self.n_channels)
        spike_generator = Spikes(n_channels, waveforms, params)
        spike_generator.spikes_buffer = RingBuffer(max_samples=100, n_channels=1)

        with mock.patch.object(
            # spike chance is 100%
            Spikes,
            "_get_spikes",
            lambda obj, rates, samples: np.ones(samples),
        ):
            spikes = spike_generator.generate_spikes(spike_rates, n_samples)
            assert np.count_nonzero(spikes.time_idx) == 3

    def test_generate_spikes_returns_no_spikes(
        self, spike_generator_params, spike_rates, waveform_params
    ):
        """Test that no spikes are returned if the rates are 0."""
        waveforms = Waveforms(waveform_params, self.n_channels)
        spike_generator = Spikes(self.n_channels, waveforms, spike_generator_params)

        with mock.patch.object(
            # spike chance is 0%
            Spikes,
            "_get_spikes",
            lambda obj, rates, samples: np.zeros(samples),
        ):
            spikes = spike_generator.generate_spikes(spike_rates, self.n_samples)
            assert len(spikes.time_idx) == 0

    def test_get_continuous_data(
        self,
        continuous_data_params,
        spike_generator_params,
        spike_rates,
        mock_noise_data,
    ):
        """Test that the continuous data contains both spikes and noise."""
        continuous_data = ContinuousData(
            mock_noise_data, self.n_channels, continuous_data_params
        )

        n_samples = 48
        waveform_params = Waveforms.Params(
            prototypes_definitions={1: np.ones((n_samples))},
            unit_prototype_mapping={"default": 1},
            n_samples=n_samples,
        )
        waveforms = Waveforms(waveform_params, self.n_channels)
        spike_generator = Spikes(self.n_channels, waveforms, spike_generator_params)

        with mock.patch.object(
            Spikes, "_get_spikes", lambda obj, rates, samples: np.ones(samples)
        ):
            spikes = spike_generator.generate_spikes(spike_rates, self.n_samples)
            data = continuous_data.get_continuous_data(self.n_samples, spikes)
            assert np.array_equal(
                np.full(
                    (self.n_samples, self.n_channels), 11.0
                ),  # 10. (noise) + 1. (waveform) == 11. (output)
                data,
            )
            next_data = continuous_data.get_continuous_data(self.n_samples, spikes)
            assert np.array_equal(
                np.full((self.n_samples, self.n_channels), 12.0),
                next_data,
            )
            lfp_data = continuous_data.get_lfp_data(data)
            assert np.allclose(
                [[1.00000898, 1.00000898, 1.00000898]],
                lfp_data,
            )
