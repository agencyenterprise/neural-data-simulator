"""Test timing.py module."""
import time
from unittest import mock

import numpy as np
import pytest

from neural_data_simulator.core import timing


@pytest.fixture
def mock_time(monkeypatch):
    """Patches timing with a unittest mock.

    Returns:
        mock.Mock: a mock useful to check the calls to the timing module
    """
    time_mock = mock.Mock()
    monkeypatch.setattr(timing, "time", time_mock)
    return time_mock


class TestTimer:
    """Test timing class."""

    def test_sleep_when_buffer_is_zero(self, mock_time):
        """Test the CPU bound sleep is not activated when max_cpu_buffer_ns is zero."""
        mock_time.sleep.return_value = None
        mock_time.perf_counter_ns.side_effect = [0, 20e6, 31e6, 31.5e6]
        timer = timing.get_timer(0.030, 0.0)
        timer.start()
        timer.wait()
        assert mock_time.sleep.call_count == 1
        assert mock_time.sleep.mock_calls == [mock.call(0.01)]
        assert mock_time.perf_counter_ns.call_count == 4
        assert timer.jitter_ns == 1.5e6

    def test_total_elapsed_time(self, mock_time):
        """Test that total_elapsed_time returns the difference between now and start."""
        mock_time.perf_counter_ns.side_effect = [0, 20e6, 31e6, 31.5e6, 32e6]
        timer = timing.get_timer(0.030, 0.0)
        timer.start()
        timer.wait()
        assert timer.total_elapsed_time() == 0.032

    def test_sleep_when_buffer_is_above_zero(self, mock_time):
        """Test CPU buffer.

        The CPU bound sleep should activate when max_cpu_buffer_ns is above zero.
        """
        mock_time.sleep.return_value = None
        mock_time.perf_counter_ns.side_effect = [0, 20e6, 20.1e6, 30.1e6, 31.5e6]
        timer = timing.get_timer(0.030, 0.005)
        timer.start()
        timer.wait()
        assert mock_time.sleep.call_count == 1
        assert mock_time.sleep.mock_calls == [mock.call(0.005)]
        assert mock_time.perf_counter_ns.call_count == 5
        assert timer.jitter_ns == 1.5e6

    def test_no_sleep_when_encoder_takes_too_long(self, mock_time):
        """Test that no sleep is called when simulation takes too long.

        when the simulation loop takes longer than the expected loop time, no further
        sleep should be performed.
        """
        mock_time.sleep.return_value = None
        mock_time.perf_counter_ns.side_effect = [0, 35e6, 35.1e6, 35.2e6]
        timer = timing.get_timer(0.030, 0.0)
        timer.start()
        timer.wait()
        assert mock_time.sleep.mock_calls == [mock.call(0)]


@pytest.mark.jitter
class TestIntegrationJitter:
    """Test what is the jitter of our timing class."""

    @pytest.fixture
    def timer(self):
        """Create an instance for FakeTimer."""
        return timing.get_timer(0.030, 0.010)

    def test_wait_jitter(self, timer):
        """Test that the jitter produced is less than 1ms.

        To prevent the test from failing you must choose an appropriate
        max_cpu_buffer_ns value.
        """
        trials = 0
        timer.start()
        jitters = []
        while trials < 1000:
            timer.wait()
            # simulate some dag processing time
            time.sleep(0.015)
            jitters.append(timer.jitter_ns)
            trials += 1
        jitters = np.array(jitters)
        max_jitter = np.abs(jitters).max()
        assert max_jitter < 1e6
