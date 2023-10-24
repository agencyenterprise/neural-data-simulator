"""Test runner.py module."""
from unittest import mock

from neural_data_simulator.core import runner


class TestRunner:
    """Test runner class."""

    def test_run(self):
        """Test run 2 seconds of simulation."""
        timer_mock = mock.Mock()
        timer_mock.total_elapsed_time.side_effect = [1, 2, 3]

        encoder_mock = mock.Mock()
        encoder_mock.connect.return_value = mock.Mock(
            __enter__=encoder_mock, __exit__=mock.Mock()
        )

        runner.run(
            encoder=encoder_mock, timer=timer_mock, total_seconds_of_simulation=2
        )

        assert timer_mock.start.call_count == 1
        assert timer_mock.wait.call_count == 2
        assert encoder_mock.iterate.call_count == 2
