"""Implement an accurate timer class that works across platforms."""

import time


class Timer:
    """A simple timer class.

    It has a custom implementation for the python's sleep method by adding a cpu
    bound routine to wait for the last `n` ms before the next loop execution.

    """

    def __init__(self, period: float, max_cpu_buffer: float = 0.005):
        """Initialize timer class.

        Args:
            period: Expected time between returns from the wait function (in seconds).
            max_cpu_buffer: Maximum time to stay in cpu bound loop (i.e. a while loop
              that does nothing) waiting for correct time to return from a `wait`
              call (in seconds). Defaults to 0.005.
        """
        self.period_ns = period * 1e9
        self.max_cpu_buffer_ns = max_cpu_buffer * 1e9

    def start(self) -> None:
        """Start the timer."""
        self._start_time_ns = time.perf_counter_ns()
        self._next_loop_ns = self._start_time_ns + self.period_ns
        self.jitter_ns = 0.0

    def wait(self) -> None:
        """Wait until the end of the next timer loop.

        It uses very small sleep intervals to avoid jitters caused by the
        :meth:`time.sleep` function.
        """
        previous_loop = self._next_loop_ns
        self._sleep()
        self._next_loop_ns += self.period_ns
        self.jitter_ns = -(previous_loop - time.perf_counter_ns())

    def total_elapsed_time(self) -> float:
        """Get total time since the `start` function call.

        Returns:
            Elapsed time (in seconds).
        """
        return (time.perf_counter_ns() - self._start_time_ns) / 1e9

    def _sleep(self) -> None:
        """CPU-bound alternative for the time.sleep method.

        It uses nanosecond precision and it is more accurate for small intervals
        above 0ms. It uses max_cpu_buffer_ns to compensate the python's
        time.sleep jitter.
        """
        sleep_ns = max(
            (self._next_loop_ns - self.max_cpu_buffer_ns) - time.perf_counter_ns(), 0
        )
        time.sleep(sleep_ns / 1e9)
        while time.perf_counter_ns() < self._next_loop_ns:
            pass


def get_timer(loop_time: float = 0.02, max_cpu_buffer: float = 0.005) -> Timer:
    """Get timer object.

    Args:
        loop_time: expected time between returns from the wait function (in seconds).
        max_cpu_buffer: Maximum time to stay in cpu bound loop (i.e. a while loop
          that does nothing) waiting for correct time to return from a `wait`
          call (in seconds). Defaults to 0.005.

    Returns:
        An instance of the :class:`neural_data_simulator.timing.Timer` class based on
        input parameters.
    """
    return Timer(loop_time, max_cpu_buffer)
