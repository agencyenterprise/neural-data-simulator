"""Health checker for NDS components."""
import logging
import time

import numpy as np

from nds.util.circular_dequeue import CircularDeque

logger = logging.getLogger(__name__)


class HealthChecker:
    """Class to monitor the processing time consistency.

    It prints messages to the console when the data processing time is
    increasing or is taking longer than expected.
    """

    def __init__(self, queue_size: int, optimal_num_samples_per_iteration: int) -> None:
        """Initialize the HealthChecker class.

        Args:
            queue_size: size of the queue to store the number of samples processed in
                each iteration. The queue acts a sliding window of the last n
                iterations.
            optimal_num_samples_per_iteration: the number of samples that should be
                processed in each iteration.
        """
        self._num_samples_processed = CircularDeque(queue_size)
        self._optimal_num_samples_per_iteration = optimal_num_samples_per_iteration

    def record_processed_samples(self, n_samples: int):
        """Record the samples processed and check the health.

        Args:
            n_samples: number of samples processed in the current iteration.
        """
        self._num_samples_processed.append(n_samples)
        self._check_n_samples()

    def _check_n_samples(self):
        if (
            self._num_samples_processed.is_full()
            or time.time() > self._num_samples_processed.time_last_clear + 5
        ):
            # checking if mean is bigger than threshold
            mean = np.mean(list(self._num_samples_processed.deque))
            if mean > self._optimal_num_samples_per_iteration + 10:
                logger.warning(
                    (
                        "Not running in real-time. Processed samples "
                        "per iteration: %.2f, expected: %.2f"
                    ),
                    mean,
                    self._optimal_num_samples_per_iteration,
                )

            # checking if the values are always increasing
            diff = np.diff(list(self._num_samples_processed.deque))
            increasing = (diff <= 0).sum() <= 0.2 * len(
                self._num_samples_processed.deque
            )
            if increasing:
                logger.error(
                    (
                        "Not able to keep up. Number of processed "
                        "samples per iteration keeps increasing."
                    )
                )

            self._num_samples_processed.clear()
