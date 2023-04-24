"""Collect and plot velocities resulted from running the task."""
import math
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
from scipy import signal
from sklearn.metrics import r2_score
from tasks.center_out_reach.scalers import PixelsToMetersConverter


class MetricsCollector:
    """Collect and plot velocities resulted from running the task."""

    def __init__(
        self,
        window_rect: Tuple[int, int],
        target_size: float,
        unit_converter: PixelsToMetersConverter,
        actual_cursor_color,
        decoded_cursor_color,
    ):
        """Create a new instance.

        Args:
            window_rect: The size of the task window.
            target_size: The radius of the target in meters.
            unit_converter: The unit converter to use for converting from meters
              to pixels.
            actual_cursor_color: The color to use for plotting the actual cursor.
            decoded_cursor_color: The color to use for plotting the decoded cursor.
        """
        self.window_rect = window_rect
        self.target_size = target_size
        self.unit_converter = unit_converter

        self.decoded_velocities = np.array([]).reshape(-1, 2)
        self.decoded_velocities_timestamps: List[float] = []
        self.actual_velocities = np.array([]).reshape(-1, 2)
        self.actual_velocities_timestamps: List[float] = []

        self.actual_positions: List[Tuple[int, int]] = []
        self.decoded_positions: List[Tuple[int, int]] = []
        self.trial_counts: List[int] = []

        self.actual_cursor_color = actual_cursor_color
        self.decoded_cursor_color = decoded_cursor_color

    def clear_data(self):
        """Remove all recorded data so far."""
        self.actual_velocities = np.array([]).reshape(-1, 2)
        self.actual_velocities_timestamps.clear()
        self.decoded_velocities = np.array([]).reshape(-1, 2)
        self.decoded_velocities_timestamps.clear()
        self.actual_positions.clear()
        self.decoded_positions.clear()
        self.trial_counts.clear()

    def record_decoded_velocities(self, decoded_velocities, timestamps):
        """Record decoded velocities.

        Args:
            decoded_velocities: List of decoded velocities.
            timestamps: The timestamps corresponding to the decoded velocities.
        """
        self.decoded_velocities = np.concatenate(
            (self.decoded_velocities, decoded_velocities)
        )
        self.decoded_velocities_timestamps.extend(timestamps)

    def record_actual_velocities(self, actual_velocities, timestamps):
        """Record actual velocities.

        Args:
            actual_velocities: List of actual velocities.
            timestamps: The timestamps corresponding to the actual velocities.
        """
        self.actual_velocities = np.concatenate(
            (self.actual_velocities, actual_velocities)
        )
        self.actual_velocities_timestamps.extend(timestamps)

    def record_cursor_positions(self, trial_count, actual_position, decoded_position):
        """Record cursor positions.

        Args:
            trial_count: The number of the current trial.
            actual_position: The position of the actual cursor.
            decoded_position: The position of the decoded cursor.
        """
        self.actual_positions.append(
            (
                actual_position[0] - self.window_rect[0] / 2,
                self.window_rect[1] / 2 - actual_position[1],
            )
        )
        self.decoded_positions.append(
            (
                decoded_position[0] - self.window_rect[0] / 2,
                self.window_rect[1] / 2 - decoded_position[1],
            )
        )
        self.trial_counts.append(trial_count)

    def _plot_velocities(
        self,
        actual_velocities: ndarray,
        actual_velocities_timestamps: ndarray,
        decoded_velocities: ndarray,
        r2: ndarray,
        axis: int,
    ):
        mean = np.mean(decoded_velocities[:, axis])
        std = np.std(decoded_velocities[:, axis])
        if axis == 0:
            plt.title(
                (
                    f"Horizontal direction: r2 = {r2[axis]:.2f}, "
                    f"mean = {mean:.2f}, std = {std:.2f}"
                )
            )
        else:
            plt.title(
                (
                    f"Vertical direction: r2 = {r2[axis]:.2f}, "
                    f"mean = {mean:.2f}, std = {std:.2f}"
                )
            )

        plt.plot(
            # use actual_velocities_timestamps instead of actual_velocities_timestamps
            # because decoded velocities were aligned to actual velocities
            actual_velocities_timestamps,
            decoded_velocities[:, axis],
            self.decoded_cursor_color,
            label="Decoded (from simulated spikes)",
        )
        plt.plot(
            actual_velocities_timestamps,
            actual_velocities[:, axis],
            color=self.actual_cursor_color,
            label="Input",
        )
        plt.ylabel("Velocity (mm/s)")
        plt.xlabel("Time (s)")
        plt.legend()

    def _get_lag(self, x: ndarray, y: ndarray):
        correlation = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        lag = lags[np.argmax(correlation)]
        return abs(lag)

    def _plot_positions(self, targets):
        actual_positions = np.array(
            [
                self.unit_converter.pixels_to_millimeters(np.array(p))
                for p in self.actual_positions
            ]
        ).reshape(-1, 2)
        decoded_positions = np.array(
            [
                self.unit_converter.pixels_to_millimeters(np.array(p))
                for p in self.decoded_positions
            ]
        ).reshape(-1, 2)

        xs = [
            self.unit_converter.pixels_to_millimeters((t[0] - self.window_rect[0] / 2))
            for t in targets
        ]
        ys = [
            self.unit_converter.pixels_to_millimeters((t[1] - self.window_rect[1] / 2))
            for t in targets
        ]

        target_size = self.unit_converter.meters_to_pixels(self.target_size)
        plt.axis("equal")
        plt.plot(0, 0, "black", label="Decoded position")
        plt.plot(
            0,
            0,
            "black",
            label="Input position",
            linestyle="--",
        )
        plt.scatter(
            xs,
            ys,
            color="k",
            s=target_size * target_size * math.pi / 4,
            label="Target",
            edgecolors="none",
            alpha=0.2,
        )
        trial_counts = np.array(self.trial_counts)

        colors = list(mcolors.XKCD_COLORS.keys())
        cursor = 0
        for trial_count in np.unique(trial_counts):
            end_of_task = np.where(trial_counts == trial_count)[0][-1]
            plt.plot(
                decoded_positions[cursor:end_of_task, 0],
                decoded_positions[cursor:end_of_task, 1],
                colors[trial_count],
                alpha=0.6,
            )
            plt.plot(
                actual_positions[cursor:end_of_task, 0],
                actual_positions[cursor:end_of_task, 1],
                colors[trial_count],
                linestyle="--",
                alpha=0.6,
            )
            cursor = end_of_task + 1

        plt.ylabel("mm")
        plt.xlabel("mm")
        lim = np.max(ys) * 1.2
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
        legend = plt.legend(
            loc="upper left", frameon=False, labelspacing=0.5, handletextpad=0.5
        )
        legend.legendHandles[-1].set_sizes([40])

    def plot_metrics(self, targets):
        """Show velocities plot and R-values.

        Args:
            targets: List of target positions in pixels.
        """
        h_lag = self._get_lag(
            self.actual_velocities[:, 0], self.decoded_velocities[:, 0]
        )

        actual_velocities = self.actual_velocities[:, :]
        actual_velocities_timestamps = np.array(self.actual_velocities_timestamps)[:]
        actual_velocities_timestamps = (
            actual_velocities_timestamps - actual_velocities_timestamps[0]
        )
        decoded_velocities = self.decoded_velocities[h_lag:, :]

        # cut behavior and decoder streams to the same length
        min_samples = min(decoded_velocities.shape[0], actual_velocities.shape[0])
        decoded_velocities = decoded_velocities[:min_samples, :]
        actual_velocities = actual_velocities[:min_samples, :]
        actual_velocities_timestamps = actual_velocities_timestamps[:min_samples]

        r2 = r2_score(
            actual_velocities,
            decoded_velocities,
            multioutput="raw_values",
        )

        # default dpi for matplotlib is 100
        dpi = 100
        fig_size = (
            self.window_rect[0] / dpi,
            self.window_rect[1] / dpi,
        )
        plt.figure(num="Velocities overview", dpi=dpi, figsize=fig_size)

        plt.subplot(2, 1, 1)
        self._plot_velocities(
            actual_velocities,
            actual_velocities_timestamps,
            decoded_velocities,
            r2,
            axis=0,
        )

        plt.subplot(2, 1, 2)
        self._plot_velocities(
            actual_velocities,
            actual_velocities_timestamps,
            decoded_velocities,
            r2,
            axis=1,
        )

        plt.tight_layout()
        plt.show()

        plt.figure(num="Trajectories overview", dpi=dpi, figsize=fig_size)
        self._plot_positions(targets)
        plt.tight_layout()
        plt.show()
