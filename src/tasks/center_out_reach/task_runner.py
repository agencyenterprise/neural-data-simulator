"""Run trial rounds in a loop.

An iteration of the loop consists of:
    1. Polling for input events.
    2. Advancing the state.
    3. Updating the window.
    4. Pausing so that the GUI doesn't run faster than the targeted \
    sampling rate.
    5. Repeat until the loop is signaled to stop.
"""

from typing import Optional, Protocol

from numpy import ndarray
import numpy as np
import pylsl
from tasks.center_out_reach.input_events import InputEvent
from tasks.center_out_reach.input_events import InputHandler
from tasks.center_out_reach.metrics import MetricsCollector
from tasks.center_out_reach.task_state import TaskState

from neural_data_simulator import inputs
from neural_data_simulator import outputs
from neural_data_simulator.filters import LowpassFilter
from neural_data_simulator.samples import Samples
from neural_data_simulator.timing import Timer


class VelocityScaler(Protocol):
    """Scales the cursor velocity.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def transform(self, X: ndarray) -> ndarray:
        """Scales the real velocity."""
        ...

    def inverse_transform(self, X: ndarray) -> ndarray:
        """Scales the decoded velocity."""
        ...


class TaskRunner:
    """The BCI task runner."""

    def __init__(
        self,
        sample_rate: float,
        decoded_cursor_input: Optional[inputs.Input],
        actual_cursor_output: Optional[outputs.Output],
        velocity_scaler: VelocityScaler,
        with_decoded_cursor: bool,
        metrics_collector: Optional[MetricsCollector],
    ):
        """Create a new instance."""
        self.decoded_cursor_input = decoded_cursor_input
        self.actual_cursor_output = actual_cursor_output
        self.with_decoded_cursor = with_decoded_cursor
        self.should_stop_loop = False

        self.sample_rate = sample_rate

        self.velocity_filter = LowpassFilter(
            name="lp_filter",
            filter_order=2,
            critical_frequency=20,
            sample_rate=self.sample_rate,
            num_channels=2,
            enabled=True,
        )

        self.velocity_scaler = velocity_scaler
        self.metrics_collector = metrics_collector
        self.timer = Timer(1 / self.sample_rate)

    def _get_decoded_velocity(self) -> ndarray:
        if self.decoded_cursor_input is not None:
            samples = self.decoded_cursor_input.read()
            if not samples.empty:
                decoded_velocities = np.array(samples.data).reshape(-1, 2)
                if metrics_collector := self.metrics_collector:
                    metrics_collector.record_decoded_velocities(
                        decoded_velocities, list(samples.timestamps)
                    )
                transformed_velocity = self.velocity_scaler.inverse_transform(
                    decoded_velocities
                )
                return transformed_velocity
        return np.array([])

    def _send_actual_velocity(self, actual_velocity: ndarray):
        if self.actual_cursor_output is not None:
            scaled_velocity = self.velocity_scaler.transform(actual_velocity)
            filtered_velocity = self.velocity_filter.execute(scaled_velocity)[0]
            velocities = filtered_velocity.reshape(1, 2)
            timestamps = np.array([pylsl.local_clock()])
            if metrics_collector := self.metrics_collector:
                metrics_collector.record_actual_velocities(velocities, timestamps)
            self.actual_cursor_output.send(
                Samples(timestamps=timestamps, data=velocities)
            )

    def stop(self):
        """Signal the loop that it should stop."""
        self.should_stop_loop = True

    def run(self, task_state: TaskState):
        """Start the loop.

        Args:
            task_state: The state machine that should be updated by the loop.
        """
        user_input = InputHandler()
        task_window = task_state.task_window

        user_input.set_handler_for_event(InputEvent.EXIT, self.stop)
        user_input.set_handler_for_event(InputEvent.RESET, task_window.reset_cursor)
        user_input.set_handler_for_event(
            InputEvent.TOGGLE_CURSOR, task_window.toggle_actual_cursor
        )
        user_input.set_handler_for_event(
            InputEvent.MOUSE_BUTTON_PRESSED, task_window.try_press_button
        )
        if metrics_collector := self.metrics_collector:
            user_input.set_handler_for_event(
                InputEvent.CLEAR_METRICS, metrics_collector.clear_data
            )

        self.timer.start()
        while not self.should_stop_loop:
            user_input.poll()
            task_state.advance()

            actual_velocity = np.array([user_input.get_cursor_relative_position()])

            self.timer.wait()

            if not task_window.show_menu_screen:
                self._send_actual_velocity(actual_velocity)

                if self.with_decoded_cursor:
                    decoded_velocity = self._get_decoded_velocity()
                else:
                    decoded_velocity = actual_velocity.copy()

                actual_position, decoded_position = task_window.update_cursor(
                    list(actual_velocity), list(decoded_velocity)
                )
                if metrics_collector := self.metrics_collector:
                    metrics_collector.record_cursor_positions(
                        task_state.trial_counter, actual_position, decoded_position
                    )

                if (
                    self.actual_cursor_output is None
                    or not self.actual_cursor_output.has_consumers()
                ):
                    task_window.show_hint(
                        [{"color": "red", "text": "No consumer for cursor output"}]
                    )
                else:
                    task_window.show_hint(None)

            # Make the GUI think that it's running at twice the frame rate.
            # The rest of the time we use our precise timer to wait so that
            # we can output at exactly sample_rate with as little jitter as possible
            task_window.tick(self.sample_rate * 2)
