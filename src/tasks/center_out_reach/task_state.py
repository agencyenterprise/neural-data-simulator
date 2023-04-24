"""The state machine for the BCI task."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
import time
from typing import Optional, Protocol

from tasks.center_out_reach.task_window import TaskWindow


class State(Protocol):
    """The model of a state in a state machine.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def is_valid_next_state(self, s: State) -> bool:
        """Check if a transition is possible to the given next state.

        Args:
            s: The next state to validate the transition to.

        Returns:
            True if a transition is possible from the current
            state to the given next state.
        """
        ...

    def enter(self, previous_state: Optional[State] = None):
        """Enter the state.

        This method is called when the state machine transitioned to this state.

        Args:
            previous_state: The previous state that the state machine was in.
        """
        ...

    def exit(self, next_state: Optional[State] = None):
        """Leave the state.

        This method is called when the state machine transitioned from this state.

        Args:
            next_state: The next state that the state machine is transitioning to.
        """
        ...


@dataclass
class StateParams:
    """Configuration for the state machine."""

    delay_to_begin: float
    """The delay before the trial begins."""

    delay_waiting_for_cue: float
    """The time between the target is shown and the go cue."""

    target_holding_time: float
    """The time required for the mouse to hover over the target."""

    max_trial_time: int
    """The time allocate for a trial to be completed."""


class BaseState(State):
    """Base class for all states in the state machine."""

    @property
    def time_in_state(self):
        """Get the time spent in the current state."""
        if self.time_entered_state is None:
            return 0
        return time.time() - self.time_entered_state

    @property
    def trial_timed_out(self) -> bool:
        """Check if the trial has timed out."""
        if self.time_started_trial is None:
            return False
        return (time.time() - self.time_started_trial) > self.params.max_trial_time

    def __init__(
        self,
        task_window: TaskWindow,
        params: StateParams,
    ) -> None:
        """Create a new instance."""
        self.task_window = task_window
        self.params = params
        self.time_entered_state: Optional[float] = None
        self.time_started_trial: Optional[float] = None

    def enter(self, previous_state: Optional[State] = None):
        """Transitioned to the state."""
        self.time_entered_state = time.time()

    def exit(self, next_state: Optional[State] = None):
        """Transitioned from the state."""
        self.time_entered_state = None

    @abstractmethod
    def is_valid_next_state(self, s: State) -> bool:
        """Check if a transition is possible to the given next state.

        Args:
            s: The next state to validate the transition to.

        Returns:
            True if a transition is possible from the current
            state to the given next state.
        """
        pass


class MenuScreen(BaseState):
    """The state that the state machine starts in.

    It represents the time before the task start when a menu is presented on the screen.
    """

    def is_valid_next_state(self, s: State) -> bool:
        """Check if a transition is possible to the given next state.

        The only valid transition is to the WaitingToBegin state after the
        GUI windows is no longer showing the menu.
        """
        return isinstance(s, WaitingToBegin) and not self.task_window.show_menu_screen


class WaitingToBegin(BaseState):
    """The state before the first trial."""

    def is_valid_next_state(self, s: State) -> bool:
        """Check if a transition is possible to the given next state.

        The only valid transition is to the WaitingForCue state after the
        delay_to_begin.
        """
        return (
            isinstance(s, WaitingForCue)
            and self.time_in_state > self.params.delay_to_begin
        )


class WaitingForCue(BaseState):
    """The state that the state machine is at the start of every trial round."""

    def is_valid_next_state(self, s: State) -> bool:
        """Check if a transition is possible to the given next state.

        The only valid transition is to the Reaching state after the
        delay_waiting_for_cue.
        """
        return (
            isinstance(s, Reaching)
            and self.time_in_state > self.params.delay_waiting_for_cue
        )

    def enter(self, previous_state: Optional[State] = None):
        """Enter the state.

        Position the target in a random location or in center depending on
        the previous position.
        Inform the task window that the target is not ready so that its
        appearance can be updated.
        """
        super().enter(previous_state)
        if self.task_window.is_target_centered:
            self.task_window.randomize_target()
            self.task_window.reset_cursor()
        else:
            self.task_window.center_target()
        self.task_window.set_target_ready(False)

    def exit(self, next_state: Optional[State] = None):
        """Exit the state.

        Inform the task window that the target is ready so that its
        appearance can be updated.
        """
        super().exit(next_state)
        self.task_window.set_target_ready(True)


class Reaching(BaseState):
    """In this state the cursor is trying reach the target."""

    def is_valid_next_state(self, s: State) -> bool:
        """Check if a transition is possible to the given next state.

        Valid transitions are:
         - to the InTarget state if the cursor is on the target
         - to the WaitingForCue state if the trial has timed out.
        """
        return (isinstance(s, WaitingForCue) and self.trial_timed_out) or (
            isinstance(s, InTarget) and self.task_window.is_cursor_on_target
        )

    def enter(self, previous_state: Optional[State] = None):
        """Enter the state.

        If the previous state was WaitingForCue, then set the time that the
        trial started to be the time this state was entered.
        """
        super().enter(previous_state)
        if isinstance(previous_state, WaitingForCue):
            self.time_started_trial = self.time_entered_state

    def exit(self, next_state: Optional[State] = None):
        """Exit the state.

        Reset the time that the trial started if the next state is not InTarget.
        """
        super().exit(next_state)
        if not isinstance(next_state, InTarget):
            self.time_started_trial = None


class InTarget(BaseState):
    """In this state the cursor is hovering over the target."""

    def is_valid_next_state(self, s: State) -> bool:
        """Check if a transition is possible to the given next state.

        Valid transitions are:
          - to the WaitingForCue state if the cursor was hovering the target \
              for the target_holding_time.
          - to the WaitingForCue state if the trial has timed out.
          - to the Reaching state if the cursor is no longer over the target.
        """
        return (
            isinstance(s, WaitingForCue)
            and (
                self.time_in_state > self.params.target_holding_time
                or self.trial_timed_out
            )
        ) or (isinstance(s, Reaching) and not self.task_window.is_cursor_on_target)

    def enter(self, previous_state: Optional[State] = None):
        """Enter the state.

        If the previous state was the Reaching state, then copy the time that
        the trial started.

        Inform the task window that the cursor is on the target so that its
        appearance can be updated.
        """
        super().enter(previous_state)
        if isinstance(previous_state, Reaching):
            self.time_started_trial = previous_state.time_started_trial
        self.task_window.set_decoded_cursor_on_target(True)

    def exit(self, next_state: Optional[State] = None):
        """Exit the state.

        If the next state is not the Reaching state, then reset the time
        that the trial started.

        Inform the task window that the cursor is no longer on the target so
        that its appearance can be updated.
        """
        super().exit(next_state)
        if not isinstance(next_state, Reaching):
            self.time_started_trial = None
        self.task_window.set_decoded_cursor_on_target(False)


class StateMachine:
    """The state machine that controls the BCI task.

    It is responsible for transitioning between states and calling the
    enter and exit methods on the states.
    It also provides a method to get the next state that should be transitioned to.
    """

    def __init__(self, states: list[State]):
        """Initialize the state machine with the given states."""
        self.states = states
        self.current_state: Optional[State] = None

    def get_next_state(self) -> Optional[State]:
        """Get the next state that should be transitioned to."""
        if self.current_state is not None:
            for state in self.states:
                if self.current_state.is_valid_next_state(state):
                    return state
        return None

    def enter(self, s: State) -> bool:
        """Transition to the given state."""
        if self.current_state is not None:
            self.current_state.exit(s)
        s.enter(self.current_state)
        self.current_state = s
        return True


class TaskState:
    """The state of the task during each trial round."""

    def __init__(self, task_window: TaskWindow, params: StateParams):
        """Initialize the state with parameters and display it in given window."""
        self.task_window = task_window
        self.trial_counter = 0

        states: list[State] = [
            MenuScreen(task_window, params),
            WaitingToBegin(task_window, params),
            WaitingForCue(task_window, params),
            Reaching(task_window, params),
            InTarget(task_window, params),
        ]

        self.state_machine = StateMachine(states)
        self.state_machine.enter(states[0])

    def advance(self):
        """Try to advance the state machine to the next state.

        If the state machine is able to transition to the next state, then
        transition to it.

        Do nothing if the state machine is not able to transition to the next state.
        """
        next_state = self.state_machine.get_next_state()

        if next_state is not None:
            if (
                isinstance(next_state, WaitingForCue)
                and self.task_window.is_target_centered
            ):
                self.trial_counter += 1
            self.state_machine.enter(next_state)
