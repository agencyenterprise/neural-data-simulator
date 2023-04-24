"""Module for handling events such as key presses."""

from enum import Enum
from typing import Callable, Tuple

import pygame
from tasks.center_out_reach.joystick import JoystickInput


class InputEvent(Enum):
    """An enumeration of the possible input events."""

    NONE = 0
    EXIT = 1
    RESET = 2
    TOGGLE_CURSOR = 3
    CLEAR_METRICS = 4
    MOUSE_BUTTON_PRESSED = 5


class InputHandler:
    """Listeners for input event handling."""

    def __init__(self):
        """Create a new instance."""
        self.event_handlers = dict()
        self.joystick_input = JoystickInput()

    def set_handler_for_event(self, event: InputEvent, handler: Callable):
        """Set a function as a handler for a specific input event."""
        self.event_handlers[event] = handler

    def get_cursor_relative_position(self) -> Tuple[int, int]:
        """Get the relative position of the joystick or mouse cursor.

        The position is relative to the previous position when this
        function was last called. If a joystick was detected at the start
        of the GUI, the joystick position is returned. Otherwise, the
        mouse position is returned.

        Returns:
            The relative position of the cursor as a tuple.
        """
        if joystick_rel_position := self.joystick_input.relative_position:
            return joystick_rel_position
        return pygame.mouse.get_rel()

    def poll(self):
        """Poll for input events and call the appropriate handler.

        This method should be called once per iteration of the main loop.
        """
        event = self._get_next_event()
        handler = self.event_handlers.get(event)
        if handler is not None:
            handler()

    def _get_next_event(self) -> InputEvent:
        for event in pygame.event.get():
            self.joystick_input.process_event(event)

            if event.type == pygame.MOUSEBUTTONUP:
                return InputEvent.MOUSE_BUTTON_PRESSED
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return InputEvent.EXIT
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE) or (
                event.type == pygame.JOYBUTTONDOWN and event.button == 3
            ):
                return InputEvent.RESET
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_c) or (
                event.type == pygame.JOYBUTTONDOWN and event.button == 2
            ):
                return InputEvent.TOGGLE_CURSOR
            if event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                return InputEvent.CLEAR_METRICS

        return InputEvent.NONE
