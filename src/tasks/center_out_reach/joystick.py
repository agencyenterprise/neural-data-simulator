"""Module for handling joysticks and gamepads."""
from typing import Optional, Tuple

import pygame


class JoystickInput:
    """Captures joystick movement and converts it to relative positions."""

    def __init__(self):
        """Create a new instance."""
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            self.joystick = None
        self.joystick_dx, self.joystick_dy = 0, 0

    @property
    def relative_position(self) -> Optional[Tuple[int, int]]:
        """Get joystick position relative to the last event poll.

        Returns:
            The relative position of the joystick as a tuple.
        """
        if self.joystick is not None:
            return self.joystick_dx, self.joystick_dy
        return None

    def _update_joystick_position(self, axis, value):
        deadzone = 0.25
        multiplier = 10

        value = value * abs(value) * multiplier

        if axis in (0, 2):
            if abs(value) < deadzone:
                self.joystick_dx = 0
            else:
                self.joystick_dx = value
        if axis in (1, 3):
            if abs(value) < deadzone:
                self.joystick_dy = 0
            else:
                self.joystick_dy = value

    def process_event(self, event: pygame.event.Event):
        """Handle input event."""
        if event.type == pygame.JOYAXISMOTION:
            axis, value = event.axis, event.value
            self._update_joystick_position(axis, value)
