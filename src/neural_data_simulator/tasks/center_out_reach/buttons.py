"""A button as a Sprite. Pressing the button will execute the associated action."""
from __future__ import annotations

from typing import Tuple

import pygame


class Button(pygame.sprite.Sprite):
    """A button object that can be drawn on the screen.

    It consists of a rectangle with a text label in the center.
    """

    def __init__(
        self,
        text: str,
        color: str,
        font: pygame.font.Font,
        size: Tuple[int, int],
        xy: Tuple[int, int],
        action=None,
    ):
        """Initialize the Button class.

        Args:
            text: The text to display on the button.
            color: The color of the button.
            font: The font to use for the text.
            size: The size of the button.
            xy: The position of the button.
            action: An optional action to execute when the button is pressed.
        """
        super().__init__()

        self.action = action

        self._width, self._height = size
        self._color = color
        self._position = xy

        self._font = font
        self._text = self._font.render(text, True, "black")

        self._make_image()
        self._draw()

    @property
    def is_mouse_over(self) -> bool:
        """Check if the mouse cursor is hovering the button."""
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def change_color(self, color):
        """Set a new color for the button.

        Args:
            color: The new button color.
        """
        self._color = color
        self._draw()

    def _make_image(self):
        self.image = pygame.Surface((self._width, self._height))
        self.rect = self.image.get_rect(center=self._position)

    def _draw(self):
        pygame.draw.rect(
            self.image,
            self._color,
            self.image.get_rect(),
        )

        text_rect = self._text.get_rect(center=self.image.get_rect().center)
        self.image.blit(self._text, text_rect)

    def press(self):
        """Call the action associated with the button."""
        if self.action is not None:
            self.action()
