"""Circular shapes that are being drawn in the window on the screen."""
from __future__ import annotations

from typing import Tuple

import pygame


class Sprite(pygame.sprite.Sprite):
    """An object that can be drawn on the screen.

    This class is a wrapper around pygame.sprite.Sprite that is used
    to represent a circle of a given color and radius.
    """

    def __init__(self, color: str, radius: int, xy: Tuple[int, int]):
        """Create a sprite with a given color, radius, and position.

        Args:
            color: The color of the sprite.
            radius: the radius of the sprite.
            xy: The position of the sprite.
        """
        super().__init__()

        self.radius = radius
        self._color = color
        self._position = xy

        self._make_image()
        self._draw()

    @property
    def position(self) -> Tuple[int, int]:
        """Get the sprite coordinates in the window.

        Returns:
            The sprite coordinates as a tuple.
        """
        return self._position

    @position.setter
    def position(self, xy: Tuple[int, int]):
        """Set the sprite coordinates in the window.

        Args:
            xy: The new sprite coordinates.
        """
        self._position = xy
        self.rect = self.image.get_rect(center=self._position)

    def collides_with(self, other_sprite: Sprite) -> bool:
        """Check if this sprite collides with another sprite.

        Args:
            other_sprite: The sprite to check for collision with.

        Returns:
            True if the sprites collide, False otherwise.
        """
        return pygame.sprite.collide_circle(self, other_sprite)

    def update_position(self, xy: Tuple[float, float]):
        """Adjust the position of the sprite by a given amount.

        Args:
            xy: The delta to adjust the position by.
        """
        self._position = (
            int(self._position[0] + xy[0]),
            int(self._position[1] + xy[1]),
        )
        self.rect.center = int(self._position[0]), int(self._position[1])

    def change_color(self, color):
        """Set a new color for the sprite.

        Args:
            color: The new sprite color.
        """
        self._color = color
        self._draw()

    def _make_image(self):
        self.image = pygame.Surface((self.radius * 2, self.radius * 2))
        # the color key is the color that will be transparent, so it's not
        # important what color we choose here
        self.image.fill("white")
        self.image.set_colorkey("white")
        self.rect = self.image.get_rect(center=self._position)

    def _draw(self):
        pygame.draw.circle(
            self.image, self._color, (self.radius, self.radius), self.radius
        )
