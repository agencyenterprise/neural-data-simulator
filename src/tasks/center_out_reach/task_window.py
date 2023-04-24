"""The BCI task GUI."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pygame
from tasks.center_out_reach.buttons import Button
from tasks.center_out_reach.sprites import Sprite


class RichText(NamedTuple):
    """A single rich text surface."""

    surface: pygame.surface.Surface
    """The text surface."""

    rect: pygame.rect.Rect
    """The text bounds."""


class TaskWindow:
    """Implements the BCI task GUI using pygame.

    This class is responsible for showing sprites on the screen
    and updating their positions.
    """

    @dataclass
    class Params:
        """Task window specific configuration."""

        target_radius: int
        """The radius of the target in pixels."""

        cursor_radius: int
        """The radius of the cursor in pixels."""

        radius_to_target: int
        """The distance from the center of the window to the target in pixels."""

        number_of_targets: int
        """The number of targets to show."""

        background_color: str
        """The window background color."""

        decoded_cursor_color: str
        """The color of the decoded cursor."""

        decoded_cursor_on_target_color: str
        """The color of the decoded cursor when it is hovering over the target."""

        actual_cursor_color: str
        """The color of the actual cursor."""

        target_color: str
        """The color of the target."""

        target_waiting_color: str
        """The color of the target when it is waiting for the cue."""

        font_size: int
        """The font size in pixels."""

        button_size: Tuple[int, int]
        """The button width and height in pixels."""

        button_spacing: int
        """The vertical spacing between buttons in pixels."""

        button_offset_top: int
        """The top offset of the buttons from the center of the window in pixels."""

        button_color: str = "gray"
        """The button background color."""

        button_color_on_hover: str = "lightgray"
        """The button background color when the mouse is hovering over it."""

    def __init__(
        self,
        window_rect: Tuple[int, int],
        params: Params,
        menu_text: Optional[List[Dict[str, str]]] = None,
    ):
        """Create a new instance."""
        pygame.init()

        self.show_menu_screen = True
        self.font = pygame.font.Font(pygame.font.get_default_font(), params.font_size)
        self.text: Optional[pygame.surface.Surface] = None
        self.textRect: Optional[pygame.rect.Rect] = None

        self.window_size = (window_rect[0], window_rect[1])
        self.params = params

        self.screen = self._set_window_size_background_and_title()
        self.clock = pygame.time.Clock()

        self._initialize_menu()

        self.target = Sprite(
            self.params.target_color, self.params.target_radius, self.window_center
        )
        self.target_sprite_group: pygame.sprite.GroupSingle = (
            pygame.sprite.GroupSingle()
        )
        self.target_sprite_group.add(self.target)
        self.actual_cursor = Sprite(
            self.params.actual_cursor_color,
            self.params.cursor_radius,
            self.window_center,
        )
        self.decoded_cursor = Sprite(
            self.params.decoded_cursor_color,
            self.params.cursor_radius,
            self.window_center,
        )

        self.cursor_sprite_group: pygame.sprite.Group = pygame.sprite.Group()
        self.cursor_sprite_group.add(self.decoded_cursor)
        self.cursor_sprite_group.add(self.actual_cursor)

        self.target_positions = (
            self._calculate_all_possible_target_positions_in_pixels()
        )

        self.show_hint(menu_text)

    @property
    def window_center(self):
        """Get the center of the window."""
        return int(self.window_size[0] / 2), int(self.window_size[1] / 2)

    @property
    def is_cursor_on_target(self) -> bool:
        """Check if the cursor is hovering over the target."""
        return self.target.collides_with(self.decoded_cursor)

    def _initialize_menu(self):
        window_width, window_height = self.window_size
        self.start_button = Button(
            "Start",
            self.params.button_color,
            self.font,
            self.params.button_size,
            (
                int(window_width / 2),
                int(
                    window_height / 2
                    - self.params.button_size[1] / 2
                    - self.params.button_spacing
                    + self.params.button_offset_top
                ),
            ),
            self.start_task,
        )

        self.quit_button = Button(
            "Quit",
            self.params.button_color,
            self.font,
            self.params.button_size,
            (
                int(window_width / 2),
                int(
                    window_height / 2
                    + self.params.button_size[1] / 2
                    + self.params.button_spacing
                    + self.params.button_offset_top
                ),
            ),
            lambda: pygame.event.post(pygame.event.Event(pygame.QUIT)),
        )

        self.menu_button_sprite_group = pygame.sprite.Group()
        self.menu_button_sprite_group.add(self.start_button)
        self.menu_button_sprite_group.add(self.quit_button)

    def show_hint(self, hint: Optional[List[Dict[str, str]]]):
        """Show/Hide a rich text at the top of the screen.

        Args:
            hint: The rich text to show. If None, the hint is hidden.
            The list should be a list of rich text elements.
            Each rich text element should be a dictionary with the following keys:
            - text: The text to show
            - color: The color of the text
        """
        self.hint = hint

    def start_task(self):
        """Start the task."""
        self.show_menu_screen = False
        self._grab_and_hide_cursor()
        self.show_hint(None)

    def _grab_and_hide_cursor(self):
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def _set_window_size_background_and_title(self) -> pygame.surface.Surface:
        screen = pygame.display.set_mode(self.window_size)
        screen.fill(self.params.background_color)
        pygame.display.set_caption("NDS Center-out reach")
        pygame.display.flip()
        return screen

    def _calculate_all_possible_target_positions_in_pixels(self) -> List[tuple]:
        n = self.params.number_of_targets
        target_positions_x = (
            self.window_size[0] / 2
            + np.cos(np.linspace(0, 2 * np.pi * (n - 1) / n, n))
            * self.params.radius_to_target
        )
        target_positions_y = (
            self.window_size[1] / 2
            + np.sin(np.linspace(0, 2 * np.pi * (n - 1) / n, n))
            * self.params.radius_to_target
        )
        return [(tx, ty) for tx, ty in zip(target_positions_x, target_positions_y)]

    def reset_cursor(self):
        """Reset the cursor position to the center of the screen."""
        self.actual_cursor.position = self.window_center
        self.decoded_cursor.position = self.window_center

    def center_target(self):
        """Position the target in the center of the screen.

        Also resets the target color to the default color.
        """
        self.target_sprite_group.add(self.target)
        self.target.position = self.window_center
        self.target.change_color(self.params.target_color)

    def randomize_target(self):
        """Place the target in a random position.

        The new position is selected from a predefined list of
        possible positions.
        """
        random_choice = np.random.choice(len(self.target_positions))
        self.target.position = self.target_positions[random_choice]

    @property
    def is_target_centered(self) -> bool:
        """Check if the target is in the center of the screen."""
        return self.target.position == self.window_center

    def reset_target_color(self):
        """Reset the target color to the default color."""
        self.target.change_color(self.params.target_color)

    def set_decoded_cursor_on_target(self, on_target: bool):
        """Set the decoded cursor color depending on whether it is on the target."""
        if on_target:
            self.decoded_cursor.change_color(self.params.decoded_cursor_on_target_color)
        else:
            self.decoded_cursor.change_color(self.params.decoded_cursor_color)

    def set_target_ready(self, ready: bool):
        """Set the target color depending on whether it is ready for reaching."""
        if ready:
            self.target.change_color(self.params.target_color)
        else:
            self.target.change_color(self.params.target_waiting_color)

    def toggle_actual_cursor(self):
        """Toggle the visibility of the actual (real) cursor."""
        if self.cursor_sprite_group.has(self.actual_cursor):
            self.cursor_sprite_group.remove(self.actual_cursor)
        else:
            self.cursor_sprite_group.add(self.actual_cursor)

    def update_cursor(
        self,
        actual_velocity: list[Tuple[float, float]],
        decoded_velocity: list[Tuple[float, float]],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Adjust the position of the actual and decoded cursors.

        Args:
            actual_velocity: History of velocities for the actual (real) cursor.
            decoded_velocity: History of velocities for the decoded cursor.

        Returns:
            Updated actual and decoded positions.
        """
        for vel in actual_velocity:
            self.actual_cursor.update_position(vel)
        for vel in decoded_velocity:
            self.decoded_cursor.update_position(vel)

        return self.actual_cursor.position, self.decoded_cursor.position

    def _check_mouse_hover_over_button(self):
        for button in self.menu_button_sprite_group.sprites():
            if button.is_mouse_over:
                button.change_color(self.params.button_color)
            else:
                button.change_color(self.params.button_color_on_hover)

    def try_press_button(self):
        """Try to press a button if the mouse is hovering over it."""
        for button in self.menu_button_sprite_group.sprites():
            if button.is_mouse_over:
                button.press()

    def _render_hint(self, hint: List[Dict[str, str]]):
        rich_text_lines: Dict[int, List[RichText]] = defaultdict(list)
        # assign each entry from the hint dictionary to a line of text
        line_number = 1
        for hint_part in hint:
            color = hint_part["color"]
            lines = hint_part["text"].splitlines()
            for i, line in enumerate(lines):
                if i > 0:
                    line_number += 1
                text = self.font.render(line, True, color)
                text_rect = text.get_rect(
                    center=(
                        self.window_center[0],
                        self.params.font_size * line_number,
                    )
                )
                rich_text_lines[line_number].append(RichText(text, text_rect))

        # center the text in each line by compensating the other text in the line
        for li in rich_text_lines.keys():
            rich_text_in_a_line = rich_text_lines[li]
            for i in range(len(rich_text_in_a_line)):
                width_after = 0
                for j in range(i + 1, len(rich_text_in_a_line)):
                    width_after += rich_text_in_a_line[j].rect.width
                width_before = 0
                for j in range(i):
                    width_before += rich_text_in_a_line[j].rect.width

                rich_text = rich_text_in_a_line[i]
                rich_text.rect.left += int(width_before / 2)
                rich_text.rect.left -= int(width_after / 2)
                self.screen.blit(rich_text.surface, rich_text.rect)

    def draw(self):
        """Draw all sprites on the screen."""
        self.screen.fill(self.params.background_color)

        if self.show_menu_screen:
            self._check_mouse_hover_over_button()
            self.menu_button_sprite_group.draw(
                self.screen,
            )
        else:
            self.target_sprite_group.draw(self.screen)
            self.cursor_sprite_group.draw(self.screen)

        if hint := self.hint:
            self._render_hint(hint)

        pygame.display.flip()

    def tick(self, framerate):
        """Tick the screen update clock.

        This method should be called for every frame in order update the screen
        and limit the game speed to match the frame rate.
        """
        self.draw()
        self.clock.tick(framerate)

    def leave(self):
        """Quit pygame nicely."""
        pygame.quit()
