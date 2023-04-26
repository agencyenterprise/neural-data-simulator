"""Implementation of a circular dequeue."""
from collections import deque
import time


class CircularDeque:
    """Implementation of a circular dequeue."""

    def __init__(self, num_elements: int) -> None:
        """Instantiate a CircularDeque."""
        self.num_elements = num_elements
        self.deque: deque = deque([])
        self.time_last_clear = time.time()

    def append(self, value) -> None:
        """Append a value to the deque."""
        if len(self.deque) > self.num_elements:
            self.deque.popleft()
        self.deque.append(value)

    def to_list(self) -> list:
        """Convert the deque into a list."""
        return list(self.deque)

    def clear(self):
        """Clear the dequeue and set last clear time."""
        self.deque.clear()
        self.time_last_clear = time.time()

    def is_full(self) -> bool:
        """Return True if the dequeue is full."""
        return len(self.deque) == self.num_elements
