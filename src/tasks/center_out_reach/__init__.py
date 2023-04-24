"""Center-out reaching task.

During the task, the GUI keeps track of 2 cursors: the `real cursor` that is
controlled by the user, and the `decoded cursor` that is controlled by the
output of the decoder. The task consists of a sequence of trials.
During each trial, the user has to make the decoded cursor reach the target.
The trial ends when the decoded cursor hovers over the target for a configurable
amount of time. The trials alternate between a "reaching out" trial (from center
outwards) and a "back to center" trial (from current position back to center).
The user can reset the cursor positions to the center of the screen at any time by
pressing the space bar. The real cursor can be toggled on and off by pressing
the 'c' key. The task can be stopped by pressing the 'ESC' key.

At the end of the task, the cursor velocities and trajectories are plotted.
"""
