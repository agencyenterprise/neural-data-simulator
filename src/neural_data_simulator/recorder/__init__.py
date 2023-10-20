"""A utility to connect to LSL streams and save the data with timestamps to a file.

The `recorder` is useful for capturing multiple streams for offline analysis.
It's meant for short sessions (minutes) as it stores all data in memory and
writes it to disk at the end of the session.
"""
