"""Shared test setup.

The message window (``bot/handlers/messages.py``: messages sent together wait
``MESSAGE_WINDOW_SECONDS`` for each other and become one turn) is off unless a test turns it on
(``tests/test_message_window.py``), so a handler queues or injects a message at once, as the
other tests expect. Set before ``bot.config`` is imported, which reads it once.
"""
import os

os.environ["MESSAGE_WINDOW_SECONDS"] = "0"
