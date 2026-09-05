"""Telegram layer of GALL.AI, built on aiogram 3.

Keep this package free of import-time side effects: modules here must not
import ``agents``, ``stats`` or ``secure_container`` at import time so the
package can be imported in tests and before the sandbox is initialised.
"""
