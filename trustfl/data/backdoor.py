"""Backdoor data poisoning helpers.

Thin re-export of the trigger/poisoning primitives implemented in
``trustfl.attacks.data_attacks`` so callers can import them from the data
package (where the trigger lives alongside the datasets it stamps).
"""
from __future__ import annotations

from ..attacks.data_attacks import add_pixel_trigger, poison_backdoor

__all__ = ["add_pixel_trigger", "poison_backdoor"]
