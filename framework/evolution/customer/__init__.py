"""Minimal open-ended Customer search core.

Legacy co-evolution and service-repair APIs remain in their existing modules;
the Customer-only research path imports only this package's small core.
"""

from .policy import AdversaryPolicy

__all__ = ["AdversaryPolicy"]
