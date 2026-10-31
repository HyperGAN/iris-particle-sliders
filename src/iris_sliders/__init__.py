"""Iris glue; exactly one installed product profile owns this namespace."""
from .profiles import installation

__version__ = installation()[0].version
