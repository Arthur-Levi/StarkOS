"""Compatibility shim for the AutoEngineer module.

The repository stores the implementation in core/Auto.Engineer.py, but the
rest of the codebase imports it as core.auto_engineer. This module re-exports
that implementation under the expected import name so the runtime can boot
without changing every import site.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_module_path = Path(__file__).with_name("Auto.Engineer.py")
_spec = importlib.util.spec_from_file_location("core._auto_engineer_impl", _module_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load AutoEngineer implementation from {_module_path}")

_impl_module: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _impl_module
_spec.loader.exec_module(_impl_module)

for _name in dir(_impl_module):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl_module, _name)
