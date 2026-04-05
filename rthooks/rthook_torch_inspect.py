# Runtime hook: patch stdlib and library internals that break in frozen PyInstaller bundles.

import sys
import types

# ── 1. inspect.getsource ────────────────────────────────────────────────────
# torch._inductor.config calls inspect.getsource(self_module) at startup.
import inspect as _inspect
_orig_getsource = _inspect.getsource
def _safe_getsource(obj):
    try:
        return _orig_getsource(obj)
    except OSError:
        return ""
_inspect.getsource = _safe_getsource

# ── 2. importlib.metadata.version ───────────────────────────────────────────
# transformers/dependency_versions_check.py uses importlib.metadata to check
# its deps. dist-info dirs are not bundled by PyInstaller; return 0.0.0.
import importlib.metadata as _importlib_metadata
_orig_metadata_version = _importlib_metadata.version
def _safe_metadata_version(pkg):
    try:
        return _orig_metadata_version(pkg)
    except _importlib_metadata.PackageNotFoundError:
        return "0.0.0"
_importlib_metadata.version = _safe_metadata_version

# ── 3. transformers.utils.import_utils.create_import_structure_from_path ────
# transformers >=4.44 calls this at __init__ time to discover lazy-import
# names by scanning .py source files. Frozen bundles have no source files.
# We must patch it BEFORE transformers/__init__.py runs, but importing
# transformers.utils.import_utils normally triggers that __init__ first.
# Fix: install temporary stub packages so the submodule loads in isolation.
if hasattr(sys, "_MEIPASS"):
    _saved = {}
    _stubs = ["transformers", "transformers.utils"]
    for _n in _stubs:
        if _n in sys.modules:
            _saved[_n] = sys.modules[_n]
    _transformers_stub = types.ModuleType("transformers")
    _transformers_stub.__path__ = []
    _utils_stub = types.ModuleType("transformers.utils")
    _utils_stub.__path__ = []
    sys.modules["transformers"] = _transformers_stub
    sys.modules["transformers.utils"] = _utils_stub
    try:
        import transformers.utils.import_utils as _im
        _orig_cisf = _im.create_import_structure_from_path
        def _safe_cisf(*args, **kwargs):
            try:
                return _orig_cisf(*args, **kwargs)
            except (FileNotFoundError, OSError):
                return {}
        _im.create_import_structure_from_path = _safe_cisf
    except Exception:
        pass
    finally:
        for _n in _stubs:
            if _n in _saved:
                sys.modules[_n] = _saved[_n]
            else:
                sys.modules.pop(_n, None)

# ── 4. torch.nn in transformers/integrations/accelerate.py ──────────────────
# In PyInstaller frozen bundles, `transformers/integrations/accelerate.py`
# references `nn` at module-level (line 62+), which was supposed to be bound
# by an earlier `import torch.nn as nn` in the same file. PyInstaller's frozen
# import machinery can lose that binding. Fix: pre-import torch and torch.nn
# and inject them into builtins so any `NameError: name 'nn'` lookup falls back
# to finding it there.  Also patch the accelerate module's globals directly
# after it loads via sys.meta_path hook so the name is always available.
try:
    import torch as _torch
    import torch.nn as _torch_nn
    import builtins as _builtins
    # Inject into builtins as a last-resort fallback for any unbound `nn` name.
    if not hasattr(_builtins, 'nn'):
        _builtins.nn = _torch_nn
    # Pre-register modules so PyInstaller's FrozenImporter finds them immediately.
    if 'torch.nn' not in sys.modules:
        sys.modules['torch.nn'] = _torch_nn
    # Patch the transformers.integrations.accelerate module's globals if already loaded.
    _acc = sys.modules.get('transformers.integrations.accelerate')
    if _acc is not None and not hasattr(_acc, 'nn'):
        _acc.nn = _torch_nn
    # Install a post-import hook so the patch is applied even if the module loads later.
    class _NNPatcher:
        def find_module(self, name, path=None):
            return self if name == 'transformers.integrations.accelerate' else None
        def load_module(self, name):
            import importlib
            mod = importlib.import_module(name)
            if not hasattr(mod, 'nn'):
                mod.nn = _torch_nn
            return mod
    sys.meta_path.insert(0, _NNPatcher())
except Exception:
    pass
