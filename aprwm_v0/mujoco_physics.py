"""Physics-only MuJoCo import that avoids classic GL/EGL renderer init.

Headless hosts without OSMesa/EGL still need ``mj_step`` / ``mj_fullM``.
We register a proper namespace package placeholder first so private bindings
load without executing the public ``__init__.py`` (which can raise
``AttributeError`` from OpenGL).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


def _mujoco_package_dir() -> Path:
    spec = importlib.util.find_spec("mujoco")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("mujoco package is not installed")
    return Path(list(spec.submodule_search_locations)[0]).resolve()


def _ensure_namespace_package() -> types.ModuleType:
    existing = sys.modules.get("mujoco")
    if existing is not None and getattr(existing, "physics_only", False) and hasattr(
        existing, "MjModel"
    ):
        return existing

    public = sys.modules.get("mujoco")
    if public is not None and not getattr(public, "physics_only", False):
        if not hasattr(public, "mj_step"):
            sys.modules.pop("mujoco", None)

    if "mujoco" in sys.modules and hasattr(sys.modules["mujoco"], "mj_step"):
        return sys.modules["mujoco"]

    pkg_dir = _mujoco_package_dir()
    module = types.ModuleType("mujoco")
    module.__file__ = str(pkg_dir / "__init__.py")  # type: ignore[attr-defined]
    module.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    module.__package__ = "mujoco"  # type: ignore[attr-defined]
    module.physics_only = True  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(
        name="mujoco",
        loader=None,
        is_package=True,
    )
    spec.submodule_search_locations = [str(pkg_dir)]
    module.__spec__ = spec  # type: ignore[attr-defined]
    sys.modules["mujoco"] = module
    return module


def _build_physics_module() -> types.ModuleType:
    module = _ensure_namespace_package()
    import mujoco._constants as constants
    import mujoco._enums as enums
    import mujoco._functions as functions
    import mujoco._specs as specs
    import mujoco._structs as structs

    # Prefer structs/specs last so classmethods and MjVfs type casters remain intact.
    for source in (constants, enums, functions, specs, structs):
        for name in dir(source):
            if name.startswith("_"):
                continue
            setattr(module, name, getattr(source, name))

    module.MjModel = structs.MjModel
    module.MjData = structs.MjData
    if hasattr(specs, "MjVfs"):
        module.MjVfs = specs.MjVfs

    try:
        from importlib import metadata

        module.__version__ = metadata.version("mujoco")  # type: ignore[attr-defined]
    except Exception:
        module.__version__ = "unknown"  # type: ignore[attr-defined]
    return module


def _patch_qM_property(data_cls: Any) -> None:
    """Expose a legacy ``qM`` attribute that recovers the owning MjData."""

    if hasattr(data_cls, "qM") and getattr(data_cls, "_aprwm_qM_property", False):
        return

    def _qM(self):  # noqa: N802 - match MuJoCo field name
        # Prefer the underlying mjData when this is a robosuite wrapper.
        return getattr(self, "_data", self)

    data_cls.qM = property(_qM)
    data_cls._aprwm_qM_property = True


def _install_robosuite_mujoco311_compat(module: Any) -> None:
    """Bridge robosuite 1.5.2's legacy ``mj_fullM(..., data.qM)`` call to MuJoCo 3.11.

    MuJoCo 3.11 removed ``mjData.qM`` and changed ``mj_fullM(model, dst, qM)``
    into ``mj_fullM(model, data, dst)``.  robosuite 1.5.2 still uses the old
    calling convention in OSC controllers.  Its ``binding_utils.MjData`` metaclass
    snapshots public ``mujoco.MjData`` attributes at import time, so we also patch
    the wrapper class when it is already loaded.
    """

    _patch_qM_property(module.MjData)

    if not getattr(module, "_aprwm_qM_compat", False):
        original = module.mj_fullM

        def mj_fullM(model, a, b=None):  # noqa: N802
            def _raw_data(obj: Any) -> Any:
                return getattr(obj, "_data", obj)

            # New API: mj_fullM(model, data, dst)
            if b is not None and hasattr(_raw_data(a), "M") and hasattr(_raw_data(a), "qacc"):
                return original(model, _raw_data(a), b)
            # Legacy robosuite API: mj_fullM(model, dst, data.qM)
            if b is not None and hasattr(_raw_data(b), "M") and hasattr(_raw_data(b), "qacc"):
                return original(model, _raw_data(b), a)
            if b is None:
                return original(model, a)
            return original(model, a, b)

        module.mj_fullM = mj_fullM
        module._aprwm_qM_compat = True

    # If robosuite already imported, its wrapper class missed the late qM property.
    binding = sys.modules.get("robosuite.utils.binding_utils")
    if binding is not None and hasattr(binding, "MjData"):
        _patch_qM_property(binding.MjData)


def _install_renderer_stubs() -> None:
    # Keep classic GL stubs. Do not stub mujoco.egl / mujoco.osmesa: V7D
    # (and any camera_obs path) needs the real offscreen backends.
    for name in (
        "mujoco.rendering",
        "mujoco.rendering.classic",
        "mujoco.rendering.classic.renderer",
        "mujoco.rendering.classic.gl_context",
    ):
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)

        def _fail(attr: str, _name: str = name) -> Any:
            raise ImportError(f"{_name} unavailable in physics-only mode")

        stub.__getattr__ = _fail  # type: ignore[attr-defined]
        sys.modules[name] = stub


def _install_render_api(module: Any) -> None:
    """Expose MuJoCo C render types (MjrContext, …) without classic OpenGL."""
    if getattr(module, "_aprwm_render_api", False):
        return
    import mujoco._render as render

    for name in dir(render):
        if name.startswith("_"):
            continue
        setattr(module, name, getattr(render, name))
    module._aprwm_render_api = True


def prepare_offscreen_gl() -> None:
    """Select a headless GL backend before robosuite first import."""
    import os

    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    for name in ("mujoco.egl", "mujoco.osmesa"):
        stub = sys.modules.get(name)
        if stub is not None and getattr(stub, "__file__", None) is None:
            sys.modules.pop(name, None)
    existing = sys.modules.get("mujoco")
    if existing is not None and hasattr(existing, "MjModel"):
        _install_render_api(existing)


def import_mujoco(*, register: bool = False) -> Any:
    """Return a MuJoCo API object usable for pure dynamics."""

    existing = sys.modules.get("mujoco")
    if (
        existing is not None
        and hasattr(existing, "mj_step")
        and hasattr(existing, "MjModel")
        and not getattr(existing, "physics_only", False)
    ):
        # Real package already imported successfully.
        if register:
            _install_robosuite_mujoco311_compat(existing)
        return existing

    if (
        existing is not None
        and getattr(existing, "physics_only", False)
        and hasattr(existing, "MjModel")
    ):
        if register:
            _install_renderer_stubs()
            _install_robosuite_mujoco311_compat(existing)
        return existing

    module = _build_physics_module()
    if register:
        sys.modules["mujoco"] = module
        _install_renderer_stubs()
        _install_robosuite_mujoco311_compat(module)
    return module


def import_mujoco_with_renderer() -> Any:
    """Load the public MuJoCo package including ``Renderer`` (EGL/OSMesa).

    Physics-only imports deliberately omit classic GL. VIS-X must call this
    instead of ``import_mujoco`` before creating cameras.
    """

    import os

    prepare_offscreen_gl()
    existing = sys.modules.get("mujoco")
    if (
        existing is not None
        and hasattr(existing, "Renderer")
        and not getattr(existing, "physics_only", False)
    ):
        return existing

    # Drop physics-only placeholder and any classic-GL stubs so public init runs.
    for key in list(sys.modules):
        if key == "mujoco" or key.startswith("mujoco."):
            mod = sys.modules[key]
            if key.startswith("mujoco.rendering") and getattr(mod, "__file__", None) is None:
                del sys.modules[key]
            elif key == "mujoco" and getattr(mod, "physics_only", False):
                del sys.modules[key]
            elif key.startswith("mujoco.") and getattr(mod, "physics_only", False):
                del sys.modules[key]

    # If physics-only root remains without Renderer, force a clean public import.
    existing = sys.modules.get("mujoco")
    if existing is not None and not hasattr(existing, "Renderer"):
        for key in list(sys.modules):
            if key == "mujoco" or key.startswith("mujoco."):
                del sys.modules[key]

    import mujoco

    if not hasattr(mujoco, "Renderer"):
        raise ImportError(
            "mujoco.Renderer unavailable; set MUJOCO_GL=egl|osmesa for VIS-X"
        )
    return mujoco
