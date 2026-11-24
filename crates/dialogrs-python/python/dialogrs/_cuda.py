"""Find the NVIDIA driver and make pip-installed cuBLAS and cuDNN visible to the CUDA backends.

scriptrs and speakrs open the driver, cuBLAS and cuDNN at run time by name:
``libcuda.so.1``, ``libcublas.so.12`` and ``libcudnn.so.9`` on Linux, and
``nvcuda.dll``, ``cublas64_12.dll`` and ``cudnn64_9.dll`` on Windows. The
``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12`` wheels install those libraries
inside site-packages, where neither loader looks, so they are loaded here first.
Both loaders reuse a library the process already holds under the name they ask
for: glibc matches the soname lookup to the loaded copy, and on Windows the
backends check for a loaded module of that name before they search. The wheels'
copies therefore win over a CUDA toolkit's, which ships the same names.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

from ._errors import DialogrsError
from ._native import BACKEND

_WINDOWS = sys.platform == "win32"
# the driver's library, which the NVIDIA driver installs on the loader's own path
DRIVER = "nvcuda.dll" if _WINDOWS else "libcuda.so.1"
# each package directory inside the wheels' ``nvidia`` namespace package, with its
# libraries in dependency order, since each load must find what it links against
# already loaded. cuDNN is optional: only speakrs's CUDA fallback layers use it
_LIBRARIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        ("cublas", ("cublasLt64_12.dll", "cublas64_12.dll")),
        (
            "cudnn",
            (
                "cudnn_graph64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_heuristic64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll",
                "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll",
                "cudnn64_9.dll",
            ),
        ),
    )
    if _WINDOWS
    else (
        ("cublas", ("libcublasLt.so.12", "libcublas.so.12")),
        (
            "cudnn",
            (
                "libcudnn_graph.so.9",
                "libcudnn_ops.so.9",
                "libcudnn_heuristic.so.9",
                "libcudnn_engines_precompiled.so.9",
                "libcudnn_engines_runtime_compiled.so.9",
                "libcudnn_cnn.so.9",
                "libcudnn_adv.so.9",
                "libcudnn.so.9",
            ),
        ),
    )
)
_LIBRARY_DIR = "bin" if _WINDOWS else "lib"
_prepared = False
# a second thread must wait for the first to finish loading, not skip ahead
_lock = threading.Lock()


def prepare() -> None:
    """Check for the driver and load the pip CUDA libraries once, before the first model loads.

    The driver check runs before any download, so a machine without an NVIDIA
    GPU fails fast instead of after fetching models it cannot run.
    """
    global _prepared
    if BACKEND != "cuda":
        return
    with _lock:
        if not _prepared:
            _check_driver()
            _load_libraries()
            _prepared = True


def driver_path() -> str:
    """The driver to load: System32's copy on Windows, the loader's soname elsewhere."""
    if not _WINDOWS:
        return DRIVER
    # a full System32 path, as the backends load the driver, since a bare name would
    # also search the interpreter's directory for a planted nvcuda.dll
    buffer = ctypes.create_unicode_buffer(260)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise OSError("could not find the System32 directory")
    return str(Path(buffer.value) / DRIVER)


def _check_driver() -> None:
    try:
        ctypes.CDLL(driver_path())
    except OSError:
        raise DialogrsError(
            "cuda",
            f"no NVIDIA driver found ({DRIVER}); on Linux and Windows dialogrs runs on NVIDIA GPUs only",
        ) from None


def _load_libraries() -> None:
    try:
        import nvidia  # namespace package shared by the nvidia-* wheels
    except ImportError:
        return

    for root in getattr(nvidia, "__path__", []):
        for package, names in _LIBRARIES:
            lib_dir = Path(root) / package / _LIBRARY_DIR
            for name in names:
                path = lib_dir / name
                if not path.is_file():
                    continue
                try:
                    # Windows ignores the mode; an absolute path there also searches
                    # the DLL's own directory for its imports
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # a broken install falls back to the system search path
                    pass
