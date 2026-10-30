"""
dep_scanner.py — scan one of a model provider's scripts (adaptor.py,
preprocessing.py) for missing dependencies and install them with UV before
the stage that runs it.

Invoked by the Go pipeline as:  python3 /app/dep_scanner.py <script.py>
(exit 0 = ready; non-zero = install failed, stderr carries the uv output).

Only third-party packages that are not already importable are installed.
stdlib modules are filtered via sys.stdlib_module_names. Imports inside a
try block that catches ImportError (or a broader exception) are optional by
the script's own declaration, so they are not installed: the BODH templates
import e.g. nibabel that way, only for NIfTI inputs. Downloads reach
pypi.org / files.pythonhosted.org, both download-only in the network policy.

_ALWAYS_PRESENT is trimmed to what THIS
image actually ships (see requirements.txt) — the original list described a
kitchen-sink image and would wrongly skip installs here. Correctness does
not depend on the list: anything not listed still passes through the
importlib.find_spec check before an install is attempted.
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

# Maps import name → PyPI package name where they differ.
# None means "skip this import entirely" (provided by the platform or not
# installable from PyPI).
_IMPORT_TO_PACKAGE: dict[str, str | None] = {
    # imaging
    "cv2":              "opencv-python-headless",
    "PIL":              "Pillow",
    "skimage":          "scikit-image",
    # ML
    "sklearn":          "scikit-learn",
    "xgboost":          "xgboost",
    "lightgbm":         "lightgbm",
    "catboost":         "catboost",
    # data / misc
    "bs4":              "beautifulsoup4",
    "yaml":             "PyYAML",
    "dotenv":           "python-dotenv",
    "attr":             "attrs",
    "dateutil":         "python-dateutil",
    "sentence_transformers": "sentence-transformers",
    # Google SDKs — platform-provided access patterns; never install
    "google":           None,
    "googleapiclient":  None,
}

# Packages this image definitely ships (requirements.txt + their transitive
# deps) — skipped without an import probe. Everything else falls through to
# the find_spec check.
_ALWAYS_PRESENT: frozenset[str] = frozenset({
    "numpy", "scipy", "pandas",
    "sklearn", "scikit_learn",
    "PIL", "cv2",
    "torch", "torchvision",
    "transformers", "tokenizers", "huggingface_hub", "safetensors",
    "onnx", "onnxruntime",
    "pydicom", "pylibjpeg",
})


def _stdlib_names() -> frozenset[str]:
    if hasattr(sys, "stdlib_module_names"):
        return sys.stdlib_module_names  # Python 3.10+
    return frozenset(sys.builtin_module_names)


# Exception names whose handler makes the imports in its try block optional.
_OPTIONAL_IMPORT_HANDLERS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:  # bare except:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(t, ast.Name) and t.id in _OPTIONAL_IMPORT_HANDLERS for t in types)


def _import_roots(node) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.split(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module:  # relative imports are intra-package
        return {node.module.split(".")[0]}
    return set()


def scan_imports(script_path: Path) -> tuple[set[str], set[str]]:
    """Parse script_path with ast; return (required, optional) import root
    names. Optional ones sit in a try block whose handler catches ImportError."""
    source = script_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(script_path))
    except SyntaxError as exc:
        raise ValueError(f"Could not parse {script_path.name}: {exc}") from exc
    guarded: set[int] = set()
    try_types = tuple(t for t in (getattr(ast, "Try", None), getattr(ast, "TryStar", None)) if t)
    for node in ast.walk(tree):
        if isinstance(node, try_types) and any(_catches_import_error(h) for h in node.handlers):
            for stmt in node.body:
                guarded.update(id(sub) for sub in ast.walk(stmt) if isinstance(sub, (ast.Import, ast.ImportFrom)))
    required: set[str] = set()
    optional: set[str] = set()
    for node in ast.walk(tree):
        (optional if id(node) in guarded else required).update(_import_roots(node))
    return required, optional - required


def extract_imports(script_path: Path) -> set[str]:
    """The import root names script_path needs (optional imports excluded)."""
    return scan_imports(script_path)[0]


def _is_importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def install_missing_deps(script_path: Path) -> tuple[list[str], list[str]]:
    """
    Scan script_path, determine which third-party packages are missing,
    and install them with `uv pip install --system`.

    Returns (installed, skipped) — both are lists of package specs.
    Raises RuntimeError with captured stderr if uv exits non-zero.
    """
    stdlib = _stdlib_names()
    raw_imports, optional = scan_imports(script_path)

    to_install: list[str] = []
    skipped: list[str] = [f"{name} (optional: imported under try/except ImportError)" for name in sorted(optional)]

    for name in sorted(raw_imports):
        if name in stdlib:
            skipped.append(f"{name} (stdlib)")
            continue
        if name in _IMPORT_TO_PACKAGE and _IMPORT_TO_PACKAGE[name] is None:
            skipped.append(f"{name} (provided by platform)")
            continue
        if name in _ALWAYS_PRESENT:
            skipped.append(f"{name} (image dep)")
            continue
        if _is_importable(name):
            skipped.append(f"{name} (already installed)")
            continue
        package = _IMPORT_TO_PACKAGE.get(name, name)
        to_install.append(package)

    if not to_install:
        return [], skipped

    cmd = ["/bin/uv", "pip", "install", "--system"] + to_install
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"UV failed to install {script_path.name} dependencies "
            f"({', '.join(to_install)}):\n{result.stderr}"
        )

    return to_install, skipped


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: dep_scanner.py <adaptor.py>", file=sys.stderr)
        return 2
    script_path = Path(sys.argv[1])
    if not script_path.exists():
        print(f"dep_scanner: script not found: {script_path}", file=sys.stderr)
        return 2
    try:
        installed, skipped = install_missing_deps(script_path)
    except (RuntimeError, ValueError) as exc:
        print(f"dep_scanner: {exc}", file=sys.stderr)
        return 1
    if installed:
        print(f"dep_scanner: installed: {', '.join(installed)}")
    print(f"dep_scanner: skipped (stdlib/image/present): {len(skipped)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())