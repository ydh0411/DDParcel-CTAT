"""Run CTAT core tests without requiring pytest to be installed."""

from __future__ import annotations

import importlib.util
import pathlib


def main() -> None:
    test_path = pathlib.Path("tests/test_ctat_core.py")
    spec = importlib.util.spec_from_file_location("test_ctat_core", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {test_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tests = sorted(name for name in dir(module) if name.startswith("test_"))
    for name in tests:
        print(f"RUN {name}")
        getattr(module, name)()

    print(f"passed {len(tests)}")


if __name__ == "__main__":
    main()
