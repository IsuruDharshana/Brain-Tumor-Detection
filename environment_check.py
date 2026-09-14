"""
environment_check.py
--------------------
Quick sanity check for the Brain Tumor MRI Classification project.

Usage:
    .venv\\Scripts\\python environment_check.py   (Windows)
    .venv/bin/python environment_check.py        (Linux / macOS)
"""

import platform
import sys


def separator(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{'--' * 2}  {title}  " + "-" * max(0, width - len(title) - 8))
    else:
        print("-" * width)


def check_python() -> None:
    separator("Python")
    print(f"  Version     : {sys.version}")
    print(f"  Executable  : {sys.executable}")


def check_os() -> None:
    separator("Operating System")
    print(f"  System      : {platform.system()} {platform.release()}")
    print(f"  Version     : {platform.version()}")
    print(f"  Machine     : {platform.machine()}")
    print(f"  Processor   : {platform.processor()}")


def check_numpy() -> None:
    separator("NumPy")
    try:
        import numpy as np  # noqa: PLC0415

        print(f"  Version     : {np.__version__}")
    except ImportError as exc:
        print(f"  [ERROR] NumPy not available: {exc}")


def check_tensorflow() -> None:
    separator("TensorFlow / Keras")
    try:
        import tensorflow as tf  # noqa: PLC0415

        print(f"  TF Version  : {tf.__version__}")

        # Keras version (bundled with TF 2.x)
        try:
            import keras  # noqa: PLC0415
            print(f"  Keras       : {keras.__version__}")
        except ImportError:
            print("  Keras       : (bundled - not separately importable)")

        # GPU detection
        separator("GPU / Device Info")
        gpus = tf.config.list_physical_devices("GPU")
        cpus = tf.config.list_physical_devices("CPU")

        print(f"  CPU devices : {len(cpus)}")
        for d in cpus:
            print(f"    {d}")

        if gpus:
            print(f"  GPU devices : {len(gpus)}")
            for g in gpus:
                print(f"    {g}")
        else:
            print("  GPU devices : 0  (none visible - expected on this laptop)")
            print("  NOTE: GPU training will be performed on the RTX 3060 workstation.")

        # Confirm TF can build a trivial graph (smoke test)
        separator("Smoke Test")
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.reduce_sum(a)
        print(f"  tf.reduce_sum([1,2,3]) = {b.numpy()}  [OK]")

    except ImportError as exc:
        print(f"  [ERROR] TensorFlow not available: {exc}")
        print("  Run:  pip install -r requirements.txt")


def main() -> None:
    print("\n" + "=" * 60)
    print("  Brain Tumor MRI Classification -- Environment Check")
    print("=" * 60)
    check_python()
    check_os()
    check_numpy()
    check_tensorflow()
    separator()
    print("  Environment check complete.\n")


if __name__ == "__main__":
    main()
