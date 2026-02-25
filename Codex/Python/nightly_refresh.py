import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str) -> None:
    cmd = [sys.executable, str(BASE_DIR / script_name)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def main() -> None:
    run_step("refresh_catalog.py")
    run_step("rebuild_predictions.py")


if __name__ == "__main__":
    main()
