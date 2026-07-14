import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from famous_cnns.cli import architecture_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(architecture_main("efficientnet", "infer"))
