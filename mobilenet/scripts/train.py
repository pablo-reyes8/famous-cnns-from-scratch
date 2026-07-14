import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from famous_cnns.cli import architecture_main  # noqa: E402

if __name__ == "__main__":
    model = "mobilenet_v2" if "--v2" in sys.argv else "mobilenet_v1"
    if "--v2" in sys.argv:
        sys.argv.remove("--v2")
    raise SystemExit(architecture_main(model, "train"))
