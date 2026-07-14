"""Root training CLI. Equivalent to ``famous-cnns train``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from famous_cnns.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
