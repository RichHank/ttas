"""Build the static GitHub Pages snapshot for TTAS."""

from __future__ import annotations

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "figures"
TARGET = ROOT / "docs" / "figures"


def build() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for html_file in SOURCE.glob("*.html"):
        shutil.copy2(html_file, TARGET / html_file.name)


if __name__ == "__main__":
    build()
