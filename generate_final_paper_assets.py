#!/usr/bin/env python3
"""Generate the frozen, publication-ready SIGMETRICS asset package."""

from pathlib import Path

from checkrcq_eval.final_paper_assets import generate_final_paper_assets


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(generate_final_paper_assets(root, root / "Results/paper_assets_final"))
