#!/usr/bin/env python3
from pathlib import Path
from checkrcq_eval.final_paper_assets import generate_final_paper_assets

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(generate_final_paper_assets(root))
