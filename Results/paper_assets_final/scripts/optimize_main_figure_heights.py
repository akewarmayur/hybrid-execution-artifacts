#!/usr/bin/env python3
from pathlib import Path
from checkrcq_eval.final_paper_height_optimizer import optimize_main_figure_heights

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[3]
    print(optimize_main_figure_heights(root))
