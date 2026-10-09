from __future__ import annotations

import pandas as pd

from checkrcq_eval.common.latex_tables import dataframe_to_latex


def test_latex_table_uses_booktabs_and_row_colors() -> None:
    frame = pd.DataFrame([{"metric_name": "save_latency_s", "value": 0.12}])
    latex = dataframe_to_latex(frame, caption="Demo_table", label="tab:demo", row_colors=True)
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "\\rowcolors{2}{gray!10}{white}" in latex
    assert "Demo\\_table" in latex
