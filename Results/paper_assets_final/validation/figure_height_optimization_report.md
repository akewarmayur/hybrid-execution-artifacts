# Main-Figure Height Optimization Report

Layout-only optimization of the six final SIGMETRICS main-paper figures. The accepted root-level package is the before-state; the optimized package is under `Results/paper_assets_final/`.

| Figure | Old width x height (in) | New width x height (in) | Height reduction | Layout changes | Data unchanged |
|---|---:|---:|---:|---|---|
| `fig_rq1_boundary_value` | 6.40 x 2.10 | 6.40 x 1.55 | 26.2% | Shortened panel headings and tightened the existing horizontal two-panel composition. | Yes; source CSV SHA-256 unchanged |
| `fig_rq2_overhead_scaling` | 6.40 x 2.35 | 6.40 x 1.75 | 25.5% | Kept three horizontal panels, changed profile ticks to compact labels, and shortened headings. | Yes; source CSV SHA-256 unchanged |
| `fig_rq3_exact_work` | 6.40 x 2.10 | 6.40 x 1.55 | 26.2% | Kept the low-height two-panel layout and shortened panel headings. | Yes; source CSV SHA-256 unchanged |
| `fig_rq4_decision_quality` | 6.40 x 2.20 | 6.40 x 1.70 | 22.7% | Kept both independent-scenario panels horizontal and shortened headings without removing annotations. | Yes; source CSV SHA-256 unchanged |
| `fig_rq5_evidence_sufficiency` | 6.40 x 2.20 | 6.40 x 1.70 | 22.7% | Kept footprint and action-flip panels horizontal and shortened headings. | Yes; source CSV SHA-256 unchanged |
| `fig_rq6_generalization_hardware` | 6.40 x 2.75 | 6.40 x 2.00 | 27.3% | Kept all three panels in one row and shortened headings while retaining every cell and path count. | Yes; source CSV SHA-256 unchanged |

- Sum of old heights: **13.70 in**
- Sum of new heights: **10.25 in**
- Total height reduction: **25.2%**
- Figure width remains 6.40 in for every main figure.
- Typography remains DejaVu Sans; axis labels are 8.1 pt, ticks and legends are 7.5 pt, and panel titles are 8.8 pt.
- Panel ordering, values, categories, colors, markers, lines, hatches, legends, and numerical annotations are unchanged.
- All six optimized PDFs were independently rendered to 300-DPI PNG for clipping and readability inspection.
- No raw/processed scientific result, table data, appendix figure, or retired asset was modified; profile labels and associated captions were updated presentation-only.
- Live IBM jobs submitted: 0; experiments rerun: 0.
