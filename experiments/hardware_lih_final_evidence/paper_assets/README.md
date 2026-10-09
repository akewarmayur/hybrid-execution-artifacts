# Frozen Hardware Paper Assets

These assets are generated only from `experiments/hardware_lih_final_evidence/{manifests,processed,raw}`. The generator performs no IBM connection, simulation, calibration, threshold fitting, or scientific execution.

## Regenerate

```bash
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/private/tmp/checkrcq-mpl
python experiments/hardware_lih_final_evidence/paper_assets/scripts/generate_hardware_paper_assets.py
```

Generation fails if the campaign is not complete, the 330 provider IDs are not unique, any frozen artifact hash fails, any incomplete tier enters final results, a headline value differs from authoritative records, or an authoritative file changes during generation.

## Recommended Main Paper Assets

- `figures/figure_hardware_main.pdf`: double-column, 7.15 x 5.35 in; one compact view of qualification, exact reuse, replay/migration outcomes, and temporal drift.
- `tables/table_hardware_summary.tex`: double-column; exact denominators and interpretation guards across qualification and RQ1--RQ6.
- Optional second table: `tables/table_hardware_rq4_secondary.tex`, only if the secondary changed-context diagnostic is discussed in the main text.

## Recommended Appendix Assets

- `figure_hardware_overhead.pdf`
- `figure_hardware_rq3_exact_reuse.pdf`
- `figure_hardware_rq6_backend_direction.pdf`
- `figure_hardware_rq5_ablation.pdf`
- `table_hardware_rq5_ablation.tex`
- `table_hardware_campaign_resource.tex`
- `table_hardware_qualification_details.tex`

The standalone qualification and temporal figures duplicate panels (a) and (d) of the main figure and should not both be included unless readability requires replacing the multi-panel figure.

## LaTeX Includes

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{experiments/hardware_lih_final_evidence/paper_assets/figures/figure_hardware_main.pdf}
  \caption{<use the candidate caption in captions.md>}
  \label{fig:hardware-main}
\end{figure*}

\input{experiments/hardware_lih_final_evidence/paper_assets/tables/table_hardware_summary.tex}
\input{experiments/hardware_lih_final_evidence/paper_assets/tables/table_hardware_rq4_secondary.tex}

\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{experiments/hardware_lih_final_evidence/paper_assets/figures/figure_hardware_overhead.pdf}
  \caption{<use the candidate caption in captions.md>}
  \label{fig:hardware-overhead}
\end{figure}
```

Tables require `booktabs`; the summary and wide appendix tables are intended for `table*` placement. Exact machine-readable figure data are in `data/`, and `figures/contact_sheet.png` previews all candidates.
