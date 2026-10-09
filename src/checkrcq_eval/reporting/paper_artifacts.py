"""Top-level paper artifact builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from checkrcq_eval.analysis.hardware.e3_hw_analysis import analyze as analyze_e3_hw
from checkrcq_eval.analysis.ideal.e1_analysis import analyze as analyze_e1
from checkrcq_eval.analysis.ideal.e2_analysis import analyze as analyze_e2
from checkrcq_eval.analysis.ideal.e4_analysis import analyze as analyze_e4
from checkrcq_eval.analysis.noisy.e3_analysis import analyze as analyze_e3
from checkrcq_eval.analysis.helpers import write_table_bundle
from checkrcq_eval.constants import PAPER_OUTPUTS_DIR, PROCESSED_DATA_DIR
from checkrcq_eval.io_utils import ensure_dir
from checkrcq_eval.reporting.evaluation_timeline import build_evaluation_timeline
from checkrcq_eval.reporting.manifest_report import write_command_manifest
from checkrcq_eval.reporting.summary_report import write_summary_json


def build() -> dict[str, str]:
    """Rebuild all paper-facing artifacts from processed data."""
    output_dir = ensure_dir(PAPER_OUTPUTS_DIR)
    tables_dir = ensure_dir(output_dir / "tables")
    summaries_dir = ensure_dir(output_dir / "summaries")
    outputs: list[str] = []
    notes: list[str] = []
    analysis_count = 0
    hardware_processed = PROCESSED_DATA_DIR / "hardware" / "e3_hw_records.csv"

    analyze_e1()
    analysis_count += 1
    analyze_e2()
    analysis_count += 1
    analyze_e4()
    analysis_count += 1
    analyze_e3()
    analysis_count += 1
    if hardware_processed.exists():
        analyze_e3_hw()
        analysis_count += 1
    else:
        notes.append(
            "Hardware validation artifacts were not rebuilt because "
            f"{hardware_processed} does not exist yet."
        )

    eval_setup = pd.DataFrame(
        [
            {"item": "Workloads", "values": "LiH VQE; H2 VQE; ADAPT-VQE"},
            {"item": "Execution modes", "values": "M1 ideal; M2 noisy; M3 selected hardware validation"},
            {
                "item": "Interruption scenarios",
                "values": "HPC preemption; grouped-measurement failure; same-backend replay delay; cross-backend migration; ADAPT ansatz-expansion interruption",
            },
            {"item": "Checkpoint policy sweep", "values": "Semantic boundaries B1-B6 with cadence sweep k in {1,2,4,8}"},
            {"item": "Post-restore budget", "values": "Fixed continuation budget B with stable window"},
            {"item": "Reporting rule", "values": "Results reported separately by mode and never pooled"},
            {"item": "Repetition units", "values": "Seeds for ideal/noisy; hardware windows for hardware validation"},
        ]
    )
    outputs.extend(
        str(path)
        for path in write_table_bundle(
            eval_setup,
            tables_dir,
            "table_eval_setup",
            "Evaluation setup summary.",
            "tab:eval_setup",
        )
    )

    mode_mapping = pd.DataFrame(
        [
            {"mode": "M1", "primary_role": "Ideal simulation for controlled failure injection and ablations", "reported_in": "E1, E2, E4"},
            {"mode": "M2", "primary_role": "Noisy replay under recorded backend properties and controlled drift", "reported_in": "E3 noisy summary"},
            {"mode": "M3", "primary_role": "Selected hardware validation on IBM-style backends", "reported_in": "E3 hardware validation only"},
        ]
    )
    outputs.extend(
        str(path)
        for path in write_table_bundle(
            mode_mapping,
            tables_dir,
            "table_mode_mapping",
            "Mode-to-result mapping.",
            "tab:mode_mapping",
        )
    )

    baselines = pd.DataFrame(
        [
            {"name": "full_contract", "role": "CheckRCQ full restart contract", "missing_artifacts": "None"},
            {"name": "no_checkpoint", "role": "No restart contract", "missing_artifacts": "G0-GH"},
            {"name": "parameter_only", "role": "Persist parameters without contract context", "missing_artifacts": "GB-GH except GA"},
            {"name": "workflow_lite", "role": "Lightweight workflow snapshot", "missing_artifacts": "GD, GE, GF, GH"},
            {"name": "no_optimizer_memory", "role": "Ablation of optimizer memory", "missing_artifacts": "GB"},
            {"name": "no_measurement_ledger", "role": "Ablation of grouped-measurement ledger", "missing_artifacts": "GD"},
            {"name": "no_estimation_mitigation_state", "role": "Ablation of estimator/mitigation state", "missing_artifacts": "GE"},
            {"name": "no_backend_snapshot", "role": "Ablation of backend evidence", "missing_artifacts": "GF"},
            {"name": "no_geometry", "role": "Ablation of continuation geometry", "missing_artifacts": "GH"},
            {"name": "no_adapt_history", "role": "ADAPT-VQE only ablation of operator history", "missing_artifacts": "GA (adaptive history component)"},
        ]
    )
    outputs.extend(
        str(path)
        for path in write_table_bundle(
            baselines,
            tables_dir,
            "table_baselines_ablations",
            "Baselines and ablations used in the evaluation.",
            "tab:baselines_ablations",
        )
    )

    outputs.extend(str(path) for path in build_evaluation_timeline(output_dir))

    summary = pd.DataFrame(
        [
            {"artifact_group": "paper_tables", "count": 3},
            {"artifact_group": "paper_figures", "count": 1},
            {"artifact_group": "analysis_commands", "count": analysis_count},
            {"artifact_group": "hardware_processed_available", "count": int(hardware_processed.exists())},
        ]
    )
    summary_path = write_summary_json(
        summaries_dir / "paper_artifacts_summary.json",
        title="Paper artifact bundle",
        frame=summary,
        notes=[
            "Static paper tables plus per-experiment analyses regenerated from processed data.",
            *notes,
        ],
    )
    outputs.append(str(summary_path))
    manifest = write_command_manifest(
        setting="paper",
        evaluation_question="paper_artifacts",
        command="build-paper-artifacts",
        output_dir=output_dir,
        inputs=[],
        outputs=outputs,
        notes=[
            "Cross-setting tables preserve separate blocks and do not pool statistics.",
            *notes,
        ],
    )
    outputs.append(str(manifest))
    return {
        "output_dir": str(output_dir),
        "hardware_status": "available" if hardware_processed.exists() else "pending",
    }
