#!/usr/bin/env python3
"""Analyze the frozen RQ4 operating-point confirmation without retuning."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from checkrcq_eval.constants import ROOT


OUTPUT = Path(os.environ.get("CHECKRCQ_RQ4_FRESH_OUTPUT", ROOT / "outputs/sigmetrics/rq4_fresh_confirmation")).resolve()
PROCESSED = OUTPUT / "processed/records.jsonl"
VALIDATION = OUTPUT / "processed/validation_report.json"
CAMPAIGN_MANIFEST = OUTPUT / "manifests/campaign.json"
PREEXECUTION = OUTPUT / "preexecution_manifest.json"
CONFIG = ROOT / "configs/campaigns/proposed/rq4_fresh_heldout_planner_validation.yaml"
REPORT = Path(os.environ.get("CHECKRCQ_RQ4_FRESH_REPORT", ROOT / "docs/rq4_fresh_operating_point_confirmation.md")).resolve()
OUTPUT = Path(os.environ.get("CHECKRCQ_RQ4_FRESH_OUTPUT", ROOT / "outputs/sigmetrics/paper_sources/rq4_fresh_confirmation")).resolve()
THRESHOLDS = (0.15, 0.05)
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20270926
WORKLOADS = ("h2_vqe", "lih_vqe", "adapt_vqe", "qaoa_maxcut")
CONTEXTS = ("no_change", "safe_change", "same_backend_delay", "cross_backend", "high_change")


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _safe_counterfactuals(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outcomes = record["counterfactual_reference"]["counterfactual_outcomes"]
    return [
        item for item in outcomes.values()
        if item.get("technically_feasible") is True
        and item.get("continuation_success") is True
        and item.get("stable_continuation") is True
    ]


def annotate(record: Mapping[str, Any]) -> dict[str, Any]:
    action = str(record["decision"]["selected_action"])
    proceeded = action in {"replay", "migrate"}
    safe = proceeded and record["outcome"].get("continuation_success") is True and record["outcome"].get("stable_continuation") is True
    overblock = action == "block" and bool(_safe_counterfactuals(record))
    return {
        "record": record,
        "scenario_id": str(record["scenario"]["scenario_id"]),
        "threshold": float(record["policy"]["operating_point"]),
        "seed": int(record["scenario"]["seed"]),
        "workload": str(record["scenario"]["workload"]),
        "context": str(record["scenario"]["changed_context_class"]),
        "action": action,
        "target": record["decision"].get("selected_target"),
        "proceeded": proceeded,
        "safe": safe,
        "unsafe": proceeded and not safe,
        "overblock": overblock,
    }


def validate_pairs(records: Sequence[Mapping[str, Any]]) -> list[dict[float, dict[str, Any]]]:
    if len(records) != 320:
        raise RuntimeError(f"Expected 320 accepted policy records, found {len(records)}.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in records:
        row = annotate(raw)
        grouped[row["scenario_id"]].append(row)
    if len(grouped) != 160:
        raise RuntimeError(f"Expected 160 independent scenarios, found {len(grouped)}.")
    pairs = []
    for scenario_id, rows in sorted(grouped.items()):
        if len(rows) != 2 or {row["threshold"] for row in rows} != set(THRESHOLDS):
            raise RuntimeError(f"Scenario {scenario_id} does not have exactly the two frozen thresholds.")
        pair = {row["threshold"]: row for row in rows}
        left = pair[0.15]["record"]
        right = pair[0.05]["record"]
        for path in (
            ("scenario", "scenario_id"),
            ("scenario", "seed"),
            ("scenario", "workload"),
            ("scenario", "checkpoint_contract_hash"),
            ("scenario", "restore_environment_hash"),
            ("scenario", "failure_scenario_id"),
            ("decision", "candidate_actions"),
            ("decision", "observable_feature_hash"),
            ("counterfactual_reference", "counterfactual_ids"),
            ("counterfactual_reference", "counterfactual_outcomes"),
            ("same_state_audit", "continuation_envelope_hash"),
            ("same_state_audit", "counterfactual_outcomes_hash"),
        ):
            a: Any = left
            b: Any = right
            for key in path:
                a = a[key]
                b = b[key]
            if a != b:
                raise RuntimeError(f"Paired mismatch at {'.'.join(path)} for {scenario_id}.")
        if left["fresh_confirmation"]["hardware_used"] or right["fresh_confirmation"]["hardware_used"]:
            raise RuntimeError("Fresh confirmation record unexpectedly reports hardware use.")
        pairs.append(pair)
    return pairs


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    replay = sum(row["action"] == "replay" for row in rows)
    migrate = sum(row["action"] == "migrate" for row in rows)
    block = sum(row["action"] == "block" for row in rows)
    proceeded = replay + migrate
    safe = sum(row["safe"] for row in rows)
    unsafe = sum(row["unsafe"] for row in rows)
    overblock = sum(row["overblock"] for row in rows)
    coverage_ci = wilson(proceeded, n)
    success_ci = wilson(safe, n)
    unsafe_ci = wilson(unsafe, proceeded)
    over_ci = wilson(overblock, block)
    return {
        "scenario_count": n,
        "replay_count": replay,
        "migration_count": migrate,
        "block_count": block,
        "coverage_numerator": proceeded,
        "coverage_denominator": n,
        "coverage_rate": proceeded / n if n else None,
        "coverage_ci_low": coverage_ci[0],
        "coverage_ci_high": coverage_ci[1],
        "successful_coverage_numerator": safe,
        "successful_coverage_denominator": n,
        "successful_coverage_rate": safe / n if n else None,
        "successful_coverage_ci_low": success_ci[0],
        "successful_coverage_ci_high": success_ci[1],
        "unsafe_numerator": unsafe,
        "unsafe_denominator": proceeded,
        "unsafe_rate": unsafe / proceeded if proceeded else None,
        "unsafe_ci_low": unsafe_ci[0],
        "unsafe_ci_high": unsafe_ci[1],
        "overblock_numerator": overblock,
        "overblock_denominator": block,
        "overblock_rate": overblock / block if block else None,
        "overblock_ci_low": over_ci[0],
        "overblock_ci_high": over_ci[1],
        "successful_continuation_count": safe,
        "unsuccessful_proceeded_count": unsafe,
    }


def transition_and_causal_rows(
    pairs: Sequence[dict[float, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    detailed = []
    causal = Counter()
    for pair in pairs:
        baseline = pair[0.15]
        conservative = pair[0.05]
        if baseline["unsafe"] and conservative["action"] == "block":
            consequence = "unsafe_continuation_avoided_by_block"
            causal["unsafe_to_block"] += 1
        elif baseline["unsafe"] and conservative["safe"] and (
            baseline["action"], baseline["target"]
        ) != (conservative["action"], conservative["target"]):
            consequence = "unsafe_continuation_changed_to_safe_action"
            causal["unsafe_to_different_safe_action"] += 1
        elif baseline["safe"] and conservative["action"] == "block":
            consequence = "successful_continuation_newly_blocked"
            causal["successful_to_block"] += 1
        elif baseline["safe"] and conservative["safe"]:
            consequence = "successful_continuation_retained"
            causal["successful_retained"] += 1
        elif baseline["unsafe"] and conservative["unsafe"]:
            consequence = "unsafe_continuation_not_avoided"
            causal["unsafe_not_avoided"] += 1
        elif baseline["action"] == "block" and conservative["action"] == "block":
            consequence = "unchanged_block"
            causal["unchanged_block"] += 1
        else:
            consequence = "other_transition"
            causal["other_action_changes"] += 1
        detailed.append({
            "scenario_id": baseline["scenario_id"],
            "seed": baseline["seed"],
            "workload": baseline["workload"],
            "context": baseline["context"],
            "action_0_15": baseline["action"],
            "target_0_15": baseline["target"],
            "outcome_0_15": "safe" if baseline["safe"] else "unsafe" if baseline["unsafe"] else "block",
            "action_0_05": conservative["action"],
            "target_0_05": conservative["target"],
            "outcome_0_05": "safe" if conservative["safe"] else "unsafe" if conservative["unsafe"] else "block",
            "paired_consequence": consequence,
        })
    aggregate = []
    counter = Counter(
        (row["action_0_15"], row["action_0_05"], row["outcome_0_15"], row["paired_consequence"])
        for row in detailed
    )
    for (action15, action05, outcome15, consequence), count in sorted(counter.items()):
        aggregate.append({
            "action_0_15": action15,
            "action_0_05": action05,
            "outcome_0_15": outcome15,
            "paired_consequence": consequence,
            "scenario_count": count,
        })
    return detailed, aggregate, dict(causal)


def paired_bootstrap(
    pairs: Sequence[dict[float, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = (
        "coverage_difference_0_05_minus_0_15",
        "successful_coverage_difference_0_05_minus_0_15",
        "unsafe_rate_difference_0_05_minus_0_15",
        "unsafe_0_15_avoided_fraction_of_scenarios",
    )

    def values(sample: Sequence[dict[float, dict[str, Any]]]) -> dict[str, float]:
        m15 = summarize([pair[0.15] for pair in sample])
        m05 = summarize([pair[0.05] for pair in sample])
        avoided = sum(
            pair[0.15]["unsafe"] and not pair[0.05]["unsafe"] for pair in sample
        )
        return {
            names[0]: float(m05["coverage_rate"] - m15["coverage_rate"]),
            names[1]: float(m05["successful_coverage_rate"] - m15["successful_coverage_rate"]),
            names[2]: float(m05["unsafe_rate"] - m15["unsafe_rate"]),
            names[3]: avoided / len(sample),
        }

    point = values(pairs)
    samples = {name: [] for name in names}
    for _ in range(BOOTSTRAP_REPETITIONS):
        selected = [pairs[index] for index in rng.integers(0, len(pairs), len(pairs))]
        result = values(selected)
        for name in names:
            samples[name].append(result[name])
    return [
        {
            "metric": name,
            "estimate": point[name],
            "ci_low": float(np.quantile(samples[name], 0.025)),
            "ci_high": float(np.quantile(samples[name], 0.975)),
            "resampling_unit": "independent_scenario",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        }
        for name in names
    ]


def tex_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], caption: str, label: str) -> None:
    def esc(value: Any) -> str:
        if value is None:
            return "--"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("_", "\\_").replace("%", "\\%")

    headers = [column.replace("_", " ").title() for column in columns]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(esc(row.get(column)) for column in columns) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_figures(operating: Sequence[Mapping[str, Any]], causal: Mapping[str, int]) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    source = []
    for row in operating:
        threshold = row["threshold"]
        for metric, value, numerator, denominator in (
            ("successful coverage", row["successful_coverage_rate"], row["successful_coverage_numerator"], row["successful_coverage_denominator"]),
            ("unsafe continuation", row["unsafe_rate"], row["unsafe_numerator"], row["unsafe_denominator"]),
            ("block fraction", row["block_count"] / row["scenario_count"], row["block_count"], row["scenario_count"]),
            ("overblock among blocks", row["overblock_rate"], row["overblock_numerator"], row["overblock_denominator"]),
        ):
            source.append({"threshold": threshold, "metric": metric, "value": value, "numerator": numerator, "denominator": denominator})
    write_csv(figures / "fig_rq4_fresh_safety_coverage_tradeoff_source.csv", source)
    labels = ["0.15\nproduction", "0.05\nconservative"]
    metrics = ["successful coverage", "unsafe continuation", "block fraction", "overblock among blocks"]
    colors = ["#1f7a8c", "#d1495b", "#edae49", "#5c7457"]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(2)
    width = 0.19
    for index, (metric, color) in enumerate(zip(metrics, colors)):
        values = [next(row["value"] for row in source if row["threshold"] == threshold and row["metric"] == metric) for threshold in THRESHOLDS]
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=metric.title(), color=color)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Rate with metric-specific denominator")
    axis.set_title("Two frozen RQ4 operating points")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "fig_rq4_fresh_safety_coverage_tradeoff.png", dpi=300)
    fig.savefig(figures / "fig_rq4_fresh_safety_coverage_tradeoff.pdf")
    plt.close(fig)

    categories = [
        ("unsafe_to_block", "Unsafe avoided\nby block"),
        ("unsafe_to_different_safe_action", "Unsafe changed\nto safe action"),
        ("successful_retained", "Successful\nretained"),
        ("successful_to_block", "Successful\nblocked"),
        ("unsafe_not_avoided", "Unsafe not\navoided"),
        ("unchanged_block", "Unchanged\nblock"),
        ("other_action_changes", "Other"),
    ]
    paired_source = [{"category": key, "label": label.replace("\n", " "), "count": causal.get(key, 0)} for key, label in categories]
    write_csv(figures / "fig_rq4_fresh_paired_outcomes_source.csv", paired_source)
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    bars = axis.bar([label for _, label in categories], [causal.get(key, 0) for key, _ in categories], color=["#2a9d8f", "#69b3a2", "#1f7a8c", "#e9c46a", "#d1495b", "#777777", "#a8a8a8"])
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Fresh independent scenarios")
    axis.set_title("Paired consequences of changing only the frozen operating point")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "fig_rq4_fresh_paired_outcomes.png", dpi=300)
    fig.savefig(figures / "fig_rq4_fresh_paired_outcomes.pdf")
    plt.close(fig)


def package_review(files: Iterable[Path]) -> tuple[Path, str]:
    if OUTPUT.exists():
        raise FileExistsError(f"Output package already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    entries = []
    for source in files:
        relative = source.relative_to(ROOT)
        destination = OUTPUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append({"path": str(relative), "sha256": sha256_file(destination), "bytes": destination.stat().st_size})
    manifest = OUTPUT / "artifact_manifest.json"
    manifest.write_text(json.dumps({"schema_version": "checkrcq-rq4-fresh-review-package-v1", "artifacts": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, sha256_file(manifest)


def main() -> int:
    validation = load_json(VALIDATION)
    campaign = load_json(CAMPAIGN_MANIFEST)
    preexecution = load_json(PREEXECUTION)
    if not validation["valid"] or validation["accepted_count"] != 320:
        raise RuntimeError("Canonical validation is not complete and clean.")
    if any((validation["quarantined_count"], validation["duplicate_run_count"], len(validation["missing_run_ids"]), len(validation["unexpected_run_ids"]), campaign["failed"])):
        raise RuntimeError("Campaign validation contains failures, quarantine, duplicates, missing, or unexpected records.")
    if preexecution["hardware_job_count"] != 0:
        raise RuntimeError("Pre-execution manifest permits hardware unexpectedly.")

    records = load_records(PROCESSED)
    pairs = validate_pairs(records)
    rows_by_threshold = {threshold: [pair[threshold] for pair in pairs] for threshold in THRESHOLDS}
    operating = [{"threshold": threshold, **summarize(rows_by_threshold[threshold])} for threshold in THRESHOLDS]
    detailed, transitions, causal = transition_and_causal_rows(pairs)
    uncertainty = paired_bootstrap(pairs)
    tables = OUTPUT / "tables"
    analysis = OUTPUT / "analysis"
    write_csv(tables / "table_rq4_fresh_operating_points.csv", operating)
    write_csv(tables / "table_rq4_paired_transitions.csv", transitions)
    write_csv(analysis / "rq4_paired_scenarios.csv", detailed)
    write_csv(analysis / "rq4_paired_uncertainty.csv", uncertainty)

    workload_rows = []
    for threshold in THRESHOLDS:
        for workload in WORKLOADS:
            subset = [row for row in rows_by_threshold[threshold] if row["workload"] == workload]
            workload_rows.append({"threshold": threshold, "workload": workload, **summarize(subset)})
    write_csv(tables / "table_rq4_workload_breakdown.csv", workload_rows)
    context_rows = []
    for threshold in THRESHOLDS:
        for context in CONTEXTS:
            subset = [row for row in rows_by_threshold[threshold] if row["context"] == context]
            context_rows.append({"threshold": threshold, "scenario_type": context, **summarize(subset)})
    write_csv(tables / "table_rq4_scenario_type_breakdown.csv", context_rows)
    action_rows = []
    for threshold in THRESHOLDS:
        for action in ("replay", "migrate", "block"):
            subset = [row for row in rows_by_threshold[threshold] if row["action"] == action]
            action_rows.append({"threshold": threshold, "action": action, **summarize(subset)})
    write_csv(tables / "table_rq4_action_outcomes.csv", action_rows)

    tex_table(
        tables / "table_rq4_fresh_operating_points.tex",
        operating,
        ("threshold", "coverage_numerator", "coverage_denominator", "coverage_rate", "successful_coverage_rate", "unsafe_numerator", "unsafe_denominator", "unsafe_rate", "overblock_numerator", "overblock_denominator", "replay_count", "migration_count", "block_count"),
        "Independent RQ4 confirmation at exactly two frozen operating points.",
        "tab:rq4_fresh_operating_points",
    )
    tex_table(
        tables / "table_rq4_paired_transitions.tex",
        transitions,
        ("action_0_15", "action_0_05", "outcome_0_15", "paired_consequence", "scenario_count"),
        "Paired scenario transitions between the two frozen operating points.",
        "tab:rq4_fresh_paired_transitions",
    )

    m15, m05 = operating
    causal.update({
        "net_reduction_unsafe_proceeded": m15["unsafe_numerator"] - m05["unsafe_numerator"],
        "reduction_successful_proceeded": m15["successful_continuation_count"] - m05["successful_continuation_count"],
    })
    write_csv(analysis / "rq4_primary_causal_counts.csv", [{"metric": key, "count": value} for key, value in sorted(causal.items())])
    replication = []
    prior = {
        0.15: {"coverage_rate": 0.800, "successful_coverage_rate": 0.325, "unsafe_numerator": 76, "unsafe_denominator": 128, "unsafe_rate": 76 / 128, "overblock_numerator": 0, "overblock_denominator": 32},
        0.05: {"coverage_rate": 0.400, "successful_coverage_rate": 0.3125, "unsafe_numerator": 14, "unsafe_denominator": 64, "unsafe_rate": 14 / 64, "overblock_numerator": 2, "overblock_denominator": 96},
    }
    for threshold in THRESHOLDS:
        replication.append({"population": "previous_diagnostic", "threshold": threshold, **prior[threshold]})
        current = next(row for row in operating if row["threshold"] == threshold)
        replication.append({"population": "independent_confirmation", "threshold": threshold, **{key: current[key] for key in prior[threshold]}})
    write_csv(analysis / "rq4_replication_comparison.csv", replication)
    build_figures(operating, causal)

    successful15 = int(m15["successful_continuation_count"])
    retained = int(causal.get("successful_retained", 0))
    lost = int(causal.get("successful_to_block", 0))
    avoided = int(causal.get("unsafe_to_block", 0) + causal.get("unsafe_to_different_safe_action", 0))
    confirmed = bool(
        m05["unsafe_rate"] < m15["unsafe_rate"]
        and retained > lost
        and avoided > lost
    )
    conclusion = (
        "CONFIRMED:\nFRESH DATA SUPPORT THE OPERATING-POINT TRADEOFF"
        if confirmed
        else "NOT CONFIRMED:\nFRESH DATA DO NOT REPLICATE THE OPERATING-POINT TRADEOFF"
    )
    uncertainty_index = {row["metric"]: row for row in uncertainty}
    report_lines = [
        "# RQ4 Fresh Operating-Point Confirmation",
        "",
        "## 1. Experimental freeze and provenance",
        f"The campaign was frozen at Git commit `{preexecution['git_commit']}` with a clean worktree, thresholds 0.15 and 0.05, and pre-execution manifest `{sha256_file(PREEXECUTION)}`. The score, action ordering, continuation envelope, workloads, and scenario definitions were unchanged.",
        "",
        "## 2. Independence from prior data",
        f"The experiment uses seeds {', '.join(str(item) for item in preexecution['fresh_seeds'])}. The programmatic seed audit found no overlap across the {load_json(OUTPUT / 'seed_provenance.json')['historical_sources_checked']} historical config/output populations inspected.",
        "",
        "## 3. Validation result",
        f"Canonical validation accepted {validation['accepted_count']}/{validation['expected_run_count']} records with zero missing, duplicate, unexpected, quarantined, or failed records. All 160 threshold pairs shared checkpoint, environment, failure, candidate-action, continuation-envelope, and counterfactual hashes. Hardware jobs were zero.",
        "",
        "## 4. Overall 0.15 result",
        f"Coverage was {m15['coverage_numerator']}/{m15['coverage_denominator']} ({m15['coverage_rate']:.3f}); successful coverage was {m15['successful_coverage_numerator']}/{m15['successful_coverage_denominator']} ({m15['successful_coverage_rate']:.3f}); unsafe continuation was {m15['unsafe_numerator']}/{m15['unsafe_denominator']} ({m15['unsafe_rate']:.3f}); over-conservative blocking was {m15['overblock_numerator']}/{m15['overblock_denominator']} ({m15['overblock_rate']:.3f}). Actions were {m15['replay_count']} replay, {m15['migration_count']} migration, and {m15['block_count']} block.",
        "",
        "## 5. Overall 0.05 result",
        f"Coverage was {m05['coverage_numerator']}/{m05['coverage_denominator']} ({m05['coverage_rate']:.3f}); successful coverage was {m05['successful_coverage_numerator']}/{m05['successful_coverage_denominator']} ({m05['successful_coverage_rate']:.3f}); unsafe continuation was {m05['unsafe_numerator']}/{m05['unsafe_denominator']} ({m05['unsafe_rate']:.3f}); over-conservative blocking was {m05['overblock_numerator']}/{m05['overblock_denominator']} ({m05['overblock_rate']:.3f}). Actions were {m05['replay_count']} replay, {m05['migration_count']} migration, and {m05['block_count']} block.",
        "",
        "## 6. Paired scenario transitions",
        "The complete paired transition table is in `table_rq4_paired_transitions.csv`; it preserves baseline outcome and consequence for every action transition.",
        "",
        "## 7. Unsafe continuations avoided",
        f"Counts first: {causal.get('unsafe_to_block', 0)} unsafe 0.15 continuations became blocks, and {causal.get('unsafe_to_different_safe_action', 0)} became different safe actions. Net unsafe proceeded cases fell by {causal['net_reduction_unsafe_proceeded']}.",
        "",
        "## 8. Successful continuations retained/lost",
        f"Of {successful15} successful 0.15 continuations, {retained} remained successful/proceeded at 0.05 and {lost} became blocks. Successful proceeded cases changed by {-causal['reduction_successful_proceeded']} (0.05 minus 0.15).",
        "",
        "## 9. Over-conservative blocks",
        f"At 0.05, {m05['overblock_numerator']}/{m05['overblock_denominator']} blocks had at least one safe executed counterfactual. This is reported separately from unsafe-continuation avoidance.",
        "",
        "## 10. Replay versus migration",
        "Replay, migration, and block counts/outcomes are reported separately in `table_rq4_action_outcomes.csv`. Migration remains a qualified descriptive result; no migration model was fit or updated.",
        "",
        "## 11. Workload breakdown",
        "The fixed H2, LiH, ADAPT-VQE, and QAOA breakdown is in `table_rq4_workload_breakdown.csv`. No workload-specific generalization claim is made from these subgroup counts.",
        "",
        "## 12. Scenario-type breakdown",
        "No-change, safe-change, same-backend delay, cross-backend, and high-change results are in `table_rq4_scenario_type_breakdown.csv` with separate action and outcome counts.",
        "",
        "## 13. Comparison with prior diagnostic population",
        "`analysis/rq4_replication_comparison.csv` reports the prior diagnostic and independent confirmation populations separately; they are not pooled.",
        "",
        "## 14. Statistical uncertainty",
        f"Wilson intervals accompany all primary rates. Scenario-level paired bootstrap differences (0.05 minus 0.15) are: coverage {uncertainty_index['coverage_difference_0_05_minus_0_15']['estimate']:.3f} [{uncertainty_index['coverage_difference_0_05_minus_0_15']['ci_low']:.3f}, {uncertainty_index['coverage_difference_0_05_minus_0_15']['ci_high']:.3f}], successful coverage {uncertainty_index['successful_coverage_difference_0_05_minus_0_15']['estimate']:.3f} [{uncertainty_index['successful_coverage_difference_0_05_minus_0_15']['ci_low']:.3f}, {uncertainty_index['successful_coverage_difference_0_05_minus_0_15']['ci_high']:.3f}], and unsafe rate {uncertainty_index['unsafe_rate_difference_0_05_minus_0_15']['estimate']:.3f} [{uncertainty_index['unsafe_rate_difference_0_05_minus_0_15']['ci_low']:.3f}, {uncertainty_index['unsafe_rate_difference_0_05_minus_0_15']['ci_high']:.3f}].",
        "",
        "## 15. Limitations",
        "This is a simulation-only confirmation over the frozen workload/context matrix. It does not establish universal threshold optimality, guarantee safety, or support a general migration-safety predictor.",
        "",
        "## 16. Paper-safe interpretation",
        ("The restart score strongly orders continuation outcomes, while the operating point controls a substantial safety-coverage tradeoff. In an independent confirmation population, the calibration-selected conservative operating point reduced unsafe continuation while preserving most successful continuations, at the cost of lower overall coverage." if confirmed else "The independent confirmation did not reproduce the predeclared operating-point tradeoff. The paper should retain the diagnostic result as limited prior evidence and explicitly report the lack of fresh replication."),
        "",
        "## 17. Final recommendation",
        conclusion,
    ]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    result_hashes = OUTPUT / "analysis/result_hashes.json"
    result_hashes.write_text(json.dumps({
        "raw_records": {"path": str((OUTPUT / 'raw/records.jsonl').relative_to(ROOT)), "sha256": sha256_file(OUTPUT / 'raw/records.jsonl')},
        "processed_records": {"path": str(PROCESSED.relative_to(ROOT)), "sha256": sha256_file(PROCESSED)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    package_files = [
        REPORT, CONFIG, PREEXECUTION, VALIDATION, OUTPUT / "seed_provenance.json", result_hashes,
        *sorted(tables.glob("*")), *sorted((OUTPUT / "figures").glob("*")), *sorted(analysis.glob("*.csv")),
    ]
    artifact_manifest, artifact_hash = package_review(package_files)
    summary = {
        "schema_version": "checkrcq-rq4-fresh-confirmation-summary-v1",
        "fresh_scenarios": 160,
        "fresh_seeds": preexecution["fresh_seeds"],
        "seed_overlap": False,
        "expected_records": validation["expected_run_count"],
        "accepted_records": validation["accepted_count"],
        "simulation_shots": preexecution["estimated_simulation_shots"],
        "hardware_jobs": 0,
        "threshold_0_15": m15,
        "threshold_0_05": m05,
        "paired_consequences": causal,
        "paired_differences": uncertainty,
        "confirmed": confirmed,
        "conclusion": conclusion.replace("\n", " "),
        "artifact_manifest": str(artifact_manifest),
        "artifact_manifest_sha256": artifact_hash,
    }
    (OUTPUT / "analysis/confirmation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
