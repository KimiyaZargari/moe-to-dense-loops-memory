#!/usr/bin/env python3
"""
summarize_results.py — Compare multiple evaluation runs in one table.

Usage:
  python summarize_results.py results/run_A results/run_B ...
  python summarize_results.py results/          # scan a directory for run folders
  python summarize_results.py --help

The script reads results.json (and optionally config.yaml) from each folder,
groups tasks under a REASONING or KNOWLEDGE section, and prints a comparison
table with one column per run.

Tasks are categorised as:
  REASONING  gsm8k, math, bbh, hellaswag, winogrande, ...
  KNOWLEDGE  triviaqa, arc_easy, arc_challenge, mmlu_*, ...
  OTHER      anything that does not match either category

MMLU per-subject rows are indented under the mmlu aggregate row. A
"(subject mean)" row gives the unweighted mean accuracy over the subjects
(n = number of subjects); the aggregate row is lm_eval's sample-weighted score.
The table is also written to results/summary.txt (override with --output); that
file is git-ignored by the `results/*` rule.

By default only the MMLU summary rows are shown; pass --show_all_mmlu to also
show the sub-group (stem/other/...) and per-subject rows.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "summary.txt"

# ── Task categorisation ────────────────────────────────────────────────────────

_REASONING_PREFIXES = (
    "gsm8k",
    "math",
    "bbh",
    "hellaswag",
    "winogrande",
    "piqa",
    "siqa",
    "openbookqa",
    "boolq",
)

_KNOWLEDGE_PREFIXES = (
    "triviaqa",
    "arc_easy",
    "arc_challenge",
    "mmlu",
    "nq_open",
    "naturalqs",
    "web_questions",
)

# Metrics to display (in this priority order); others are skipped in the brief table.
_PRIMARY_METRICS = (
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match",
    "acc_norm",
    "acc",
)

# Random-guess baseline for 4-option MCQ tasks.
_CHANCE: dict[str, float] = {}
for _pfx in ("mmlu", "arc_easy", "arc_challenge"):
    _CHANCE[_pfx] = 0.25


def _category(task: str) -> str:
    t = task.lower()
    for p in _REASONING_PREFIXES:
        if t.startswith(p):
            return "REASONING"
    for p in _KNOWLEDGE_PREFIXES:
        if t.startswith(p):
            return "KNOWLEDGE"
    return "OTHER"


def _chance_for(task: str) -> float | None:
    t = task.lower()
    for pfx, val in _CHANCE.items():
        if t.startswith(pfx):
            return val
    return None


def _is_mmlu_subject(task: str) -> bool:
    """True for leaf MMLU tasks (e.g. mmlu_abstract_algebra_continuation)."""
    t = task.lower()
    return t.startswith("mmlu_") and "::" not in t and t not in (
        "mmlu_continuation",
        "mmlu_stem",
        "mmlu_other",
        "mmlu_humanities",
        "mmlu_social_sciences",
        "mmlu_stem_continuation",
        "mmlu_other_continuation",
        "mmlu_humanities_continuation",
        "mmlu_social_sciences_continuation",
    )


def _is_mmlu_subgroup(task: str) -> bool:
    """True for MMLU sub-category aggregates (stem / other / etc.)."""
    t = task.lower()
    return t.startswith("mmlu_continuation::") or t in (
        "mmlu_stem",
        "mmlu_other",
        "mmlu_humanities",
        "mmlu_social_sciences",
        "mmlu_stem_continuation",
        "mmlu_other_continuation",
        "mmlu_humanities_continuation",
        "mmlu_social_sciences_continuation",
    )


# ── Results loading ────────────────────────────────────────────────────────────


def _find_results_json(path: Path) -> Path | None:
    candidate = path / "results.json"
    if candidate.exists():
        return candidate
    # Also accept lm_eval's nested output (model_name/results_timestamp.json)
    for p in sorted(path.rglob("results*.json")):
        return p
    return None


def load_run(path: Path) -> dict:
    """Load a single run directory. Returns a dict with keys: label, tasks."""
    results_file = _find_results_json(path)
    if results_file is None:
        print(f"WARNING: no results.json found under {path}", file=sys.stderr)
        return {}

    with open(results_file) as f:
        raw = json.load(f)

    # Derive a short label from config or directory name.
    config_file = path / "config.yaml"
    label = path.name
    if config_file.exists():
        try:
            import yaml

            cfg = yaml.safe_load(config_file.read_text()) or {}
            model_short = Path(cfg.get("model", "")).name or cfg.get("model", "")
            run_name = cfg.get("run_name", "")
            if model_short:
                label = f"{model_short}\n({run_name})" if run_name else model_short
        except Exception:
            pass

    return {"label": label, "raw": raw, "path": path}


# ── Table rendering ────────────────────────────────────────────────────────────


def _primary_metric(metrics: dict) -> tuple[str, float, float | None] | None:
    """Pick the most informative metric from a task result dict."""
    for name in _PRIMARY_METRICS:
        if name in metrics and isinstance(metrics[name], float):
            stderr = metrics.get(name + "_stderr")
            if not isinstance(stderr, float):
                stderr = None
            return name, metrics[name], stderr
    # Fall back to first numeric metric.
    for key, val in metrics.items():
        if key in ("alias", "sample_len") or key.endswith("_stderr"):
            continue
        if isinstance(val, float):
            stderr = metrics.get(key + "_stderr")
            if not isinstance(stderr, float):
                stderr = None
            return key, val, stderr
    return None


def _all_metrics(metrics: dict) -> list[tuple[str, float, float | None]]:
    """Return all reportable (metric, value, stderr) tuples."""
    out = []
    for key, val in metrics.items():
        if key in ("alias", "sample_len") or key.endswith("_stderr"):
            continue
        if not isinstance(val, float):
            continue
        stderr = metrics.get(key + "_stderr")
        if not isinstance(stderr, float):
            stderr = None
        out.append((key, val, stderr))
    return out


def _fmt_cell(value: float, stderr: float | None, chance: float | None) -> str:
    v = f"{value:.4f}"
    s = f"±{stderr:.4f}" if stderr is not None else ""
    flag = ""
    if chance is not None and stderr is not None and stderr > 0:
        if abs(value - chance) <= 2 * stderr:
            flag = "*"
    return f"{v}{flag:1s} {s:<8}".rstrip()


def _n_eff(run: dict, task: str) -> str:
    n_samples = run.get("raw", {}).get("n-samples", {}).get(task, {})
    eff = n_samples.get("effective")
    if eff is not None:
        return str(eff)
    # fallback: sample_len in task metrics
    m = run.get("raw", {}).get("results", {}).get(task, {})
    sl = m.get("sample_len")
    return str(sl) if sl is not None else "?"


def _mmlu_subject_mean(run: dict) -> tuple[float, int] | None:
    """Unweighted mean accuracy over the MMLU subject tasks in a run, and the subject count."""
    accs = [
        m["acc,none"]
        for t, m in run.get("raw", {}).get("results", {}).items()
        if _is_mmlu_subject(t) and isinstance(m.get("acc,none"), float)
    ]
    return (sum(accs) / len(accs), len(accs)) if accs else None


def print_table(runs: list[dict], show_all_mmlu: bool = False) -> None:
    if not runs:
        print("No runs to compare.")
        return

    labels = [r["label"] for r in runs]
    col_w = 18  # width of each model column

    # Collect all task names present in any run.
    all_tasks: set[str] = set()
    for r in runs:
        all_tasks.update(r.get("raw", {}).get("results", {}).keys())

    # Partition and sort tasks.
    def sort_key(t: str) -> tuple[int, str]:
        cat = _category(t)
        order = {"REASONING": 0, "KNOWLEDGE": 1, "OTHER": 2}
        return (order[cat], t)

    sorted_tasks = sorted(all_tasks, key=sort_key)

    # Build rows: (indent, task, metric, [(value, stderr)])
    Row = tuple  # (indent: int, task: str, metric_label: str, cells: list)
    rows: list[Row] = []
    current_cat = None

    for task in sorted_tasks:
        cat = _category(task)
        is_mmlu_subgroup = _is_mmlu_subgroup(task)
        is_subject = _is_mmlu_subject(task)

        if not show_all_mmlu and (is_subject or is_mmlu_subgroup):
            continue

        if cat != current_cat:
            rows.append(("HEADER", cat))
            current_cat = cat

        cells = []
        for r in runs:
            task_metrics = r.get("raw", {}).get("results", {}).get(task)
            cells.append(task_metrics)

        # Decide which metrics to show: all metrics for the task.
        metric_names: list[str] = []
        for task_metrics in cells:
            if task_metrics:
                for key, val in task_metrics.items():
                    if key not in ("alias", "sample_len") and not key.endswith("_stderr") and isinstance(val, float):
                        if key not in metric_names:
                            metric_names.append(key)

        if not metric_names:
            continue

        chance = _chance_for(task)
        indent = 2 if is_subject else (1 if is_mmlu_subgroup else 0)
        display_task = ("  " * indent) + task

        for metric in metric_names:
            row_cells = []
            for task_metrics in cells:
                if task_metrics and metric in task_metrics and isinstance(task_metrics[metric], float):
                    val = task_metrics[metric]
                    stderr = task_metrics.get(metric + "_stderr")
                    if not isinstance(stderr, float):
                        stderr = None
                    row_cells.append(_fmt_cell(val, stderr, chance))
                else:
                    row_cells.append("  —  ")
            rows.append(("DATA", display_task, metric, row_cells))

        if task in ("mmlu", "mmlu_continuation"):
            means = [_mmlu_subject_mean(r) for r in runs]
            if any(means):
                cells_avg = [f"{m[0]:.4f} (n={m[1]})" if m else "  —  " for m in means]
                rows.append(("DATA", f"{task} (subject mean)", "acc,none", cells_avg))

    # Compute column widths.
    task_col_w = max((len(r[1]) for r in rows if r[0] == "DATA"), default=30)
    metric_col_w = max((len(r[2]) for r in rows if r[0] == "DATA"), default=30)
    cell_w = max(col_w, *(len(c) for r in rows if r[0] == "DATA" for c in r[3]), 18)

    sep = "─" * (task_col_w + metric_col_w + (cell_w + 3) * len(runs) + 6)

    print()
    print("── Benchmark Comparison ────────────────────────────────────────────────")
    print()
    # Header row.
    header = f"  {'Task':<{task_col_w}}  {'Metric':<{metric_col_w}}"
    for lbl in labels:
        # Replace newlines for single-line header.
        short = lbl.replace("\n", " / ")
        header += f"  {short:<{cell_w}}"
    print(header)
    print("  " + sep)

    for row in rows:
        if row[0] == "HEADER":
            print()
            print(f"  ── {row[1]} {'─' * (len(sep) - len(row[1]) - 4)}")
        else:
            _, task_label, metric, cells = row
            line = f"  {task_label:<{task_col_w}}  {metric:<{metric_col_w}}"
            for cell in cells:
                line += f"  {cell:<{cell_w}}"
            print(line)

    print()
    if any(_chance_for(t) is not None for t in all_tasks):
        print("  * = result within 2 stderr of the random-guess baseline")
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "paths",
        nargs="+",
        metavar="DIR",
        help="One or more run directories (each must contain results.json). "
        "If a single directory is given and it does not contain results.json "
        "directly, its subdirectories are treated as individual run folders.",
    )
    p.add_argument(
        "--show_all_mmlu",
        action="store_true",
        help="Show MMLU sub-group and per-subject rows; by default only the summary rows are shown.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="File the table is also written to (default: results/summary.txt, git-ignored).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    paths: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            print(f"ERROR: path does not exist: {p}", file=sys.stderr)
            sys.exit(1)
        # If this is a directory without a results.json, treat its children
        # as individual run folders (scan mode).
        if p.is_dir() and not (p / "results.json").exists() and not list(p.rglob("results*.json")):
            children = sorted(p.iterdir())
            paths.extend(c for c in children if c.is_dir())
        else:
            paths.append(p)

    runs = [load_run(p) for p in paths]
    runs = [r for r in runs if r]  # drop failed loads

    if not runs:
        print("No valid run directories found.", file=sys.stderr)
        sys.exit(1)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_table(runs, show_all_mmlu=args.show_all_mmlu)
    table = buf.getvalue()
    print(table, end="")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"  saved to {args.output}")


if __name__ == "__main__":
    main()
