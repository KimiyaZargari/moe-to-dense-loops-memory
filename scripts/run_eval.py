#!/usr/bin/env python3
"""
run_eval.py — Run LM Evaluation Harness via the Python API (lm_eval.simple_evaluate).

Quick start:
  python run_eval.py --dry-run                          # see resolved config, no model load
  python run_eval.py --config configs/granite_gsm8k_n500.yaml
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml  # PyYAML — install via: pip install pyyaml

# ── device helpers ────────────────────────────────────────────────────────────


def resolve_device(device: str) -> str:
    """
    Translate 'auto' to the best backend available on this machine:
      CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    """
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def device_display_name(device: str) -> str:
    """Human-readable label for the environment record."""
    try:
        import torch

        if device.startswith("cuda"):
            idx = int(device.split(":")[1]) if ":" in device else 0
            return torch.cuda.get_device_name(idx)
        if device == "mps":
            return "Apple MPS (Metal Performance Shaders)"
    except Exception:
        pass
    return "CPU"


# ── git helpers ───────────────────────────────────────────────────────────────


def git_commit_hash(path: str | Path) -> str | None:
    """Return the short HEAD commit hash if `path` is inside a git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ── environment snapshot ──────────────────────────────────────────────────────


def capture_environment(cfg: dict) -> dict:
    """
    Record everything needed to reproduce this run later:
    Python version, key library versions, device, and git state.
    """
    env: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": cfg["device"],
        "device_name": device_display_name(cfg["device"]),
    }

    # git commit of the current working directory (useful when this script lives
    # in a research repo and you want to track which version of the code ran)
    env["repo_git_commit"] = git_commit_hash(Path.cwd())

    # git commit of the model directory — only relevant for locally converted models
    model_path = Path(cfg["model"])
    if model_path.exists() and model_path.is_dir():
        env["model_git_commit"] = git_commit_hash(model_path)

    return env


# ── config plumbing ───────────────────────────────────────────────────────────


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def normalize_tasks(raw) -> list[dict]:
    """
    Accept tasks in any of these forms and return a uniform list of dicts:

      "gsm8k"                               → [{"name": "gsm8k", "num_fewshot": None}]
      ["gsm8k", "mmlu"]                     → [{"name": "gsm8k", ...}, ...]
      [{"name": "gsm8k", "num_fewshot": 5}] → kept as-is
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    result = []
    for item in raw:
        if isinstance(item, str):
            result.append({"name": item, "num_fewshot": None})
        elif isinstance(item, dict):
            name = item.get("name") or item.get("task")
            if not name:
                raise ValueError(f"Task entry has no 'name' key: {item!r}")
            result.append({"name": name, "num_fewshot": item.get("num_fewshot"), "limit": item.get("limit")})
        else:
            raise ValueError(f"Unexpected task entry type: {item!r}")
    return result


def build_config(args: argparse.Namespace) -> dict:
    """
    Merge configuration from two sources (lowest to highest priority):
      1. YAML file          — provided via --config
      2. CLI flags          — only those the user explicitly passed (argparse
                              defaults are None so we can tell them apart)
    """
    if args.config is None:
        raise SystemExit("error: --config is required (e.g. --config configs/my_run.yaml)")
    cfg = load_yaml_config(args.config)
    if "tasks" in cfg:
        cfg["tasks"] = normalize_tasks(cfg["tasks"])

    # CLI flag overrides — skip when argparse left the default as None
    cli = vars(args)
    for key in (
        "model",
        "dtype",
        "max_length",
        "device",
        "batch_size",
        "seed",
        "output_dir",
        "run_name",
    ):
        if cli.get(key) is not None:
            cfg[key] = cli[key]

    if cli.get("tasks"):  # list of strings from CLI
        cfg["tasks"] = normalize_tasks(cli["tasks"])

    if cli.get("num_fewshot") is not None:  # override all per-task values
        for t in cfg.get("tasks", []):
            t["num_fewshot"] = cli["num_fewshot"]

    if cli.get("limit") is not None:
        # 0 is a sentinel meaning "full test set"
        cfg["limit"] = None if cli["limit"] == 0 else cli["limit"]

    if cli.get("log_samples") is not None:  # --no-log-samples sets this to False
        cfg["log_samples"] = cli["log_samples"]

    # Validate all required keys are present before proceeding
    required = ("model", "dtype", "max_length", "device", "batch_size", "seed",
                "output_dir", "run_name", "tasks", "log_samples")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(
            f"error: the following required config keys are missing: {', '.join(missing)}\n"
            "Set them in your YAML config or pass them as CLI flags."
        )

    # Translate 'auto' → best available backend
    cfg["device"] = resolve_device(cfg["device"])

    return cfg


# ── argument parser ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--config",
        metavar="PATH",
        help="YAML config file. CLI flags override values in this file.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and exit. The model is NOT loaded.",
    )

    # ── model ─────────────────────────────────────────────────────────────────
    g_model = p.add_argument_group("Model")
    g_model.add_argument(
        "--model",
        metavar="ID_OR_PATH",
        help="HuggingFace model ID (e.g. ibm-granite/granite-3.1-1b-a400m-base) "
        "or a local folder path (e.g. ./converted_models/my_model).",
    )
    g_model.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        help="Weight dtype. float32 is safest on CPU. "
        "bfloat16 is preferred on modern CUDA GPUs.",
    )
    g_model.add_argument(
        "--max-length",
        dest="max_length",
        type=int,
        help="Maximum context length in tokens (truncates longer prompts).",
    )
    g_model.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        help="PyTorch compute device. 'auto' picks cuda > mps > cpu.",
    )

    # ── tasks ─────────────────────────────────────────────────────────────────
    g_tasks = p.add_argument_group("Tasks")
    g_tasks.add_argument(
        "--tasks",
        nargs="+",
        metavar="TASK",
        help="One or more benchmark names (e.g. --tasks gsm8k mmlu). "
        "For per-task few-shot counts, set them in a YAML config instead.",
    )
    g_tasks.add_argument(
        "--num-fewshot",
        dest="num_fewshot",
        type=int,
        help="Few-shot examples to prepend. Overrides per-task values in YAML.",
    )

    # ── run ───────────────────────────────────────────────────────────────────
    g_run = p.add_argument_group("Run settings")
    g_run.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Inference batch size. Reduce if you hit out-of-memory errors.",
    )
    g_run.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N examples per task. "
        "Use 0 to run the full test set. "
        "Any value > 0 is for debugging — metrics will NOT be final.",
    )
    g_run.add_argument(
        "--seed",
        type=int,
        help="RNG seed applied to Python random, NumPy, PyTorch, and few-shot sampling.",
    )
    g_run.add_argument(
        "--no-log-samples",
        dest="log_samples",
        action="store_false",
        default=None,
        help="Disable per-sample output logging (saves disk space on large runs).",
    )

    # ── output ────────────────────────────────────────────────────────────────
    g_out = p.add_argument_group("Output")
    g_out.add_argument(
        "--output-dir",
        dest="output_dir",
        metavar="DIR",
        help="Root folder for all results.",
    )
    g_out.add_argument(
        "--run-name",
        dest="run_name",
        metavar="NAME",
        help="Prefix for the timestamped results subfolder "
        "(<output_dir>/<run_name>_<timestamp>/).",
    )

    return p


# ── output saving ─────────────────────────────────────────────────────────────


def save_outputs(run_dir: Path, results: dict, cfg: dict, env: dict) -> None:
    """Write all run artifacts to run_dir (created here if needed)."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Full results dict from simple_evaluate (metrics, config echo, git hash …)
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results.json      → {results_path}")

    # Per-sample outputs as JSONL (one line = one example)
    # Each line contains the prompt, model response, gold label, etc.
    samples = results.get("samples") or {}
    for task_name, task_samples in samples.items():
        samples_path = run_dir / f"samples_{task_name}.jsonl"
        with open(samples_path, "w") as f:
            for sample in task_samples:
                f.write(json.dumps(sample, default=str) + "\n")
        print(f"  samples_{task_name}.jsonl → {samples_path}")

    # Resolved configuration — paste this into a YAML config to re-run identically
    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  config.yaml       → {config_path}")

    # Environment record — Python/library versions and device info
    env_path = run_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"  environment.json  → {env_path}")


# ── summary table ─────────────────────────────────────────────────────────────

# 4-option multiple-choice tasks share the same random-guess baseline.
_CHANCE_BASELINE: dict[str, float] = {}
_CHANCE_BASELINE["arc_easy"] = 0.25
_CHANCE_BASELINE["arc_challenge"] = 0.25
# All mmlu_*_continuation and mmlu_* leaf tasks use 4 choices.
_MMLU_CHANCE = 0.25


def _chance_for_task(task_name: str) -> float | None:
    if task_name in _CHANCE_BASELINE:
        return _CHANCE_BASELINE[task_name]
    t = task_name.lower()
    if t.startswith("mmlu_") or t == "mmlu":
        return _MMLU_CHANCE
    return None


def print_summary(results: dict, limit: int | None) -> None:
    """Print per-task metrics with n_examples, chance baseline, and near-chance flag."""
    if limit is not None:
        print()
        print("!" * 72)
        print(f"  WARNING: --limit {limit} was used.")
        print("  These metrics are based on a subset of the test set and are")
        print("  NOT comparable to published benchmarks.")
        print("  Re-run without --limit (or with --limit 0) for final numbers.")
        print("!" * 72)

    print()
    print("── Evaluation Results ──────────────────────────────────────────────────")

    # make_table is lm_eval's own formatter — it handles all the metric/filter
    # naming conventions correctly (e.g. exact_match,strict-match).
    try:
        from lm_eval.utils import make_table

        print(make_table(results))
    except Exception as e:
        print(f"  (make_table unavailable: {e})")

    # ── Extended per-task summary ─────────────────────────────────────────────
    task_results = results.get("results", {})
    n_samples = results.get("n-samples", {})
    if not task_results:
        print("  (no results)")
        print()
        return

    print()
    print("── Extended Summary ────────────────────────────────────────────────────")
    hdr = f"  {'Task':<38} {'Metric':<34} {'Value':>8}  {'±Stderr':>8}  {'N':>6}  {'Chance':>7}  Note"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for task in sorted(task_results):
        metrics = task_results[task]
        n_info = n_samples.get(task, {})
        n_eff = n_info.get("effective", metrics.get("sample_len", "?"))
        chance = _chance_for_task(task)

        for key, value in metrics.items():
            if key in ("alias", "sample_len") or key.endswith("_stderr"):
                continue
            if not isinstance(value, float):
                continue
            stderr = metrics.get(key + "_stderr")
            v_str = f"{value:.4f}"
            s_str = f"{stderr:.4f}" if isinstance(stderr, float) else "     N/A"
            c_str = f"{chance:.4f}" if chance is not None else "    N/A"
            note = ""
            if (
                chance is not None
                and isinstance(stderr, float)
                and stderr > 0
                and abs(value - chance) <= 2 * stderr
            ):
                note = "<<CHANCE  (within 2 stderr of random-guess baseline)"
            print(
                f"  {task:<38} {key:<34} {v_str:>8}  {s_str:>8}  {n_eff!s:>6}  {c_str:>7}  {note}"
            )
    print()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Step 1: Resolve configuration ────────────────────────────────────────
    cfg = build_config(args)

    # ── Step 2: Dry run — show config and exit ────────────────────────────────
    if args.dry_run:
        print(
            "── Resolved Configuration (--dry-run) ──────────────────────────────────"
        )
        print(yaml.dump(cfg, default_flow_style=False, sort_keys=False), end="")
        print(
            "────────────────────────────────────────────────────────────────────────"
        )
        print("Dry run complete. No model was loaded.")
        return

    # ── Step 3: Build the timestamped output directory path ───────────────────
    # Using UTC to avoid ambiguity across time zones.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg["output_dir"]) / f"{cfg['run_name']}_{timestamp}"
    print(f"Run directory: {run_dir}")

    # ── Step 4: Capture environment before any heavy imports ──────────────────
    env = capture_environment(cfg)

    # ── Step 5: Build lm_eval API arguments ───────────────────────────────────
    import lm_eval  # imported here so --dry-run stays fast

    # model_args is the dict that goes to HuggingFace's AutoModelForCausalLM.
    # 'model' in lm_eval is always "hf" (the backend type); the model ID or
    # local path goes into model_args["pretrained"].
    model_args: dict = {
        "pretrained": cfg["model"],
        "dtype": cfg["dtype"],
        "max_length": cfg["max_length"],
    }

    # lm_eval's HFLM backend defaults to device_map={"": "mps"} when device=mps,
    # which tells from_pretrained to load weights directly to MPS and segfaults.
    # Passing any explicit device_map triggers the else-branch in _get_accelerate_args,
    # which sets device_map=None so the model loads to CPU, then .to("mps") is called.
    if cfg["device"] == "mps":
        model_args["device_map"] = "cpu"

    task_names = [t["name"] for t in cfg["tasks"]]
    fewshot_values = [
        t["num_fewshot"] for t in cfg["tasks"] if t["num_fewshot"] is not None
    ]
    all_same_fewshot = len(set(fewshot_values)) <= 1
    has_per_task_limits = any(t.get("limit") is not None for t in cfg["tasks"])

    # limit=None means 'evaluate the full test set' in lm_eval
    limit = cfg.get("limit")  # already normalised to None or int in build_config

    # Use simple_evaluate (fast path) only when every task shares the same
    # num_fewshot AND no task overrides the global limit.
    use_per_task_path = not all_same_fewshot or has_per_task_limits

    # ── Step 6: Run evaluation ────────────────────────────────────────────────
    print()
    print("── Starting Evaluation ─────────────────────────────────────────────────")
    print(f"  Model:      {cfg['model']}")
    print(f"  dtype:      {cfg['dtype']}   max_length: {cfg['max_length']}")
    if not use_per_task_path:
        num_fewshot = fewshot_values[0] if fewshot_values else None
        print(f"  Tasks:      {task_names}  (num_fewshot={num_fewshot})")
    else:
        per_task_str = ", ".join(
            f"{t['name']}={t['num_fewshot']}-shot"
            + (f"/limit={t['limit']}" if t.get("limit") is not None else "")
            for t in cfg["tasks"]
        )
        print(f"  Tasks:      {per_task_str}  (per-task path)")
    print(f"  Device:     {cfg['device']}")
    print(f"  Batch size: {cfg['batch_size']}   global limit: {limit}   seed: {cfg['seed']}")
    print()

    if not use_per_task_path:
        # Fast path: single simple_evaluate call.
        num_fewshot = fewshot_values[0] if fewshot_values else None
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            limit=limit,
            log_samples=cfg["log_samples"],
            random_seed=cfg["seed"],
            numpy_random_seed=cfg["seed"],
            torch_random_seed=cfg["seed"],
            fewshot_random_seed=cfg["seed"],
        )
    else:
        # Per-task path: handles mixed num_fewshot and/or per-task limits.
        # Initialises the model once, then runs one lm_eval.evaluate() call per
        # distinct effective limit, merging all results into a single dict.
        from collections import defaultdict
        from lm_eval.api.registry import get_model
        from lm_eval.evaluator import add_env_info, add_tokenizer_info, get_git_commit_hash
        from lm_eval.tasks import TaskManager

        lm = get_model("hf").create_from_arg_obj(
            model_args,
            {"batch_size": cfg["batch_size"], "max_batch_size": None, "device": cfg["device"]},
        )

        def _leaf_names(name: str, loaded: dict) -> list[str]:
            """Recursively collect leaf task names under a group or task name."""
            if name in loaded["tasks"]:
                return [name]
            return [
                leaf
                for child in loaded.get("group_map", {}).get(name, [])
                for leaf in _leaf_names(child, loaded)
            ]

        # Group task specs by their effective limit (per-task overrides global).
        limit_groups: dict = defaultdict(list)
        for t in cfg["tasks"]:
            effective = t["limit"] if t.get("limit") is not None else limit
            limit_groups[effective].append(t)

        start_date = datetime.now(timezone.utc).isoformat()
        results = None
        task_manager = TaskManager()

        for group_limit, group_specs in limit_groups.items():
            group_names = [t["name"] for t in group_specs]
            loaded = task_manager.load(group_names)

            for spec in group_specs:
                num_fs = spec.get("num_fewshot")
                if num_fs is None:
                    continue
                for leaf in _leaf_names(spec["name"], loaded):
                    loaded["tasks"][leaf].set_config("num_fewshot", num_fs)
                    loaded["tasks"][leaf].set_fewshot_seed(seed=cfg["seed"])

            group_results = lm_eval.evaluate(
                lm=lm,
                task_dict=loaded,
                limit=group_limit,
                log_samples=cfg["log_samples"],
                bootstrap_iters=100000,
            )

            if group_results is None:
                continue

            if results is None:
                results = group_results
            else:
                for key in ("results", "n-samples", "n-shot", "samples", "configs",
                            "versions", "higher_is_better", "group_subtasks"):
                    if key in group_results:
                        results.setdefault(key, {}).update(group_results[key])

        if results is not None:
            results["config"] = {
                "model": cfg["model"],
                "model_args": model_args,
                "batch_size": cfg["batch_size"],
                "batch_sizes": list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else [],
                "device": cfg["device"],
                "use_cache": None,
                "limit": limit,
                "bootstrap_iters": 100000,
                "gen_kwargs": None,
                "random_seed": cfg["seed"],
                "numpy_seed": cfg["seed"],
                "torch_seed": cfg["seed"],
                "fewshot_seed": cfg["seed"],
            }
            results["git_hash"] = get_git_commit_hash()
            results["date"] = start_date
            add_env_info(results)
            add_tokenizer_info(results, lm)

    if results is None:
        # This happens on non-rank-0 processes in distributed runs — safe to exit.
        print("No results returned (non-primary process). Exiting.")
        return

    # ── Step 7: Save all outputs ──────────────────────────────────────────────
    print()
    print(f"── Saving Outputs ──────────────────────────────────────────────────────")
    save_outputs(run_dir, results, cfg, env)

    # ── Step 8: Print summary table ───────────────────────────────────────────
    print_summary(results, limit)


if __name__ == "__main__":
    main()
