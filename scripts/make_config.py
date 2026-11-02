#!/usr/bin/env python3
"""
make_config.py — Create a runnable eval config from a template.

Templates live in configs/templates/ (committed). Output goes to
configs/generated/ (git-ignored), ready for scripts/run_eval.py.

Examples:
  python scripts/make_config.py knowledge --size quick
  python scripts/make_config.py reasoning --size full --name baseline_reasoning
  python scripts/make_config.py knowledge --model models/compressed_v1 --name compressed_knowledge
  python scripts/make_config.py knowledge --set device=cuda --set batch_size=16
  python scripts/make_config.py --list
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "configs" / "templates"
GENERATED = ROOT / "configs" / "generated"


def list_templates() -> list[str]:
    return sorted(p.stem for p in TEMPLATES.glob("*.yaml"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("template", nargs="?", help="Template name (see --list).")
    p.add_argument("--list", action="store_true", help="List available templates and exit.")
    p.add_argument("--size", default="quick", help="Size preset defined in the template's `sizes` (default: quick).")
    p.add_argument("--name", help="run_name and output filename (default: <template>_<size>).")
    p.add_argument("--model", help="HF model ID or local path.")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                   help="Override any top-level key; VALUE is parsed as YAML. Repeatable.")
    args = p.parse_args()

    if args.list:
        print("\n".join(list_templates()))
        return
    if not args.template:
        p.error("template is required (use --list to see options)")

    tpl_path = TEMPLATES / f"{args.template}.yaml"
    if not tpl_path.exists():
        sys.exit(f"error: no template '{args.template}'. Available: {', '.join(list_templates())}")

    cfg = yaml.safe_load(tpl_path.read_text())

    sizes = cfg.pop("sizes", {})
    if args.size not in sizes:
        sys.exit(f"error: size '{args.size}' not in template. Available: {', '.join(sizes)}")
    cfg["limit"] = sizes[args.size]

    if args.model:
        cfg["model"] = args.model
    for item in args.sets:
        if "=" not in item:
            sys.exit(f"error: --set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)

    name = args.name or f"{args.template}_{args.size}"
    cfg["run_name"] = name

    GENERATED.mkdir(parents=True, exist_ok=True)
    out = GENERATED / f"{name}.yaml"
    header = f"# Generated from configs/templates/{args.template}.yaml (size={args.size}). Do not commit.\n"
    out.write_text(header + yaml.dump(cfg, default_flow_style=False, sort_keys=False))

    rel = out.relative_to(ROOT)
    print(f"wrote {rel}")
    print(f"run:  python scripts/run_eval.py --config {rel}")


if __name__ == "__main__":
    main()
