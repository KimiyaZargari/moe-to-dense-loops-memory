# MoE -> Dense -> Loop -> Memory

University lab project. Pipeline:

1. **Baseline** — load a Mixture-of-Experts model and benchmark it on reasoning and knowledge tasks. *(implemented)*
2. **Compress** — shrink the MoE model into a smaller / dense one. *(planned)*
3. **Loop** — iteratively improve the compressed model, re-evaluating each round. *(planned)*
4. **Memory** — add a memory mechanism to recover or improve quality. *(planned)*

Base model: `ibm-granite/granite-3.1-1b-a400m-base`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# sanity check: load the model and generate once
python scripts/smoke_test.py

# create a runnable config from a template (written to configs/generated/, git-ignored)
python scripts/make_config.py --list
python scripts/make_config.py knowledge --size quick --name baseline_knowledge_quick
python scripts/make_config.py reasoning --size quick --name baseline_reasoning_quick
# inspect the resolved config without loading the model
python scripts/run_eval.py --config configs/generated/baseline_knowledge_quick.yaml --dry-run

# run a benchmark
python scripts/run_eval.py --config configs/generated/baseline_knowledge_quick.yaml

# compare runs side by side (reasoning vs knowledge)
python scripts/summarize_results.py results/
```

CLI flags (`--model`, `--tasks`, `--limit`, `--device`, ...) override the YAML.
Any run with `--limit > 0` is a debugging run; its metrics are not comparable to published numbers.

## Config templates

Templates in `configs/templates/` (`knowledge.yaml`, `reasoning.yaml`) are committed and hold the task lists and defaults.
`make_config.py` fills in a size preset (`quick` / `medium` / `full`, defined in each template), the run name and any overrides
(`--model`, `--set key=value`) and writes the result to `configs/generated/`, which is git-ignored.
To change which tasks run, edit the template; to change a one-off setting, override it at generation time.

## Older hand-written configs (`configs/eval/`)

| File | Purpose |
|---|---|
| `knowledge_quick.yaml` | Smoke test: triviaqa, arc_easy, mmlu_continuation, 10 examples/task |
| `knowledge_medium.yaml` | Larger subset |
| `knowledge_full.yaml` | Full test sets, for final numbers |
| `granite_gsm8k_n500.yaml` | Reasoning: GSM8K, 500 examples |

## Layout

```
configs/
  templates/   committed config templates (knowledge, reasoning)
  generated/   configs made from templates (git-ignored)
  eval/        older hand-written configs
  compress/    compression configs        (stage 2)
  loop/        improvement-loop configs   (stage 3)
  memory/      memory configs             (stage 4)
scripts/       command-line entry points
  smoke_test.py         load model, generate once
  make_config.py        template -> configs/generated/<name>.yaml
  run_eval.py           lm_eval runner; saves to results/<run_name>_<timestamp>/
  summarize_results.py  comparison table across runs
models/        compressed/converted checkpoints (git-ignored)
results/       one folder per run: results.json, config.yaml, environment.json, samples_*.jsonl
```

## Conventions

- Every run writes a timestamped folder in `results/` with the resolved config and environment, so any number can be traced back to how it was produced.
- Use the **same task configs** for the baseline and every later stage so results stay comparable.
- Name runs by stage, e.g. `baseline_*`, `compressed_*`, `loop_r1_*`, `memory_*`.

## Next steps

- [ ] Run full reasoning + knowledge baseline and record it
- [ ] Pick a compression method and implement it
- [ ] Define the loop's improvement step and stopping criterion
- [ ] Define the memory design
