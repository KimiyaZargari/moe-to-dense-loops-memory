#!/usr/bin/env python
"""Collect per-expert calibration statistics for MoE-to-dense conversion.

Follows Kim et al., "Pruning and Distilling Mixture-of-Experts into Dense
Language Models" (arXiv 2605.28207). For every MoE layer we gather, from a
forward pass of the *unmodified* teacher over calibration text:

  cp[l, e]    conditional probability: mean router weight of expert e over the
              tokens that routed to it (Granite's router is a softmax over the
              top-k logits only, so weights sum to 1 across the k picks)
  rms[l, e]   sqrt(mean ||f_e(t)||^2) over the tokens routed to e
  gram[l,e,f] G_ij = mean_t <f_i(t), f_j(t)> over ALL calibration tokens, where
              f_e(t) is expert e's output for token t (computed for every
              expert, whether or not the router picked it)
  freq[l, e]  selection frequency (for reference / ablations)

Output: a torch file consumed by scripts/moe_to_dense.py.

Architecture facts (transformers 5.x GraniteMoe, verified):
  layer.block_sparse_moe.router.weight        [E, H]
  layer.block_sparse_moe.experts.gate_up_proj [E, 2*I, H]  (gate first, then up)
  layer.block_sparse_moe.experts.down_proj    [E, H, I]
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_batches(tokenizer, dataset, config, split, text_field, n_tokens, seq_len, batch_size):
    ds = load_dataset(dataset, config, split=split, streaming=True)
    buf, seqs, total = [], [], 0
    for ex in ds:
        buf.extend(tokenizer(ex[text_field])["input_ids"] + [tokenizer.eos_token_id])
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len])
            buf = buf[seq_len:]
            total += seq_len
            if len(seqs) == batch_size:
                yield torch.tensor(seqs)
                seqs = []
        if total >= n_tokens:
            break
    if seqs:
        yield torch.tensor(seqs)


class Collector:
    """Forward pre-hook on a GraniteMoeMoE block; accumulates stats in float64."""

    def __init__(self, moe, chunk=2048):
        self.moe = moe
        self.chunk = chunk
        e = moe.router.num_experts
        self.E, self.k = e, moe.router.top_k
        self.n_tokens = 0
        self.cp_sum = torch.zeros(e, dtype=torch.float64)
        self.sq_sum = torch.zeros(e, dtype=torch.float64)
        self.count = torch.zeros(e, dtype=torch.float64)
        self.gram = torch.zeros(e, e, dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, module, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        for s in range(0, x.shape[0], self.chunk):
            self._update(x[s : s + self.chunk])

    @torch.no_grad()
    def _update(self, x):
        moe = self.moe
        _, w, idx = moe.router(x)  # logits, weights [T,k], indices [T,k]
        T = x.shape[0]
        ex = moe.experts
        gu = torch.einsum("th,eoh->teo", x, ex.gate_up_proj)  # [T,E,2I]
        gate, up = gu.chunk(2, dim=-1)
        act = ex.act_fn(gate) * up  # [T,E,I]
        out = torch.einsum("tei,ehi->teh", act, ex.down_proj).float()  # [T,E,H]

        # routed mask / router weight matrix [T,E]
        mask = torch.zeros(T, self.E, device=x.device)
        wfull = torch.zeros(T, self.E, device=x.device)
        mask.scatter_(1, idx, 1.0)
        wfull.scatter_(1, idx, w.float())
        sqnorm = (out**2).sum(-1)  # [T,E]

        self.cp_sum += (wfull).sum(0).double().cpu()
        self.count += mask.sum(0).double().cpu()
        self.sq_sum += (sqnorm * mask).sum(0).double().cpu()
        a = out.permute(1, 0, 2).reshape(self.E, -1)  # [E, T*H]
        self.gram += (a @ a.T).double().cpu()
        self.n_tokens += T

    def result(self):
        c = self.count.clamp(min=1)
        return {
            "cp": (self.cp_sum / c).float(),
            "rms": (self.sq_sum / c).sqrt().float(),
            "freq": (self.count / max(self.n_tokens, 1)).float(),
            "gram": (self.gram / max(self.n_tokens, 1)).float(),  # mean_t <f_i,f_j>
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default="models/granite_calib_stats.pt")
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    # MPS segfaults when loading directly (see run_eval.py); load on CPU then move.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cpu")
    model.to(device).eval()

    collectors = []
    for layer in model.model.layers:
        c = Collector(layer.block_sparse_moe)
        layer.block_sparse_moe.register_forward_pre_hook(c)
        collectors.append(c)

    t0, seen = time.time(), 0
    with torch.no_grad():
        for batch in build_batches(
            tok, args.dataset, args.dataset_config, args.split, args.text_field,
            args.tokens, args.seq_len, args.batch_size,
        ):
            model.model(input_ids=batch.to(device))
            seen += batch.numel()
            print(f"{seen:>10,} tokens  {time.time()-t0:6.0f}s", flush=True)

    stats = {k: torch.stack([c.result()[k] for c in collectors]) for k in ("cp", "rms", "freq", "gram")}
    stats["meta"] = {"model": args.model, "tokens": seen, "dataset": args.dataset, "seq_len": args.seq_len}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(stats, args.out)
    print(f"saved {args.out}: cp {tuple(stats['cp'].shape)}, gram {tuple(stats['gram'].shape)}", flush=True)
    # Skip interpreter teardown: the streaming dataset's background thread otherwise
    # races with shutdown and aborts with a PyGILState_Release error (core dump).
    os._exit(0)


if __name__ == "__main__":
    main()
