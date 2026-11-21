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

Output file layout (``torch.save`` of a dict; L = #MoE layers, E = #experts):

  "cp"    float32 [L, E]
  "rms"   float32 [L, E]
  "freq"  float32 [L, E]
  "gram"  float32 [L, E, E]
  "meta"  dict: model name, number of tokens actually seen, dataset, seq_len

How the pieces fit together:
  1. ``build_batches`` streams text, tokenizes it, and packs it into fixed-length
     sequences (documents are joined with EOS; no padding).
  2. A ``Collector`` is attached as a forward *pre*-hook on every MoE block. It
     sees the block's input hidden states, re-runs the router, and additionally
     evaluates *all* E experts densely on those inputs. The model's own forward
     output is untouched (the hook returns None), so downstream layers see the
     exact teacher activations.
  3. After the calibration tokens are consumed, per-layer results are stacked and saved.

Usage:
  python scripts/calibrate_experts.py --tokens 1000000 --out models/granite_calib_stats.pt

Notes:
  * All accumulators live on CPU in float64 to avoid precision loss when summing
    over millions of tokens; only the per-chunk math runs on ``--device``.
  * Memory: the dense expert evaluation materializes [chunk, E, 2*I] and
    [chunk, E, H] tensors per layer. Lower ``Collector.chunk`` if you hit OOM.

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
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeDecoderLayer


def build_batches(tokenizer, dataset, config, split, text_field, n_tokens, seq_len, batch_size):
    """Stream a HF dataset and yield packed token batches.

    Documents are tokenized, terminated with EOS, and concatenated into a running
    buffer that is cut into non-overlapping sequences of exactly ``seq_len`` tokens
    (so there is no padding and no attention mask is needed).

    Args:
        tokenizer: HF tokenizer (must define ``eos_token_id``).
        dataset, config, split: arguments forwarded to ``load_dataset`` (streaming).
        text_field: name of the column holding raw text.
        n_tokens: approximate token budget. Checked once per document, so the
            total can overshoot by up to one document's worth of sequences.
        seq_len: length of each sequence.
        batch_size: sequences per yielded batch.

    Yields:
        LongTensor [batch_size, seq_len]. The final batch may contain fewer
        sequences; leftover tokens shorter than ``seq_len`` are dropped.
    """
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
    """Forward pre-hook on a GraniteMoeMoE block; accumulates stats in float64.

    One Collector is attached per MoE layer. On each forward pass it receives the
    block's input hidden states, and for every token:
      * runs the router to find which k experts are selected and their weights;
      * evaluates *all* E experts (not just the selected ones) so that the Gram
        matrix of expert outputs is defined over every calibration token.

    Running sums kept (all float64, on CPU), with E = number of experts:
      cp_sum [E]    sum of router weights over tokens routed to each expert
      count  [E]    number of tokens routed to each expert
      sq_sum [E]    sum of ||f_e(t)||^2 over tokens routed to each expert
      gram   [E,E]  sum over ALL tokens of <f_i(t), f_j(t)>
      n_tokens      total tokens processed (denominator for freq and gram)

    Args:
        moe: the ``GraniteMoeMoE`` module (``block_sparse_moe``) to observe.
        chunk: number of tokens evaluated at once. The dense evaluation of every
            expert is memory-hungry ([chunk, E, 2*I] activations), so this bounds
            peak GPU memory.
    """

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
        """Pre-hook entry point; ``args[0]`` is the hidden states [B, S, H].

        Returns None so the module's input (and thus the model's forward pass)
        is left unchanged.
        """
        x = args[0].reshape(-1, args[0].shape[-1])  # flatten to [B*S, H]
        for s in range(0, x.shape[0], self.chunk):
            self._update(x[s : s + self.chunk])

    @torch.no_grad()
    def _update(self, x):
        """Fold one chunk of tokens ``x`` [T, H] into the running statistics."""
        moe = self.moe
        _, w, idx = moe.router(x)  # logits, weights [T,k], indices [T,k]
        T = x.shape[0]
        ex = moe.experts
        # Evaluate every expert on every token. gate_up_proj packs gate then up
        # along dim 1, so split in half after the matmul.
        gu = torch.einsum("th,eoh->teo", x, ex.gate_up_proj)  # [T,E,2I]
        gate, up = gu.chunk(2, dim=-1)
        act = ex.act_fn(gate) * up  # [T,E,I]  (gated MLP hidden activation)
        out = torch.einsum("tei,ehi->teh", act, ex.down_proj).float()  # [T,E,H]

        # Dense [T,E] versions of the router's sparse top-k output:
        #   mask[t,e]  = 1 if token t was routed to expert e
        #   wfull[t,e] = router weight of that pick (0 where not routed)
        mask = torch.zeros(T, self.E, device=x.device)
        wfull = torch.zeros(T, self.E, device=x.device)
        mask.scatter_(1, idx, 1.0)
        wfull.scatter_(1, idx, w.float())
        sqnorm = (out**2).sum(-1)  # [T,E]  ||f_e(t)||^2

        # Routed-only statistics (cp, rms, freq): wfull/mask are zero off-route.
        self.cp_sum += (wfull).sum(0).double().cpu()
        self.count += mask.sum(0).double().cpu()
        self.sq_sum += (sqnorm * mask).sum(0).double().cpu()
        # Gram over ALL tokens and ALL experts: flatten each expert's outputs
        # across tokens into one row, so (a @ a.T)[i,j] = sum_t <f_i(t), f_j(t)>.
        a = out.permute(1, 0, 2).reshape(self.E, -1)  # [E, T*H]
        self.gram += (a @ a.T).double().cpu()
        self.n_tokens += T

    def result(self):
        """Normalize the running sums into the final float32 statistics.

        Returns a dict with:
          cp   [E]    mean router weight over routed tokens
          rms  [E]    sqrt(mean ||f_e||^2) over routed tokens
          freq [E]    fraction of tokens routed to each expert (sums to k)
          gram [E,E]  mean over all tokens of <f_i, f_j>

        Experts that never received a token get cp = rms = 0 (count is clamped
        to 1 to avoid division by zero).
        """
        c = self.count.clamp(min=1)
        return {
            "cp": (self.cp_sum / c).float(),
            "rms": (self.sq_sum / c).sqrt().float(),
            "freq": (self.count / max(self.n_tokens, 1)).float(),
            "gram": (self.gram / max(self.n_tokens, 1)).float(),  # mean_t <f_i,f_j>
        }


def main():
    """CLI entry point: load the teacher, hook every MoE layer, run calibration, save stats."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default="models/granite_calib_stats.pt")
    args = p.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()

    # Attach one collector per decoder layer's MoE block (index = layer index).
    collectors: list[Collector] = []
    layer: GraniteMoeDecoderLayer
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
            # Run the base model only (no LM head needed); the hooks do the work.
            model.model(input_ids=batch.to(device))
            seen += batch.numel()
            print(f"{seen:>10,} tokens  {time.time()-t0:6.0f}s", flush=True)

    # Stack per-layer results along a new leading layer dim: [L, E] / [L, E, E].
    # (result() is recomputed per key; it is cheap relative to calibration.)
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
