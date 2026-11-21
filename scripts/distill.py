#!/usr/bin/env python
"""Distill the original Granite MoE (teacher) into the pruned dense model (student).

Follows the recovery step of Kim et al. (arXiv 2605.28207): the dense student
produced by scripts/moe_to_dense.py is trained to match the frozen MoE teacher.

Loss = kl_weight * T^2 * KL(teacher || student)         (logit distillation)
     + ce_weight * CE(student, next token)               (hard labels)
     + hidden_weight * mean_l MSE(h_s^l, h_t^l) / Var(h_t^l)   (optional)

Hidden-state matching works because the student keeps the teacher's hidden size
and layer count, so layer l of the student maps onto layer l of the teacher.

Control run (no teacher signal, same token budget):
  python scripts/distill.py --kl-weight 0 --hidden-weight 0 --ce-weight 1 --out models/dense-ce-only

Usage:
  python scripts/distill.py --student models/granite-1b-dense-doacp-k8 \
      --tokens 100_000_000 --out models/granite-1b-dense-doacp-k8-distilled
  # pilot
  python scripts/distill.py --tokens 50_000_000 --out models/distill_pilot
  # resume after interruption (same args + --resume)
  python scripts/distill.py --tokens 100_000_000 --out ... --resume

Training setup:
  * Student weights are kept in fp32 and run under bf16 autocast; the frozen
    teacher is loaded directly in bf16 (it only does forward passes).
  * Effective batch = --micro-batch x --accum sequences of --seq-len tokens per
    optimizer step; total steps = --tokens // tokens-per-step.
  * LR schedule: linear warmup then cosine decay to --min-lr-frac x peak.
  * If kl-weight and hidden-weight are both 0 the teacher is not loaded at all
    (CE-only control run).
  * Checkpoints (train_state.pt) hold student + optimizer + step + stream position;
    the file is deleted when training completes and final weights are saved to --out.

Outputs (in --out): final HF student + tokenizer (+ selection.json copied from the
student dir), train_args.json, and train_log.jsonl (one JSON record per log step:
loss, kl, ce, hid, lr, gnorm, throughput, docs consumed).

Data: streamed FineWeb-Edu, skipping the first --skip-docs documents so the
distillation text does not overlap the calibration text used for expert
selection (calibrate_experts.py reads from the start of the same stream).
"""
import argparse
import json
import math
import os
import shutil
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(name):
    """Return ``name`` unchanged, or resolve "auto" to cuda if available, else cpu."""
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def int_arg(s):
    """argparse type: int that tolerates underscores (e.g. ``100_000_000``)."""
    return int(str(s).replace("_", ""))


# ------------------------------------------------------------------- data ---
class PackedStream:
    """Streams text, packs it into fixed-length sequences, tracks docs consumed
    so a resumed run can continue from the same place in the stream.

    Documents are tokenized, EOS-terminated, and concatenated into a buffer that is
    cut into exact ``seq_len`` sequences (no padding). Resuming re-skips
    ``docs_consumed`` documents; tokens left in the buffer at checkpoint time are
    lost, which is negligible at this scale.

    Args:
        skip_docs: number of leading documents to skip (also the initial value of
            ``docs_consumed``), used both to avoid the calibration text and to resume.
    """

    def __init__(self, tokenizer, dataset, config, split, text_field, seq_len, skip_docs):
        self.tok, self.field, self.seq_len = tokenizer, text_field, seq_len
        ds = load_dataset(dataset, config, split=split, streaming=True)
        self.docs_consumed = skip_docs
        if skip_docs:
            ds = ds.skip(skip_docs)
        self.it = iter(ds)
        self.buf = []

    def next_seq(self):
        """Return one list of ``seq_len`` token ids, pulling more documents as needed."""
        while len(self.buf) < self.seq_len:
            ex = next(self.it)
            self.docs_consumed += 1
            self.buf.extend(self.tok(ex[self.field])["input_ids"] + [self.tok.eos_token_id])
        seq, self.buf = self.buf[: self.seq_len], self.buf[self.seq_len:]
        return seq

    def next_batch(self, n):
        """Return a LongTensor [n, seq_len] of consecutive packed sequences."""
        return torch.tensor([self.next_seq() for _ in range(n)])


# ------------------------------------------------------------------- loss ---
def _token_loss(s_logits, t_logits, labels, temperature, want_kl):
    """Sum of CE and of T^2*KL(teacher || student) over a chunk of tokens (fp32).

    Args:
        s_logits, t_logits: [n, vocab] student / teacher logits (t_logits unused
            when ``want_kl`` is False).
        labels: [n] next-token targets.
        temperature: softening temperature for the KL term; CE always uses T=1.
        want_kl: skip the KL computation (returns a zero) when False.

    Returns:
        (ce_sum, kl_sum) scalar tensors, summed (not averaged) over the chunk.
        The T^2 factor keeps KL gradient magnitudes comparable across temperatures.
    """
    s_lp = F.log_softmax(s_logits.float(), dim=-1)
    ce = F.nll_loss(s_lp, labels, reduction="sum")
    if not want_kl:
        return ce, torch.zeros_like(ce)
    if temperature != 1.0:
        s_lp = F.log_softmax(s_logits.float() / temperature, dim=-1)
    t_lp = F.log_softmax(t_logits.float() / temperature, dim=-1)
    kl = (t_lp.exp() * (t_lp - s_lp)).sum() * temperature**2
    return ce, kl


def lm_losses(s_logits, t_logits, labels, temperature, want_kl, chunk=512):
    """Mean CE and KL over all tokens. Computed in checkpointed chunks so the
    [tokens, vocab] fp32 intermediates are never all alive at once.

    Args:
        s_logits: student logits [B, S, V] (already shifted to align with labels).
        t_logits: teacher logits [B, S, V], or None when KL is not used.
        labels: [B, S] target ids.
        chunk: tokens per checkpointed chunk (bounds peak memory).

    Returns:
        (mean CE, mean KL) over all B*S tokens; KL is 0.0 if ``want_kl`` is False.
    """
    s_logits = s_logits.reshape(-1, s_logits.size(-1))
    labels = labels.reshape(-1)
    t_logits = t_logits.reshape(-1, t_logits.size(-1)) if t_logits is not None else None
    ce = kl = 0.0
    for i in range(0, s_logits.size(0), chunk):
        sl = s_logits[i:i + chunk]
        tl = t_logits[i:i + chunk] if t_logits is not None else sl
        c, k = grad_checkpoint(_token_loss, sl, tl, labels[i:i + chunk], temperature, want_kl,
                               use_reentrant=False)
        ce, kl = ce + c, kl + k
    n = labels.numel()
    return ce / n, kl / n


def hidden_loss(s_hidden, t_hidden, layers):
    """MSE between hidden states, normalised by the teacher's second moment so
    layers with different scales contribute comparably.

    Args:
        s_hidden, t_hidden: tuples of [B, S, H] hidden states (index 0 = embeddings,
            index l = output of layer l).
        layers: indices into those tuples to match.

    Returns:
        Scalar: average over ``layers`` of MSE(s, t) / mean(t^2).
    """
    total = 0.0
    for l in layers:
        s, t = s_hidden[l].float(), t_hidden[l].float()
        total = total + F.mse_loss(s, t) / t.pow(2).mean().clamp(min=1e-8)
    return total / len(layers)


def lr_at(step, total, warmup, peak, floor_frac):
    """Learning rate at optimizer ``step``: linear warmup, then cosine decay.

    Rises linearly to ``peak`` over ``warmup`` steps, then follows a half cosine
    down to ``floor_frac * peak`` at step ``total``.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


# ------------------------------------------------------------------- misc ---
def save_student(student, tokenizer, out_dir, src_dir, dtype):
    """Save the final student (cast to ``dtype``) and tokenizer to ``out_dir``.

    Also copies ``selection.json`` (expert-selection provenance written by
    moe_to_dense.py) from ``src_dir`` if present. Note this casts the model
    in place, so call it only once training is finished.
    """
    os.makedirs(out_dir, exist_ok=True)
    student.to(dtype=dtype).save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    sel = os.path.join(src_dir, "selection.json")
    if os.path.exists(sel):
        shutil.copy(sel, os.path.join(out_dir, "selection.json"))


def main():
    """CLI entry point: set up models/optimizer/data, run the training loop, save."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="ibm-granite/granite-3.1-1b-a400m-base")
    p.add_argument("--student", default="models/granite-1b-dense-doacp-k8")
    p.add_argument("--out", default="models/granite-1b-dense-doacp-k8-distilled")
    # data
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--skip-docs", type=int_arg, default=20_000,
                   help="docs to skip so we don't reuse the calibration text")
    p.add_argument("--seq-len", type=int, default=1024)
    # budget / optimisation
    p.add_argument("--tokens", type=int_arg, default=100_000_000, help="total training tokens")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=16, help="grad accumulation steps")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-checkpoint", action="store_true", help="save memory, ~30%% slower")
    # loss
    p.add_argument("--kl-weight", type=float, default=0.9)
    p.add_argument("--ce-weight", type=float, default=0.1)
    p.add_argument("--hidden-weight", type=float, default=0.5, help="0 disables hidden-state matching")
    p.add_argument("--hidden-every", type=int, default=6, help="match every N-th layer (+ the last)")
    p.add_argument("--temperature", type=float, default=1.0)
    # io
    p.add_argument("--save-every", type=int, default=500, help="optimizer steps between checkpoints")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", help="autocast / saved-model dtype")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    use_teacher = args.kl_weight > 0 or args.hidden_weight > 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "train_state.pt")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    # student: fp32 master weights, bf16 autocast (bf16 weights would swallow small lr updates)
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32, attn_implementation="sdpa")
    student.to(device).train()
    student.config.use_cache = False
    if args.grad_checkpoint:
        student.gradient_checkpointing_enable()

    teacher = None
    if use_teacher:
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=dtype, attn_implementation="sdpa")
        teacher.to(device).eval().requires_grad_(False)
        assert teacher.config.vocab_size == student.config.vocab_size
        assert teacher.config.num_hidden_layers == student.config.num_hidden_layers
        assert teacher.config.hidden_size == student.config.hidden_size

    n_layers = student.config.num_hidden_layers
    # hidden_states has n_layers+1 entries (embeddings, then each layer output)
    h_layers = sorted(set(range(args.hidden_every, n_layers + 1, args.hidden_every)) | {n_layers})
    if args.hidden_weight <= 0:
        h_layers = []

    # Weight decay only on matrices; biases and norm gains (ndim < 2) are exempt.
    decay = [q for n, q in student.named_parameters() if q.ndim >= 2]
    no_decay = [q for n, q in student.named_parameters() if q.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"),
    )

    toks_per_step = args.micro_batch * args.accum * args.seq_len
    total_steps = max(1, args.tokens // toks_per_step)

    # Resume restores weights, optimizer state, step counter, and stream position.
    step, docs_consumed = 0, args.skip_docs
    if args.resume and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location="cpu")
        student.load_state_dict(st["student"])
        opt.load_state_dict(st["opt"])
        step, docs_consumed = st["step"], st["docs_consumed"]
        print(f"resumed from step {step} (docs consumed {docs_consumed})")

    stream = PackedStream(tokenizer, args.dataset, args.dataset_config, args.split,
                          args.text_field, args.seq_len, docs_consumed)

    n_params = sum(q.numel() for q in student.parameters())
    print(f"student {n_params/1e6:.1f}M params | teacher: {'yes' if use_teacher else 'none (CE-only control)'} | "
          f"{toks_per_step:,} tok/step x {total_steps} steps = {total_steps*toks_per_step/1e6:.1f}M tokens | "
          f"hidden layers {h_layers}")

    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "a")

    def checkpoint():
        """Write resumable training state (reads step/stream from the enclosing scope)."""
        torch.save({"student": student.state_dict(), "opt": opt.state_dict(),
                    "step": step, "docs_consumed": stream.docs_consumed}, ckpt_path)

    t0 = time.time()
    meter = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "hid": 0.0}
    # One iteration = one optimizer step = `accum` micro-batches.
    while step < total_steps:
        lr = lr_at(step, total_steps, args.warmup_steps, args.lr, args.min_lr_frac)
        for g in opt.param_groups:
            g["lr"] = lr

        for _ in range(args.accum):
            ids = stream.next_batch(args.micro_batch).to(device)
            with torch.autocast(device_type=device, dtype=dtype):
                s_out = student(input_ids=ids, output_hidden_states=bool(h_layers))
                if teacher is not None:
                    with torch.no_grad():
                        t_out = teacher(input_ids=ids, output_hidden_states=bool(h_layers))

            # drop the last position (no target); everything below is in fp32
            want_kl = args.kl_weight > 0
            ce, kl = lm_losses(s_out.logits[:, :-1], t_out.logits[:, :-1] if want_kl else None,
                               ids[:, 1:], args.temperature, want_kl)
            loss = args.ce_weight * ce + args.kl_weight * kl
            hid = torch.zeros((), device=device)  # stays 0 when hidden matching is off
            if h_layers:
                hid = hidden_loss(s_out.hidden_states, t_out.hidden_states, h_layers)
                loss = loss + args.hidden_weight * hid

            # Divide by accum so gradients average over the micro-batches.
            (loss / args.accum).backward()
            meter["loss"] += loss.item() / args.accum
            meter["kl"] += kl.item() / args.accum
            meter["ce"] += ce.item() / args.accum
            meter["hid"] += hid.item() / args.accum

        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        # Log averaged meters; n = steps accumulated since the last log (the final
        # log may cover fewer than log_every steps).
        if step % args.log_every == 0 or step == total_steps:
            n =args.log_every if step % args.log_every == 0 else step % args.log_every
            rec = {"step": step, "tokens": step * toks_per_step, "lr": lr, "gnorm": gnorm,
                   **{k: v / n for k, v in meter.items()},
                   "tok_per_s": n * toks_per_step / (time.time() - t0), "docs": stream.docs_consumed}
            meter = {k: 0.0 for k in meter}
            t0 = time.time()
            print(f"step {step}/{total_steps} loss {rec['loss']:.4f} kl {rec['kl']:.4f} ce {rec['ce']:.4f} "
                  f"hid {rec['hid']:.4f} lr {lr:.2e} |g| {gnorm:.2f} {rec['tok_per_s']:.0f} tok/s", flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        if step % args.save_every == 0 and step < total_steps:
            checkpoint()

    save_student(student, tokenizer, args.out, args.student, dtype)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # training finished; final weights are in --out
    print("saved", args.out, flush=True)
    log_f.close()
    # the streaming dataset's worker threads can abort during interpreter
    # teardown (core dump after a successful save); skip teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
