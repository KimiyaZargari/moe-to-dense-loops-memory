#!/usr/bin/env python
"""Convert Granite MoE -> dense GraniteForCausalLM (Kim et al., arXiv 2605.28207).

Recipe (paper's best): per layer pick the top-k experts with DO-ACP
(K = k, pure pruning), concatenate them into one dense SwiGLU FFN
(gate/up row-concat, down column-concat, down scaled by 1/k), and copy all
non-MoE weights unchanged. Distillation (scripts/distill.py) follows.

Why the conversion is (nearly) function-preserving per expert:
  A MoE layer computes sum_e w_e * Down_e(act(Gate_e x) * Up_e x). Stacking the
  chosen experts' gate/up rows gives a hidden vector that is the concatenation of
  each expert's hidden activation; multiplying by the column-concatenated down
  matrices then sums the experts' outputs. Scaling down by alpha = 1/k turns that
  sum into a uniform average, i.e. the MoE output with all router weights = 1/k.
  The router (and its input-dependent weights) is dropped, so the dense model
  only approximates the teacher; distillation recovers the gap.

Inputs:  calibration stats from scripts/calibrate_experts.py (cp, rms, freq, gram).
Outputs (in --out): dense HF checkpoint, tokenizer, and selection.json
         ({"method", "k", "selection": [[expert ids] per layer]}).

Methods (--method):
  acp     top-k by ACP score = cp * rms            (importance only)
  cp      top-k by conditional probability
  sf      top-k by selection frequency
  do-acp  D-optimal selection with ACP importance   (importance + diversity; default)
  do-cp   D-optimal selection with cp importance

Usage:
  python scripts/moe_to_dense.py --stats models/granite_calib_stats.pt \
      --out models/granite-1b-dense-doacp-k8
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, GraniteConfig, GraniteForCausalLM

# GraniteMoe config fields that must not be passed to the dense GraniteConfig.
MOE_KEYS = {
    "num_local_experts", "num_experts_per_tok", "output_router_logits",
    "router_aux_loss_coef", "model_type", "architectures",
}


# ---------------------------------------------------------------- scoring ---
def acp_scores(stats, layer):
    """ACP importance [E] for one layer: conditional probability x output RMS.

    Combines how strongly the router weights an expert when it is picked (cp)
    with how large its output is (rms), approximating its contribution magnitude.
    """
    return stats["cp"][layer] * stats["rms"][layer]


def cp_scores(stats, layer):
    """Conditional-probability importance [E] for one layer."""
    return stats["cp"][layer]


def freq_scores(stats, layer):
    """Selection-frequency importance [E] for one layer."""
    return stats["freq"][layer]


def d_optimal_select(importance, gram, k, reg=1e-3):
    """Greedy maximisation of logdet(K_S + reg*I), K_ij = sqrt(I_i I_j) G_ij.

    The objective is monotone submodular, so greedy has a (1-1/e) guarantee.
    G is normalised by its mean diagonal so `reg` is scale-free.

    Intuition: logdet of a Gram-like matrix grows with the volume spanned by the
    chosen experts' outputs, so it rewards experts that are both important
    (large I) and non-redundant (low correlation in G) with those already chosen.

    Args:
        importance: [E] per-expert importance (e.g. ACP or cp scores).
        gram: [E, E] mean output inner products from calibration.
        k: number of experts to select.
        reg: ridge term keeping each sub-matrix positive definite.

    Returns:
        List of k expert indices, in the order they were picked.
    """
    imp = importance.double().clamp(min=1e-12)  # avoid sqrt(0) for dead experts
    g = gram.double()
    g = g / g.diagonal().mean()
    s = imp.sqrt()
    K = s[:, None] * g * s[None, :]  # K_ij = sqrt(I_i I_j) G_ij
    K = K / K.diagonal().mean()
    chosen = []
    for _ in range(k):
        # Greedy step: add the expert that increases logdet the most.
        best, best_val = None, -float("inf")
        for e in range(K.shape[0]):
            if e in chosen:
                continue
            idx = chosen + [e]
            sub = K[idx][:, idx] + reg * torch.eye(len(idx), dtype=K.dtype)
            val = torch.linalg.slogdet(sub)[1].item()
            if val > best_val:
                best, best_val = e, val
        chosen.append(best)
    return chosen


def select_experts(stats, k, method="do-acp", reg=1e-3):
    """Choose k experts per layer.

    Args:
        stats: dict loaded from calibrate_experts.py output.
        k: experts to keep per layer.
        method: "do-acp" | "do-cp" (greedy D-optimal, see ``d_optimal_select``) or
            "acp" | "cp" | "sf" (plain top-k by the corresponding score).
        reg: ridge term for the D-optimal methods (ignored otherwise).

    Returns:
        List (one entry per layer) of lists of k selected expert indices.
    """
    n_layers = stats["cp"].shape[0]
    sel = []
    for l in range(n_layers):
        if method == "do-acp":
            sel.append(d_optimal_select(acp_scores(stats, l), stats["gram"][l], k, reg))
        elif method == "do-cp":
            sel.append(d_optimal_select(cp_scores(stats, l), stats["gram"][l], k, reg))
        else:
            score = {"acp": acp_scores, "cp": cp_scores, "sf": freq_scores}[method](stats, l)
            sel.append(score.topk(k).indices.tolist())
    return sel


# ------------------------------------------------------------- conversion ---
def dense_config(moe_cfg, k):
    """Derive a dense GraniteConfig from the MoE config.

    Drops MoE-only fields and sets the FFN width to k * (per-expert width), since
    the k selected experts are concatenated into a single MLP.
    """
    d = {key: v for key, v in moe_cfg.to_dict().items() if key not in MOE_KEYS}
    d["intermediate_size"] = k * moe_cfg.intermediate_size
    d["mlp_bias"] = False
    return GraniteConfig(**d)


@torch.no_grad()
def convert(moe, selection, k):
    """Build the dense student from the teacher and per-layer expert selection.

    Args:
        moe: the loaded GraniteMoeForCausalLM teacher.
        selection: per-layer lists of expert indices (from ``select_experts``).
        k: experts per layer; sets the dense intermediate size and the 1/k scale.

    Returns:
        (dense GraniteForCausalLM, number of non-MLP tensors copied from the teacher).
    """
    cfg = dense_config(moe.config, k)
    dense = GraniteForCausalLM(cfg).to(dtype=next(moe.parameters()).dtype)
    I = moe.config.intermediate_size
    alpha = 1.0 / k  # uniform down-projection scaling (paper's default)

    # non-MoE parameters (embeddings, attention, norms, lm_head): copy by name.
    # Dense MLP weights (".mlp.") are filled per layer below.
    moe_sd = moe.state_dict()
    dense_sd = dense.state_dict()
    copied = 0
    for name in dense_sd:
        if ".mlp." in name:
            continue
        dense_sd[name].copy_(moe_sd[name])
        copied += 1

    for l, experts in enumerate(selection):
        ex = moe.model.layers[l].block_sparse_moe.experts
        # gate_up_proj[e] is [2I, H]: rows [:I] are gate, rows [I:] are up.
        gate = torch.cat([ex.gate_up_proj[e, :I] for e in experts], dim=0)  # [kI, H]
        up = torch.cat([ex.gate_up_proj[e, I:] for e in experts], dim=0)  # [kI, H]
        down = torch.cat([ex.down_proj[e] * alpha for e in experts], dim=1)  # [H, kI]
        mlp = dense.model.layers[l].mlp
        mlp.gate_proj.weight.copy_(gate)
        mlp.up_proj.weight.copy_(up)
        mlp.down_proj.weight.copy_(down)
    dense.tie_weights()
    return dense, copied


@torch.no_grad()
def check_layer(moe, dense, selection, layer=0):
    """Sanity check: dense MLP must equal the uniform-weight (1/k) sum of the chosen experts.

    Feeds random inputs through the dense MLP of ``layer`` and through the selected
    teacher experts (averaged), and prints/returns the max absolute difference.
    Expect ~0 up to dtype rounding; a large value indicates a conversion bug.
    """
    x = torch.randn(16, moe.config.hidden_size, dtype=next(moe.parameters()).dtype)
    ex = moe.model.layers[layer].block_sparse_moe.experts
    I = moe.config.intermediate_size
    ref = 0
    for e in selection[layer]:
        g, u = torch.nn.functional.linear(x, ex.gate_up_proj[e]).chunk(2, dim=-1)
        ref = ref + torch.nn.functional.linear(ex.act_fn(g) * u, ex.down_proj[e]) / len(selection[layer])
    out = dense.model.layers[layer].mlp(x)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"layer {layer} max abs diff dense vs. 1/k expert sum: {err:.3e}")
    return err


def main():
    """CLI: load teacher + stats, select experts, convert, verify, and save."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    p.add_argument("--stats", default="models/granite_calib_stats.pt")
    p.add_argument("--method", default="do-acp", choices=["do-acp", "do-cp", "acp", "cp", "sf"])
    p.add_argument("--k", type=int, default=None, help="experts kept per layer (default: top-k of model)")
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default="models/granite-1b-dense-doacp-k8")
    args = p.parse_args()

    moe = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="cpu")
    k = args.k or moe.config.num_experts_per_tok
    stats = torch.load(args.stats)
    selection = select_experts(stats, k, args.method, args.reg)
    dense, copied = convert(moe, selection, k)

    n_moe = sum(p.numel() for p in moe.parameters())
    n_dense = sum(p.numel() for p in dense.parameters())
    print(f"params: MoE {n_moe/1e6:.1f}M -> dense {n_dense/1e6:.1f}M (copied {copied} non-MLP tensors)")
    check_layer(moe, dense, selection, 0)  # quick correctness check on layer 0

    os.makedirs(args.out, exist_ok=True)
    dense.save_pretrained(args.out)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump({"method": args.method, "k": k, "selection": selection}, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
