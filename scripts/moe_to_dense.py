#!/usr/bin/env python
"""Convert Granite MoE -> dense GraniteForCausalLM (Kim et al., arXiv 2605.28207).

Recipe (paper's best): per layer pick the top-k experts with DO-ACP
(K = k, pure pruning), concatenate them into one dense SwiGLU FFN
(gate/up row-concat, down column-concat, down scaled by 1/k), and copy all
non-MoE weights unchanged. Distillation (scripts/distill.py) follows.

Usage:
  python scripts/moe_to_dense.py --stats models/granite_calib_stats.pt \
      --out models/granite-1b-dense-doacp-k8
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, GraniteConfig, GraniteForCausalLM

MOE_KEYS = {
    "num_local_experts", "num_experts_per_tok", "output_router_logits",
    "router_aux_loss_coef", "model_type", "architectures",
}


# ---------------------------------------------------------------- scoring ---
def acp_scores(stats, layer):
    return stats["cp"][layer] * stats["rms"][layer]


def cp_scores(stats, layer):
    return stats["cp"][layer]


def freq_scores(stats, layer):
    return stats["freq"][layer]


def d_optimal_select(importance, gram, k, reg=1e-3):
    """Greedy maximisation of logdet(K_S + reg*I), K_ij = sqrt(I_i I_j) G_ij.

    The objective is monotone submodular, so greedy has a (1-1/e) guarantee.
    G is normalised by its mean diagonal so `reg` is scale-free.
    """
    imp = importance.double().clamp(min=1e-12)
    g = gram.double()
    g = g / g.diagonal().mean()
    s = imp.sqrt()
    K = s[:, None] * g * s[None, :]
    K = K / K.diagonal().mean()
    chosen = []
    for _ in range(k):
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
    d = {key: v for key, v in moe_cfg.to_dict().items() if key not in MOE_KEYS}
    d["intermediate_size"] = k * moe_cfg.intermediate_size
    d["mlp_bias"] = False
    return GraniteConfig(**d)


@torch.no_grad()
def convert(moe, selection, k):
    """Build the dense student from the teacher and per-layer expert selection."""
    cfg = dense_config(moe.config, k)
    dense = GraniteForCausalLM(cfg).to(dtype=next(moe.parameters()).dtype)
    I = moe.config.intermediate_size
    alpha = 1.0 / k  # uniform down-projection scaling (paper's default)

    # non-MoE parameters: copy by name
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
    """Dense MLP must equal the uniform-weight (1/k) sum of the chosen experts."""
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
    p = argparse.ArgumentParser()
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
    check_layer(moe, dense, selection, 0)

    os.makedirs(args.out, exist_ok=True)
    dense.save_pretrained(args.out)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    with open(os.path.join(args.out, "selection.json"), "w") as f:
        json.dump({"method": args.method, "k": k, "selection": selection}, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
