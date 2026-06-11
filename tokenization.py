"""Experiment 2 — A SHARED knowledge base is what integrates observations.

N agents observe the SAME concept, each through an independent noisy channel.
We compare three ways of integrating their observations, scored as cosine
alignment of the integrated estimate to the true concept direction:

  1. Raw consensus            -- average the raw belief vectors (no tokens).
  2. Stigmergy w/o shared KB  -- every agent DOES tokenize, but each has its own
       PRIVATE knowledge base, so the same token index means a different
       direction for each agent. The field pools token indices and exposes a
       dominant one, but every agent decodes it through its own KB.
  3. Stigmergy w/ shared KB   -- all agents tokenize through the SAME knowledge
       base, so a token is a common symbol; the field's dominant token decodes
       to one concept for everyone.

The point: tokenization only integrates if the knowledge base is SHARED.
- Raw consensus works at low noise but its centroid degrades as noise grows.
- Private-KB tokenization fails at ALL noise levels (even sigma=0): the symbols
  carry no common meaning, so the dominant token decodes to a random direction
  for each agent -> alignment ~ chance.
- Shared-KB tokenization grounds every agent's heterogeneous observation in one
  common symbol set, so they integrate into a single correct concept, robustly.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import SharedKnowledgeBase

# ── Parameters ──────────────────────────────────────────────
DIM = 512
M = 10            # knowledge-base size (CIFAR-10 categories)
N = 50            # agents, all observing the SAME concept
SIGMAS = np.linspace(0.0, 2.5, 26)   # observation-noise / heterogeneity sweep
RUNS = 250        # random (concept, noise, private-KB) draws per sigma
SEED = 0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial:bold",
    "mathtext.it": "Arial:italic:bold",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "bf",
    "font.size": 9,
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
})

KB = SharedKnowledgeBase(num_tokens=M, dim=DIM, seed=0)


def _dominant(tokens: torch.Tensor) -> int:
    return int(torch.bincount(tokens, minlength=M).argmax())


def trial(sigma: float, gen: torch.Generator):
    """One draw: a concept, N heterogeneous noisy observations, and the three
    integrated estimates' alignment to the true concept direction."""
    c = int(torch.randint(0, M, (1,), generator=gen))
    truth = KB.E[c]
    obs = F.normalize(truth.unsqueeze(0) + sigma * torch.randn(N, DIM, generator=gen), dim=1)

    # 1) raw consensus: average the raw belief vectors
    cons = (F.normalize(obs.mean(dim=0), dim=0) @ truth).item()

    # 2) stigmergy w/o shared KB: each agent has its OWN private knowledge base
    priv = F.normalize(torch.randn(N, M, DIM, generator=gen), dim=2)   # (N, M, dim)
    t_priv = torch.einsum("nd,nmd->nm", obs, priv).argmax(dim=1)       # each agent's own token
    dom_p = _dominant(t_priv)
    # every agent decodes the dominant token index through its OWN KB
    nokb = (F.normalize(priv[:, dom_p, :], dim=1) @ truth).mean().item()

    # 3) stigmergy w/ shared KB: all agents tokenize through the same KB
    t_sh = KB.tokenize_batch(obs)
    shared = (KB.E[_dominant(t_sh)] @ truth).item()

    return shared, cons, nokb


def main():
    gen = torch.Generator().manual_seed(SEED)
    shared, cons, nokb = [], [], []
    for s in SIGMAS:
        r = np.array([trial(s, gen) for _ in range(RUNS)])
        shared.append(r[:, 0].mean())
        cons.append(r[:, 1].mean())
        nokb.append(r[:, 2].mean())
        print(f"sigma={s:4.2f} | shared KB={shared[-1]:.3f} | consensus={cons[-1]:.3f} | no-share KB={nokb[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(SIGMAS, shared, "o-", color="tab:red", linewidth=2.4, markersize=4.5,
            label="Stigmergy w/ shared KB")
    ax.plot(SIGMAS, cons, "s-", color="tab:blue", linewidth=2.2, markersize=4.5,
            label="Raw consensus")
    ax.plot(SIGMAS, nokb, "^--", color="0.5", linewidth=2.0, markersize=4.5,
            label="Stigmergy w/o shared KB")

    ax.set_xlabel(r"Observation noise $\sigma$", fontsize=10)
    ax.set_ylabel("Alignment to true concept", fontsize=10)
    ax.set_xlim(0, SIGMAS[-1])
    ax.set_ylim(-0.08, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout(pad=0.4)
    plt.savefig("figures/Fig_tokenization.pdf")
    plt.savefig("figures/Fig_tokenization.png")
    print("Saved figures/Fig_tokenization.pdf/.png")


if __name__ == "__main__":
    main()
