"""Experiment 3 — Dynamic environment (field decay enables adaptation).

The environment is a ground-truth knowledge-base category that changes every
SHIFT_INTERVAL rounds. Two agent roles:
  - Adopters (sensors): observe the environment and deposit their token.
  - Passives (consumers): only read the field's dominant token and align to it
    (no observation), so they can track the environment ONLY if the field does.

With field decay, stale token counts fade and the dominant token follows each
environment shift, so passives re-align quickly. Without decay, the first
category's counts accumulate and freeze the dominant token, so passives stay
stuck on the old environment. Consensus (passives average the adopters' current
beliefs directly) is the ideal-adaptation reference, at O(N^2) cost.

Read-before-deposit (C2) ordering is used throughout.
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from stigmergy import SharedKnowledgeBase

# ── Parameters ──────────────────────────────────────────────
DIM = 512
M = 10
N = 50
K = 3             # adopters (sensors), all identical; the rest are passives
BETA = 0.3        # adopter observation weight
ALPHA = 0.3       # passive field-reading weight
SIGMA = 0.0       # observation noise (0 = noiseless observation)
THETA = 0.3       # gating threshold (unused for adopters here; kept for ref)
SHIFT_INTERVAL = 50             # longer phases (belief_spread scale)
N_PHASES = 4
TOTAL_ROUNDS = SHIFT_INTERVAL * N_PHASES   # = 200 rounds
ENV_SEQ = [0, 1, 2, 3]          # each phase is a NEW distinct category

# Evaporation-rate sweep: counts *= (1 - rho) each round.
# Higher rho forgets faster -> adapts faster; rho=0 = no decay.
RHOS = [0.3, 0.03, 0.0]
RHO_COLORS = ["tab:green", "tab:orange", "tab:red"]  # 0.3 green, 0.03 orange, 0.0 red
RUNS = 20

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

# Orthonormal categories: exactly orthogonal concept directions, so an
# old-aligned belief has *exactly* 0 similarity to a new (shifted) environment
# instead of a small +/- residual from near-orthogonal random vectors.
KB = SharedKnowledgeBase(num_tokens=M, dim=DIM, seed=0, orthonormal=True)


def env_cat(t: int) -> int:
    return ENV_SEQ[min(t // SHIFT_INTERVAL, N_PHASES - 1)]


def adopter_obs(t: int) -> torch.Tensor:
    """All K adopters observe the SAME current environment (one sensor role).
    With SIGMA>0 each sees it through independent observation noise."""
    env = KB.E[env_cat(t)].unsqueeze(0).expand(K, DIM)
    if SIGMA > 0:
        return F.normalize(env + SIGMA * torch.randn(K, DIM), dim=1)
    return env


# ── Stigmergy (token field), evaporation rate rho ───────────
def run_stigmergy(rho: float) -> np.ndarray:
    adopter = F.normalize(torch.randn(K, DIM), dim=1)
    passive = F.normalize(torch.randn(N - K, DIM), dim=1)
    counts = torch.zeros(M)
    sims = []

    for t in range(TOTAL_ROUNDS):
        c = env_cat(t)
        env = KB.E[c]

        # adopters (all identical) observe the current environment
        obs = adopter_obs(t)
        adopter = F.normalize((1 - BETA) * adopter + BETA * obs, dim=1)
        a_toks = (adopter @ KB.E.t()).argmax(dim=1)

        # passives read the current dominant token and align (read before deposit)
        dom = None if counts.max() < 1e-8 else int(counts.argmax())
        if dom is not None:
            passive = F.normalize((1 - ALPHA) * passive + ALPHA * KB.E[dom], dim=1)

        # field: adopters deposit, then decay
        counts += torch.bincount(a_toks, minlength=M).float()
        counts *= (1.0 - rho)

        sims.append((passive @ env).mean().item())

    return np.array(sims)


# ── Consensus baseline — averaging WITHOUT forgetting ────────────────
def run_consensus() -> np.ndarray:
    """Direct averaging with no forgetting factor: agents read a running
    cumulative average of all sensed signals. Without decay this hub drifts
    toward the centroid of all past environments, so its alignment to the
    current one progressively collapses round after round — the averaging
    analogue of no-decay stigmergy. (A standard re-averaging consensus instead
    sits at a constant low plateau; it does not progressively degrade.)"""
    adopter = F.normalize(torch.randn(K, DIM), dim=1)
    passive = F.normalize(torch.randn(N - K, DIM), dim=1)
    hub = torch.zeros(DIM)
    sims = []

    for t in range(TOTAL_ROUNDS):
        c = env_cat(t)
        env = KB.E[c]

        obs = adopter_obs(t)
        adopter = F.normalize((1 - BETA) * adopter + BETA * obs, dim=1)

        hub = hub + adopter.mean(dim=0)                       # accumulate, no forgetting
        passive = F.normalize((1 - ALPHA) * passive + ALPHA * F.normalize(hub, dim=0), dim=1)

        sims.append((passive @ env).mean().item())

    return np.array(sims)


def avg(fn, *args):
    arr = np.stack([fn(*args) for _ in range(RUNS)])
    return arr.mean(axis=0), arr.std(axis=0)


# ── Main ─────────────────────────────────────────────────────
def main():
    print("Running dynamic-environment experiment...")
    stig = {r: avg(run_stigmergy, r) for r in RHOS}
    c_mean, _ = avg(run_consensus)

    rounds = np.arange(1, TOTAL_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))   # match Fig_scalability size/aspect

    # evaporation sweep: faster forgetting (higher rho) -> faster re-adaptation
    for rho, color in zip(RHOS, RHO_COLORS):
        m, _ = stig[rho]
        lbl = f"Stigmergy $\\rho$={rho}" + (" (w/o decay)" if rho == 0.0 else "")
        ax.plot(rounds, m, color=color, linewidth=1.6, label=lbl, zorder=3)

    ax.plot(rounds, c_mean, color="0.5", linestyle="--", linewidth=1.2,
            label="Consensus", zorder=2)

    for i in range(1, N_PHASES):
        ax.axvline(i * SHIFT_INTERVAL, color="black", linewidth=0.7,
                   linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Similarity to current environment", fontsize=10)
    ax.set_ylim(-0.15, 1.05)
    ax.tick_params(labelsize=8)
    # legend as a horizontal row above the axes (outside the data area)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, columnspacing=1.3, handlelength=1.7,
              handletextpad=0.5, labelspacing=0.3)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/Fig_dynamic_env.pdf", bbox_inches="tight")
    plt.savefig("figures/Fig_dynamic_env.png", bbox_inches="tight")
    print("Saved figures/Fig_dynamic_env.pdf/.png")


if __name__ == "__main__":
    main()
