"""Experiment 1 — Scalability.

Token / knowledge-base stigmergy vs. continuous-vector baselines (pairwise gossip,
gossip k=3, all-to-all consensus). We count messages to convergence as N grows.

Stigmergy now exchanges a *single token index* per write (plus N dominant-token
reads per round), so it stays O(N) in message count while the write payload
collapses from a 512-d vector to one symbol. Consensus is O(N^2).
"""
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from stigmergy import SharedKnowledgeBase

# ── Parameters ──────────────────────────────────────────────
DIM = 512
M = 10            # knowledge-base size (CIFAR-10 categories)
DELTA = 0.001     # convergence threshold
MAX_ROUNDS = 500
RUNS = 60         # runs per N for averaging (high -> smooth Monte-Carlo curves)
SEED = 0          # global seed: makes the whole sweep deterministic/reproducible

BETA = 0.0        # no observation step (communication-complexity focus)
ALPHA = 0.5       # absorption strength (shared across protocols)
THETA = 0.3       # gating threshold (stigmergy)
DECAY = 0.9       # field token-frequency decay

N_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

#figure style (sans-serif, matching Fig_belief_spread)
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


# ── Helpers ──────────────────────────────────────────────────
def make_concepts(n: int, dim: int) -> torch.Tensor:
    v = torch.randn(n, dim)
    return F.normalize(v, dim=1)


def assign_tokens(n: int) -> torch.Tensor:
    """Assign each agent one of the M knowledge-base categories."""
    return torch.randint(0, M, (n,))


# ── Stigmergy (token / knowledge base, vectorized) ───────────────────
def run_stigmergy(n: int, gate: bool = True) -> tuple[int, bool]:
    """gate=True: selective transmission (deposit only if belief diverges from
    dominant by > THETA). gate=False: every agent deposits every round."""
    init_tokens = assign_tokens(n)
    B = KB.E[init_tokens].clone()          # (N, dim) beliefs start at category dirs
    counts = torch.zeros(M)
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev = B.clone()

        dom = None if counts.max() < 1e-8 else int(counts.argmax())

        # (1) read dominant + update belief FIRST (removes phase-lag oscillation)
        if dom is not None:
            B = F.normalize((1 - ALPHA) * B + ALPHA * KB.E[dom], dim=1)

        # (2) tokenize updated belief
        toks = (F.normalize(B, dim=1) @ KB.E.t()).argmax(dim=1)   # (N,)

        # (3) deposit decision
        if dom is None or not gate:
            deposit_mask = torch.ones(n, dtype=torch.bool)         # seed / no selection
        else:
            dist = 1.0 - (F.normalize(B, dim=1) @ KB.E[dom])     # cos-dist belief vs dominant
            deposit_mask = dist > THETA                            # selective transmission

        dep_toks = toks[deposit_mask]
        if dep_toks.numel() > 0:
            counts += torch.bincount(dep_toks, minlength=M).float()

        counts *= DECAY
        total_messages += int(deposit_mask.sum()) + n   # writes (1 token each) + N reads

        if dom is not None and (prev - B).norm(dim=1).max().item() < DELTA:
            return total_messages, True

    return total_messages, False


# ── Gossip (pairwise) ────────────────────────────────────────
def run_gossip(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    """Pairwise gossip: random pair matching per round, symmetric push-pull."""
    beliefs = [concepts[i].clone() for i in range(n)]
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev_beliefs = [b.clone() for b in beliefs]
        perm = np.random.permutation(n)
        new_beliefs = [b.clone() for b in beliefs]
        n_pairs = n // 2

        for k in range(n_pairs):
            i, j = int(perm[2 * k]), int(perm[2 * k + 1])
            bi, bj = beliefs[i], beliefs[j]
            new_beliefs[i] = F.normalize((1 - ALPHA) * bi + ALPHA * bj, dim=0)
            new_beliefs[j] = F.normalize((1 - ALPHA) * bj + ALPHA * bi, dim=0)

        total_messages += 2 * n_pairs
        beliefs = new_beliefs

        if max((p - c).norm().item() for p, c in zip(prev_beliefs, beliefs)) < DELTA:
            return total_messages, True

    return total_messages, False


# ── Consensus (all-to-all) ───────────────────────────────────
def run_consensus(n: int, concepts: torch.Tensor) -> tuple[int, bool]:
    B = concepts.clone()
    total_messages = 0

    for _ in range(MAX_ROUNDS):
        prev_B = B.clone()
        total = B.sum(dim=0, keepdim=True)
        mean_others = (total - B) / (n - 1)
        mean_others = F.normalize(mean_others, dim=1)
        B = F.normalize((1 - ALPHA) * B + ALPHA * mean_others, dim=1)
        total_messages += n * (n - 1)

        if (prev_B - B).norm(dim=1).max().item() < DELTA:
            return total_messages, True

    return total_messages, False


# ── Main ─────────────────────────────────────────────────────
CACHE = "figures/_exp1_cache.npz"

STYLES = {
    "consensus":        ("s-",  "Consensus",                         "tab:blue",   1.8),
    "gossip":           ("^--", "Gossip (pairwise)",                 "tab:green",  1.8),
    "stigmergy_nogate": ("D-",  "Stigmergy (w/o selective deposit)", "tab:orange", 1.8),
    "stigmergy":        ("o-",  "Stigmergy (w/ selective deposit)",  "tab:red",    2.2),
}


def _signature() -> str:
    return "|".join(str(x) for x in
                    [N_VALUES, RUNS, SEED, ALPHA, THETA, DECAY, MAX_ROUNDS, DELTA, DIM, M])


def compute_results():
    torch.manual_seed(SEED)      # deterministic, reproducible sweep
    np.random.seed(SEED)
    results = {k: [] for k in STYLES}
    converged = {k: [] for k in STYLES}
    for n in N_VALUES:
        for key in results:
            msgs_list, conv_list = [], []
            for _ in range(RUNS):
                if key == "stigmergy":
                    msgs, conv = run_stigmergy(n, gate=True)
                elif key == "stigmergy_nogate":
                    msgs, conv = run_stigmergy(n, gate=False)
                else:
                    fn = {"gossip": run_gossip, "consensus": run_consensus}[key]
                    msgs, conv = fn(n, make_concepts(n, DIM))
                msgs_list.append(msgs)
                conv_list.append(conv)
            results[key].append(np.mean(msgs_list))
            converged[key].append(all(conv_list))

        print(
            f"N={n:5d} | stigmergy={results['stigmergy'][-1]:9.0f}"
            f" | stig_nogate={results['stigmergy_nogate'][-1]:9.0f}"
            f" | gossip={results['gossip'][-1]:9.0f}"
            f" | consensus={results['consensus'][-1]:11.0f}"
            f"{'  [NOT CONVERGED]' if not converged['consensus'][-1] else ''}"
        )
    return results, converged


def load_or_compute():
    """Cache results so figure-style tweaks don't re-run the slow sweep.
    Delete figures/_exp1_cache.npz (or change a parameter) to recompute."""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        if str(d["sig"]) == _signature():
            print("Loaded cached results (delete figures/_exp1_cache.npz to recompute).")
            return d["results"].item(), d["converged"].item()
    results, converged = compute_results()
    np.savez(CACHE, results=results, converged=converged, sig=_signature())
    return results, converged


def main():
    results, converged = load_or_compute()
    per_agent = {k: [m / n for m, n in zip(results[k], N_VALUES)] for k in results}

    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    for key, (style, label, color, lw) in STYLES.items():
        ax.plot(N_VALUES, per_agent[key], style, label=label, color=color,
                linewidth=lw, markersize=4.5, zorder=3)
        nc_x = [N_VALUES[i] for i, c in enumerate(converged[key]) if not c]
        nc_y = [per_agent[key][i] for i, c in enumerate(converged[key]) if not c]
        if nc_x:
            ax.scatter(nc_x, nc_y, marker="x", s=70, color="black", zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Number of agents $N$", fontsize=10)
    ax.set_ylabel("Messages per agent", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # ── inset: zoom on small N so the message-count gaps are visible (linear y).
    # Sits in the empty band between Consensus (above) and the flat lines (below);
    # connector lines are hidden so nothing crosses the main curves. ──
    axins = ax.inset_axes([0.63, 0.27, 0.34, 0.34])
    for key in ["gossip", "stigmergy_nogate", "stigmergy"]:
        style, _, color, lw = STYLES[key]
        axins.plot(N_VALUES, per_agent[key], style, color=color,
                   linewidth=lw, markersize=4.0)
    axins.set_xscale("log")
    axins.set_xlim(8, 125)            # cut just below 10 and just above 100
    axins.set_ylim(10, 30)
    axins.set_xticks([10, 20, 50, 100])
    axins.xaxis.set_major_formatter(plt.ScalarFormatter())
    axins.set_yticks([10, 20, 30])
    axins.minorticks_off()
    axins.tick_params(labelsize=7)
    axins.set_title("$N$ = 10–100 (zoom)", fontsize=7.5, fontweight="bold")
    indicator = ax.indicate_inset_zoom(axins, edgecolor="0.4", linewidth=0.8, alpha=0.8)
    for c in indicator.connectors:
        c.set_visible(False)          # hide auto connectors (they cross the curves)
    # custom connectors: from the box's top-right corner up into the empty band
    # to the inset's left edge, so nothing crosses the main curves.
    for inset_corner in [(0.0, 0.0), (0.0, 1.0)]:
        ax.add_artist(ConnectionPatch(
            xyA=(125, 32), coordsA=ax.transData,          # box top-right corner
            xyB=inset_corner, coordsB=axins.transAxes,    # inset left edge
            color="0.4", linewidth=0.8, alpha=0.8))

    ax.grid(False)
    axins.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout(pad=0.4)
    plt.savefig("figures/Fig_scalability.pdf")
    plt.savefig("figures/Fig_scalability.png")
    print("Saved figures/Fig_scalability.pdf/.png")


if __name__ == "__main__":
    main()
