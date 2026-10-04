"""
neighborhood_centrality.py
===========================
Implementation of the method from:

    Liu, Y., Tang, M., Zhou, T., & Do, Y. (2016).
    "Identify influential spreaders in complex networks, the role of
    neighborhood." Physica A, 450, 636-666.

What this script gives you
---------------------------
1. Degree centrality               k(i)
2. Coreness centrality             ks(i)   (k-shell decomposition)
3. Neighborhood centrality         C_n(theta)  -- the paper's main proposal
4. An SIR spreading simulation to get each node's real spreading efficiency
5. The paper's two evaluation tools:
     - Imprecision function       epsilon(p)
     - Kendall's tau ranking correlation + "improved tau ratio" eta


Run it as-is to see a demo on the Zachary Karate Club graph:

    python neighborhood_centrality.py

Feed your own edge list (two node-ids per line, whitespace/comma separated):

    python neighborhood_centrality.py --edgelist my_network.txt

Feed your own edge list as a CSV with a header (columns: source,target):

    python neighborhood_centrality.py --edgelist my_network.csv --csv

Useful knobs:

    --steps 1 2 3 4       which neighborhood depths to compute/compare
    --a 0.2               the decay parameter a in Eq. (1) of the paper
    --benchmark degree    or: coreness   (theta in the paper)
    --lam 0               infection probability (0 = auto: 1.5 x lambda_c)
    --runs 200            number of SIR simulation runs per seed node
    --top 20              print the top-N nodes by each measure
    --plot                save a PNG of the imprecision curves
    --seed 42             RNG seed, for reproducibility

Requirements: networkx, numpy, scipy (for Kendall's tau), matplotlib (optional, for --plot)
    pip install networkx numpy scipy matplotlib
"""

import argparse
import random
import sys
from collections import defaultdict

import networkx as nx
import numpy as np

try:
    from scipy.stats import kendalltau
except ImportError:  # fall back to a manual implementation if scipy is missing
    kendalltau = None


# ---------------------------------------------------------------------------
# 1-2. Benchmark centralities: degree and coreness
# ---------------------------------------------------------------------------

def degree_centrality(G):
    """k(i): plain node degree. Returns {node: value}."""
    return dict(G.degree())


def coreness_centrality(G):
    """ks(i): k-shell index via iterative pruning (k-core decomposition)."""
    return nx.core_number(G)


# ---------------------------------------------------------------------------
# 3. Neighborhood centrality  --  Eq. (1) of the paper
#
#    C_n(theta_i) = theta_i
#                   + a   * sum(theta_j for j in 1-step neighbors)
#                   + a^2 * sum(theta_j for j in 2-step neighbors)
#                   + ...
#                   + a^n * sum(theta_j for j in n-step neighbors)
#
#    "step" = shortest-path distance from i, found via BFS layers so no
#    node is double counted across steps.
# ---------------------------------------------------------------------------

def neighborhood_centrality(G, theta, n_steps, a):
    """
    theta    : dict {node: benchmark centrality value}, e.g. from
               degree_centrality(G) or coreness_centrality(G)
    n_steps  : how many steps of neighbors to include (the paper tests 1-4)
    a        : decay parameter in [0, 1] (paper's default a = 0.2)

    Returns {node: C_n(theta) value}.
    """
    C = {}
    for node in G.nodes():
        total = theta[node]
        visited = {node}
        frontier = {node}
        for step in range(1, n_steps + 1):
            next_frontier = set()
            for u in frontier:
                next_frontier.update(G.neighbors(u))
            next_frontier -= visited
            if not next_frontier:
                break
            visited.update(next_frontier)
            weight = a ** step
            total += weight * sum(theta[v] for v in next_frontier)
            frontier = next_frontier
        C[node] = total
    return C


def all_neighborhood_centralities(G, theta, max_steps, a):
    """Convenience: returns {step: {node: C_step(theta)}} for step = 1..max_steps."""
    return {s: neighborhood_centrality(G, theta, s, a) for s in range(1, max_steps + 1)}


# ---------------------------------------------------------------------------
# 4. SIR spreading simulation -> each node's real spreading efficiency M
# ---------------------------------------------------------------------------

def epidemic_threshold(G):
    """lambda_c = <k> / (<k^2> - <k>), the heterogeneous mean-field threshold
    used by the paper to pick a meaningful infection probability."""
    degrees = np.array([d for _, d in G.degree()])
    k_mean = degrees.mean()
    k2_mean = (degrees ** 2).mean()
    denom = k2_mean - k_mean
    if denom <= 0:
        return 0.1  # fallback for degenerate/regular graphs
    return k_mean / denom


def sir_single_run(G, seed_node, lam, mu, rng):
    """One SIR cascade starting at seed_node. Returns the fraction of the
    network that ends up Recovered (= the run's spreading outcome)."""
    state = {n: "S" for n in G.nodes()}
    state[seed_node] = "I"
    infected = {seed_node}
    n_recovered = 0

    while infected:
        newly_infected = set()
        for u in infected:
            for v in G.neighbors(u):
                if state[v] == "S" and rng.random() < lam:
                    newly_infected.add(v)

        newly_recovered = {u for u in infected if rng.random() < mu}

        for v in newly_infected:
            state[v] = "I"
        for u in newly_recovered:
            state[u] = "R"
            n_recovered += 1

        infected = (infected - newly_recovered) | newly_infected

    return n_recovered / G.number_of_nodes()


def spreading_efficiency(G, lam, mu=1.0, n_runs=100, rng=None):
    """
    Runs the SIR model from every node in G as the seed, n_runs times each,
    and returns {node: average fraction of network eventually infected}.
    This is the ground-truth "M" the paper ranks centralities against.
    """
    if rng is None:
        rng = random.Random()
    M = {}
    for node in G.nodes():
        outcomes = [sir_single_run(G, node, lam, mu, rng) for _ in range(n_runs)]
        M[node] = sum(outcomes) / len(outcomes)
    return M


# ---------------------------------------------------------------------------
# 5a. Imprecision function  --  Eq. used in Results section
#
#     epsilon(p) = 1 - M(p) / Meff(p)
#
#     M(p)    = average real spreading efficiency of the pN nodes the
#               CENTRALITY under test ranks highest
#     Meff(p) = average real spreading efficiency of the pN nodes that are
#               ACTUALLY the best spreaders (oracle ranking)
# ---------------------------------------------------------------------------

def imprecision_curve(centrality, spreading_eff, p_values):
    """
    centrality     : dict {node: score} for the measure being evaluated
    spreading_eff  : dict {node: M}, ground truth from spreading_efficiency()
    p_values       : iterable of p in (0, 1], e.g. np.arange(0.01, 0.21, 0.01)

    Returns list of epsilon(p), same order as p_values.
    """
    N = len(spreading_eff)
    nodes_by_centrality = sorted(centrality, key=centrality.get, reverse=True)
    nodes_by_true_eff = sorted(spreading_eff, key=spreading_eff.get, reverse=True)

    eps = []
    for p in p_values:
        pN = max(1, round(p * N))
        top_pred = nodes_by_centrality[:pN]
        top_true = nodes_by_true_eff[:pN]
        M_p = np.mean([spreading_eff[n] for n in top_pred])
        Meff_p = np.mean([spreading_eff[n] for n in top_true])
        eps.append(1 - M_p / Meff_p if Meff_p > 0 else 0.0)
    return eps


# ---------------------------------------------------------------------------
# 5b. Kendall's tau + the paper's "improved tau ratio" eta
# ---------------------------------------------------------------------------

def _kendall_tau(x, y):
    """Kendall's tau-b. Uses scipy if available, else a plain O(n^2) fallback."""
    if kendalltau is not None:
        tau, _ = kendalltau(x, y)
        return tau
    # manual fallback (fine for the small/medium node counts used here)
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs if total_pairs else 0.0


def tau_for_top_p(centrality, spreading_eff, p):
    """Kendall's tau between a centrality's ranking and true spreading
    efficiency, restricted to the top-p fraction of nodes BY CENTRALITY
    (matches the paper's "top ranked pN nodes" setup)."""
    N = len(centrality)
    pN = max(2, round(p * N))
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:pN]
    x = [centrality[n] for n in top_nodes]
    y = [spreading_eff[n] for n in top_nodes]
    return _kendall_tau(x, y)


def improved_tau_ratio(tau_C, tau_theta):
    """eta = (tau_C - tau_theta) / |tau_theta|, expressed as a percentage."""
    if tau_theta == 0:
        return float("inf") if tau_C != 0 else 0.0
    return (tau_C - tau_theta) / abs(tau_theta) * 100


# ---------------------------------------------------------------------------
# Loading your own network
# ---------------------------------------------------------------------------

def load_graph(edgelist_path, is_csv=False):
    if is_csv:
        import csv
        G = nx.Graph()
        with open(edgelist_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 2 and row[0] != "" and row[1] != "":
                    G.add_edge(row[0], row[1])
        return G
    return nx.read_edgelist(edgelist_path)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def print_top(label, scores, top_n):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    print(f"\nTop {top_n} nodes by {label}:")
    for rank, (node, val) in enumerate(ranked, 1):
        print(f"  {rank:>2}. node {node!s:<8} {label} = {val:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Neighborhood centrality (Liu, Tang, Zhou & Do, 2016) — "
                     "compute it, and validate it with an SIR simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--edgelist", type=str, default=None,
                         help="path to your own edge list (default: demo graph)")
    parser.add_argument("--csv", action="store_true",
                         help="treat --edgelist as CSV with a header row (source,target)")
    parser.add_argument("--benchmark", choices=["degree", "coreness"], default="degree",
                         help="theta in the paper: which centrality to build the neighborhood version from")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4],
                         help="neighborhood depths to compute and compare")
    parser.add_argument("--a", type=float, default=0.2, help="decay parameter a in [0,1]")
    parser.add_argument("--lam", type=float, default=0.0,
                         help="SIR infection probability (0 = auto-pick as 1.5x epidemic threshold)")
    parser.add_argument("--mu", type=float, default=1.0, help="SIR recovery probability")
    parser.add_argument("--runs", type=int, default=100, help="SIR runs per seed node")
    parser.add_argument("--top", type=int, default=10, help="print top-N nodes per measure")
    parser.add_argument("--plot", action="store_true", help="save imprecision-vs-p plot to a PNG")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ---- 1. Load the network ------------------------------------------------
    if args.edgelist:
        G = load_graph(args.edgelist, is_csv=args.csv)
        print(f"Loaded network from {args.edgelist}: "
              f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        G = nx.karate_club_graph()
        print("No --edgelist given -> using the built-in Zachary Karate Club demo graph "
              f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()  # SIR needs one component

    # ---- 2. Benchmark + neighborhood centralities ---------------------------
    theta = degree_centrality(G) if args.benchmark == "degree" else coreness_centrality(G)
    print(f"\nUsing '{args.benchmark}' as the benchmark centrality theta.")

    C_by_step = all_neighborhood_centralities(G, theta, max(args.steps), args.a)

    print_top(args.benchmark, theta, args.top)
    for s in args.steps:
        print_top(f"C{s}({args.benchmark})", C_by_step[s], args.top)

    # ---- 3. SIR simulation for ground-truth spreading efficiency ------------
    lam_c = epidemic_threshold(G)
    lam = args.lam if args.lam > 0 else 1.5 * lam_c
    print(f"\nEpidemic threshold lambda_c = {lam_c:.4f}  ->  using lambda = {lam:.4f}")
    print(f"Running SIR ({args.runs} runs/node) ... this may take a moment for larger graphs.")
    M = spreading_efficiency(G, lam=lam, mu=args.mu, n_runs=args.runs, rng=rng)
    print_top("true spreading efficiency M", M, args.top)

    # ---- 4. Evaluate: imprecision + Kendall's tau ----------------------------
    p_values = np.arange(0.05, 0.55, 0.05)

    print("\n--- Imprecision epsilon(p): lower is better -------------------------")
    header = "p    " + "".join(f"{args.benchmark:>10}") + "".join(f"C{s}(.)".rjust(10) for s in args.steps)
    print(header)
    eps_theta = imprecision_curve(theta, M, p_values)
    eps_by_step = {s: imprecision_curve(C_by_step[s], M, p_values) for s in args.steps}
    for i, p in enumerate(p_values):
        row = f"{p:.2f} " + f"{eps_theta[i]:>10.3f}"
        row += "".join(f"{eps_by_step[s][i]:>10.3f}" for s in args.steps)
        print(row)

    print("\n--- Kendall's tau and improved tau ratio (top 20% of nodes) ----------")
    p_eval = 0.2
    tau_theta = tau_for_top_p(theta, M, p_eval)
    if np.isnan(tau_theta):
        print(f"tau({args.benchmark}) = undefined (many tied {args.benchmark} values in the "
              f"top {int(p_eval*100)}% -> no rank variance). Try a coarser benchmark, a larger "
              f"--top-p slice, or compare C-steps to each other instead.")
    else:
        print(f"tau({args.benchmark}) = {tau_theta:.4f}")
    for s in args.steps:
        tau_C = tau_for_top_p(C_by_step[s], M, p_eval)
        if np.isnan(tau_C) or np.isnan(tau_theta):
            tau_str = "undefined" if np.isnan(tau_C) else f"{tau_C:.4f}"
            print(f"tau(C{s}({args.benchmark})) = {tau_str}   improved tau ratio eta = n/a (ties)")
        else:
            eta = improved_tau_ratio(tau_C, tau_theta)
            print(f"tau(C{s}({args.benchmark})) = {tau_C:.4f}   "
                  f"improved tau ratio eta = {eta:+.1f}%")

    # ---- 5. Optional plot -----------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.plot(p_values, eps_theta, "k-s", label=args.benchmark)
        for s in args.steps:
            plt.plot(p_values, eps_by_step[s], "-o", label=f"C{s}({args.benchmark})")
        plt.xlabel("p (fraction of network)")
        plt.ylabel("imprecision  \u03b5(p)")
        plt.title("Neighborhood centrality vs. benchmark: imprecision")
        plt.legend()
        plt.tight_layout()
        out_path = "imprecision_curve.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()