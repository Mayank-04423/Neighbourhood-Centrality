"""
===============================================================================
 NEIGHBORHOOD CENTRALITY — Implementation & SIR Validation Toolkit
===============================================================================

PURPOSE
-------
Implements and validates the centrality measure proposed in:

    Liu, Y., Tang, M., Zhou, T., & Do, Y. (2016).
    "Identify influential spreaders in complex networks, the role of
    neighborhood." Physica A: Statistical Mechanics and its Applications,
    450, 636-666.

Given any network, this script lets you:
    1. Compute Degree centrality            -> k(i)
    2. Compute Coreness centrality           -> ks(i)   (k-shell decomposition)
    3. Compute Neighborhood Centrality       -> C_n(theta)   [the paper's method]
    4. Simulate SIR epidemic spreading to get each node's REAL spreading power
    5. Evaluate predictions against that ground truth using:
           - Imprecision function   epsilon(p)
           - Kendall's tau          + "improved tau ratio" eta
    6. Visualize results (optional PNG plot of imprecision vs. p)


-------------------------------------------------------------------------------
HOW IT WORKS (high level)
-------------------------------------------------------------------------------
    Step 1   Load a graph (your own edge list, or the built-in demo graph)
    Step 2   Compute a "benchmark" centrality theta  (degree or coreness)
    Step 3   Compute neighborhood centrality C_n(theta) for n = 1..4 steps,
             using decay parameter `a`:

                 C_n(theta_i) = theta_i
                              + a    * sum(theta_j : j in 1-step neighbors)
                              + a^2  * sum(theta_j : j in 2-step neighbors)
                              + ...
                              + a^n  * sum(theta_j : j in n-step neighbors)

    Step 4   Run an SIR (Susceptible-Infected-Recovered) simulation from
             every node as seed, many times, to measure each node's TRUE
             average spreading power M(i).
    Step 5   Compare theta and each C_n(theta) against M using imprecision
             and Kendall's tau, to see how much neighborhood information
             improves ranking accuracy.


-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
Requires Python 3.9+. Install dependencies with:

    pip install networkx numpy scipy matplotlib

    networkx    -> required  (graph data structure, k-shell, BFS, etc.)
    numpy       -> required  (array math)
    scipy       -> optional  (exact Kendall's tau; falls back to a manual
                              O(n^2) implementation if not installed)
    matplotlib  -> optional  (only needed for the --plot flag)


-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
Run the built-in demo (Zachary Karate Club graph), no arguments needed:

    python neighborhood_centrality.py

Use your own network — a plain edge list (two node IDs per line):

    python neighborhood_centrality.py --edgelist my_network.txt

Use your own network — a CSV with a header row (columns: source,target):

    python neighborhood_centrality.py --edgelist my_network.csv --csv

Full example with custom options:

    python neighborhood_centrality.py \
        --edgelist my_network.txt \
        --benchmark coreness \
        --steps 1 2 3 \
        --a 0.3 \
        --runs 200 \
        --top 15 \
        --plot

See all available options:

    python neighborhood_centrality.py --help


-------------------------------------------------------------------------------
COMMAND-LINE ARGUMENTS
-------------------------------------------------------------------------------
--edgelist PATH     Path to your own network file. Omit to use the demo graph.
--csv               Treat --edgelist as a CSV with a header row (source,target).
--benchmark {degree,coreness}
                     Which centrality (theta) to build the neighborhood
                     version from. Default: degree.
--steps N [N ...]   Neighborhood depths to compute and compare.
                     Default: 1 2 3 4  (as tested in the paper).
--a FLOAT            Decay parameter a in [0, 1] from Eq. (1).
                     Default: 0.2 (the paper's default).
--lam FLOAT          SIR infection probability lambda.
                     Default: 0 -> auto-picked as 1.5 x the epidemic
                     threshold lambda_c for the loaded network.
--mu FLOAT           SIR recovery probability. Default: 1.0.
--runs N             Number of SIR simulation runs per seed node.
                     Default: 100. Lower this for very large graphs.
--top N              How many top-ranked nodes to print per measure.
                     Default: 10.
--plot               If set, saves imprecision_curve.png comparing the
                     benchmark vs. each neighborhood-centrality step.
--seed N             RNG seed, for reproducible runs. Default: 42.


-------------------------------------------------------------------------------
FUNCTION REFERENCE
-------------------------------------------------------------------------------
degree_centrality(G)
    Returns {node: degree}.

coreness_centrality(G)
    Returns {node: k-shell index}, via networkx's core_number().

neighborhood_centrality(G, theta, n_steps, a)
    Computes C_n(theta) for every node using BFS layers (no double-counting
    of nodes across steps). Returns {node: score}.

all_neighborhood_centralities(G, theta, max_steps, a)
    Convenience wrapper -> {step: {node: score}} for step = 1..max_steps.

epidemic_threshold(G)
    Returns lambda_c = <k> / (<k^2> - <k>), the heterogeneous mean-field
    epidemic threshold, used to auto-pick a meaningful lambda.

sir_single_run(G, seed_node, lam, mu, rng)
    Runs ONE SIR cascade from seed_node. Returns the fraction of the
    network that ends up Recovered.

spreading_efficiency(G, lam, mu, n_runs, rng)
    Runs sir_single_run() from every node, n_runs times each, and
    averages the outcome. Returns {node: M}, the ground-truth spreading
    power used to evaluate all centrality measures against.

imprecision_curve(centrality, spreading_eff, p_values)
    Computes epsilon(p) = 1 - M(p)/Meff(p) for each p in p_values.
    Lower epsilon = more accurate ranking.

tau_for_top_p(centrality, spreading_eff, p)
    Kendall's tau between a centrality's ranking and true spreading
    power, restricted to the top-p fraction of nodes by that centrality.

improved_tau_ratio(tau_C, tau_theta)
    eta = (tau_C - tau_theta) / |tau_theta| * 100 (%). Positive means the
    neighborhood version ranks nodes better than the plain benchmark.

load_graph(path, is_csv)
    Loads a graph from a plain edge list or a source/target CSV.


-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------
Printed to the console:
    - Top-N nodes by each centrality measure (benchmark + each C_n step)
    - Top-N nodes by true SIR spreading efficiency
    - A table of imprecision epsilon(p) for p = 0.05 .. 0.50
    - Kendall's tau and improved tau ratio (eta) at the top-20% level

If --plot is passed:
    - imprecision_curve.png saved to the working directory


-------------------------------------------------------------------------------
NOTES & LIMITATIONS
-------------------------------------------------------------------------------
    - SIR simulation only works on connected graphs; the script
      automatically keeps the largest connected component if the input
      graph is disconnected.
    - Kendall's tau can come back as "undefined" for a top-p slice where
      the benchmark has many tied values (e.g., coreness on small graphs)
      — this is a real statistical edge case, not a bug, and the script
      reports it explicitly rather than printing a misleading number.
    - Runtime scales with --runs x number of nodes x network size; reduce
      --runs for quick checks on large networks before running the full
      simulation count.

===============================================================================
"""