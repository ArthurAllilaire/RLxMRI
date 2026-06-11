# Seed-stability check for the CR-optimal fixed schedule (Run B baseline config).
# Re-solves the exact reported configuration across rng seeds 0..3 and reports
# how much the objective L and chosen schedule move.

using MRISystemPhantom
using Random
using Printf
using Statistics: mean, std

const M = MRISystemPhantom

T1s = [1.879, 1.432, 1.027, 0.7513, 0.527, 0.3841, 0.2723,
       0.1945, 0.1378, 0.0947, 0.067, 0.04814, 0.03435, 0.02416]
budget_s   = 420.0
Npe        = 32
block_grid = [4, 6, 8, 10, 14, 18]
n_starts   = 1000
n_refine   = 10
TR_lo_floor = 0.5
TE_s        = 0.02
TR_headroom = 0.9

results = []
for seed in 0:3
    res = M.cr_optimize_sweep(T1s;
        budget_s = budget_s, Npe = Npe,
        n_block_grid = block_grid,
        n_starts = n_starts, n_refine = n_refine,
        rng = MersenneTwister(seed),
        TR_lo_floor = TR_lo_floor, TE_s = TE_s, TR_headroom = TR_headroom)
    sched = res.schedule
    push!(results, (seed = seed, n_blocks = res.n_blocks,
                    L = sched.L, TIs = sort(sched.TIs)))
    @printf("seed %d:  n_blocks=%2d   L=%.6f\n", seed, res.n_blocks, sched.L)
end

Ls = [r.L for r in results]
@printf("\nL across seeds: min=%.6f  max=%.6f  mean=%.6f  std=%.6f  spread=%.2f%%\n",
        minimum(Ls), maximum(Ls), mean(Ls), std(Ls),
        100 * (maximum(Ls) - minimum(Ls)) / mean(Ls))
nbs = [r.n_blocks for r in results]
@printf("n_blocks across seeds: %s\n", string(nbs))

println("\nSorted TI schedules per seed:")
for r in results
    @printf("seed %d (n=%2d): %s\n", r.seed, r.n_blocks,
            join((@sprintf("%.3f", t) for t in r.TIs), ", "))
end
