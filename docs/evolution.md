# Evolution: objectives, search, and what it found

Code: `src/evolve.jl` (objectives, `evaluate`, NSGA-II, recipes), `bin/evolve.jl`
(experiments and post-hoc modes), `test/evolve.jl`. Archives: `results/evolve-<run>-<mode>.jsonl`,
one line per evaluated rule (θ, z, σ, parents, seeds, physics, eval config, objectives,
diagnostics, wall time). All results below are **one-ring** rules (`Wnb2 = 0`; the
second shell was not yet wired into the step).

## Measurement

One run per rule from a fixed initial condition (World seed 1, `h_scale 0.1`, 32×32×24
slab, T = 1000 steps at dt 0.1, i.e. t = 100), then three continuations of 250 steps
from the same state: control, activity reset (`h ←` 0.01·noise, `s n w` kept), and a
wound (sphere of radius N/6 at the most solid layer: `s → 0` with its mass returned to
`n`, `h` reset). Every objective is "larger = more of it".

| objective | definition | range |
|---|---|---|
| persistence | mean over the last quarter of `exp(−d/0.05)`, d = distance of `mean_s` from [0.05, 0.6], × `min(1, std_s(end)/max std_s over last half)` | [0,1] |
| structure | lateral (per-layer x–y DFT, layer mean removed) power share in 2π/16 ≤ \|k\| ≤ 2π/4, excess over white noise's share, × `min(1, rms/0.05)` | [0,1] |
| temporal | normalized spectral entropy of 16 voxels × C channels of `h` (last T/2, every 5 steps, detrended, Hann), × `min(1, std/0.02)`; fixed points score 0, line spectra low | [0,1] |
| memory_s / memory_h | Pearson correlation of `s` / `h` between control and activity-reset continuations | [−1,1] |
| recovery | `1 − (|m_e−m_c| + |σ_e−σ_c|)/(m_c+σ_c)` of `s` inside the wound, wounded vs control (phase-free) | [−1,1] |
| light | `4x(1−x)`, x = fraction of voxels with `L > 0.1 L0` at the end | [0,1] |

Selection uses the five objectives persistence, structure, temporal, memory_h, recovery
(clipped to [0,1]). memory_s and light are recorded but not selected: memory_s is ≈1 for
almost every rule because the material is slow (median 0.99), so it does not discriminate.

Machine checks (`test/evolve.jl`): a saturating builder (b = 3, v_dep = 5) gives
mean_s > 0.8 with persistence and structure < 0.05; the zero rule has temporal = 0 and
every probe fixed; a λ = 8 stripe scores structure > 0.9 and white noise < 0.1; noise ≫
sine on temporal; mass drift < 1e-3; Pareto rank, crowding, MC hypervolume on toy cases;
search re-runnable from its seed; recipe round trip.

## Search

(μ+λ) NSGA-II in block-normalized coordinates z = θ ./ scale (the `random_rule` scales):
binary tournament on (rank, crowding), block-uniform crossover (p 0.5), self-adaptive
step `σ' = σ·exp(0.2N)` clipped to [0.02, 1]. Pop 32 (gens 6, or 5 in the low-mass runs).
`--mode random` draws fresh rules from the same prior for the same budget and the same
seed (so generation 0 is shared). **No weighted sum** anywhere in selection.

**Explicit-Euler check.** `Analysis.stable_dt` finds that 10% (g1 −0.5) to 35% (g1 +0.5) of
Gaussian-prior rules need dt < 0.1, and that these rules **game the temporal objective**
(median temporal 0.26 vs 0.05 for dt-stable rules). In the g1 +0.5 search 62% of the
last generation were dt-unstable. `evaluate_one` now records `stable_dt`/`dt_ok` and
`objvec` makes dt-unstable rules infeasible; the archives below predate that, so the
dt-checked numbers are recomputed post hoc (`--correlate`, `--hv`).

## Random vs evolved (equal budget, MC hypervolume over the five, ref 0)

| run (physics) | evals/mode | HV random | HV evolved | HV random, dt ok | HV evolved, dt ok |
|---|---|---|---|---|---|
| g1neg: g1 −0.5, α_s 0.5, α_p 0.1, n0 1 | 192 | 0.0189 | 0.0406 | 0.0137 | **0.0406** |
| g1pos: g1 +0.5, same | 192 | 0.0026 | 0.0123 | 0.0024 | 0.0011 |
| prone: g1 −0.5, α_s 0.05, α_p 0.1, n0 0.5, Turing-prone prior | 160 | 0.0107 | 0.0059 | 0.0107 | 0.0059 |
| lowmass: same physics, Gaussian prior | 160 | 0.0038 | — | 0.0031 | — |

Per-objective best (dt ok), random → evolved: g1neg structure 0.158 → 0.170, temporal
0.353 → 0.376; g1pos structure 0.027 → 0.115, temporal 0.475 → 0.607. Observations:
- g1neg: evolution beats random (×3 hypervolume once dt-unstable rules are removed);
  gains are in co-occurrence (a rule good on all five) more than in any single maximum.
- g1pos: the apparent ×5 win was **entirely dt-unstable rules**; on dt-stable rules
  evolution lost. Conducting solid makes 98% of Gaussian rules grid-scale (`--classify-prior`).
- prone: random draws from lane A's `turing_prone_rule` reach the highest structure in
  the project (0.395) and beat the ES, because Gaussian mutation in z destroys the
  prior's positive-semidefinite `W_nb`; at equal physics the prone prior beats the
  Gaussian prior (HV 0.0107 vs 0.0031, best structure 0.395 vs 0.041).
- One seed per arm; the differences above are machine measurements on that seed.

## Trade-offs (Pearson over each run's archive)

temporal ↔ memory_h −0.35 (g1neg): rules whose activity keeps moving do not re-form it
from the material. structure ↔ recovery −0.31 (prone): sharp textures regrow the amount
but not the texture in 25 time units. persistence ↔ recovery +0.29/+0.48. memory_s ↔
memory_h +0.95 under the prone prior (activity slaved to material).

## Does the linear atlas predict structure? (analysis as a lever)

- Post hoc, every archived rule classified by `Analysis` (top layer, own physics):
  Spearman(structure, max Re λ) = −0.17 (g1neg), −0.12 (g1pos), −0.23 (prone). The 88
  Turing-classified prone rules have mean structure 0.008 (2.3% > 0.1) vs 0.003 for stable
  ones. Reason: their median finite-k growth rate is 2e-5 per unit time (quartiles
  7e-7 … 2e-4), so over t = 100 the predicted mode grows by < 1%. The prediction is about
  sign, not about whether a pattern appears within a run.
- Atlas recipes in their own physics and dt (t = 50; `results/evolve-atlas-own-physics.json`): 1 of 10
  Turing recipes (turing-10) patterns, and at the predicted wavelength (4.2 predicted, 4.4
  measured, structure 0.33). The rest score ≈ 0.
- As a prior, the analysis is a strong lever: the prone sampler raised best structure ten-fold.

## The evolved recipes (`recipes/evolved-*.toml`, explicit θ, all `stable_dt` ≥ 0.27 > dt)

| recipe | structure | temporal | memory_h | recovery | persistence | mean_s |
|---|---|---|---|---|---|---|
| prone-structure | 0.395 (seed 2: 0.388) | 0.05 | 1.00 | 0.53 | 1.00 | 0.23 |
| prone-nsga-structure | 0.177 (0.178) | 0.04 | 0.99 | 0.41 | 1.00 | 0.27 |
| g1neg-balanced | 0.170 (0.194) | 0.31 | 0.84 | 0.85 | 0.99 | 0.46 |
| g1neg-temporal | 0.000 | 0.38 | 0.31 | 0.83 | 1.00 | 0.44 |
| g1pos-temporal | 0.000 | 0.61 | −0.33 | 0.83 | 0.90 | 0.58 |

prone-structure, observed: a near-binary solid (std_s 0.30 at mean 0.23) arranged in
lateral domains ~7–14 cells across; activity is static (temporal 0.05) and fully re-forms
from the material after reset (memory_h 1.0): the material holds the pattern. At
128×128×24 (t = 300) it keeps the texture (structure 0.49, λ 6.7, mean_s 0.31, std 0.44,
31% of voxels > 0.5, light reaches the floor at 0.31). At 128³ the same rule builds a
textured canopy in the top ~40 layers (k=118: mean_s 0.31, std 0.45, lateral λ ≈ 10)
over nearly empty depth (mean_s ≈ 0.01–0.02 below k ≈ 85); whole-volume structure 0.52,
mass drift −1e-6 (`results/evolve-best-128.json`, captures in `outputs/d/`).
g1neg-temporal and g1pos-temporal have grid-scale (λ ≈ 1.5) salt-and-pepper solid
with broadband activity; g1pos-temporal's activity does not re-form after reset (memory_h −0.33).

## Limits

32×32×24, t = 100 per evaluation: slow instabilities (most Turing-classified rules) cannot
show. One world seed in search (seed 2 rechecks only the recipes). dt 0.1 fixed; one
Physics per run; α_s 0.5 (g1 runs) vs 0.05 (prone/lowmass) are different light regimes.
Budget ≈ 1350 evaluations total, ~5 s each at 4 concurrent Metal worlds (no WorldBatch).
Structure measures lateral texture only; recovery is phase-free; temporal entropy
cannot tell chaos from noise. These are machine measurements of a field, not claims of life.
