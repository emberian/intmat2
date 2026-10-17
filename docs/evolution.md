# Evolution: objectives, search, and what it found

Code: `src/evolve.jl`, `bin/evolve.jl` (experiments and post-hoc modes `--correlate --hv
--tie --recipe --eval-recipes --track`), `test/evolve.jl`. Archives:
`results/evolve-<run>-<mode>.jsonl`, one line per evaluated rule (θ, search coordinates,
σ, parents, seeds, physics incl. its dt, config, objectives, diagnostics, wall time).
Step used: g1neg, g1pos, prone, lowmass **one-ring** (before 69b4888); tied, tiedbase,
life-* **two-shell**. No weighted sum anywhere in selection.

## Part I: lattice-era objectives (32×32×24, fixed world seed, t = 50–100)

One run per rule plus three continuations from its end state: control, activity reset
(`h ←` 0.01·noise), wound (sphere of radius N/6 at the most solid layer: `s → 0`, mass to
`n`, `h` reset). Larger is "more of it".

| objective | definition | range |
|---|---|---|
| persistence | last-quarter mean of `exp(−d/0.05)`, d = distance of `mean_s` from [0.05, 0.6], × texture not collapsing | [0,1] |
| structure | lateral (per-layer DFT) power share at 2π/16 ≤ \|k\| ≤ 2π/4 above white noise's share, × `min(1, rms/0.05)` | [0,1] |
| temporal | normalized spectral entropy of 16 voxels' `h` (last half), × `min(1, std/0.02)`; fixed point 0 | [0,1] |
| memory_h (memory_s) | correlation of `h` (`s`) between control and activity-reset continuations | [−1,1] |
| recovery | `1 − (|Δmean| + |Δstd|)/(mean_c + std_c)` of `s` in the wound vs control (phase-free) | [−1,1] |
| light | `4x(1−x)`, x = lit fraction (`L > 0.1 L0`) | [0,1] |

Search: (μ+λ) NSGA-II (pop 32), binary tournament on rank and crowding, block crossover,
self-adaptive Gaussian step in block-normalized z = θ ./ scale; `--mode random` spends the
same budget on fresh prior draws (generation 0 shared). memory_s ≈ 1 almost always.

**Explicit-Euler check.** 10% (g1 −0.5) to 35% (g1 +0.5) of `random_rule` rules need dt
< 0.1 (`Analysis.stable_dt`) and they game temporal (median 0.26 vs 0.05). Such rules are
now infeasible; older archives are re-scored post hoc.

| run (physics; step) | evals/mode | HV random (dt ok) | HV evolved (dt ok) |
|---|---|---|---|
| g1neg: g1 −0.5, α_s 0.5, α_p 0.1, n0 1, dt 0.1; one-ring | 192 | 0.0137 | **0.0406** |
| g1pos: g1 +0.5, same | 192 | 0.0024 | 0.0011 (raw 0.0123, all dt-unstable) |
| prone: g1 −0.5, α_s 0.05, n0 0.5, `turing_prone_rule` prior | 160 | 0.0107 | 0.0059 |
| lowmass: same physics, Gaussian prior | 160 | 0.0031 | — |
| tied: same, dt 0.05, `tied_rule` prior; two-shell | 160 | 0.0913 | 0.1132 |
| tiedbase: same step, Gaussian one-ring prior | 160 | 0.0041 | — |

- Priors from analysis dominate search: best structure 0.041 (Gaussian) → 0.395 (prone)
  → 0.994 (tied). Flat Gaussian mutation breaks those priors: in the tied run the fitted
  tie residual |Wnb2 + βWnb|/|Wnb2| went 0 → 0.77, sym(Wnb) lost PSD and median
  structure fell 0.32 → 0 (`results/evolve-tied-tie.json`) — a mutation artifact, fixed below.
- Trade-offs: temporal ↔ memory_h −0.35, structure ↔ recovery −0.31, persistence ↔ recovery +0.3/+0.5.
- Atlas vs structure: one-ring Turing predictions do not correlate (Spearman −0.12 … −0.23;
  growth rates ~2e-5 per unit time cannot act within t = 100). Two-shell they do: Turing-
  classified rules mean structure 0.46 vs 0.21 stable, and measured/predicted wavelength
  has quartiles 0.98 / 1.02 / 1.06 over 100 patterned rules.

Lattice recipes (`recipes/evolved-*.toml`, explicit θ; seed-2 recheck in parentheses):
tied-structure 0.994 (0.995) and tied-balanced 0.93 (0.93) need the two-shell step;
prone-structure 0.395 (0.388), prone-nsga-structure, g1neg-balanced, g1neg-temporal,
g1pos-temporal are one-ring. Observed: tied-structure is a lateral solid lattice of period
5.3–5.7 cells (predicted 5.32) whose static activity re-forms from the material (memory_h
0.98); at 128³ the same period runs through all 128 layers (structure 0.987,
`results/evolve-best-128.json`). prone-structure is a near-binary solid in 7–14-cell domains.

## Part II: regions (cells, splitting) as objectives

**Manifold search.** `TiedSpace` searches `(Wself, P, log nb1, β, Win, b, vdep, vero,
vpig[, D])` with `Wnb = nb1·PPᵀ/C`, `Wnb2 = −β·Wnb + nb1·D/C`: mutation stays on the
PSD tied manifold; β clamped to [0.2, 0.6]; prior nb1 log-uniform [1, 16], β ∈ [0.28, 0.5].
`untied` lets mutation move the small free matrix D (initially 0) so selection decides.
**Per-rule dt** = 0.8·`stable_dt`, rounded down to 0.1·2⁻ᵏ, floor 0.004; rules with equal
dt share a `WorldBatch`, and every rule runs the same physical time (t = 100 + 25).

**Regions.** Captures of `s` every 2 time units; threshold at the midpoint of the 10th and
90th percentiles; 6-connected components (periodic x,y; ≥ 8 voxels); tracked over the
second half by voxel overlap. Events: birth (no predecessor), death (no successor),
split / merge (k ≥ 2 successors / predecessors count k−1).

| objective | definition |
|---|---|
| population | `4x(1−x)`, x = clamp(mean count / (volume/8³), 0, 1) |
| localization | `(1 − PR)·min(1, rms/0.05)`, PR the participation ratio of `(s − layer mean)²` |
| turnover | `4y(1−y)`, y = clamp(events per region per unit time / 0.2, 0, 1) (peak at 0.1) |
| lifetime_spread | normalized entropy of identity lifetimes in log2 bins |

Selection: {localization, population, turnover, temporal, recovery}. Test: two blobs joined
by a bridge give exactly one merge (and the reverse one split).

| run (g1 −0.5, α_s 0.05, α_p 0.1) | evals | HV | best turnover |
|---|---|---|---|
| n0 0.5: random (tied prior) / NSGA tied / NSGA untied | 224 each | 0.085 / 0.085 / 0.099 | 1.00 / 1.00 / 1.00 |
| n0 0.3: random / NSGA tied / NSGA untied (cut at 128 by its timeout) | 224/224/128 | 0.075 / 0.072 / 0.071 | 1.00 each |

Turnover > 0 is common (about half of the population-in-band rules exceed 0.5); seven
generations of NSGA did not beat the tied prior except with the untied option at n0 0.5.
Untied: the tie residual stays ≈ 0.06 (mutation-sized) for 6 generations, β 0.40 → 0.41–
0.44, sym(Wnb) stays PSD and |Wnb| doubles (14.5 → 26–30): selection pushed coupling up,
not off the manifold (`results/evolve-life-tie.json`).
Tensions (Pearson, life runs): memory_h vs localization / structure / temporal −0.4 … −0.55;
temporal vs persistence −0.4 … −0.52; localization tracks structure (+0.83 … +0.88), so
as defined it rewards a contrasty lattice as much as isolated blobs.

**Long runs** (`recipes/evolved-turnover-*.toml`, 128×128×48, t = 300, captures every 2;
series in `results/evolve-track-turnover-*.json`, captures in `outputs/d/`):

| recipe (dt) | count over time | births / deaths / splits / merges (whole run) | second half rate |
|---|---|---|---|
| n05-176 (0.05) | 113 (t 20) → 6 → 447 (t 118) → 79 | 2098 / 1301 / 1667 / 2385 | 0.10 per region per unit time |
| n03-untied-92 (0.025) | 0 → 802 (t 56) → 67 | 628 / 1350 / 2469 / 1680 | 0.06, no births after t ≈ 150 |
| n05-188 (0.0125) | 1 → 530 (t 80) → 980, still rising | 316 / 935 / 2307 / 708 | 0.014, all four kinds present |

What the regions do (observation): in all three, one connected component holds 84–100%
of the above-threshold solid: a sponge-like canopy network, densest under the light. The
"population" is that network plus many small fragments (median 11–15 voxels, centred in
the lower half, z ≈ 6–20 of 48) that pinch off its lower fringe. In n05-176 and
n03-untied-92 this is a transient: fragments bloom, then die or rejoin and the count
decays while mean_s slowly falls (0.27 → 0.20, 0.22 → 0.18). In n05-188 the sheet
fragments after t ≈ 70 and settles to a slowly growing population of ~900–980 small
regions with a few births, deaths, splits and merges every capture (rate 0.014): the
nearest thing to a standing population, but slower than the 32³ score (0.097) implied —
the short evaluation scored the fragmentation transient.

## Limits

Evaluation grids are 32×32×24 for t ≤ 125; turnover measured over t 50–100 rewards
transients (above). Count-based population is satisfied by one giant component plus
dust; a next version should score the size distribution (e.g. penalize the giant
fraction). One world seed during search; one Physics family per run; ~1 hour per
experiment on a shared GPU; seven generations is short. Thresholded regions are a
measurement choice, not organisms. These are machine measurements of a field, not
claims of life.
