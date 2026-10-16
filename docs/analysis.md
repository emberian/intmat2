# Analysis: steady state, dispersion, atlas

`src/analysis.jl` (`Intmat2.Analysis`), tests in `test/analysis.jl`, atlas driver
`bin/atlas.jl`. State order is `z = [h₁…h_C, n, s, w]`.

## Homogeneous steady state

`steady_state(rule, p; M, depth=1, L=nothing)` solves the spatially uniform fixed
point at mass per voxel `M = n + s + w`:

- material, for fixed `h`: `k_dep A_d n(1−s) = k_ero A_e s = k_photo L w`
  (`A_d = softplus(v_dep·h)`, `A_e = softplus(v_ero·h)`). With `n, w` eliminated,
  `s + e s/(k_photo L(s)) + e s/(a(1−s)) = M` is strictly increasing on `[0,1)`, so
  `s*` is unique and found by bisection;
- activity: `h = tanh(W_self h + W_in [n,s,w,L](h) + b)`, damped Newton from
  `h = 0` (falls back to relaxation `h ← h + 0.2 G` plus Newton restarts when Newton
  stalls). Several fixed points may exist; this is the one reached from `h = 0`.

**Light (choice).** A uniform slab is not uniform in depth, so the steady state is
taken at layer `depth` (1 = top) of a slab whose layers above are identical:
`L* = L0 exp(−(depth−½)(α_s s* + α_p σ(v_pig·h*)))`, solved self-consistently with
`s*`. It is exactly the steady state of an `N×N×1` world and the top layer of a
deep slab before structure forms; deeper layers are a different `depth`. `L=` fixes
`L*` instead. `clamped` flags a steady state at which a `min` in the material
update binds at the given `dt` (0.7% of atlas points); there the stepper's fixed
point differs.

## Linearization

For a horizontal mode `exp(i k·x)`, every k enters through `κ = 2Σ(1 − cos kᵢ)`:

- coupling `m → −g* κ δh`, `g* = clamp(g0 + g1 s*, 0, 1)`. The conductance
  perturbation `∂g/∂s · δs` multiplies `h_y − h_x = 0` at the uniform state and
  vanishes at linear order;
- transport `→ −D(s*) κ` on `n, w`; `∂D/∂s · δs` multiplies `∇n = 0`, vanishes;
- `δu = S(W_self − g* κ W_nb) δh + S W_in δ[n,s,w,L]`, `S = diag(1 − h*²)`;
- fluxes: `∂dep/∂h = k_dep σ(v_dep·h) n(1−s) v_dep`, `∂dep/∂n = k_dep A_d (1−s)`,
  `∂dep/∂s = −k_dep A_d n`, `∂ero/∂h = k_ero σ(v_ero·h) s v_ero`,
  `∂ero/∂s = k_ero A_e`, `∂photo/∂w = k_photo L`, `∂photo/∂L = k_photo w`.

`jacobian(rule, p, ss, κ)` is the continuous-time `(C+3)×(C+3)` Jacobian;
`stepmap(rule, p, ss, κ)` is the exact linearization of one `step!` (activity, then
transport, then material on the new `h` and transported `n, w`, with last step's
`L`). Both are affine in κ (`Pencil` caches the two matrices); `stepmap = I + dt J
+ O(dt²)` is tested. The mass mode (`δn+δs+δw`) has eigenvalue exactly 0 at κ = 0;
it is removed by restricting to the invariant subspace `δn+δs+δw = 0`.

**Three approximations.**
1. *Light local.* `light=:frozen` (default) sets `δL = 0`. `light=:local` lets `L`
   respond to the voxel's own `δs`, `δpig` through the depth factor
   (`δL = −L*(depth−½)(α_s δs + α_p σ'(·) v_pig·δh)`), which assumes the column
   above moves with the voxel. For `N×N×1` worlds `:local` is exact, which is why
   the validation uses it. Atlas: 96.7% of points get the same class under both.
2. *Horizontal k.* Modes are uniform in z (`k_z = 0`), so the z-coupling and
   z-transport drop out. A slab's vertical structure (light gradient, walls) is
   not in the dispersion relation.
3. *Steady state at a chosen depth.* All atlas results are for `depth = 1`.

`dispersion(...)` returns `Re λ_max(κ)` on a κ grid (refined at the max), κ*,
the axis wavelength `2π/(2 asin(√κ*/2))`, `Im λ` at κ*, the unstable band
`[band_lo, band_hi]`, and the discrete growth `log|μ_max|/dt`. `classify`:
`stable`; `turing` (κ = 0 stable, finite κ* ≤ 4 unstable, real); `wave` (same,
complex); `gridscale` (κ* > 4: no axis mode longer than 2 voxels);
`hopf` / `homogeneous` (κ = 0 unstable, complex / real, and fastest);
`mixed` (κ = 0 unstable, a finite κ faster).

## Validation against `step!`

CPU Float64, `32×32×1`, steady state plus zero-mean noise `1e-6` in every field,
`light=:local`. Per-mode growth = `log(a(t2)/a(t1))/(t2−t1)` of the all-field
DFT amplitude, compared with `log|μ_max(κ)|/dt` on the lattice.

| case (C=4) | predicted | stepper: fastest mode | median \|Δrate\| | s-power peak |
|---|---|---|---|---|
| Turing, conserved long-wave, seed 838841668, dt 1/6, t 50→150 | κ* 1.530 (λ_wave 4.7), λ 0.03703, map 0.03784; lattice best κ 1.572 at 0.03785 | κ 1.572 at 0.03785 | 5.8e-6 (767 modes) | κ 1.648 |
| Turing, band 0.83–3.4, seed 1184891871, dt 0.2, t 100→400 | κ* 2.019 (λ_wave 4.0), λ 0.00963, map 0.00997; lattice best κ 2.038 at 0.009968 | κ 2.000 at 0.009974 | 4.5e-4 (1023 modes) | κ 2.038 |
| stable, seed 1443778830, dt 1/6, t 20→80 | slowest lattice mode κ 0.0384 at −0.00555 | — | 7.4e-6 (36 modes) | — |

Stable case: the norm decays at −0.157 over t ∈ [0, 20] and −0.0138 over
[20, 80], approaching the slowest mode (−0.00555, the long-wave mass mode) from
below, as a sum of modes must. Measured norm ratio over [20, 80] is 0.4122;
predicted from the measured spectrum at t = 20 and the predicted rates, 0.4125.
Also tested: the steady state is a fixed point of `step!` to 1e-11 over 200 steps;
one `step!` on a single cos mode matches `stepmap · v` to 1e-5 relative.

## Atlas

`julia -t 8 --project=. bin/atlas.jl 20000 20261016` → `results/atlas-20261016.json`
(rows in `columns` order; 90 s on the M2 Max). C ∈ {4, 8}; rule scales self
{0.5,1,1.5,2}, nb {0,0.1,0.2,0.4,0.8}, inp {0.5,1,2,4}, bias {0.1,0.5}, read
{0.5,1,2,4}; k_dep {0.1,0.3,1,3}, τ_ero/τ_dep {3,10,30}, k_photo {0.05,0.2,1},
D0 {0.05,0.2,1}, g1/g0 {−1,0,1} (g0 = 0.5), M {0.5,1,2};
`dt = min(0.05, 1/(6 D0), τ_h)`; depth 1.

| class | frozen light | local light |
|---|---|---|
| stable | 0.7796 | 0.7794 |
| gridscale | 0.1456 | 0.1459 |
| mixed | 0.0350 | 0.0355 |
| homogeneous | 0.0243 | 0.0238 |
| hopf | 0.0076 | 0.0082 |
| turing | 0.0062 | 0.0055 |
| no steady state | 0.0017 | 0.0017 |
| wave | 0.0001 | 0.0001 |

- `gridscale` is `W_nb` anti-diffusion: `−g*κ W_nb` has a growing eigenvalue
  whenever `W_nb` has one with negative real part, fastest at κ = 8. Its share
  rises with `nb` (1.3% at nb 0 → 41% at nb 0.8).
- 110 of 123 Turing points are *conserved long-wave*: the band reaches κ → 0
  because mass conservation pins the mass branch to λ(0) = 0, so the instability
  is a phase separation of mass with a finite fastest κ*. Turing is commoner with
  large read-out scale (0.14% at read 0.5 → 1.3% at read 4) and low mass
  (1.35% at M 0.5 → 0.12% at M 2).
- 57 points are continuous-stable but the one-step map is unstable at their dt
  (`map_rate_max > 0`); the stepper would show a numerical instability there.
- Turing growth rates are small (1e-3 to 1e-2 per unit time): 5 e-folds take
  9 000–90 000 steps at dt 0.05. A quiet short run of these recipes is expected.

Recipes (`recipes/atlas-20261016-*.toml`; `random_rule(MersenneTwister(seed), C,
Float64; rule_scales...)`, `n0 = M`, `steps` = 5 e-folds; frozen-light prediction,
the other model's class in `other_light_model_class`):

| recipe | class | λ_max | κ* | λ_wave | Im λ |
|---|---|---|---|---|---|
| turing-01 | turing | 0.0112 | 1.249 | 5.3 | 0 |
| turing-02 | turing | 0.00556 | 1.652 | 4.5 | 0 |
| turing-03 | turing | 0.00348 | 1.923 | 4.1 | 0 |
| turing-04 | turing | 0.00251 | 0.160 | 15.6 | 0 |
| turing-05 | turing | 0.00226 | 0.606 | 7.9 | 0 |
| turing-06 | turing | 0.00168 | 1.858 | 4.2 | 0 |
| turing-07 | turing | 0.00147 | 0.494 | 8.7 | 0 |
| turing-08 | turing | 0.00136 | 1.096 | 5.7 | 0 |
| turing-09 | turing | 0.00135 | 0.985 | 6.1 | 0 |
| turing-10 | turing | 0.00108 | 1.867 | 4.2 | 0 |
| osc-01 | wave (local: mixed) | 0.00544 | 0.012 | 58 | −0.032 |
| osc-02 | hopf | 0.00628 | 0 | — | −0.928 |
| osc-03 | hopf | 0.00656 | 0 | — | −0.235 |
| osc-04 | hopf | 0.0110 | 0 | — | −0.105 |
| osc-05 | hopf | 0.0120 | 0 | — | −0.373 |

All recipes use α_s = α_p = 2, so only the top few layers of a deep slab are lit.
In a thick slab the prediction applies to the top layer; rerun `steady_state` /
`dispersion` with the recipe's `depth` or optics before trusting it deeper.
