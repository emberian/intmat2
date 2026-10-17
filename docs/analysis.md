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

## Time-step bound: `stable_dt`

The Physics bounds (`6 D0 dt ≤ 1`, `dt ≤ τ_h`) are necessary, not sufficient.
`stable_dt(rule, p; M)` bisects for the largest `dt ≤ min(τ_h, 1/(6 D0))` at which
the exact one-step map grows no faster than the continuous system at every
`κ ∈ [0, 12]` (12 = 3D grid scale): `log|μ_max|/dt ≤ 1.5 max(Re λ, 0) + 10⁻³/τ_h`.
For a stable steady state that is spectral radius ≤ 1; for Turing/Hopf it admits
the physical growth only. Two mechanisms set it below the Physics bounds:

- explicit Euler on the activity coupling at large κ: rule seed 1184891871 (C=4,
  D0 0.2) has `stable_dt = 0.266`; at dt 0.5 (both Physics bounds hold) the
  checkerboard grows at 1.30 per unit time and a 16² run goes from 4e-5 to 13 in
  200 steps; at dt 0.266 it reaches 1.4e-4 (Turing growth only). Tested.
- weakly damped oscillatory modes: `|1 + dt λ| ≤ 1` needs `dt ≤ 2|Re λ|/|λ|²`.
  Recipe turing-10 has a mode `−0.0041 − 1.03i` at κ ≈ 10.6 (3D only), so
  `stable_dt = 0.0024`; at dt 0.05 that mode grows at +0.0023 in the stepper while
  the continuous system decays. osc-02 (Hopf, `0.0063 − 0.93i`) grows 4× too fast
  at dt 0.05 (map 0.0255); `stable_dt = 0.0046`.

Recipes at dt 0.05: turing-01…09 and osc-01, 04, 05 pass (`stable_dt` 0.10–1.0);
turing-10, osc-02, osc-03 (0.021) do not. The honest bound for any run is
`stable_dt(rule, p; M)`, not the Physics docstring.

## What predicts the class

`julia -t 8 --project=. bin/prone.jl 2000` → `results/atlas-20261016-prone.json`.
One-vs-rest L2 logistic regression on standardized features over the 19 458
converged atlas points (frozen light). Since this commit `steady_state` also
rejects degenerate saturation (`1 − s* < 1e-9` or `|h*| ≈ 1`), which moves ~3% of
random-rule points to `no_steady_state`. Largest coefficients:

| class (base rate, AUC) | coefficients |
|---|---|
| gridscale (0.149, 0.935) | nb anti-diffusion +1.92, ρ(W_self) −0.55, nb +0.36, C +0.26, M −0.24 |
| stable (0.777, 0.893) | nb anti-diffusion −2.36, ρ(W_self) −0.45, M +0.39, read −0.27, s* −0.22 |
| turing (0.0062, 0.883) | M −2.46, s* +1.07, ‖W_in L-col‖ −0.60, inp +0.58, read +0.56, log τ_photo/τ_dep +0.53, nb anti-diffusion −0.48 |

"nb anti-diffusion" is `8 g* max(0, −min Re eig(S W_nb))`, the κ = 8 growth the
coupling can supply; the stump `≥ 2.1` is 87% gridscale. Turing wants low mass
(`M < 1`: 1.36%), strong read-outs (`read ≥ 4`: 1.29%), erosion off at the steady
state (`v_ero·h* < −6.8`: 3.1%), deposition on (`v_dep·h* ≥ 9.9`: 2.1%), and high
s*. Light input to the rule works against it.

`turing_prone_rule(rng, C, T; nb=0.4, inp=2, light_in=0.25, bias=0.5, read=3)`:
`W_nb = nb·QQᵀ/C` (positive semidefinite: no anti-diffusion), strong `W_in` with a
weak light column, strong read-outs with `v_dep` aligned and `v_ero` anti-aligned
to `tanh(W_in[0.5,0.3,0.1,0.3] + b)`. On 2000 fresh points over the atlas physics
grid: **Turing 9.2% vs 0.45%** for `random_rule` (gridscale 4.8% vs 14.0%, no
steady state 5.8% vs 3.6%).

## Deep slab: layered steady state

`layered_steady_state(rule, p, Nz; M)` marches down from the lit face, each layer in
local balance under the exact optical depth above; `relax_column` then runs the
real `step!` on a `1×1×Nz` column (vertical coupling and transport on) from it.
`julia --project=. bin/layers.jl 32 200000` on the ten Turing recipes (Nz 32,
α_s = α_p = 2, t = 10 000):

- Decoupled layers: Turing only in the top layer (top 3 for turing-08). By layer 3
  L < 0.01 and the layers hold their mass as waste (w ≈ M). Below the top: stable
  in 5 recipes, gridscale in 4 (dark layers with s ≈ 0, where g* = g0 is largest), a
  homogeneous layer in 1.
- Coupled column: the dark layers become a waste reservoir holding 62–89% of the
  column's mass (9 of 10 recipes; turing-09 keeps it as nutrient). Waste diffuses up,
  is regenerated at the top, and the top layer often crusts (s* 0.99–1.0 in
  turing-06, 07, 10). The top layer stays Turing in 2 of 10 (02, 08); it turns
  stable in 7 and homogeneous in 1 (turing-01); in turing-09 the dark layers
  become mixed.
  turing-01: top s 0.245 → 0.740, layer 2 s 0.120 → 0.048, dark w 0.48.

So a deep, opaque slab makes vertical structure first: a lit skin one to three
layers thick over a dark waste store. That store changes the top layer's balance
enough to remove the Turing prediction in most recipes. The predictions are exact
for `N×N×1` worlds; for slabs use thin optics or few layers, or re-derive the class
from the relaxed column.

## Proposed step change: two-scale coupling and ambient light

Lane C's 128³ photographs and the atlas agree: one 6-neighbour ring gives the rule
one spatial scale, so lateral structure is either the checkerboard (the ring's
anti-diffusion at κ = 8) or slow conserved long-wave separation through `s`.
`Pencil2` / `dispersion2` (`src/analysis.jl`) linearize a second shell, and
`bin/atlas2.jl` re-runs the 20 000 atlas points (frozen light, depth 1, horizontal
k grid step π/24) → `results/atlas-20261016-twoscale.json` and `…-tied.json`.

**Shell: the 6 axis voxels at distance 2.** Its symbol is `κ2(k) = 2Σ(1 − cos 2kᵢ)
= κ1(2k)`: separable, the same small-k shape as the ring (`κ2 ≈ 4|k|²`), and zero
at k = π, so it neither creates nor cures the checkerboard. For scalar weights
`w1` (ring) and `w2` (shell) the per-axis growth is `−2(1−c)[w1 + 2w2(1+c)]`,
`c = cos k`, with its maximum at `cos k* = −w1/(4w2)`. With `w1 > 0` (diffusive
ring) and `w2 = −β w1`, that is `cos k* = 1/(4β)`: the wavelength is set by β alone
(β 0.3 → 10.7 cells, 0.4 → 7.0, 0.5 → 6.0 on an axis; diagonal modes are √2 shorter)
and the peak strength is `w1 (4β−1)²/(4β)` per axis. The 18-voxel shell
(`|d|² ∈ {2, 4}`) adds face diagonals with symbol `4Σ_{i<j}(1 − cos kᵢ cos kⱼ)`:
non-separable, no closed form for k*, and 12 more loads per voxel.

Class fractions (20 000 points; Turing+wave wavelength histogram; share of all
points that are Turing or wave with λ_wave in 4–16):

| coupling | stable | turing | wave | gridscale | λ 2–4 | 4–8 | 8–16 | 16+ | 4–16 share | median Turing λ |
|---|---|---|---|---|---|---|---|---|---|---|
| ring (random W_nb), as now | 0.787 | 0.0046 | 0.0000 | 0.120 | 26 | 41 | 13 | 12 | 0.0027 | 1.1e-4 |
| + independent W_nb2 = −0.2·QQᵀ/C | 0.514 | 0.221 | 0.037 | 0.140 | 5132 | 16 | 4 | 2 | 0.0010 | 0.64 |
| + independent W_nb2, nb2 0.8 | 0.210 | 0.630 | 0.030 | 0.042 | 13194 | 0 | 2 | 0 | 0.0001 | 3.4 |
| box-filtered ring (1+2cos k)/3 | 0.890 | 0.0079 | 0.0009 | 0.013 | 135 | 32 | 6 | 4 | 0.0019 | 3.5e-3 |
| binomial-filtered ring (1+cos k)/2 | 0.893 | 0.0076 | 0.0009 | 0.011 | 135 | 20 | 6 | 8 | 0.0013 | 1.4e-3 |
| PSD ring nb·PPᵀ/C | 0.896 | 0.0097 | 0.0001 | 0.0055 | 37 | 88 | 36 | 35 | 0.0063 | 1.7e-4 |
| tied: W_nb = 1·PPᵀ/C, W_nb2 = −0.5 W_nb | 0.698 | 0.174 | 0.037 | 0.0020 | 133 | 4086 | 4 | 2 | 0.205 | 0.30 |
| tied: W_nb = 2·PPᵀ/C, β 0.5 | 0.471 | 0.382 | 0.057 | 0.0010 | 76 | 8707 | 2 | 0 | 0.436 | 0.82 |
| tied: W_nb = 4·PPᵀ/C, β 0.4 | 0.504 | 0.352 | 0.055 | 0.0008 | 31 | 8108 | 2 | 0 | 0.406 | 0.70 |
| tied: W_nb = 4·PPᵀ/C, β 0.5 | 0.285 | 0.580 | 0.046 | 0.0003 | 28 | 12487 | 1 | 0 | 0.624 | 1.9 |
| tied: W_nb = 4·PPᵀ/C, β 0.3 | 0.884 | 0.022 | 0.004 | 0.0013 | 41 | 287 | 198 | 2 | 0.024 | 1.2e-3 |

(The ring row has 12.0% gridscale rather than the atlas's 14.6% because `steady_state`
now rejects saturated points: 3.6% no steady state here.)

- An *independent* inhibitory `W_nb2` raises Turing to 22–63%, but every pattern sits
  at k = (π/2, π/2), λ ≈ 2.8: a radius-2 checkerboard. It does not lower
  gridscale, because the random ring's anti-diffusion is untouched.
- *Filtering the ring alone* cuts gridscale from 12% to 1.1–1.3% (box and binomial
  alike: the box symbol is negative for k > 2π/3 but small there). Turing barely moves
  (0.46% → 0.8%) and stays short or slow. A PSD ring alone does the same (0.55%).
- *Tied Mexican hat* (`W_nb` PSD, `W_nb2 = −β W_nb`) gives what the diagnosis asked for:
  Turing+wave at 4–16 cells in 20–62% of points, gridscale ≤ 0.2%, growth on the
  activity clock (median 0.3–1.9 per unit time, against 1e-4 for today's Turing).
  Within the atlas's k-resolution, β ≥ 0.4 lands in 4–8 cells; reaching 8–16 needs
  β ≈ 0.3, where growth falls as (4β−1)², so it needs a larger `W_nb`.

**Recommended per-voxel formula** (for voxel x, neighbours with the slab's no-flux
walls: z-offsets that leave `[1, Nz]` are omitted, as for the ring):

```
m1 = Σ_{d ∈ ±e_x, ±e_y, ±e_z}   g(s_x, s_{x+d}) (h_{x+d} − h_x)      # existing ring
m2 = Σ_{d ∈ ±2e_x, ±2e_y, ±2e_z} g(s_x, s_{x+d}) (h_{x+d} − h_x)     # new: axis shell at distance 2
u  = W_self h + W_nb m1 + W_nb2 m2 + W_in [n, s, w, L] + b
L  = L_ambient + L0 exp(−(τ_above + a/2))                           # a = α_s s + α_p σ(v_pig·h)
```

Symbols at the uniform state: `m̂ = −g*(κ1(k) W_nb + κ2(k) W_nb2) δh`,
`κ1 = 2Σ(1 − cos kᵢ)`, `κ2 = 2Σ(1 − cos 2kᵢ)`; `L_ambient` adds to `L*` and does not
enter δL. Conductance across distance 2 uses the two end voxels only. `W_nb2` is a
full learnable C×C matrix (`3C² + 8C` parameters), `L_ambient` a Physics field
(default 0 keeps today's step bit-identical when `W_nb2 = 0`). Sample new rules with
`W_nb = nb1·PPᵀ/C`, nb1 ∈ [1, 4], `W_nb2 = −β W_nb`, β ∈ [0.3, 0.5] (wavelength ≈
2π/acos(1/(4β)) on an axis); evolution can then break the tie. Cost: six more
neighbour loads of `h` and `s` per voxel. `stable_dt` does not yet include `W_nb2`;
with activity growth of 1–6 per unit time, recheck dt against the new stepmap.

## Tied two-shell rules (adopted)

`Rule.Wnb2` and `Physics.L_ambient` are now part of the step (69b4888) and of the
main analysis path. `jacobian`/`stepmap` take the shell symbol `κ2 = κ(2k)`
(keyword `κ2`). `Pencil` is affine in both symbols. `dispersion` switches to a
horizontal wavevector scan (`0 ≤ ky ≤ kx ≤ π`, step π/48, refined) whenever
`W_nb2 ≠ 0`, and then reports `wavelength = 2π/|k*|`. `stable_dt` scans the 3D
wavevector grid (step π/16) for two-shell rules. `steady_state` uses `p.L_ambient`.
One-ring rules take exactly the old path.

The fastest mode of a tied rule lies on the diagonal (`kx = ky`, each at
`cos k = 1/(4β)`), so `λ_wave` is √2 shorter than the axis formula. That gives
`λ_wave` 4.2 at β 0.5, 5.0 at β 0.4, 6.6 at β 0.32 and 8.5 at β 0.29. Growth near
β → ¼ scales as `nb1 (4β−1)²`, so 6–12 cells needs nb1 12–16.

`stable_dt` survey (`bin/tied.jl` → `results/tied-20261016.json`): C = 8, default
Physics (α_s 0.05, α_p 0.1), M 0.5, 24 seeds per cell, `tied_rule` default scales.

| nb1 | β | Turing | median stable dt [min, max] | Turing λ_wave | median λ |
|---|---|---|---|---|---|
| 1 | 0.3 / 0.4 / 0.5 | 0 / 0.04 / 0.25 | 0.185 [0.007, 0.290] | — / 5.0 / 4.2 | — / 0.060 / 0.18 |
| 2 | 0.3 / 0.4 / 0.5 | 0 / 0.21 / 0.67 | 0.097 [0.007, 0.155] | — / 5.0 / 4.2 | — / 0.17 / 0.73 |
| 4 | 0.3 / 0.4 / 0.5 | 0 / 0.58 / 0.88 | 0.050 [0.015, 0.081] | — / 5.0 / 4.2 | — / 0.55 / 2.3 |

Stable dt falls as ≈ 0.2/nb1 (explicit Euler on the coupling at 3D grid scale) and
hardly depends on β. The package default dt 0.05 is unstable for nb1 ≳ 4.

Recipes `recipes/tied-20261016-01…10.toml` (C 8, `theta` = `vec(rule)` with
`W_nb2`; N 128, Nz 48, n0 0.5, α_s 0.05, α_p 0.1; dt = min(0.05, 0.8 stable_dt);
steps = 8 e-folds). All are Turing at depths 1, 24 and 48 with the same wavelength.
They came from 600 candidates over nb1 {6,8,12,16} × β {0.29…0.32}, of which 51
qualified; up to three were picked per β, fastest first:

| recipe | nb1 | β | λ_max | λ_wave | stable dt | dt | steps |
|---|---|---|---|---|---|---|---|
| 01 | 16 | 0.29 | 0.226 | 8.5 | 0.0091 | 0.0073 | 4836 |
| 02 | 16 | 0.30 | 0.557 | 7.5 | 0.0086 | 0.0069 | 2087 |
| 03 | 16 | 0.31 | 0.723 | 7.1 | 0.0108 | 0.0086 | 1281 |
| 04 | 16 | 0.32 | 1.746 | 6.6 | 0.0069 | 0.0055 | 833 |
| 05 | 16 | 0.30 | 0.382 | 7.5 | 0.0122 | 0.0098 | 2150 |
| 06 | 16 | 0.31 | 0.340 | 7.1 | 0.0099 | 0.0080 | 2958 |
| 07 | 16 | 0.32 | 1.192 | 6.6 | 0.0091 | 0.0073 | 920 |
| 08 | 16 | 0.30 | 0.265 | 7.5 | 0.0112 | 0.0089 | 3385 |
| 09 | 16 | 0.31 | 0.322 | 7.1 | 0.0138 | 0.0111 | 2246 |
| 10 | 12 | 0.32 | 1.086 | 6.6 | 0.0102 | 0.0081 | 907 |

**Validation against the two-shell `step!`** (tied-03, CPU Float64, one layer,
`light=:local`):
- One step on a single cos mode matches the two-shell `stepmap` to 1.2–1.7e-10
  relative. The one-ring map misses it by 1–8%.
- 64×64×1 world from the steady state plus noise 1e-12, rates over t ∈ [5/λ, 10/λ]
  (1615 steps). The fastest measured mode is the predicted lattice mode
  (λ_wave 7.54): measured rate 0.6942 against predicted 0.6912. The continuous
  optimum is 0.7162 at λ_wave 7.05, off the lattice. Power in both `s` and `h₁`
  peaks at λ_wave 7.54. Median |Δrate| over the 16 fastest modes is 2.4e-3 (1.3e-2
  over the shorter window [2/λ, 7/λ]: transients between nearby eigenvalues).
- Tested on a 32² world in `test/analysis.jl`.

## Stencil anisotropy of the tied coupling

Lane C's tied runs lock to the axes because of the stencil, not the material. In
the current axis-only ring + axis-only shell, growth at fixed `|k*|` peaks on the
in-plane diagonal (φ = 45°) and is negative along the axes. Two diagonal modes
`(k,k)` and `(k,−k)` sum to `cos kx · cos ky`: an axis-aligned square weave. In 3D
the oblique mode (θ 60°, φ 45°) grows faster than any horizontal one (tied-10: 1.71
against 1.09), so a slab prefers oblique structure.

19-point symbols (6 axis neighbours at weight 1/3, 12 edge diagonals `(±1,±1,0)` at
weight 1/6; same small-k normalization as the ring, `κ ≈ |k|²`):

```
κ_iso(k)  = (2/3) Σᵢ (1 − cos kᵢ) + (2/3) Σ_{i<j} (1 − cos kᵢ cos kⱼ)
κ2_iso(k) = κ_iso(2k)            # the same stencil dilated by 2 (offsets (±2,0,0), (±2,±2,0), …)
m̂ = −g* (κ_ring(k) W_nb + κ_shell(k) W_nb2) δh,   transport keeps κ = 2Σ(1 − cos kᵢ)
```

(max − min)/mean of λ at fixed `|k*|`, over in-plane φ and 3D polar angles θ
(`aniso*.jl` probes, frozen light; same β as the recipe unless stated):

| stencil | tied-10 (nb1 12, β 0.32) | tied-03 (nb1 16, β 0.31) |
|---|---|---|
| (a) axis ring + axis shell (current) | λ_wave 6.6 at 45°, λ* 1.09; in-plane 309%, 3D 456% | 7.0 at 45°, 0.72; 326%, 495% |
| (b) 19-pt ring + 19-pt shell | λ_wave 9.0 at 45°, λ* 0.19; 23%, 3D 23% | 9.6, 0.018; 145% (near onset) |
| (c) axis ring + 19-pt shell | λ_wave 9.3 at 0°, λ* 0.15; 74%, 3D 120% | 9.9, 0.004; marginal |
| (b) with β retuned to 0.40 | λ_wave 6.6, λ* 3.0; 11.7%, 3D 11.7% | 6.6, 3.0; 11.0% |
| (b), β 0.50 / 0.75 | 5.5 / 4.4 cells; 14.8% / 24.0% | 14.3% / 23.7% |
| 19-pt ring + Gaussian shell σ 1, radius 3 (122 points) | λ_wave 6.4; 11.0%, 3D 13.0% | 6.9; 11.7%, 14.0% |

k* does shift. The 19-point shell is "softer" at the same β (longer wavelength,
much weaker growth), so β has to be retuned: β 0.40 with (b) restores the 6.6-cell
wavelength with three times the growth of the current β 0.32.

**Build (b), with β ≈ 0.4.** It removes the gross anisotropy: the absolute spread
falls 30–50×, and the 3D spread equals the in-plane one, so oblique modes no longer
win. (c) is worse than (b): an isotropic shell over an anisotropic ring leaves the
ring's k⁴ anisotropy uncancelled and turns the preference to the axes. None of the
stencils tested reaches 5%. With a tied coupling the k² terms of ring and shell
nearly cancel (`w1 + 4w2 ≈ 0` near β = ¼), and the pattern lives in the k⁴ and k⁶
terms, where any radius-2 lattice stencil is anisotropic at `|2k| ≈ 1.3–1.9`. The
residual 11% favours the diagonal; a Gaussian shell over a radius-3 ball does no
better. Getting under 5% would need shell weights fitted for isotropy over the band
`|k| ∈ [0.5, 1.2]`, or a spectral coupling. Neither is worth building before (b) has
been seen.

## Isotropic stencil (`Physics.stencil = 1`)

The analysis honours `p.stencil`. The ring symbol is `kappa_iso(k)` and the shell is
`kappa_iso(2k)` (formulas above); transport keeps `κ = 2Σ(1 − cos kᵢ)`. `Pencil` now
carries separate ring, shell and transport matrices, and `dispersion` and
`stable_dt` scan wavevectors whenever `stencil = 1` or `W_nb2 ≠ 0`.

**One-layer worlds see a different stencil.** The step keeps the 1/3, 1/6 weights
and drops neighbours cut by the z walls. For an in-plane mode, the dropped
`(±1, 0, ±1)`-type edges contribute `1 − cos kx`, not 0. So an `N×N×1` world (or a
wall layer) runs the planar symbol
`kappa_iso_planar(k) = (2/3)[(1−cos kx) + (1−cos ky) + (1−cos kx cos ky)]`
(`≈ (2/3)|k|²`). That symbol is anisotropic: axis-to-diagonal 2:1, where the
isotropic 2D 9-point stencil needs 4:1. `Pencil(...; planar=true)` uses it. A slab's
interior layers see the full 3D `kappa_iso`.

Recipes `recipes/iso-20261016-01…06.toml` (`bin/iso.jl` → `results/iso-20261016.json`):
tied rules with stencil 1, C 8, n0 0.5, α 0.05/0.1, N 128, Nz 48, dt = 0.8 stable dt,
8 e-folds, explicit theta. Turing at depths 1/24/48 (bulk symbol); 13 of 400
candidates qualified, and none at β 0.38. Spread = (max − min)/mean of λ over
in-plane φ at `|k*|` (3D polar spread is identical):

| recipe | nb1 | β | λ_max | λ_wave | stable dt | dt | steps | spread |
|---|---|---|---|---|---|---|---|---|
| 01 | 6 | 0.40 | 0.799 | 6.62 | 0.075 | 0.060 | 168 | 12.6% |
| 02 | 6 | 0.42 | 1.175 | 6.31 | 0.053 | 0.043 | 160 | 16.2% |
| 03 | 6 | 0.40 | 0.523 | 6.62 | 0.073 | 0.058 | 264 | 18.6% |
| 04 | 6 | 0.42 | 1.163 | 6.31 | 0.057 | 0.045 | 152 | 17.8% |
| 05 | 6 | 0.40 | 0.503 | 6.62 | 0.057 | 0.046 | 347 | 30.3% |
| 06 | 6 | 0.42 | 0.867 | 6.31 | 0.056 | 0.045 | 206 | 24.3% |

The isotropic stencil also relaxes dt about sixfold against the axis-stencil tied
recipes (stable dt 0.05–0.075 against 0.007–0.014), because its weight is spread
over more neighbours.

**Validation** (iso-01, CPU Float64, 64×64×1, `light=:local`, planar symbol):
- One step on a diagonal cos mode matches the planar `stepmap` to 3.6e-10. The
  axis-stencil map misses it by 10%, and the bulk `kappa_iso` map by 1.4%.
- Growth over t ∈ [5/λ, 10/λ] (463 steps): the fastest measured mode is a predicted
  fastest lattice mode (λ_wave 7.11, on an axis), rate 0.3575 against 0.3575. The
  median |Δrate| over the 44 fastest modes is 1.9e-4. `s` power peaks at
  λ_wave 7.11.
- Per-angle spread on the lattice annulus `|k*| ± 0.06` (60 modes, 0–90°): predicted
  40.5%, measured 40.5%. That is the one-layer stencil. For the same rule the bulk
  stencil gives λ* 0.80 at 6.62 cells with a 12.6% spread, and the axis stencil
  gives >300%.
- Tested on 32² in `test/analysis.jl`.

For thin worlds (`Nz` of 1–2) the 19-point stencil is not isotropic. If they matter,
weight the in-plane stencil 4:1 when z neighbours are missing.
