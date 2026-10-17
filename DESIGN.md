# Interior 2 — design

**One field. The material is the organism.**

Interior 1 (`emberian/intmat`, Oct 14–16 2026) built Fraser Price's model: discrete
agents with inherited brains, energy ledgers, contacts, construction credit. Every fix
was more accounting on the same agent-centric skeleton, and the art was Blender
isosurfaces of saved scalar fields. The original brief asked for something else:
*activity that constitutes and remodels the material, where the organism/environment
split is a parameter, not a data structure.* This repo builds that and nothing else.

## The substrate

A 3D grid, periodic in x and y, a slab in z (light enters from the top, transport
has no-flux walls at top and bottom). Every voxel carries:

| field | symbol | meaning | clock |
|---|---|---|---|
| activity | `h ∈ (-1,1)^C` | recurrent internal state, C channels | fast, `τ_h` |
| solid | `s ∈ [0,1]` | deposited matter; insulates or conducts activity, blocks transport, absorbs light | slow, `1/k_dep`, `1/k_ero` |
| nutrient | `n ≥ 0` | feedstock for deposition, diffuses | `dx²/D₀` |
| waste | `w ≥ 0` | product of erosion, diffuses, regenerated into nutrient by light | `1/(k_photo L₀)` |
| light | `L ≥ 0` | derived each step: top illumination attenuated by solid and pigment above | instantaneous |

Mass `n + s + w` is conserved per voxel up to transport, and transport is a
symmetric face-flux scheme, so **total mass is a tested invariant**. Nothing is
created; a pattern has to be paid for in nutrient, and nutrient only comes back
through light hitting waste. Structure shades what is beneath it. That is the whole
ecology: canopy, not creatures.

### Activity (the rule)

For voxel x with six neighbours y, conductance `g(x,y) = clamp(g₀ + g₁·(s_x+s_y)/2, 0, 1)`
so solid changes who talks to whom (g₁ < 0: solid insulates; g₁ > 0: solid conducts).

```
m   = Σ_y g(x,y) · (h_y − h_x)                      # solid-modulated coupling
u   = W_self·h + W_nb·m + W_in·[n, s, w, L] + b
h  += dt/τ_h · (tanh(u) − h)
```

A second shell couples the six axis voxels at distance 2 with its own matrix:

```
m2  = Σ_{d=±2eᵢ} g(x,x+d) · (h_{x+d} − h_x)
u  += W_nb2·m2
```

`θ = {W_self, W_nb, W_nb2 ∈ R^{C×C}, W_in ∈ R^{C×4}, b, v_dep, v_ero, v_pig ∈ R^C}` is the
**learnable local rule**: `3C² + 8C` numbers (248 at C=8). It is shared by every
voxel. The second shell exists because the analysis said so (2026-10-16): with one
ring the rule has one spatial scale, and over 20,000 sampled rules only 0.5% formed
patterns at all while 12% collapsed to grid-scale checkerboards. Tying
`W_nb2 = −β·W_nb` with `W_nb` positive semidefinite is short-range activation with
longer-range inhibition; it puts the fastest mode at `cos k* = 1/(4β)` per axis and
raised the patterning fraction to 20–62%, at 4–16 cells, growing on the activity
clock. Measured wavelengths match the prediction within a few percent across a
hundred rules. The tie is a sampling prior; search may break it. There is no separate organism: a "cell" is whatever coherent region the
dynamics make, reproduction is a region splitting, inheritance is continuity of
state and deposit, evolution is which regimes persist.

### Material

```
a_dep = softplus(v_dep·h)      a_ero = softplus(v_ero·h)      pig = σ(v_pig·h)
dep   = min(k_dep·a_dep·n·(1−s)·dt, n, 1−s)
ero   = min(k_ero·a_ero·s·dt, s)
photo = min(k_photo·L·w·dt, w)
s += dep − ero        n += −dep + photo + ∇·(D(s)∇n)        w += ero − photo + ∇·(D(s)∇w)
D(s) = D₀ (1−s)²   with harmonic-mean face conductances
L(i,j,k) = L_amb + L₀ exp(−Σ_{k'>k} (α_s s + α_p pig) − ½(α_s s + α_p pig)_k)
```

Absorption is per cell and small (`α_s = 0.05`): the material is translucent, and
light reaches the floor of a 48-deep slab. At `α_s = 2` the slab went dark three
layers down and became a waste store under a crust, which removed the predicted
patterns. `L_amb` lets a specimen be lit without a vertical gradient.

Light is both an input to the rule (`W_in`) and the only source of regeneration, and
the rendered image uses the *same* `L` as its illumination. Light is inside the loop.

## Design coordinates, not sliders

| ratio | question |
|---|---|
| `τ_dep / τ_h` | does activity outrun its own traces? |
| `τ_ero / τ_dep` | how permanent is memory? |
| `τ_photo / τ_dep` | replenishment vs consumption: can the canopy sustain itself? |
| `√(D₀ τ_dep) / dx` | interaction length: many independent domains or one? |
| `g₁ / g₀` | does building insulate or connect? |

## Analyze before running

`Intmat2.Analysis` linearizes the system about its homogeneous steady state
`(h*, s*, n*, w*)` and computes the dispersion relation `λ(k)` of the
`(C+3)×(C+3)` Jacobian in Fourier space (`m → −g* κ(k) h`, `κ(k) = 2Σ(1−cos kᵢ)`,
transport `→ −D(s*) κ(k)`). Finite-`k` instability with stable `k=0` is Turing-type
patterning; `Im λ ≠ 0` is waves. A **stability atlas** over the design coordinates and
random rules is computed *before* any 3D run, and recipes are chosen from it. A
quiet run is a point in the atlas, not a verdict.

## Learnability

The rule is 184 numbers and the whole system is one field, so it is cheap to
search. World batching on the GPU runs many independent worlds per launch.
Objectives are a repertoire, never one score: persistence without saturation,
spectral power at finite wavenumber, temporal richness of `h`, memory
(correlation of `s` before and after a perturbation), recovery. Evolution
strategies first (no adjoint code, trivially parallel); BPTT through the step is a
second route since every operation is differentiable.

## Performance from the representation

Dense struct-of-arrays, `h` channel-first `(C, Nx, Ny, Nz)`, Float32 on device,
Float64 CPU reference as oracle. Four KernelAbstractions kernels per step
(activity, transport, material, light), no host round trips, no per-voxel
allocation. Same source runs on Metal (M2 Max) and AMD (hbox RX 6750 XT via the
verified pinned environment). Throughput is measured as **complete steps per
second at a given N³**, and separately as **worlds per second** for batches.

## What the first day found

- The linear theory is exact for this stepper: predicted and measured growth rates
  agree to three decimals; a quiet run of a predicted-slow rule is expected.
- `6 D₀ dt ≤ 1` is not the dt bound; the explicit neighbour coupling is. `stable_dt`
  computes the real one. Rules that violate it game temporal-richness scores.
- Evolution over a repertoire beats random search when the search vector respects
  the structure the analysis found, and loses when Gaussian mutation destroys it.
  Mutate on the manifold.
- The best structured rule so far is a frozen lattice with near-perfect material
  memory. Structure, memory and ongoing change trade off; the alife question is the
  corner where all three hold, and the objectives that measure it are regions that
  persist, split, merge and die, not spectral power.

## What is deliberately not here

Particles. Contact mechanics. Per-agent genomes. Energy ledgers. Credit
assignment. Blender. A tracker file. If one of these earns its way back it comes
with an analysis of why the field formulation could not express it.
