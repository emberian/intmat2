# Rendering: photographs lit by the simulation's own light

Each photograph is lit only by the light field `L` that `step!` computes: a column march
from the lit top face through `α_s·s + α_p·σ(v_pig·h)`. The brightness of a voxel in
an image is the light that voxel received during the run. Nothing else lights the scene.

## Optical model (`src/render.jl`)

One work-item per pixel marches its ray through the slab, a finite block `[0,Nx]×[0,Ny]×[0,Nz]`
(no periodic tiling, z up, voxel centres at `i−½`), with step `step` voxels and per-pixel jitter.
At each sample, `s`, `pig` and `L` are trilinear and `p̃ = max(pig − pig_floor, 0)/(1 − pig_floor)`:

    extinction  σ_c = density·(α_s·s + (α_p + stain)·p̃·pigment_absorb_c)
    scattering  β_c = density·(α_s·s·solid_c + (α_p + stain)·p̃·pigment_c)
    radiance   += T·β_c·(1 − e^{−σ_c Δ})/σ_c·(L + ambient);   T *= e^{−σ_c Δ}

followed by `exposure`, a shoulder `v/(1 + shoulder·v)`, the background through what is
left of `T`, and gamma. `α_s`, `α_p` come from the capture's `Physics`.
- `density = 1` means the camera path absorbs exactly as the light path does. Other values
  change how deep the camera sees, never the illumination.
- `stain` and `pig_floor` are camera-only, like a histological stain: they choose which
  pigment the camera picks out. `pig_floor = "q0.95"` takes a quantile of the capture.
- `prefilter = true` blurs `s` and `pig` with a 3³ binomial kernel before marching. Many rules
  settle into single-cell alternation (checkerboards), which a camera cannot resolve and
  which otherwise aliases into moiré. The domain walls between checkerboards survive the blur.
- `ambient` defaults to 0. Solid is a pale neutral and pigment is the only colour.
- Clip: a box `lo..hi` plus one half-space `normal·x ≤ offset` cut the slab open, so
  you can photograph its interior.

## Export format (`src/export.jl`)

`capture(wd, rule, p, dir)` copies a world from any backend to the host and writes
`<f>.f32` (Float32, little-endian, column-major `(Nx,Ny,Nz)`; `h` is `(C,Nx,Ny,Nz)`,
optional) for `s n w L pig` (`pig = σ(v_pig·h)`). It also writes `manifest.json`, which holds
dims, C, t, step, the rule vector, every Physics field, `total_mass`, `summarize`, the
producing commit (`-dirty` if the tree was dirty) and the file table. Next to these it puts
UInt8 copies under `u8/ds{1,2,4}/` (box-averaged, quantized over `[0, hi]`).
`load_capture(dir)` round-trips. `Recorder(dir, rule, p; every)` + `record!(rec, wd)`
writes `dir/NNNNNN/` and keeps `dir/series.json` current. `viewer_volume(rundir, out; ds,
max_frames, optics)` stages a run for the browser (`index.json` + `fNNN_<field>.u8`).

## Running and photographing

    julia --project=env/metal bin/run.jl recipes/example.toml outputs/x [steps=4000 N=128 ...]
    julia --project=env/metal bin/photograph.jl outputs/x/004000 optics.toml out.png
    julia --project=env/metal bin/photograph.jl --many outputs/x/004000 outdir a.toml b.toml
    julia --project=env/metal bin/photograph.jl --series outputs/x optics.toml framesdir
    julia --project=env/metal bin/film.jl recipe.toml optics.toml outdir every=10 steps=4000

A recipe has `C, seed, N|dims, steps, capture_every, n0, h_scale, backend`, a `[rule_scales]`
table (with optional `nb_diag`, which adds `nb_diag·I` to `W_nb`) or an explicit `theta`,
and Physics fields at the top level or under `[physics]`. Lane A's `recipes/atlas-*.toml`
run unchanged. An optional `note` is written into `series.json`. `optics.toml` has
`width, height, backend, prefilter` and the `[optics]`, `[camera]` and `[clip]` tables. Camera
`pos`/`target` and clip `lo`/`hi` are fractions of the slab dims (see `photo_setup`).
`film.jl` renders the live world on the device every `every` steps and encodes an mp4 with ffmpeg.

Metal is loaded at top level before `main` runs (`invokelatest`). Loading it inside a
function leaves `step!` on stale methods, and the run crawls at about 1 step/s.

## Viewer and Pages

`web/viewer.html?v=<volume dir>` is a WebGL2 ray marcher over the staged UInt8 volumes. It
uses the same model as above in "as lit" mode, plus false-colour views of s, pig, n, w and L,
a time scrubber, a clip plane (`&clip=axis,pos`), density and exposure. `scripts/build_pages.py`
stages `site/` and `web/` into `_site/`, and `.github/workflows/pages.yml` deploys it on
push to `dev`: https://emberian.github.io/intmat2/. Raw runs stay in `outputs/` (gitignored).

## Gallery (`site/collection.json`)

| name | run | what it shows |
|---|---|---|
| lamellae | tied-10 (two-shell), 128×128×48, 5442 steps | through-depth sheets in x/y stripe grains, λ 6.9 (pred. 6.6); film |
| lamellae-section | same, diagonal cut | sheets on the cut face, dimming with depth |
| crossgrain | tied-09, 13476 steps | stripe grains sharpening while solid erodes to waste |
| lamellae-sheet | tied-10, Nz 16 | same wavelength and mosaic in a thin sheet |
| veined-floor | atlas turing-04, α 0.05/0.02, 20000 steps | chambers walled in pigment, in the dim bottom; plus a film |
| strata-section | same run, diagonal cut | crust, blotch band, clear band, vein network |
| receding-front | turing-04, α 0.05/0.1, 8000 steps | lobed solid front over a dark, fully pigmented lower half |
| crust-underside | turing-06, α 0.05/0.1, 4000 steps | 12-layer crust seen from the empty, nutrient-rich space below |
| lit-skin | turing-02, original α 2.0 | crust with light shafts over dark waste |

One-ring random and atlas rules mostly give depth strata or single-cell checkerboards.
The tied two-shell recipes give the predicted 6–8 cell lateral pattern at every depth.
Tied recipes need `fast=false` on Metal: fast tanh returns NaN in one step.
