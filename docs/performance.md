# Performance of the step

Float32, C = 8, complete steps (`run!`, one sync at the end) unless stated. Raw numbers,
commands and caveats: `results/perf-metal-20261016.json`, `results/perf-hbox-20261016.json`.
The M2 Max was shared with other lanes' Metal runs all session (30–98% busy), and hbox
clocks float (auto DPM). Every before/after below is interleaved in one process and
best-of-5, so both sides see the same contention. Absolute Metal numbers vary up to 2x.

## What the step is now

Two passes (`src/step.jl`). **update**: one work-item per voxel does activity
(ring + optional second shell), 6-face transport, material, and writes the absorbance
of the new state into `L`. **light**: one work-item per column turns that absorbance
into `L`. Was: four kernels (activity, transport, material, light).

| change | what it bought | evidence |
|---|---|---|
| Linear Int32 indexing (no KA `@index(Global, NTuple)`) | a plain 256³ copy runs at 303 GB/s, against 51 GB/s with the NTuple index (64-bit div/mod per item) | scratch copy kernel, Metal |
| Fuse activity, transport, material and absorbance | ~23 → ~14 bytes/voxel/step of traffic; precise math is no faster at 256³ | the kernel is compute-bound, see ablation |
| Rule read from a flat θ at the point of use, staged in threadgroup memory | the by-value `Rule` was loaded whole (184 floats) at kernel entry: register pressure, 384-thread cap | IR: 184 loads at entry |
| Fast `tanh`/`exp`/`log` on GPUs (`fast=true` default; tanh argument clamped to ±9) | 256³ update 35.8 → 16.1 ms on Metal; no change on AMD | gate drift 5.8e-7 (Metal), 3.9e-7 (AMD) |
| `run!` without a per-step sync (`step!(…; sync=false)`) | a Metal sync costs ~1 ms: 64³ 1.7 → 0.25 ms per step | `bench/kernels.jl` |
| Second shell skipped when every rule has Wnb2 = 0 (runtime flag) | one-ring rules don't pay the shell | AMD 256³ 64 → 142 steps/s |

Ablation of the fused kernel at 256³ on Metal (precise, ms): full 32.7; with fast tanh
16.2; without the matvecs 17.5; without the 6 neighbours 33.5 (so the neighbours cost
nothing); memory only 4.7; fast math plus rule in threadgroup memory 11.0.

## Before / after (steps/s)

| N³ | Metal before | Metal after | AMD before | AMD after | AMD after, two-shell rule |
|---|---|---|---|---|---|
| 64 | 589 | 4060 | 5770 | 8630 | 7690 |
| 128 | 184 | 524 | 886 | 1157 | 1016 |
| 256 | 23.4 | 49.1 | 93.6 | 141.8 | 127.4 |

"Before" is the 09d1bd7 four-kernel step synchronizing every step, kept verbatim in
`bench/step_09d1bd7.jl`. Metal comes from the quietest session (commit 1f4a9e8, one-ring);
later sessions measured 37–48 at 256³ and 407–524 at 128³ under heavier contention. AMD
comes from commit bca05ac. Adding the `stencil` branch (52f7be0) costs AMD ~15% on
stencil 0 (256³ 141.8 → 121.6; two-shell 127 → 101). A compile-time `Val` stencil
recovers that, but it is slower for stencil 1 on AMD (56 → 43) and for both stencils on
Metal, so the runtime branch stays.

Per kernel on Metal (ms; before: synced individually; after: back to back, wg 32):

| N³ | activity | transport | material | light (before) | update precise | update fast | light (after) |
|---|---|---|---|---|---|---|---|
| 64 | 2.09 | 1.58 | 1.74 | 1.60 | 0.60 | 0.22 | 0.03 |
| 128 | 4.71 | 3.92 | 4.20 | 2.18 | 4.67 | 1.94 | 0.08 |
| 256 | 14.4 | 8.16 | 11.7 | 4.90 | 35.8 | 16.1 | 0.44 |

The light pass stays a column march (Nx·Ny items). It is 2–3% of the step, so a
parallel scan would not pay. On AMD the 256³ update takes 8.2 ms and light 0.13 ms.
Workgroup size: 32 on Metal (64–256 cost 10–30% more at 128³ and 256³); AMD is flat
from 64 to 256.

## Coupling variants (128³ steps/s, tied rule, fast)

| | Metal | AMD |
|---|---|---|
| one ring, stencil 0 | 522 | 993 |
| two shells, stencil 0 | 376 | 804 |
| two shells, `stencil = 1` (19-point ring and shell) | 138 (x2.7) | 456 (x1.8) |

## hbox (RX 6750 XT)

The same source runs through `env/amd`. CPU vs ROCm at 32³ over 20 steps: max |Δh|
1.8e-7, mass ratio 1.000000000. The largest grid is **480³** (9.48 GiB of world,
10.6 GiB of the device, 20.6 steps/s, 2.28 Gvox/s); 384³ and 448³ run at the same
2.3 Gvox/s, and 496³ fails to allocate. A world takes 23 Float32 fields per voxel
(h and h2 at C = 8, s n w and their copies, L).

## Batching (`WorldBatch`, worlds·steps/s, rules as a P×B device matrix)

| N³ | B | Metal | AMD | AMD, B worlds stepped one by one |
|---|---|---|---|---|
| 32 | 1 | 14 100 | 34 100 | 28 000 |
| 32 | 4 | 18 400 | 54 700 | 30 300 |
| 32 | 16 | 19 100 | 72 900 | 30 300 |
| 32 | 64 | 20 600 | 74 200 | 30 300 |
| 64 | 1 | 2 340 | 7 660 | 8 560 |
| 64 | 16 | 2 680 | 9 220 | 7 250 |
| 64 | 64 | 2 800 | 9 090 | 7 260 |

At 32³ batching is worth 2.4x on AMD, because separate small worlds are launch-bound
(~33 µs per step). From 64³ the GPU is already full, so a batch matches the single-world
voxel rate (~2.3 Gvox/s on AMD). In a quiet Metal session 32³ reached 35 000 at B = 16.
Rules passed as a by-value tuple are 25–150x slower (the tuple is copied per thread) and
fail to compile at B = 64, so `src/batch.jl` takes one `Rule`, a `Vector{Rule}` or a
device matrix. A batch is bitwise equal to its worlds stepped alone on CPU, Metal and
ROCm. That needed one kernel specialization for both (flat vectors) and a sequential fma
`dot` (StaticArrays' `@simd` dot was reassociated differently per code path).

## Gates

`bench/gate.jl` replays 64³ for 50 steps against `bench/ref-64-50.bin` (09d1bd7 kernels,
one-ring rule, alpha 2). Precise is bitwise on Metal (max |Δ| 0) and 4.7e-7 on AMD. Fast
is 5.8e-7 on Metal and 3.9e-7 on AMD. `bench/gate_cpu.jl`: CPU Float64 bitwise vs
09d1bd7. CPU Float32 differs by < 1e-5 (muladd vectorization order).

## The remaining bottleneck

On Metal the update is compute- and latency-bound, not bandwidth-bound: memory alone
takes 4.7 ms at 256³ against ~16 ms for the fast update. What remains is the three C×C
matvecs (~15 ms of the precise kernel) and the occupancy they cost: the pipeline caps
at 384 threads per group. AMD reaches 2.3 Gvox/s, about half its copy bandwidth, and the
second shell and stencil 1 add strided h loads there (stencil 1: 37 h vectors per
voxel). The next step is a threadgroup tile of h and s with a halo of 2. Each h is now
loaded by up to 13 voxels (two-shell) or 37 (stencil 1), and a tile would turn that
into one load. For the matvecs: several voxels per thread, so θ reads amortize across
voxels.
