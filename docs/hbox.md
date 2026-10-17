# hbox AMD runtime

Verified 2026-10-15 in the predecessor project; reused here unchanged.

| component | location / version |
|---|---|
| probe root | `/othersys/tmp/intmat-probe-20261014` |
| Julia | `$PROBE_ROOT/julia-1.12.4/bin/julia` |
| depot | `JULIA_DEPOT_PATH="$PROBE_ROOT/depot:"` (**keep the trailing colon**) |
| ROCm | `ROCM_PATH="$PROBE_ROOT/rocm-debs/opt/rocm-6.4.3"`, `LD_LIBRARY_PATH="$ROCM_PATH/lib"` |
| device | gfx1031 (RX 6750 XT, 11.98 GiB), run with `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| AMDGPU.jl | upstream commit `c77460e860b35f21277d8f1253e37304bd8b8993` (reports 2.8.0) |
| GPUCompiler / LLVM.jl | 2.14.1 / 10.2.1 |
| KernelAbstractions | 0.9.45 — this package must stay on KA 0.9 |
| reference env | `$PROBE_ROOT/environment112/{Project,Manifest}.toml` |

Minimal ROCm: no rocFFT/rocBLAS. Normal compiled modules on; the cold
`--compiled-modules=existing` route fails GPU compilation.

Every launch goes through the enforced cgroup wrapper:

```sh
export PROBE_ROOT=/othersys/tmp/intmat-probe-20261014
export JULIA_DEPOT_PATH="$PROBE_ROOT/depot:" JULIA_NUM_THREADS=2 JULIA_NUM_PRECOMPILE_TASKS=1
export ROCM_PATH="$PROBE_ROOT/rocm-debs/opt/rocm-6.4.3" LD_LIBRARY_PATH="$ROCM_PATH/lib" HSA_OVERRIDE_GFX_VERSION=10.3.0
SWARM_MEM_MAX=16G /home/hbox/bin/swarm-build timeout 900s taskset -c 0,1 \
  "$PROBE_ROOT/julia-1.12.4/bin/julia" --startup-file=no --project=<env/amd on hbox> <script>
```

The intmat2 environment on hbox should be a *copy* of `environment112`'s manifest
pins plus this package `develop`ed from a clone under `/othersys/tmp/intmat2/`.
Do not modify `environment112` or anything else under the probe root.

## Intmat2 on hbox (lane B, 2026-10-16)

- Clone: `/othersys/tmp/intmat2/repo`; environment script `/othersys/tmp/intmat2/env.sh`
  (the exports above, plus `JL` and **`JULIA_PKG_OFFLINE=true`**). Without the offline flag,
  `Pkg.develop` auto-updated `depot/registries/General.tar.gz` under the probe root
  (2026-10-16 19:47). Manifest pins were unaffected, but keep Pkg offline.
- `env/amd/Manifest.toml` is environment112's manifest with Interior's dependencies
  dropped and Intmat2 `develop`ed (`../..`). It contains an absolute AMDGPU path, so it
  is hbox-only.
- Run: `. ../env.sh; SWARM_MEM_MAX=24G /home/hbox/bin/swarm-build timeout 1500s taskset -c 0-3 "$JL" --startup-file=no --project=env/amd bench/<script>.jl rocm`.
- No `rocm-smi` in the minimal ROCm. For GPU load use `/sys/class/drm/card*/device/gpu_busy_percent`
  and `mem_info_vram_used`. `AMDGPU.info()` gives (free, total) = 12.87 GB total.
- Clocks are on auto DPM (`pp_dpm_sclk` 500/2660/2880 MHz), so single runs vary up to 2x.
  Interleave and take best-of.
- Largest world: 480³ at C = 8 (9.48 GiB); 496³ fails. Run each large N in a fresh process,
  because the previous world's memory is not returned in time.
- Once, during a 64/128/256 sweep, a "Memory access fault … Page not present" aborted
  the process. It did not reproduce in three identical reruns. Rule uploads are now only
  evicted after a sync, which removed one possible use-after-free with `sync=false`.
