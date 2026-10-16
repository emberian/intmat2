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
