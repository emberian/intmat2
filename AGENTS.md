# Working in Interior 2

Read `DESIGN.md` first. It is the contract; this file is how to work inside it.

## The shape of the code

- `src/rule.jl` — `Rule{C,T}` (the 2C²+8C learnable numbers, isbits), `Physics{T}`
  (timescales), `timescales(p)` (design coordinates).
- `src/state.jl` — `World{C,T}`: `h (C,Nx,Ny,Nz)`, `s n w L (Nx,Ny,Nz)`, double buffers.
- `src/step.jl` — the four KernelAbstractions kernels and `step!(w, rule, p) -> w`.
  **`step!` returns the new world; the argument is stale.**
- `src/observe.jl` — `total_mass`, `summarize`.
- `test/runtests.jl` — conservation, bounds, Float32 vs Float64 oracle. Fast (seconds).
- `env/metal`, `env/amd` — backend environments that `develop` this package.

One field, one rule, four kernels. New capability goes in a new file that *uses*
these; it does not fork the step. If you need to change the step, change it for
everyone and make the tests say why.

## Rules of the road

- **Julia only.** No Python/Torch simulation. Python is allowed for the Pages build script only.
- **Measure complete steps** (`steps/s` at N³, or `worlds/s` for batches). A kernel
  microbenchmark is not a result. Report the exact command, grid, backend, dtype.
- **Conservation is a gate.** `total_mass` before and after; any change to `step.jl`
  reruns `test/runtests.jl`. Float32 drift is reported, not hidden.
- **Analysis before soaks.** A new recipe comes with its `Analysis` dispersion
  result. A quiet run is a point in the atlas, not a verdict about the substrate.
- **No tracker file.** Results live next to the code that made them (`results/*.json`
  with the producing command) and in short `docs/*.md`. Delete what is superseded.
- **Preserve others' runs.** Never reseed, migrate or stop a run you did not start.
- **Commit named files**, never `git add -A`. Unsigned commits are fine.

## Hardware

Local: Apple M2 Max, 96 GiB, Metal via `env/metal` (`julia --project=env/metal`).

hbox (ssh `hbox`): RX 6750 XT, 12 GiB VRAM, 24 CPUs, 123 GiB RAM, ZFS. The
verified Julia 1.12.4 + AMDGPU(pinned commit c77460e) + KernelAbstractions 0.9.45 +
standalone ROCm 6.4.3 environment lives at `/othersys/tmp/intmat-probe-20261014`
(see `docs/hbox.md`). **Keep this package on KernelAbstractions 0.9** so it loads
there. Every hbox build or run goes through `/home/hbox/bin/swarm-build`
(enforced memory cap), e.g. `SWARM_MEM_MAX=16G swarm-build julia ...`. `free` lies
on hbox (ZFS ARC); check `ps -eo rss --sort=-rss | head` before calling it busy.

## Stopping processes: PID file or nothing

- **Never kill by pattern.** No `pkill -f <anything>`, no `pkill -x <bare word>`, no
  `killall`, no `kill -u`.
- **A process you started is stopped by the PID in its PID file**, and only that.
  Every detached job writes one (`nohup ... & echo $! > job.pid`). No PID file, no
  kill: say so instead.
- If you must look first, `pgrep -fl` and read the list; it is never the argument to a kill.

## Disk

Disk pressure is a pause, not a cleanup. Stop writing, report `df`/`du`, wait.
Delete only files this session created in its own scratch or output directories.
