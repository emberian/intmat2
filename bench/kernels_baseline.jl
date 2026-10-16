# Per-kernel attribution of the original four-kernel step (commit 09d1bd7 kernels).
# usage: julia --project=env/metal bench/kernels_baseline.jl
using Metal
include(joinpath(@__DIR__, "common.jl"))
using Intmat2: activity_kernel!, transport_kernel!, material_kernel!, light_kernel!
be = MetalBackend()
function ktime(f, reps)
    f(); KA.synchronize(be)
    t = @elapsed (for _ in 1:reps; f(); KA.synchronize(be); end)
    1e3 * t / reps
end
for N in (64, 128, 256)
    r = gate_rule(); p = gate_physics()
    wd = light!(gate_world(be, N), r, p)
    reps = max(10, round(Int, 1e9 / N^3))
    ta = ktime(() -> activity_kernel!(be)(wd.h2, wd.h, wd.s, wd.n, wd.w, wd.L, r, p, N, N, N; ndrange=(N, N, N)), reps)
    tt = ktime(() -> transport_kernel!(be)(wd.n2, wd.w2, wd.n, wd.w, wd.s, p, N, N, N; ndrange=(N, N, N)), reps)
    tm = ktime(() -> material_kernel!(be)(wd.s2, wd.n2, wd.w2, wd.h2, wd.s, wd.L, r, p; ndrange=(N, N, N)), reps)
    tl = ktime(() -> light_kernel!(be)(wd.L, wd.s2, wd.h2, r, p, N; ndrange=(N, N)), reps)
    sps = steps_per_s(be, N)
    @printf("N=%d  activity %.3f  transport %.3f  material %.3f  light %.3f  sum %.3f ms | step %.3f ms = %.1f steps/s\n",
            N, ta, tt, tm, tl, ta + tt + tm + tl, 1e3 / sps, sps)
end
