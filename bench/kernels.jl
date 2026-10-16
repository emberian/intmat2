# Per-pass attribution and workgroup sweep of the fused step, then complete steps/s.
# usage: julia --project=env/metal bench/kernels.jl [metal|rocm] [N...]
be_name = get(ARGS, 1, "metal")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
using Intmat2: update_kernel!, light_kernel!, device_theta
Ns = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [64, 128, 256]
function ktime(f, reps)
    f(); KA.synchronize(be)
    t = @elapsed (for _ in 1:reps; f(); end; KA.synchronize(be))  # back-to-back, one sync
    1e3 * t / reps
end
bestof(f, reps; n=5) = minimum(ktime(f, reps) for _ in 1:n)
for N in Ns
    r = gate_rule(); p = gate_physics()
    wd = light!(gate_world(be, N), r, p); X = Int32(N); θ = device_theta(be, r)
    reps = max(5, round(Int, 5e8 / N^3))
    for fast in (false, true), wg in (32, 64, 128, 256)
        t = bestof(() -> update_kernel!(be, wg)(wd.h2, wd.s2, wd.n2, wd.w2, wd.L, wd.h, wd.s, wd.n, wd.w,
                       θ, Val(8), Val(fast), p, X, X, X, Int32(1); ndrange=cld(N^3, wg) * wg), reps)
        @printf("N=%d update fast=%-5s wg=%-4d %.3f ms\n", N, fast, wg, t)
    end
    for wg in (32, 64, 128, 256)
        t = bestof(() -> light_kernel!(be, wg)(wd.L, p, X, X, X; ndrange=N^2), reps)
        @printf("N=%d light  wg=%-4d %.3f ms\n", N, wg, t)
    end
    for fast in (false, true)
        sps = maximum(steps_per_s(be, N; fast=fast) for _ in 1:2)
        @printf("N=%d STEP fast=%-5s %.3f ms = %.1f steps/s\n", N, fast, 1e3 / sps, sps)
    end
end
