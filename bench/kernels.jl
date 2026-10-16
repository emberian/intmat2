# Per-kernel attribution and workgroup sweep of the fused step.
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
shapes = [32, 64, 128, 256, 512, 1024]
for N in Ns
    r = gate_rule(); p = gate_physics()
    wd = light!(gate_world(be, N), r, p); X = Int32(N); θ = device_theta(be, r)
    reps = max(20, round(Int, 2e9 / N^3))
    for wg in shapes
        t = ktime(() -> update_kernel!(be, wg)(wd.h2, wd.s2, wd.n2, wd.w2, wd.L, wd.h, wd.s, wd.n, wd.w, θ, Val(8), p, X, X, X; ndrange=N^3), reps)
        @printf("N=%d update wg=%-12s %.3f ms\n", N, string(wg), t)
    end
    for wg in (32, 64, 128, 256)
        t = ktime(() -> light_kernel!(be, wg)(wd.L, p, X, X, X; ndrange=N^2), reps)
        @printf("N=%d light  wg=%-12s %.3f ms\n", N, string(wg), t)
    end
    sps = steps_per_s(be, N)
    @printf("N=%d STEP %.3f ms = %.1f steps/s\n", N, 1e3 / sps, sps)
end
