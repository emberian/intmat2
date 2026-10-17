# Throughput of the isotropic stencil (1) vs the axis stencil (0), tied two-shell rule,
# fast math, run! (one sync), interleaved best of 5.
# usage: julia --project=env/metal bench/stencil.jl [metal|rocm] [N...]
be_name = get(ARGS, 1, "metal")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
Ns = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [128, 256]
r = tied_rule(MersenneTwister(1), 8, Float32; nb1=12, beta=0.4)
r1 = gate_rule()
for N in Ns
    ps = [Physics{Float32}(dt=0.01, stencil=Int32(st)) for st in (0, 1)]
    ws = [run!(light!(gate_world(be, N), r, q), r, q, 2) for q in ps]
    n = max(10, round(Int, 2.5e8 / N^3))
    best = zeros(2); best1 = zeros(2)
    for _ in 1:5, i in 1:2
        t = @elapsed (ws[i] = run!(ws[i], r, ps[i], n)); best[i] = max(best[i], n / t)
        t = @elapsed (ws[i] = run!(ws[i], r1, ps[i], n)); best1[i] = max(best1[i], n / t)
    end
    @printf("N=%d  two-shell rule: stencil 0 %.1f, stencil 1 %.1f steps/s (x%.2f) | one-ring rule: stencil 0 %.1f, stencil 1 %.1f (x%.2f)\n",
            N, best[1], best[2], best[1] / best[2], best1[1], best1[2], best1[1] / best1[2])
end
