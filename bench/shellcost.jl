# Cost of the second shell: the one-ring fused step (1f4a9e8, bench/step_1f4a9e8.jl) vs
# the current step, interleaved, tied rule, Metal. usage: julia --project=env/metal bench/shellcost.jl
using Metal
include(joinpath(@__DIR__, "common.jl"))
module Prev
    using KernelAbstractions, Intmat2.StaticArrays, Intmat2.LinearAlgebra
    using Intmat2: Rule, Physics, World, backend, nparams
    const KA = KernelAbstractions
    include(joinpath(@__DIR__, "step_1f4a9e8.jl"))
end
be = MetalBackend()
r = tied_rule(MersenneTwister(1), 8, Float32; nb1=2, beta=0.4); p = Physics{Float32}(dt=0.02)
for N in (128, 256)
    a = run!(light!(gate_world(be, N), r, p), r, p, 3); b = copy(a)
    n = N == 128 ? 200 : 30
    bn = 0.0; bo = 0.0
    for _ in 1:5
        t = @elapsed (a = run!(a, r, p, n)); bn = max(bn, n / t)
        t = @elapsed (for _ in 1:n; b = Prev.step!(b, r, p; sync=false); end; KA.synchronize(be)); bo = max(bo, n / t)
    end
    @printf("N=%d  one ring %.1f steps/s  two shells %.1f steps/s  cost %.1f%%\n", N, bo, bn, 100 * (bo / bn - 1))
end
