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
r2 = tied_rule(MersenneTwister(1), 8, Float32; nb1=2, beta=0.4); r1 = gate_rule(); p = Physics{Float32}(dt=0.02)
for N in (128, 256)
    w0 = run!(light!(gate_world(be, N), r1, p), r1, p, 3)
    a = copy(w0); b = copy(w0); c = copy(w0)
    n = N == 128 ? 200 : 30
    bp = 0.0; b1 = 0.0; b2 = 0.0
    for _ in 1:5
        t = @elapsed (for _ in 1:n; b = Prev.step!(b, r1, p; sync=false); end; KA.synchronize(be)); bp = max(bp, n / t)
        t = @elapsed (a = run!(a, r1, p, n)); b1 = max(b1, n / t)
        t = @elapsed (c = run!(c, r2, p, n)); b2 = max(b2, n / t)
    end
    @printf("N=%d  pre-shell step %.1f  current, one-ring rule %.1f  current, two-shell rule %.1f steps/s\n", N, bp, b1, b2)
end
