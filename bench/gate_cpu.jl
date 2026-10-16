# CPU gate: the current kernels against the original four-kernel step (09d1bd7, kept
# verbatim in bench/step_09d1bd7.jl), Float64 and Float32, on odd non-cubic grids.
# usage: julia --project=. bench/gate_cpu.jl
using Intmat2, KernelAbstractions, Random, Printf
module Old
    using KernelAbstractions, StaticArrays, LinearAlgebra
    using Intmat2: Rule, Physics, World, backend
    const KA = KernelAbstractions
    include(joinpath(@__DIR__, "step_09d1bd7.jl"))
end
using Intmat2: light!
ok = true
for T in (Float64, Float32), dims in ((12, 10, 8), (7, 5, 9))
    C = 6
    r = random_rule(MersenneTwister(2), C, T; self=1.0, nb=1.0)
    p = Physics{T}()
    a = World(CPU(), r, dims; T=T, h_scale=0.3, s0=0.4, seed=3)
    b = copy(a)
    a = light!(a, r, p); b = Old.light!(b, r, p)
    for _ in 1:100
        a = step!(a, r, p); b = Old.step!(b, r, p)
    end
    for f in (:h, :s, :n, :w, :L)
        same = getfield(a, f) == getfield(b, f)
        global ok &= same
        @printf("%-7s %-11s %-2s bitwise=%s max|Δ|=%.2e\n", T, string(dims), f, same,
                maximum(abs.(getfield(a, f) .- getfield(b, f))))
    end
end
println(ok ? "CPU GATE PASS (bitwise)" : "CPU GATE: NOT BITWISE")
