# Shared setup for the benchmark and gate scripts. Include after `using Intmat2`
# and a backend package. The gate world: C=8, Float32, seed=3, rule seed 1.
using Intmat2, KernelAbstractions, Random, Printf
using Intmat2: run!, light!
const KA = KernelAbstractions

gate_rule(T=Float32) = random_rule(MersenneTwister(1), 8, T; self=1.0, nb=1.0)
gate_physics(T=Float32) = Physics{T}()
gate_world(be, N; T=Float32, seed=3) =
    World(be, gate_rule(T), (N, N, N); T=T, h_scale=0.3, s0=0.3, seed=seed)

"""Complete-step throughput: warm up, then time `nsteps` of `step!` (which syncs)."""
function steps_per_s(be, N; T=Float32, nsteps=nothing, warm=5, reps=3, kw...)
    r = gate_rule(T); p = gate_physics(T)
    w = light!(gate_world(be, N; T=T), r, p)
    w = run!(w, r, p, warm; kw...)
    ns = something(nsteps, max(20, round(Int, 5e8 / N^3)))
    best = Inf
    for _ in 1:reps
        t = @elapsed (w = run!(w, r, p, ns; kw...))
        best = min(best, t / ns)
    end
    1 / best
end
