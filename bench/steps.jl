# Complete steps/s, before (09d1bd7 four-kernel step, bench/step_09d1bd7.jl) and after,
# interleaved in one process so that contention from other GPU users hits both alike.
# usage: julia --project=env/metal bench/steps.jl [metal|rocm] [out.json] [N...]
be_name = get(ARGS, 1, "metal")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
module Old
    using KernelAbstractions, Intmat2.StaticArrays, Intmat2.LinearAlgebra
    using Intmat2: Rule, Physics, World, backend
    const KA = KernelAbstractions
    include(joinpath(@__DIR__, "step_09d1bd7.jl"))
end
out = get(ARGS, 2, "")
Ns = length(ARGS) > 2 ? parse.(Int, ARGS[3:end]) : [64, 128, 256]

function rate(stepper, w, n)
    t = @elapsed (for _ in 1:n; w = stepper(w); end; KA.synchronize(be))
    n / t, w
end
results = Dict{String,Any}()
for N in Ns
    r = gate_rule(); p = gate_physics()
    variants = [
        "before (4 kernels, sync/step)" => w -> Old.step!(w, r, p),
        "after precise, sync/step" => w -> step!(w, r, p; fast=false),
        "after fast, sync/step" => w -> step!(w, r, p; fast=true),
        "after precise, run!" => w -> step!(w, r, p; fast=false, sync=false),
        "after fast, run!" => w -> step!(w, r, p; fast=true, sync=false),
    ]
    ws = Dict(k => run!(light!(gate_world(be, N), r, p), r, p, 3) for (k, _) in variants)
    best = Dict(k => 0.0 for (k, _) in variants)
    n = max(10, round(Int, 2.5e8 / N^3))
    for rep in 1:5, (k, f) in variants
        sps, ws[k] = rate(f, ws[k], n)
        best[k] = max(best[k], sps)
    end
    for (k, _) in variants
        @printf("N=%-4d %-32s %8.1f steps/s  %7.3f ms  %.3f Gvox/s\n", N, k, best[k], 1e3 / best[k], best[k] * N^3 / 1e9)
    end
    results[string(N)] = best
end
if !isempty(out)
    open(out, "w") do io
        println(io, "{\"command\": \"julia --project=env/", be_name == "metal" ? "metal" : "amd",
                " bench/steps.jl ", join(ARGS, " "), "\", \"dtype\": \"Float32\", \"C\": 8,")
        println(io, " \"note\": \"best of 5 interleaved repetitions; steps/s\",")
        println(io, " \"steps_per_s\": {", join(["\"$N\": {" * join(["\"$k\": $(round(v; digits=1))" for (k, v) in sort(collect(results[string(N)]))], ", ") * "}" for N in Ns], ",\n  "), "}}")
    end
end
