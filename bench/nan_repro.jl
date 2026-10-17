# Fast-path saturation check: a large-weight tied recipe (|u| in the tens) at 64×64×16
# on the GPU, fast vs precise, plus the raw fast intrinsics at large arguments.
# usage: julia --project=env/metal bench/nan_repro.jl [metal|rocm] [recipe.toml]
be_name = get(ARGS, 1, "metal")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
using TOML
rec = TOML.parsefile(get(ARGS, 2, joinpath(@__DIR__, "..", "recipes", "tied-20261016-10.toml")))
r = Rule{8,Float32}(Float32.(rec["theta"]))
p = Physics{Float32}(; (Symbol(k) => Float32(v) for (k, v) in rec["physics"])...)
for fast in (false, true), hs in (rec["h_scale"], 1.0)
    w = light!(World(be, r, (64, 64, 16); h_scale=hs, n0=rec["n0"], seed=1), r, p)
    m0 = total_mass(w)
    w = run!(w, r, p, 1; fast=fast); c1 = KA.adapt(CPU(), w)
    w = run!(w, r, p, 199; fast=fast); c = KA.adapt(CPU(), w)
    @printf("h_scale=%.1f fast=%-5s NaN after 1 step: %s  after 200: %s  mass drift %.2e\n", hs, fast,
            any(isnan, c1.h) || any(isnan, c1.s), any(isnan, c.h) || any(isnan, c.s), total_mass(c) / m0 - 1)
end
xs = Float32[-200, -90, -50, -20, -10, -9, 0, 9, 10, 20, 50, 90, 200]
d = KA.allocate(be, Float32, length(xs)); copyto!(d, xs)
println("tanh_fast raw:     ", Array(Base.FastMath.tanh_fast.(d)))
println("act_tanh (step):   ", Array(Intmat2.act_tanh.(Val(true), d)))
println("exp_fast:  ", Array(Base.FastMath.exp_fast.(d)))
