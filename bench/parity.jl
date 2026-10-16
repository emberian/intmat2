# 32^3 CPU (precise) vs GPU parity after 20 steps, and the 64^3 50-step gate, on one backend.
# usage: julia --project=env/amd bench/parity.jl rocm   (or metal)
be_name = get(ARGS, 1, "rocm")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
r = gate_rule(); p = gate_physics()
c = run!(light!(gate_world(CPU(), 32), r, p), r, p, 20)
for fast in (false, true)
    g = KA.adapt(CPU(), run!(light!(gate_world(be, 32), r, p), r, p, 20; fast=fast))
    @printf("32^3 x20 %s fast=%-5s max|Δ| s %.2e h %.2e n %.2e w %.2e L %.2e  mass ratio %.9f\n", be_name, fast,
            (maximum(abs.(getfield(g, f) .- getfield(c, f))) for f in (:s, :h, :n, :w, :L))..., total_mass(g) / total_mass(c))
end
