# Save the 64^3 50-step Float32 reference (s, h, n, w, L) from the current kernels.
# usage: julia --project=env/metal bench/make_ref.jl <out.bin>
using Metal
include(joinpath(@__DIR__, "common.jl"))
out = ARGS[1]
r = gate_rule(); p = gate_physics()
w = light!(gate_world(MetalBackend(), 64), r, p)
w = run!(w, r, p, 50)
c = KA.adapt(KA.CPU(), w)
open(out, "w") do io
    for a in (c.s, c.h, c.n, c.w, c.L); write(io, a); end
end
println("wrote $out  mass ", total_mass(c), " mean_s ", sum(c.s)/length(c.s))
