# Kernel-change gate: 64^3, 50 steps, Float32 on Metal (or ROCm) vs bench/ref-64-50.bin.
# usage: julia --project=env/metal bench/gate.jl [metal|rocm]
be_name = get(ARGS, 1, "metal")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
r = gate_rule(); p = gate_physics()
w = light!(gate_world(be, 64), r, p)
w = run!(w, r, p, 50)
c = KA.adapt(KA.CPU(), w)
N = 64; C = 8
ref = open(joinpath(@__DIR__, "ref-64-50.bin")) do io
    rd(dims...) = (a = Array{Float32}(undef, dims...); read!(io, a); a)
    (s=rd(N, N, N), h=rd(C, N, N, N), n=rd(N, N, N), w=rd(N, N, N), L=rd(N, N, N))
end
worst = 0.0
for f in (:s, :h, :n, :w, :L)
    d = maximum(abs.(Float64.(getfield(c, f)) .- getfield(ref, f)))
    global worst = max(worst, f == :L ? 0.0 : d)
    @printf("%-2s max|Δ| = %.3e  bitwise=%s\n", f, d, getfield(c, f) == getfield(ref, f))
end
println(worst <= 1e-5 ? "GATE PASS" : "GATE FAIL", "  worst(s,h,n,w) = ", worst)
