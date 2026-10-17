# Largest N^3 that fits: allocate one world, run fast steps, report steps/s, VRAM, mass drift.
# usage: julia --project=env/amd bench/maxgrid.jl rocm N...
be_name = ARGS[1]
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
for N in parse.(Int, ARGS[2:end])
    GC.gc(true)
    try
        r = gate_rule(); p = gate_physics()
        w = light!(gate_world(be, N), r, p)
        m0 = total_mass(w)
        w = run!(w, r, p, 3)
        n = max(5, round(Int, 2e9 / N^3))
        t = @elapsed (w = run!(w, r, p, n))
        vram = be_name == "metal" ? NaN : (AMDGPU.info()[2] - AMDGPU.info()[1]) / 2^30
        @printf("N=%d  %.2f steps/s  %.3f Gvox/s  world %.2f GiB  device used %.2f GiB  mass drift %.2e\n",
                N, n / t, n * N^3 / t / 1e9, 23 * 4 * N^3 / 2^30, vram, total_mass(w) / m0 - 1)
        w = nothing
    catch e
        println("N=$N failed: ", sprint(showerror, e)[1:min(end, 200)])
    end
end
