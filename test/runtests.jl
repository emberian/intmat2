using Intmat2, Test, Random, KernelAbstractions, Statistics
using Intmat2: run!, light!, timescales

@testset "rule round trip" begin
    r = random_rule(MersenneTwister(1), 8, Float64)
    @test nparams(r) == 2 * 64 + 64
    @test Rule{8,Float64}(vec(r)) == r
    @test_throws ArgumentError Rule{8,Float64}(zeros(3))
end

@testset "conservation and bounds (Float64 CPU)" begin
    C = 6
    r = random_rule(MersenneTwister(2), C, Float64)
    p = Physics{Float64}()
    w = World(CPU(), r, (12, 10, 8); n0=1.0, h_scale=0.3, seed=3)
    w = light!(w, r, p)
    m0 = total_mass(w)
    for _ in 1:200
        w = step!(w, r, p)
    end
    @test total_mass(w) ≈ m0 atol = 1e-9 * m0
    @test all(0 .<= w.s .<= 1)
    @test all(w.n .>= 0) && all(w.w .>= 0)
    @test all(-1 .<= w.h .<= 1)
    @test all(0 .<= w.L .<= p.L0)
    # light is monotone down each column
    @test all(w.L[:, :, 2:end] .>= w.L[:, :, 1:end-1])
    @test w.step == 200 && w.t ≈ 200 * p.dt
end

@testset "Float32 agrees with Float64 oracle at short times" begin
    C = 4
    r64 = random_rule(MersenneTwister(5), C, Float64)
    r32 = convert(Rule{C,Float32}, r64)
    p64 = Physics{Float64}(); p32 = Physics{Float32}(p64)
    w64 = World(CPU(), r64, (8, 8, 8); T=Float64, h_scale=0.2, seed=7)
    w32 = World(CPU(), r32, (8, 8, 8); T=Float32, h_scale=0.2, seed=7)
    w64 = run!(light!(w64, r64, p64), r64, p64, 50)
    w32 = run!(light!(w32, r32, p32), r32, p32, 50)
    @test maximum(abs.(Float64.(w32.s) .- w64.s)) < 1e-4
    @test maximum(abs.(Float64.(w32.h) .- w64.h)) < 1e-3
end

@testset "deposition pays and shading works" begin
    # A rule whose deposition readout is strongly on should convert nutrient to solid,
    # and the solid should darken the column beneath it.
    C = 4
    θ = zeros(2C * C + 8C)
    θ[2C*C+4C+C+1:2C*C+4C+2C] .= 5.0   # vdep large positive; h≈0 → softplus(0)=log2 anyway
    r = Rule{C,Float64}(θ)
    p = Physics{Float64}(k_dep=1.0, k_ero=0.0, D0=0.0)
    w = World(CPU(), r, (4, 4, 6); n0=0.5, h_scale=0.0, seed=1)
    w = run!(light!(w, r, p), r, p, 400)
    @test mean(w.s) > 0.4
    @test mean(w.n) < 0.1
    @test w.L[1, 1, 1] < w.L[1, 1, 6]
    @test total_mass(w) ≈ 0.5 * 4 * 4 * 6 atol = 1e-10
end

@testset "copy and timescales" begin
    r = random_rule(3)
    p = Physics{Float32}()
    w = World(CPU(), r, (4, 4, 4))
    w2 = copy(w)
    @test w2.s == w.s && w2.s !== w.s
    ts = timescales(p)
    @test ts.tau_dep ≈ 10 && ts.ero_over_dep ≈ 10
end
