using Intmat2, Test, Random, KernelAbstractions, Statistics
using Intmat2: EvalConfig, evaluate, evaluate_one, structure, temporal, persistence,
    nondominated_rank, nsga_select, hypervolume, evolve, write_recipe, read_recipe, rule_scale

@testset "evolve: objective sanity" begin
    C = 4; np = 2C * C + 8C
    p = Physics{Float32}(dt=0.1f0)
    cfg = EvalConfig(dims=(12, 12, 6), T=200, cont=60, rec_every=10, probe_every=2, nprobe=6,
                     concurrency=2)
    # saturating builder: h → tanh(b) ≈ 1, deposition readout strongly on, nothing else
    θ = zeros(np)
    θ[2C*C+4C+1:2C*C+5C] .= 3.0          # b
    θ[2C*C+5C+1:2C*C+6C] .= 5.0          # vdep
    sat = Rule{C,Float32}(Float32.(θ))
    zero_rule = Rule{C,Float32}(zeros(Float32, np))
    rs = evaluate([sat, zero_rule], p, cfg)
    @test rs[1].mean_s > 0.8
    @test rs[1].persistence < 0.05
    @test rs[1].structure < 0.05
    @test rs[2].temporal == 0.0          # h relaxes to the fixed point 0
    @test rs[2].t_fixed == 1.0
    for r in rs, k in Intmat2.ALL_OBJECTIVES
        @test isfinite(r[k])
    end
    @test all(abs(r.mass_drift) < 1e-3 for r in rs)
    # structure: a lateral stripe of wavelength 8 is all in band; white noise is ~0
    s = [0.3 + 0.2 * sin(2π * i / 8) for i in 1:32, j in 1:32, k in 1:4]
    st = structure(s)
    @test st.score > 0.9 && st.lambda ≈ 8
    @test structure(0.3 .+ 0.2 .* rand(MersenneTwister(1), 32, 32, 4)).score < 0.1
    # temporal: broadband noise ≫ sinusoid > constant
    M = 200
    sine = reshape([0.3 * sin(2π * 5.0 * m / M) for m in 1:M], 1, 1, M)
    noise = reshape(0.3 .* randn(MersenneTwister(2), M), 1, 1, M)
    @test temporal(noise).score > 0.7
    @test temporal(sine).score < 0.35 && temporal(sine).periodic == 1.0
    @test temporal(zeros(1, 1, M)).score == 0.0
end

@testset "evolve: Pareto machinery and search" begin
    F = [1.0 0.0 0.5 0.4; 0.0 1.0 0.5 0.4]
    @test nondominated_rank(F) == [1, 1, 1, 2]
    ch, _, _ = nsga_select(F, 2)
    @test sort(ch) == [1, 2]               # extremes have infinite crowding
    @test hypervolume([0.5; 0.5;;]) ≈ 0.25 atol = 0.01
    C = 2
    p = Physics{Float32}(dt=0.1f0)
    cfg = EvalConfig(dims=(8, 8, 4), T=60, cont=20, rec_every=10, probe_every=2, nprobe=4)
    arch, P = evolve(C, p, cfg; pop=4, gens=2, seed=3)
    @test length(arch) == 8 && length(P) == 4
    arch2, _ = evolve(C, p, cfg; pop=4, gens=2, seed=3)
    @test [a.theta for a in arch] == [a.theta for a in arch2]   # re-runnable from the seed
    @test arch[1].theta ≈ rule_scale(C) .* arch[1].z
    path = tempname() * ".toml"
    r = Rule{C,Float32}(Float32.(arch[5].theta))
    write_recipe(path, r, p, cfg; objectives=arch[5].res)
    r2, p2, _ = read_recipe(path)
    @test r2 == r && p2 == p
end
