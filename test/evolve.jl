using Intmat2, Test, Random, KernelAbstractions, Statistics, LinearAlgebra
using Intmat2: EvalConfig, evaluate, evaluate_one, structure, temporal, persistence,
    nondominated_rank, nsga_select, hypervolume, evolve, write_recipe, read_recipe, rule_scale

@testset "evolve: objective sanity" begin
    C = 4; np = 2C * C + 8C   # one-ring layout (Wnb2 = 0)
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
    @test all(haskey(r, :dt_ok) for r in rs) && rs[2].dt_ok
    @test Intmat2.objvec(merge(rs[2], (dt_ok=false,)), Intmat2.SELECT_OBJECTIVES) == zeros(5)
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
    @test arch[1].theta[1:length(arch[1].z)] ≈ rule_scale(C) .* arch[1].z
    path = tempname() * ".toml"
    r = Rule{C,Float32}(Float32.(arch[5].theta))
    write_recipe(path, r, p, cfg; objectives=arch[5].res)
    r2, p2, _ = read_recipe(path)
    @test r2 == r && p2 == p
end

@testset "evolve: WorldBatch evaluator matches per-world evaluator" begin
    C = 4
    p = Physics{Float32}(dt=0.1f0)
    cfg = EvalConfig(dims=(8, 8, 6), T=80, cont=20, rec_every=10, probe_every=2, nprobe=4)
    rules = [random_rule(MersenneTwister(i), C) for i in 1:3]
    a = Intmat2.evaluate_batch(rules, p, cfg)
    b = [Intmat2.evaluate_one(r, p, cfg) for r in rules]
    for (x, y) in zip(a, b), k in (Intmat2.ALL_OBJECTIVES..., :mean_s, :std_h)
        @test x[k] ≈ y[k] atol = 1e-4
    end
end

@testset "evolve: regions, tracking, manifold space" begin
    using Intmat2: label_components, track, life_objectives, TiedSpace, space_theta, space_prior,
        space_scale, evaluate_life, LifeConfig, choose_dt
    # two blobs that merge: one merge event, nothing else
    a = falses(16, 16, 4); b = falses(16, 16, 4)
    a[2:4, 2:4, 1:2] .= true; a[8:10, 2:4, 1:2] .= true
    b[2:4, 2:4, 1:2] .= true; b[8:10, 2:4, 1:2] .= true; b[5:7, 3, 1] .= true
    la, sa = label_components(a); lb, sb = label_components(b)
    @test length(sa) == 2 && length(sb) == 1
    tr = track(la, lb)
    @test (tr.births, tr.deaths, tr.splits, tr.merges) == (0, 0, 0, 1)
    tr2 = track(lb, la)                                   # and back: one split
    @test (tr2.births, tr2.deaths, tr2.splits, tr2.merges) == (0, 0, 1, 0)
    # periodic wrap in x joins a blob across the boundary; z does not wrap
    c = falses(8, 8, 4); c[1, 2:3, 1:2] .= true; c[8, 2:3, 1:2] .= true
    @test length(label_components(c; min_size=1)[2]) == 1
    d = falses(8, 8, 4); d[2:3, 2:3, 1] .= true; d[2:3, 2:3, 4] .= true
    @test length(label_components(d; min_size=1)[2]) == 2
    # a frozen field: no events, turnover 0
    s = [0.1 + 0.8 * (sin(2π * i / 8) * sin(2π * j / 8) > 0.5) for i in 1:32, j in 1:32, k in 1:4]
    lo = life_objectives([s, s, s, s], 2.0)
    @test lo.turnover == 0.0 && lo.births + lo.deaths + lo.splits + lo.merges == 0
    @test lo.n_regions > 1 && lo.localization > 0.3
    # manifold: Wnb PSD, Wnb2 = −β Wnb exactly when tied
    sp = TiedSpace(4)
    u = space_prior(sp, MersenneTwister(1))
    r = Rule{4,Float64}(space_theta(sp, u))
    β = u[2 * 16 + 2]
    @test Matrix(r.Wnb2) ≈ -β .* Matrix(r.Wnb)
    @test minimum(eigvals(Symmetric(Matrix(r.Wnb)))) > -1e-9
    @test length(space_scale(TiedSpace(4; untied=true))) == length(u) + 16
    cfgL = LifeConfig(dt_max=0.1, dt_min=0.004)
    @test choose_dt(0.2, cfgL) == 0.1 && choose_dt(0.05, cfgL) == 0.025 && choose_dt(0.001, cfgL) == 0.004
    # evaluate_life runs end to end on a tiny grid with per-rule steps
    p = Physics{Float32}(g1=-0.5f0, alpha_s=0.05f0, alpha_p=0.1f0)
    cfg = LifeConfig(dims=(8, 8, 6), t_main=8.0, t_cont=2.0, cap_dt=1.0, probe_dt=0.5, nprobe=4)
    rules = [Rule{4,Float32}(Float32.(space_theta(sp, space_prior(sp, MersenneTwister(i))))) for i in 1:3]
    res = evaluate_life(rules, p, cfg)
    @test length(res) == 3 && all(r -> all(isfinite, (r.localization, r.population, r.turnover, r.recovery)), res)
    @test all(r -> r.dt <= 0.1 && r.dt >= 0.004, res)
    arch, _ = evolve(4, p, cfg; pop=3, gens=2, seed=2, eval=evaluate_life, objectives=Intmat2.LIFE_SELECT,
                     space=TiedSpace(4; untied=true))
    @test length(arch) == 6
end
