# WorldBatch: B worlds in one launch equal B worlds stepped alone (bitwise on the CPU).
using Intmat2: run!, light!, world, nworlds, device_rules

@testset "batch: bitwise equal to separate worlds (CPU)" begin
    for T in (Float64, Float32), dims in ((7, 5, 9), (6, 4, 4))   # 315 voxels: groups straddle worlds
        C = 4; B = 3
        rules = [random_rule(MersenneTwister(10 + b), C, T; self=1.0, nb=1.0) for b in 1:B]
        p = Physics{T}()
        seeds = [3, 8, 21]
        wb = WorldBatch(CPU(), C, dims, B; T=T, seeds=seeds, h_scale=0.3, s0=0.4)
        @test nworlds(wb) == B && size(wb) == dims
        wb = run!(light!(wb, rules, p), rules, p, 40)
        for b in 1:B
            w = World(CPU(), C, dims; T=T, seed=seeds[b], h_scale=0.3, s0=0.4)
            w = run!(light!(w, rules[b], p), rules[b], p, 40)
            wbb = world(wb, b)
            @test wbb.h == w.h && wbb.s == w.s && wbb.n == w.n && wbb.w == w.w && wbb.L == w.L
            @test total_mass(wb)[b] ≈ total_mass(w) rtol = 1e-12
        end
        # one shared rule, and a pre-uploaded rule matrix
        wb1 = run!(light!(WorldBatch(CPU(), C, dims, B; T=T, seeds=seeds), rules[2], p), rules[2], p, 10)
        wbm = WorldBatch(CPU(), C, dims, B; T=T, seeds=seeds)
        θ = device_rules(CPU(), fill(rules[2], B))
        wbm = run!(light!(wbm, θ, p), θ, p, 10)
        @test wb1.h == wbm.h && wb1.s == wbm.s
        w2 = run!(light!(World(CPU(), C, dims; T=T, seed=seeds[3]), rules[2], p), rules[2], p, 10)
        @test world(wb1, 3).h == w2.h
        @test wb1.step == 10 && wb1.t ≈ 10 * p.dt
    end
    @test_throws ArgumentError step!(WorldBatch(CPU(), 4, (4, 4, 4), 2), [random_rule(4)], Physics{Float32}())
end

@testset "two-shell step: conservation and the shell acts" begin
    C = 8
    r = tied_rule(MersenneTwister(1), C, Float64; nb1=2, beta=0.4)
    p = Physics{Float64}(dt=0.02)
    w = light!(World(CPU(), r, (12, 10, 8); T=Float64, h_scale=0.3, s0=0.3, seed=4), r, p)
    m0 = total_mass(w)
    w = run!(w, r, p, 200)
    @test total_mass(w) ≈ m0 atol = 1e-9 * m0
    @test all(0 .<= w.s .<= 1) && all(w.n .>= 0) && all(w.w .>= 0) && all(-1 .<= w.h .<= 1)
    # the same rule with Wnb2 dropped evolves differently; ambient light is additive
    r1 = Rule{C,Float64}(vec(r)[1:2C*C+8C])
    w1 = run!(light!(World(CPU(), r1, (12, 10, 8); T=Float64, h_scale=0.3, s0=0.3, seed=4), r1, p), r1, p, 20)
    w2 = run!(light!(World(CPU(), r, (12, 10, 8); T=Float64, h_scale=0.3, s0=0.3, seed=4), r, p), r, p, 20)
    @test w1.h != w2.h
    pa = Physics{Float64}(dt=0.02, L_ambient=0.25)
    wa = light!(World(CPU(), r, (6, 6, 6); T=Float64, seed=1), r, pa)
    wz = light!(World(CPU(), r, (6, 6, 6); T=Float64, seed=1), r, p)
    @test wa.L ≈ wz.L .+ 0.25
end
