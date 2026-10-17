using Intmat2, Test, Random, LinearAlgebra, Statistics, KernelAbstractions
# Fixtures were characterized at the original absorption alpha_s=alpha_p=2.0; the
# Physics defaults later dropped to 0.05/0.1, so every fixture sets them explicitly.
using Intmat2.Analysis
const AN = Intmat2.Analysis

# Rules and physics below come from results/atlas-20261016.json (local-light class);
# dt is raised within the stated bounds (6 D0 dt ≤ 1, dt ≤ tau_h) to keep runs short,
# and every comparison is against the exact one-step map at that dt.
const TYPEII = (C=4, seed=838841668, sc=(self=0.5, nb=0.8, inp=4.0, bias=0.5, read=2.0), M=0.5,
                p=Physics{Float64}(k_dep=0.3, k_ero=0.01, k_photo=1.0, D0=1.0, g1=-0.5, dt=1/6, alpha_s=2.0, alpha_p=2.0))
const BAND = (C=4, seed=1184891871, sc=(self=2.0, nb=0.8, inp=2.0, bias=0.5, read=0.5), M=0.5,
              p=Physics{Float64}(k_dep=0.1, k_ero=0.1/3, k_photo=0.05, D0=0.2, g1=0.0, dt=0.2, alpha_s=2.0, alpha_p=2.0))
const STABLE = (C=4, seed=1443778830, sc=(self=1.5, nb=0.2, inp=1.0, bias=0.1, read=4.0), M=1.0,
                p=Physics{Float64}(k_dep=3.0, k_ero=1.0, k_photo=1.0, D0=1.0, g1=0.5, dt=1/6, alpha_s=2.0, alpha_p=2.0))

mkrule(c) = random_rule(MersenneTwister(c.seed), c.C, Float64; c.sc...)

@testset "analysis: steady state is a fixed point of step!" begin
    for c in (TYPEII, BAND, STABLE)
        r = mkrule(c)
        ss = steady_state(r, c.p; M=c.M)
        @test ss.converged && !ss.clamped
        @test ss.n + ss.s + ss.w ≈ c.M
        # one layer: L = L0 exp(-a/2) is exactly the depth-1 light
        wd = World(CPU(), r, (3, 3, 1); T=Float64, h_scale=0.0)
        AN.set_uniform!(wd, ss)
        wd = Intmat2.light!(wd, r, c.p)
        @test maximum(abs.(wd.L .- ss.L)) < 1e-12
        wd = Intmat2.run!(wd, r, c.p, 200)
        @test maximum(abs.(wd.s .- ss.s)) < 1e-11
        @test maximum(abs.(wd.h .- ss.h)) < 1e-11
    end
end

@testset "analysis: stepmap is the linearization of step! and I + dt J to O(dt²)" begin
    c = TYPEII
    r = mkrule(c); C = c.C
    ss = steady_state(r, c.p; M=c.M)
    # a single cos mode along x through the real stepper
    Nx, m = 16, 3
    kx = 2π * m / Nx
    κ = kappa(kx, 0.0)
    A = stepmap(r, c.p, ss, κ; light=:local)
    v = randn(MersenneTwister(4), C + 3)
    ε = 1e-7
    wd = World(CPU(), r, (Nx, 2, 1); T=Float64, h_scale=0.0)
    AN.set_uniform!(wd, ss)
    for i in 1:Nx
        φ = ε * cos(kx * (i - 1))
        wd.h[:, i, :, 1] .+= φ .* v[1:C]
        wd.n[i, :, 1] .+= φ * v[C+1]; wd.s[i, :, 1] .+= φ * v[C+2]; wd.w[i, :, 1] .+= φ * v[C+3]
    end
    wd = Intmat2.light!(wd, r, c.p)
    wd = step!(wd, r, c.p)
    proj(f) = 2 / Nx * sum(f[i] * cos(kx * (i - 1)) for i in 1:Nx)
    out = vcat([proj(wd.h[cc, :, 1, 1] .- ss.h[cc]) for cc in 1:C],
               proj(wd.n[:, 1, 1] .- ss.n), proj(wd.s[:, 1, 1] .- ss.s), proj(wd.w[:, 1, 1] .- ss.w))
    @test norm(out ./ ε .- A * v) < 1e-5 * norm(A * v)
    # consistency of the continuous Jacobian with the map
    J = jacobian(r, c.p, ss, κ; light=:local)
    e(dt) = (p2 = Physics{Float64}(; (f => getfield(c.p, f) for f in fieldnames(Physics) if f != :dt)..., dt=dt);
             norm(stepmap(r, p2, ss, κ; light=:local) - I - dt * J))
    @test e(0.01) / e(0.005) ≈ 4 rtol = 0.05
end

function mode_check(c; N=32, t1, t2)
    r = mkrule(c)
    ss = steady_state(r, c.p; M=c.M)
    P = AN.Pencil(r, c.p, ss; light=:local)
    d = dispersion(P)
    Kl = AN.lattice_kappa(N, N)
    pred = [AN.map_rate(P, κ) for κ in Kl]
    m = AN.measure_mode_rates(r, c.p, ss; N, t1, t2)
    mask = trues(N, N); mask[1, 1] = false
    (d=d, Kl=Kl, pred=pred, m=m, mask=mask, cls=classify(d))
end

@testset "analysis: Turing prediction vs the real stepper" begin
    for (c, t1, t2) in ((TYPEII, 50.0, 150.0), (BAND, 100.0, 400.0))
        x = mode_check(c; t1, t2)
        @test x.cls == :turing
        top = x.mask .& (x.pred .> maximum(x.pred[x.mask]) - 0.05)
        # per-mode growth rates: dispersion relation reproduced mode by mode
        @test median(abs.(x.m.rates[top] .- x.pred[top])) < 1e-3
        # fastest measured mode sits at the predicted κ*
        jm = argmax(ifelse.(x.mask, x.m.rates, -Inf))
        @test abs(x.Kl[jm] - x.d.kappa_star) < 0.15
        @test x.m.rates[jm] ≈ x.d.map_rate_star rtol = 0.05
        # the power spectrum of s peaks near κ*
        Ps = AN.power2d(x.m.world.s[:, :, 1])
        @test abs(x.Kl[argmax(Ps)] - x.d.kappa_star) < 0.3
        # still linear
        @test x.m.norms[end] / 32 < 1e-2
    end
end

@testset "analysis: predicted-stable rule decays at the predicted rate" begin
    x = mode_check(STABLE; t1=20.0, t2=80.0)
    @test x.cls == :stable
    @test x.d.map_rate_max < 0
    slow = maximum(x.pred[x.mask])          # slowest lattice mode (long-wave mass mode)
    sel = x.mask .& (x.pred .> slow - 0.05)
    @test median(abs.(x.m.rates[sel] .- x.pred[sel])) < 1e-3
    n = x.m.norms; dt = STABLE.p.dt
    i1 = round(Int, 20 / dt) + 1; i2 = length(n)
    ts = (i1-1:i2-1) .* dt
    slope = cov(ts, log.(n[i1:i2])) / var(ts)
    # the norm is a sum over modes; it decays no slower than the slowest mode, and
    # the predicted spectrum (measured amplitudes at t1, predicted rates) gives its
    # decay over [t1, t2]
    @test slope <= slow < 0
    a1 = x.m.amp1 .* x.mask
    pred_ratio = sqrt(sum(abs2.(a1) .* exp.(2 .* x.pred .* 60.0)) / sum(abs2, a1))
    meas_ratio = norm(x.m.amp2 .* x.mask) / norm(a1)
    @test meas_ratio ≈ pred_ratio rtol = 0.02
    @test meas_ratio < 0.6
end

@testset "analysis: classify" begin
    base = (re0=-0.1, im0=0.0, lambda_max=-0.01, kappa_star=1.0, imag_star=0.0)
    @test classify(base) == :stable
    @test classify(merge(base, (lambda_max=0.01,))) == :turing
    @test classify(merge(base, (lambda_max=0.01, imag_star=0.3))) == :wave
    @test classify(merge(base, (lambda_max=0.01, kappa_star=8.0))) == :gridscale
    @test classify(merge(base, (re0=0.02, im0=0.5, lambda_max=0.02, kappa_star=1e-3))) == :hopf
    @test classify(merge(base, (re0=0.02, lambda_max=0.02, kappa_star=1e-3))) == :homogeneous
    @test classify(merge(base, (re0=0.02, lambda_max=0.2))) == :mixed
    @test axis_wavelength(2.0) ≈ 4 && axis_wavelength(4.0) ≈ 2
end

@testset "analysis: stable_dt" begin
    c = BAND
    r = mkrule(c)
    p = AN.withdt(c.p, 0.5)          # satisfies 6 D0 dt ≤ 1 and dt ≤ tau_h, yet unstable
    dt = stable_dt(r, p; M=c.M)
    @test 0 < dt < 0.5
    @test dt <= min(p.tau_h, 1 / (6p.D0))
    run200(dt) = (pp = AN.withdt(p, dt); ss = steady_state(r, pp; M=c.M);
                  m = AN.measure_mode_rates(r, pp, ss; N=16, t1=dt * 100, t2=dt * 200); m.norms)
    n_bad = run200(0.5)
    @test n_bad[end] > 1e3 * n_bad[1]        # the stepper blows up at the rejected dt
    n_ok = run200(dt)
    @test all(isfinite, n_ok) && n_ok[end] < 10 * n_ok[1]   # physical Turing growth only
end

@testset "analysis: layered steady state" begin
    r = mkrule(TYPEII)
    layers = layered_steady_state(r, TYPEII.p, 6; M=TYPEII.M)
    top = steady_state(r, TYPEII.p; M=TYPEII.M, depth=1)
    @test layers[1].s ≈ top.s && layers[1].L ≈ top.L
    @test all(l -> l.converged, layers)
    @test issorted([l.tau_above for l in layers]) && issorted([l.L for l in layers], rev=true)
    col = AN.relax_column(r, TYPEII.p, layers; steps=10)
    @test sum(col.s) + sum(col.n) + sum(col.w) ≈ 6 * TYPEII.M
end

@testset "analysis: turing_prone_rule" begin
    r = turing_prone_rule(MersenneTwister(3), 8, Float64)
    @test r == turing_prone_rule(MersenneTwister(3), 8, Float64)
    @test minimum(eigvals(Symmetric(Matrix(r.Wnb)))) >= -1e-12   # no anti-diffusion
    @test turing_prone_rule(MersenneTwister(3), 8) isa Rule{8,Float32}
end

@testset "analysis: two-scale coupling symbols" begin
    c = TYPEII
    r = mkrule(c); ss = steady_state(r, c.p; M=c.M)
    P2 = Pencil2(r, c.p, ss)
    @test norm(jacobian(r, c.p, ss, 1.3) - (P2.J0 - 1.3 * (P2.K1 + P2.KD))) < 1e-12
    d1 = dispersion(r, c.p, ss); d2 = dispersion2(P2; nk=48)
    @test d2.lambda_max ≈ d1.lambda_max rtol = 1e-3
    @test kappa2(π, 0.0) ≈ 0 atol = 1e-12          # the radius-2 shell is blind to the checkerboard
    @test kappa2(0.3, 0.2) ≈ kappa(0.6, 0.4)
    # tied Mexican hat: scalar per-axis optimum cos k* = 1/(4β)
    W = Matrix(1.0I, c.C, c.C)
    P3 = Pencil2(r, c.p, ss; Wnb=4.0 .* W, Wnb2=-0.5 .* 4.0 .* W)
    d3 = dispersion2(P3; nk=48)
    @test classify(d3) == :turing && 3.5 < d3.wavelength < 7
    ssa = steady_state(r, c.p; M=c.M, L_ambient=0.5)
    @test ssa.L > 0.5 && ssa.converged
end
