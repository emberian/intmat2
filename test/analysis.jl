using Intmat2, Test, Random, LinearAlgebra, Statistics, KernelAbstractions
using Intmat2.Analysis
const AN = Intmat2.Analysis

# Rules and physics below come from results/atlas-20261016.json (local-light class);
# dt is raised within the stated bounds (6 D0 dt ≤ 1, dt ≤ tau_h) to keep runs short,
# and every comparison is against the exact one-step map at that dt.
const TYPEII = (C=4, seed=838841668, sc=(self=0.5, nb=0.8, inp=4.0, bias=0.5, read=2.0), M=0.5,
                p=Physics{Float64}(k_dep=0.3, k_ero=0.01, k_photo=1.0, D0=1.0, g1=-0.5, dt=1/6))
const BAND = (C=4, seed=1184891871, sc=(self=2.0, nb=0.8, inp=2.0, bias=0.5, read=0.5), M=0.5,
              p=Physics{Float64}(k_dep=0.1, k_ero=0.1/3, k_photo=0.05, D0=0.2, g1=0.0, dt=0.2))
const STABLE = (C=4, seed=1443778830, sc=(self=1.5, nb=0.2, inp=1.0, bias=0.1, read=4.0), M=1.0,
                p=Physics{Float64}(k_dep=3.0, k_ero=1.0, k_photo=1.0, D0=1.0, g1=0.5, dt=1/6))

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
