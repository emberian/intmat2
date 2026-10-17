# Stencil isotropy check (2D): a 48×48×1 Float64 CPU world from the homogeneous steady
# state + 1e-6 noise, tied-10's rule retied at β = 0.4, grown in the linear regime;
# s-field power at the dominant |k| in 8 angular bins over [0, π), for stencil 0 and 1.
# usage: julia --project=. bench/isotropy.jl [recipe.toml] [beta]
using Intmat2, KernelAbstractions, Random, Printf, TOML
using Intmat2: light!, Analysis
const AN = Analysis
rec = TOML.parsefile(get(ARGS, 1, joinpath(@__DIR__, "..", "recipes", "tied-20261016-10.toml")))
β = parse(Float64, get(ARGS, 2, "0.4"))
C = 8
θ = Float64.(rec["theta"]); P1 = 2C * C + 8C
θ[P1+1:end] .= -β .* θ[C*C+1:2C*C]                     # Wnb2 = −β Wnb
rule = Rule{C,Float64}(θ)
base = Physics{Float64}(; (Symbol(k) => Float64(v) for (k, v) in rec["physics"])...)
N = 48
for stencil in (0, 1)
    p = Physics{Float64}(; (f => getfield(base, f) for f in fieldnames(Physics) if f != :stencil)..., stencil=Int32(stencil))
    ss = AN.steady_state(rule, p; depth=1)
    wd = World(CPU(), rule, (N, N, 1); T=Float64, h_scale=0.0)
    AN.set_uniform!(wd, ss)
    rng = MersenneTwister(7)
    noise() = (x = 1e-6 .* randn(rng, N, N, 1); x .- sum(x) / length(x))
    for c in 1:C; wd.h[c, :, :, :] .+= noise(); end
    wd.s .+= noise(); wd.n .+= noise(); wd.w .+= noise()
    wd = light!(wd, rule, p)
    nsteps = 0
    while maximum(abs.(wd.s .- ss.s)) < 1e-4 && nsteps < round(Int, 200 / p.dt)
        wd = step!(wd, rule, p); nsteps += 1
    end
    Ps = AN.power2d(wd.s[:, :, 1] .- ss.s)
    # lattice wavevectors in (-π, π]
    kk(a) = (a - 1 <= N ÷ 2 ? a - 1 : a - 1 - N) * 2π / N
    kmag = [hypot(kk(a), kk(b)) for a in 1:N, b in 1:N]
    ang = [mod(atan(kk(b), kk(a)), π) for a in 1:N, b in 1:N]
    dk = 2π / N
    rbin = round.(Int, kmag ./ dk)
    ringpow = [sum(Ps[rbin .== r]) for r in 1:N÷2]
    r = argmax(ringpow)
    bins = zeros(8)
    for I in eachindex(Ps)
        rbin[I] == r || continue
        bins[min(8, 1 + floor(Int, ang[I] / (π / 8)))] += Ps[I]
    end
    @printf("stencil %d: %d steps (t=%.1f), max|δs| %.1e, dominant |k| = %.3f (λ = %.1f cells), angular power max/min = %.3g (growth-rate spread ln(max/min)/2t = %.2f)  bins %s\n",
            stencil, nsteps, nsteps * p.dt, maximum(abs.(wd.s .- ss.s)), r * dk, N / r, maximum(bins) / max(minimum(bins), eps()),
            log(maximum(bins) / max(minimum(bins), eps())) / (2 * nsteps * p.dt),
            join([@sprintf("%.2g", b / maximum(bins)) for b in bins], " "))
end
