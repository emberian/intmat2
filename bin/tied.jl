# Tied two-shell rules (tied_rule): stable dt survey and recipes.
#
#   julia -t 8 --project=. bin/tied.jl
#
# Physics: package defaults (translucent alpha_s 0.05, alpha_p 0.1) with M = n0 = 0.5
# (the low-mass Turing finding). Survey: C = 8, nb1 ∈ {1,2,4}, beta ∈ {0.3,0.4,0.5},
# 24 seeds each, tied_rule's default scales. Recipes: ten Turing rules with
# λ_wave ∈ [6, 12] and growth ≥ 0.01 at depth 1 that stay Turing at depths 24 and 48 of a 48-layer slab,
# dt = min(0.05, 0.8 stable_dt), steps = 8 e-folds. Writes results/tied-<date>.json
# and recipes/tied-<date>-NN.toml.

using Intmat2, Random, LinearAlgebra, Printf, TOML, Statistics
using Intmat2.Analysis
const AN = Intmat2.Analysis
BLAS.set_num_threads(1)
const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATE = Libc.strftime("%Y%m%d", time())
const M = 0.5
const NZ = 48
const P0 = Physics{Float64}()

function analyse(r, p; depths=(1,))
    out = []
    for dep in depths
        ss = steady_state(r, p; M, depth=dep)
        ss.converged || return nothing
        d = dispersion(r, p, ss)
        push!(out, (ss=ss, d=d, cls=classify(d)))
    end
    out
end

function survey()
    rows = []
    for nb1 in (1.0, 2.0, 4.0), beta in (0.3, 0.4, 0.5)
        res = Vector{Any}(undef, 24)
        Threads.@threads for s in 1:24
            r = tied_rule(MersenneTwister(1000s), 8, Float64; nb1, beta)
            a = analyse(r, P0)
            res[s] = a === nothing ? nothing :
                (cls=a[1].cls, wl=a[1].d.wavelength, lam=a[1].d.lambda_max,
                 sdt=Float64(stable_dt(r, P0; M, ss=a[1].ss)))
        end
        ok = filter(!isnothing, res)
        sdt = [x.sdt for x in ok]
        tur = filter(x -> x.cls === :turing, ok)
        row = (nb1=nb1, beta=beta, n=length(ok), turing=length(tur) / 24,
               stable_dt_median=median(sdt), stable_dt_min=minimum(sdt), stable_dt_max=maximum(sdt),
               wl_median=isempty(tur) ? NaN : median([x.wl for x in tur]),
               lam_median=isempty(tur) ? NaN : median([x.lam for x in tur]))
        @printf("nb1 %.0f β %.1f: converged %2d, turing %.2f, stable_dt median %.3f [%.3f, %.3f], Turing λ_wave %.1f, λ %.3g\n",
                nb1, beta, row.n, row.turing, row.stable_dt_median, row.stable_dt_min, row.stable_dt_max, row.wl_median, row.lam_median)
        push!(rows, row)
    end
    rows
end

function search(ntarget=10)
    found = []
    rng = MersenneTwister(20261016)
    # The fastest mode of a tied rule is diagonal (kx = ky = k_axis*), √2 shorter than
    # the axis wavelength from cos k* = 1/(4β); λ_wave = 2π/|k*| ∈ [6, 12] needs
    # β ≈ 0.30–0.34, where growth ∝ nb1 (4β−1)²/(4β) is small, hence larger nb1.
    cands = [(seed=rand(rng, 1:2^31-1), nb1=rand(rng, (6.0, 8.0, 12.0, 16.0)), beta=rand(rng, (0.29, 0.30, 0.31, 0.32)),
              inp=rand(rng, (0.5, 1.0, 2.0)), read=rand(rng, (0.5, 1.0, 2.0))) for _ in 1:600]
    res = Vector{Any}(undef, length(cands))
    Threads.@threads for i in eachindex(cands)
        c = cands[i]
        r = tied_rule(MersenneTwister(c.seed), 8, Float64; nb1=c.nb1, beta=c.beta, inp=c.inp, read=c.read)
        a = analyse(r, P0; depths=(1, NZ ÷ 2, NZ))
        res[i] = nothing
        a === nothing && continue
        all(x -> x.cls === :turing, a) || continue
        6 <= a[1].d.wavelength <= 12 || continue
        a[1].d.lambda_max >= 0.01 || continue
        a[1].ss.clamped && continue
        sdt = Float64(stable_dt(r, P0; M, ss=a[1].ss))
        res[i] = (c=c, r=r, a=a, sdt=sdt)
    end
    good = filter(!isnothing, res)
    sort!(good, by=x -> -x.a[1].d.lambda_max)
    @printf("search: %d of %d candidates are Turing at depths 1, %d, %d with λ_wave 6–12\n", length(good), length(cands), NZ ÷ 2, NZ)
    # spread over wavelengths: round-robin over β (fastest first within each β)
    groups = [filter(x -> x.c.beta == b, good) for b in sort(unique([x.c.beta for x in good]))]
    sel = []
    while length(sel) < ntarget && any(!isempty, groups)
        for g in groups
            isempty(g) || length(sel) >= ntarget || push!(sel, popfirst!(g))
        end
    end
    sel
end

function write_recipes(sel)
    names = String[]
    for (j, x) in enumerate(sel)
        d = x.a[1].d
        dt = min(0.05, 0.8 * x.sdt)
        p = Physics{Float64}(; (f => (f === :dt ? dt : getfield(P0, f)) for f in fieldnames(Physics))...)
        name = @sprintf("tied-%s-%02d", DATE, j)
        rec = Dict{String,Any}("name" => name, "C" => 8, "seed" => x.c.seed, "rng" => "MersenneTwister",
            "construct" => "tied_rule(MersenneTwister(seed), C, Float64; nb1, beta, inp, read) == Rule{C,Float64}(theta)",
            "nb1" => x.c.nb1, "beta" => x.c.beta, "rule_scales" => Dict("self" => 0.8, "inp" => x.c.inp, "bias" => 0.1, "read" => x.c.read),
            "theta" => collect(vec(x.r)), "N" => 128, "Nz" => NZ, "n0" => M, "s0" => 0.0, "h_scale" => 0.1,
            "steps" => ceil(Int, 8 / (d.lambda_max * dt)),
            "physics" => Dict(String(f) => Float64(getfield(p, f)) for f in fieldnames(Physics)),
            "predicted" => Dict("class" => "turing", "lambda_max" => d.lambda_max, "wavelength" => d.wavelength,
                "kx" => d.kx, "ky" => d.ky, "kappa_star" => d.kappa_star, "imag_star" => d.imag_star, "re0" => d.re0,
                "stable_dt" => x.sdt, "light_model" => "frozen", "depth" => 1,
                "wavelength_depth24" => x.a[2].d.wavelength, "lambda_depth24" => x.a[2].d.lambda_max,
                "wavelength_depth48" => x.a[3].d.wavelength, "lambda_depth48" => x.a[3].d.lambda_max,
                "steady_s" => x.a[1].ss.s, "steady_n" => x.a[1].ss.n, "steady_w" => x.a[1].ss.w, "steady_L" => x.a[1].ss.L))
        open(joinpath(ROOT, "recipes", name * ".toml"), "w") do io
            println(io, "# bin/tied.jl: tied two-shell rule, predicted Turing at depth 1 (and depths $(NZ ÷ 2), $NZ).")
            println(io, "# `theta` is vec(rule) including W_nb2; `steps` = 8 e-folds at this dt (= min(0.05, 0.8 stable_dt)).")
            TOML.print(io, rec; sorted=true)
        end
        push!(names, name)
        @printf("  %s nb1 %.0f β %.2f λ %.3f λ_wave %.1f (d24 %.1f, d48 %.1f) stable_dt %.3f dt %.3f steps %d\n", name,
                x.c.nb1, x.c.beta, d.lambda_max, d.wavelength, x.a[2].d.wavelength, x.a[3].d.wavelength, x.sdt, dt, rec["steps"])
    end
    names
end

include(joinpath(@__DIR__, "atlas.jl"))   # jsonv
t0 = time()
rows = survey()
sel = search()
names = write_recipes(sel)
open(joinpath(ROOT, "results", "tied-$DATE.json"), "w") do io
    jsonv(io, Dict{String,Any}("command" => "julia -t $(Threads.nthreads()) --project=. bin/tied.jl",
        "commit" => strip(read(`git -C $ROOT rev-parse --short HEAD`, String)), "M" => M,
        "physics" => "Physics{Float64}() defaults", "survey" => [Dict(String(k) => v for (k, v) in pairs(r)) for r in rows],
        "recipes" => names, "elapsed_s" => time() - t0))
end
