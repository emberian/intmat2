# Tied two-shell rules on the isotropic 19-point stencil (Physics.stencil = 1): recipes.
#
#   julia -t 8 --project=. bin/iso.jl
#
# Candidates: tied_rule, C = 8, β ∈ {0.38, 0.40, 0.42}, nb1 ∈ {2, 3, 4, 6}, inp/read
# varied; Physics defaults (α_s 0.05, α_p 0.1) with stencil = 1, n0 = M = 0.5.
# Kept: Turing at depths 1, 24, 48 of a 48-layer slab, λ_wave ∈ [6, 9], growth ≥ 0.5,
# stable dt ≥ 0.008. Writes recipes/iso-<date>-NN.toml and results/iso-<date>.json
# (with the in-plane angular spread of the growth rate at |k*|).

using Intmat2, Random, LinearAlgebra, Printf, TOML, Statistics
using Intmat2.Analysis
const AN = Intmat2.Analysis
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "atlas.jl"))   # jsonv
const DATE2 = Libc.strftime("%Y%m%d", time())
const M = 0.5
const NZ = 48
const PI = Physics{Float64}(stencil=1)

"""(max − min)/mean of Re λ_max over in-plane directions at fixed |k|, and the same for 3D
polar angles (φ = 0, 45°)."""
function angular_spread(P, kk)
    ls = [real(AN.dominant(P, AN.symbols(P, kk * cos(φ), kk * sin(φ), 0.0)...)) for φ in range(0, π / 2, length=91)]
    l3 = [real(AN.dominant(P, AN.symbols(P, kk * sin(θ) * cos(φ), kk * sin(θ) * sin(φ), kk * cos(θ))...))
          for φ in (0.0, π / 4) for θ in range(0, π / 2, length=19)]
    a = vcat(ls, l3)
    ((maximum(ls) - minimum(ls)) / mean(ls), (maximum(a) - minimum(a)) / mean(a))
end

function main_iso(ntarget=6)
    t0 = time()
    rng = MersenneTwister(20261017)
    cands = [(seed=rand(rng, 1:2^31-1), nb1=rand(rng, (2.0, 3.0, 4.0, 6.0)), beta=rand(rng, (0.38, 0.40, 0.42)),
              inp=rand(rng, (0.5, 1.0, 2.0)), read=rand(rng, (0.5, 1.0, 2.0))) for _ in 1:400]
    res = Vector{Any}(undef, length(cands))
    Threads.@threads for i in eachindex(cands)
        c = cands[i]; res[i] = nothing
        r = tied_rule(MersenneTwister(c.seed), 8, Float64; nb1=c.nb1, beta=c.beta, inp=c.inp, read=c.read)
        a = []
        for dep in (1, NZ ÷ 2, NZ)
            ss = steady_state(r, PI; M, depth=dep)
            ss.converged || break
            P = AN.Pencil(r, PI, ss)
            d = dispersion(P)
            push!(a, (ss=ss, P=P, d=d, cls=classify(d)))
        end
        length(a) == 3 && all(x -> x.cls === :turing, a) || continue
        d = a[1].d
        (6 <= d.wavelength <= 9 && d.lambda_max >= 0.5 && !a[1].ss.clamped) || continue
        sdt = Float64(stable_dt(r, PI; M, ss=a[1].ss))
        sdt >= 0.008 || continue
        res[i] = (c=c, r=r, a=a, sdt=sdt, spread=angular_spread(a[1].P, d.kstar))
    end
    good = filter(!isnothing, res)
    @printf("%d of %d candidates qualify\n", length(good), length(cands))
    groups = [sort(filter(x -> x.c.beta == b, good), by=x -> -x.a[1].d.lambda_max) for b in (0.40, 0.38, 0.42)]
    sel = []
    while length(sel) < ntarget && any(!isempty, groups)
        for g in groups
            isempty(g) || length(sel) >= ntarget || push!(sel, popfirst!(g))
        end
    end
    names = String[]; rows = []
    for (j, x) in enumerate(sel)
        d = x.a[1].d
        dt = 0.8 * x.sdt
        p = Physics{Float64}(; (f => (f === :dt ? dt : getfield(PI, f)) for f in fieldnames(Physics))...)
        name = @sprintf("iso-%s-%02d", DATE2, j)
        rec = Dict{String,Any}("name" => name, "C" => 8, "seed" => x.c.seed, "rng" => "MersenneTwister",
            "construct" => "tied_rule(MersenneTwister(seed), C, Float64; nb1, beta, inp, read) == Rule{C,Float64}(theta)",
            "nb1" => x.c.nb1, "beta" => x.c.beta, "rule_scales" => Dict("self" => 0.8, "inp" => x.c.inp, "bias" => 0.1, "read" => x.c.read),
            "theta" => collect(vec(x.r)), "N" => 128, "Nz" => NZ, "n0" => M, "s0" => 0.0, "h_scale" => 0.1,
            "steps" => ceil(Int, 8 / (d.lambda_max * dt)),
            "physics" => Dict(String(f) => (f === :stencil ? Int(getfield(p, f)) : Float64(getfield(p, f))) for f in fieldnames(Physics)),
            "predicted" => Dict("class" => "turing", "lambda_max" => d.lambda_max, "wavelength" => d.wavelength,
                "kx" => d.kx, "ky" => d.ky, "kstar" => d.kstar, "imag_star" => d.imag_star, "re0" => d.re0,
                "stable_dt" => x.sdt, "light_model" => "frozen", "depth" => 1,
                "angular_spread_inplane" => x.spread[1], "angular_spread_3d" => x.spread[2],
                "wavelength_depth24" => x.a[2].d.wavelength, "lambda_depth24" => x.a[2].d.lambda_max,
                "wavelength_depth48" => x.a[3].d.wavelength, "lambda_depth48" => x.a[3].d.lambda_max,
                "steady_s" => x.a[1].ss.s, "steady_n" => x.a[1].ss.n, "steady_w" => x.a[1].ss.w, "steady_L" => x.a[1].ss.L))
        open(joinpath(ROOT, "recipes", name * ".toml"), "w") do io
            println(io, "# bin/iso.jl: tied two-shell rule on the isotropic 19-point stencil (stencil = 1),")
            println(io, "# predicted Turing at depths 1, $(NZ ÷ 2), $NZ. `theta` = vec(rule); dt = 0.8 stable_dt; steps = 8 e-folds.")
            TOML.print(io, rec; sorted=true)
        end
        push!(names, name)
        push!(rows, Dict("name" => name, "nb1" => x.c.nb1, "beta" => x.c.beta, "lambda_max" => d.lambda_max,
                         "wavelength" => d.wavelength, "stable_dt" => x.sdt, "dt" => dt, "steps" => rec["steps"],
                         "spread_inplane" => x.spread[1], "spread_3d" => x.spread[2]))
        @printf("  %s nb1 %.0f β %.2f λ %.3f λ_wave %.2f (d24 %.2f, d48 %.2f) stable_dt %.4f dt %.4f steps %d spread %.1f%% / 3D %.1f%%\n",
                name, x.c.nb1, x.c.beta, d.lambda_max, d.wavelength, x.a[2].d.wavelength, x.a[3].d.wavelength, x.sdt, dt,
                rec["steps"], 100x.spread[1], 100x.spread[2])
    end
    open(joinpath(ROOT, "results", "iso-$DATE2.json"), "w") do io
        jsonv(io, Dict{String,Any}("command" => "julia -t $(Threads.nthreads()) --project=. bin/iso.jl",
            "commit" => strip(read(`git -C $ROOT rev-parse --short HEAD`, String)), "M" => M,
            "physics" => "Physics{Float64}(stencil=1) defaults", "candidates" => length(cands),
            "qualified" => length(good), "recipes" => rows, "elapsed_s" => time() - t0))
    end
end

main_iso()
