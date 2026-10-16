# Stability atlas: random rules × design coordinates, classified by the linear
# dispersion relation about the homogeneous steady state (top layer, depth = 1).
#
#   julia -t 8 --project=. bin/atlas.jl [npoints=20000] [master_seed=20261016]
#
# Writes results/atlas-<date>.json (every point) and recipes/atlas-<date>-*.toml.
# Each rule is `random_rule(MersenneTwister(seed), C, Float64; rule_scales...)`.

using Intmat2, Random, LinearAlgebra, Printf, TOML
using Intmat2.Analysis
using Intmat2: timescales
const AN = Intmat2.Analysis

BLAS.set_num_threads(1)

const IS_MAIN = abspath(PROGRAM_FILE) == @__FILE__
const NPTS = IS_MAIN && length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20000
const MASTER = IS_MAIN && length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20261016
const DATE = Libc.strftime("%Y%m%d", time())
const ROOT = normpath(joinpath(@__DIR__, ".."))

const GRID = (
    C=(4, 8),
    self=(0.5, 1.0, 1.5, 2.0), nb=(0.0, 0.1, 0.2, 0.4, 0.8), inp=(0.5, 1.0, 2.0, 4.0),
    bias=(0.1, 0.5), read=(0.5, 1.0, 2.0, 4.0),
    k_dep=(0.1, 0.3, 1.0, 3.0), ero_over_dep=(3.0, 10.0, 30.0), k_photo=(0.05, 0.2, 1.0),
    D0=(0.05, 0.2, 1.0), g1_over_g0=(-1.0, 0.0, 1.0), M=(0.5, 1.0, 2.0),
)
const KAPPAS = [8.0 * (j / 96)^2 for j in 1:96]

physics_dict(p) = Dict(String(f) => Float64(getfield(p, f)) for f in fieldnames(Physics))

function sample_point(i)
    rng = MersenneTwister(hash((MASTER, i)))
    pick(t) = t[rand(rng, 1:length(t))]
    C = pick(GRID.C)
    seed = rand(rng, 1:2^31-1)
    sc = (self=pick(GRID.self), nb=pick(GRID.nb), inp=pick(GRID.inp), bias=pick(GRID.bias), read=pick(GRID.read))
    k_dep = pick(GRID.k_dep); D0 = pick(GRID.D0); g0 = 0.5
    dt = min(0.05, 1 / (6D0), 1.0)
    p = Physics{Float64}(; dt=dt, tau_h=1.0, g0=g0, g1=g0 * pick(GRID.g1_over_g0), k_dep=k_dep,
                         k_ero=k_dep / pick(GRID.ero_over_dep), k_photo=pick(GRID.k_photo), D0=D0)
    (C=C, seed=seed, scales=sc, p=p, M=pick(GRID.M))
end

function evaluate(pt)
    r = random_rule(MersenneTwister(pt.seed), pt.C, Float64; pt.scales...)
    ss = steady_state(r, pt.p; M=pt.M)
    rec = Dict{String,Any}("C" => pt.C, "seed" => pt.seed, "rule_scales" => Dict(String(k) => v for (k, v) in pairs(pt.scales)),
        "physics" => physics_dict(pt.p), "M" => pt.M,
        "design" => Dict(String(k) => Float64(v) for (k, v) in pairs(timescales(pt.p))),
        "steady" => Dict("s" => ss.s, "n" => ss.n, "w" => ss.w, "L" => ss.L, "converged" => ss.converged,
                         "clamped" => ss.clamped, "method" => String(ss.method)))
    for light in (:frozen, :local)
        if !ss.converged
            rec[String(light)] = Dict("class" => "no_steady_state")
            continue
        end
        d = dispersion(r, pt.p, ss; light, κs=KAPPAS)
        rec[String(light)] = Dict("class" => String(classify(d)), "lambda_max" => d.lambda_max,
            "kappa_star" => d.kappa_star, "kstar" => d.kstar, "wavelength" => d.wavelength,
            "imag_star" => d.imag_star, "re0" => d.re0, "im0" => d.im0,
            "map_rate_max" => d.map_rate_max, "band_lo" => d.band_lo, "band_hi" => d.band_hi)
    end
    rec
end

# ---- minimal JSON writer (no JSON dependency in the package)
jsonv(io, x::AbstractString) = print(io, '"', escape_string(x), '"')
jsonv(io, x::Bool) = print(io, x ? "true" : "false")
jsonv(io, x::Integer) = print(io, x)
jsonv(io, x::Real) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
jsonv(io, x::Symbol) = jsonv(io, String(x))
function jsonv(io, x::AbstractDict)
    print(io, '{')
    for (j, k) in enumerate(sort!(collect(keys(x))))
        j > 1 && print(io, ',')
        jsonv(io, String(k)); print(io, ':'); jsonv(io, x[k])
    end
    print(io, '}')
end
function jsonv(io, x::AbstractVector)
    print(io, '[')
    for (j, v) in enumerate(x)
        j > 1 && print(io, eltype(x) <: AbstractVector ? ",\n" : ",")
        jsonv(io, v)
    end
    print(io, ']')
end

# Points are stored as rows (one array per point, `COLUMNS` order), floats at 6
# significant digits: 20000 points in a few MB.
const COLUMNS = ["C", "seed", "self", "nb", "inp", "bias", "read",
    "dt", "tau_h", "g0", "g1", "k_dep", "k_ero", "k_photo", "D0", "alpha_s", "alpha_p", "L0", "M",
    "ss_s", "ss_n", "ss_w", "ss_L", "ss_converged", "ss_clamped",
    "frozen_class", "frozen_lambda_max", "frozen_kappa_star", "frozen_wavelength", "frozen_imag_star",
    "frozen_re0", "frozen_im0", "frozen_map_rate_max", "frozen_band_lo", "frozen_band_hi",
    "local_class", "local_lambda_max", "local_kappa_star", "local_wavelength", "local_imag_star",
    "local_re0", "local_im0", "local_map_rate_max", "local_band_lo", "local_band_hi"]
sig(x::Real) = isfinite(x) ? round(Float64(x), sigdigits=6) : x
sig(x) = x
function row(r)
    sc = r["rule_scales"]; ph = r["physics"]; st = r["steady"]
    v = Any[r["C"], r["seed"], sc["self"], sc["nb"], sc["inp"], sc["bias"], sc["read"],
        ph["dt"], ph["tau_h"], ph["g0"], ph["g1"], ph["k_dep"], ph["k_ero"], ph["k_photo"], ph["D0"],
        ph["alpha_s"], ph["alpha_p"], ph["L0"], r["M"], st["s"], st["n"], st["w"], st["L"],
        st["converged"], st["clamped"]]
    for light in ("frozen", "local")
        d = r[light]
        push!(v, d["class"])
        for k in ("lambda_max", "kappa_star", "wavelength", "imag_star", "re0", "im0", "map_rate_max", "band_lo", "band_hi")
            push!(v, get(d, k, NaN))
        end
    end
    map(sig, v)
end

function recipe(rec, light_used, name)
    d = rec[light_used]
    rate = d["class"] in ("hopf", "homogeneous") ? d["re0"] : d["lambda_max"]
    efolds_steps = ceil(Int, 5 / (rate * rec["physics"]["dt"]))   # five e-folds of the linear growth
    Dict{String,Any}("name" => name, "C" => rec["C"], "seed" => rec["seed"], "rng" => "MersenneTwister",
        "h_scale" => 0.1, "steps" => efolds_steps,
        "rule_eltype" => "Float64",
        "construct" => "random_rule(MersenneTwister(seed), C, Float64; rule_scales...)",
        "rule_scales" => rec["rule_scales"], "physics" => rec["physics"], "n0" => rec["M"],
        "s0" => 0.0,
        "predicted" => Dict("class" => d["class"], "lambda_max" => d["lambda_max"],
            "kstar" => d["kstar"], "kappa_star" => d["kappa_star"], "wavelength" => d["wavelength"],
            "imag_star" => d["imag_star"], "re0" => d["re0"], "band_lo" => d["band_lo"], "band_hi" => d["band_hi"],
            "light_model" => light_used, "depth" => 1,
            "other_light_model_class" => rec[light_used == "frozen" ? "local" : "frozen"]["class"],
            "steady_s" => rec["steady"]["s"], "steady_n" => rec["steady"]["n"], "steady_w" => rec["steady"]["w"],
            "steady_L" => rec["steady"]["L"]))
end

function main()
    t0 = time()
    pts = [sample_point(i) for i in 1:NPTS]
    recs = Vector{Dict{String,Any}}(undef, NPTS)
    Threads.@threads for i in 1:NPTS
        recs[i] = evaluate(pts[i])
    end
    elapsed = time() - t0
    summary = Dict{String,Any}()
    for light in ("frozen", "local")
        cnt = Dict{String,Int}()
        for r in recs
            c = r[light]["class"]; cnt[c] = get(cnt, c, 0) + 1
        end
        summary[light] = Dict(k => v / NPTS for (k, v) in cnt)
        println("class fractions ($light light):")
        for (k, v) in sort(collect(cnt), by=last, rev=true)
            @printf("  %-16s %6d  %.4f\n", k, v, v / NPTS)
        end
    end
    agree = count(r -> r["frozen"]["class"] == r["local"]["class"], recs)
    @printf("frozen/local class agreement: %.4f\n", agree / NPTS)
    clamped = count(r -> r["steady"]["clamped"] == true, recs)

    # Turing recipes: frozen-light Turing (the stated default), wavelength 4–16,
    # stable at the grid scale, unclamped steady state. Most Turing points are
    # conserved long-wave (band reaches κ → 0 because mass conservation pins λ(0) = 0
    # on the mass branch); that is recorded as `band_lo`. Prefer points whose
    # local-light class agrees; rank by λ_max.
    istur(r) = r["frozen"]["class"] == "turing" && 4 <= r["frozen"]["wavelength"] <= 16 &&
               r["steady"]["clamped"] == false && r["frozen"]["band_hi"] < KAPPAS[end]
    tur = filter(istur, recs)
    sort!(tur, by=r -> (r["local"]["class"] == "turing", r["frozen"]["lambda_max"]), rev=true)
    tur = tur[1:min(10, end)]
    # Oscillatory: finite-κ waves first, then homogeneous Hopf with a clean
    # oscillation (|Im| ≥ 3 Re) and modest growth (0.005 ≤ Re ≤ 0.3), nearest onset first.
    isosc(r) = r["steady"]["clamped"] == false && (r["frozen"]["class"] == "wave" ||
               (r["frozen"]["class"] == "hopf" && 0.005 <= r["frozen"]["re0"] <= 0.3 &&
                abs(r["frozen"]["im0"]) >= 3 * r["frozen"]["re0"]))
    osc = filter(isosc, recs)
    sort!(osc, by=r -> (r["frozen"]["class"] == "wave", r["local"]["class"] == r["frozen"]["class"],
                        -r["frozen"]["re0"]), rev=true)
    osc = osc[1:min(5, end)]
    mkpath(joinpath(ROOT, "recipes"))
    names = String[]
    for (tag, set) in (("turing", tur), ("osc", osc))
        for (j, r) in enumerate(set)
            name = @sprintf("atlas-%s-%s-%02d.toml", DATE, tag, j)
            open(joinpath(ROOT, "recipes", name), "w") do io
                println(io, "# From bin/atlas.jl (master seed $MASTER, $NPTS points). Linear prediction about the")
                println(io, "# homogeneous steady state at the top layer; see docs/analysis.md.")
                println(io, "# `steps` is five e-folds of the predicted linear growth at this dt.")
                TOML.print(io, recipe(r, "frozen", replace(name, ".toml" => "")); sorted=true)
            end
            push!(names, name)
            fr = r["frozen"]
            @printf("  %s  %s λ=%.4g κ*=%.3f λ_wave=%.1f Im=%.3g (local: %s)\n", name, fr["class"], fr["lambda_max"],
                    fr["kappa_star"], fr["wavelength"], fr["imag_star"], r["local"]["class"])
        end
    end
    out = Dict{String,Any}("command" => "julia -t $(Threads.nthreads()) --project=. bin/atlas.jl $NPTS $MASTER",
        "commit" => strip(read(`git -C $ROOT rev-parse --short HEAD`, String)),
        "date" => DATE, "npoints" => NPTS, "master_seed" => MASTER, "elapsed_s" => elapsed,
        "grid" => Dict(String(k) => collect(v) for (k, v) in pairs(GRID)),
        "kappa_grid" => "8 (j/96)^2, j = 1..96, golden-section refined at the max",
        "dt_rule" => "min(0.05, 1/(6 D0), tau_h); tau_h = 1, g0 = 0.5",
        "depth" => 1, "fractions" => summary, "clamped_fraction" => clamped / NPTS,
        "frozen_local_agreement" => agree / NPTS, "recipes" => names,
        "columns" => COLUMNS, "rows" => [row(r) for r in recs])
    path = joinpath(ROOT, "results", "atlas-$DATE.json")
    open(io -> jsonv(io, out), path, "w")
    @printf("wrote %s (%d points, %.1f s)\n", path, NPTS, elapsed)
end

IS_MAIN && main()
