# Proposed coupling variants over the atlas grid (frozen light, depth 1): a second,
# inhibitory radius-2 axis shell W_nb2 = −nb2·QQᵀ/C, and a filtered single ring.
#
#   julia -t 8 --project=. bin/atlas2.jl [base|tied]
#
# Same 20000 (seed, scales, physics, M) points as bin/atlas.jl. Q is drawn from
# MersenneTwister(seed + 1) and shared by every nb2 level. Writes
# results/atlas-20261016-twoscale.json.

include(joinpath(@__DIR__, "atlas.jl"))
using Statistics

const NK = 24
psd(seed, C) = (Q = randn(MersenneTwister(seed + 1), C, C); Q * Q' ./ C)
psd2(seed, C) = (Q = randn(MersenneTwister(seed + 2), C, C); Q * Q' ./ C)

# name => (W_nb source, nb2, filter, nb1, β). Sources: :rule (random_rule's W_nb),
# :psd (nb·PPᵀ/C with the point's nb), :tied (nb1·PPᵀ/C, and W_nb2 = −β·W_nb instead
# of the independent −nb2·QQᵀ/C).
const BASE_CONFIGS = [
    "ring" => (:rule, 0.0, :none, 0.0, 0.0),
    "ring+nb2=0.2" => (:rule, 0.2, :none, 0.0, 0.0),
    "ring+nb2=0.4" => (:rule, 0.4, :none, 0.0, 0.0),
    "ring+nb2=0.8" => (:rule, 0.8, :none, 0.0, 0.0),
    "box-filtered ring" => (:rule, 0.0, :box, 0.0, 0.0),
    "binomial-filtered ring" => (:rule, 0.0, :binomial, 0.0, 0.0),
    "psd ring" => (:psd, 0.0, :none, 0.0, 0.0),
    "psd ring+nb2=0.2" => (:psd, 0.2, :none, 0.0, 0.0),
    "psd ring+nb2=0.4" => (:psd, 0.4, :none, 0.0, 0.0),
    "psd ring+nb2=0.8" => (:psd, 0.8, :none, 0.0, 0.0),
    "binomial-filtered ring+nb2=0.4" => (:rule, 0.4, :binomial, 0.0, 0.0),
]
const TIED_CONFIGS = ["tied nb1=$(n1) β=$(β)" => (:tied, 0.0, :none, n1, β)
                      for n1 in (1.0, 2.0, 4.0) for β in (0.3, 0.4, 0.5, 0.75)]
const MODE = length(ARGS) >= 1 ? ARGS[1] : "base"
const CONFIGS = MODE == "tied" ? TIED_CONFIGS : BASE_CONFIGS

function eval2(pt)
    r = random_rule(MersenneTwister(pt.seed), pt.C, Float64; pt.scales...)
    ss = steady_state(r, pt.p; M=pt.M)
    ss.converged || return [(cls=:no_steady_state, wl=NaN, lam=NaN) for _ in CONFIGS]
    map(CONFIGS) do (_, (wsrc, nb2, filt, nb1, β))
        Wnb = wsrc === :psd ? pt.scales.nb .* psd2(pt.seed, pt.C) :
              wsrc === :tied ? nb1 .* psd2(pt.seed, pt.C) : nothing
        Wnb2 = wsrc === :tied ? -β .* Wnb : nb2 > 0 ? -nb2 .* psd(pt.seed, pt.C) : nothing
        d = dispersion2(Pencil2(r, pt.p, ss; Wnb, Wnb2); filter=filt, nk=NK)
        (cls=classify(d), wl=d.wavelength, lam=d.lambda_max)
    end
end

function main2()
    t0 = time()
    pts = [sample_point(i) for i in 1:NPTS]
    res = Vector{Any}(undef, NPTS)
    Threads.@threads for i in 1:NPTS
        res[i] = eval2(pts[i])
    end
    classes = ["stable", "turing", "wave", "gridscale", "hopf", "homogeneous", "mixed", "no_steady_state"]
    bins = [(2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, Inf)]
    out = Dict{String,Any}()
    @printf("%-32s %7s %7s %7s %7s %7s %7s | turing λ_wave: %s | turing 4–16\n", "config", "stable", "turing", "wave",
            "grid", "hopf+h", "mixed", join(["$(a)–$(b)" for (a, b) in bins], " "))
    for (c, (name, _)) in enumerate(CONFIGS)
        cl = [String(res[i][c].cls) for i in 1:NPTS]
        frac = Dict(k => count(==(k), cl) / NPTS for k in classes)
        twl = [res[i][c].wl for i in 1:NPTS if res[i][c].cls in (:turing, :wave)]
        hist = [count(x -> a <= x < b, twl) for (a, b) in bins]
        good = count(x -> 4 <= x <= 16, twl) / NPTS
        tl = [res[i][c].lam for i in 1:NPTS if res[i][c].cls === :turing]
        out[name] = Dict("fractions" => frac, "turing_wave_wavelength_hist" => Dict("$(a)-$(b)" => h for ((a, b), h) in zip(bins, hist)),
                         "turing_or_wave_4_16_fraction" => good,
                         "turing_lambda_median" => isempty(tl) ? NaN : median(tl))
        @printf("%-32s %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f | %s | %.4f (median λ %.3g)\n", name, frac["stable"], frac["turing"],
                frac["wave"], frac["gridscale"], frac["hopf"] + frac["homogeneous"], frac["mixed"],
                join(lpad.(hist, 5), " "), good, isempty(tl) ? NaN : median(tl))
    end
    path = joinpath(ROOT, "results", MODE == "tied" ? "atlas-20261016-twoscale-tied.json" : "atlas-20261016-twoscale.json")
    open(path, "w") do io
        jsonv(io, Dict{String,Any}("command" => "julia -t $(Threads.nthreads()) --project=. bin/atlas2.jl $MODE",
            "commit" => strip(read(`git -C $ROOT rev-parse --short HEAD`, String)), "npoints" => NPTS,
            "light" => "frozen", "depth" => 1, "k_grid" => "0 ≤ ky ≤ kx ≤ π, step π/$NK",
            "Wnb2" => "−nb2·QQᵀ/C, Q = randn(MersenneTwister(seed+1), C, C); tied: −β·W_nb",
            "tied_Wnb" => "nb1·PPᵀ/C, P = randn(MersenneTwister(seed+2), C, C)",
            "psd_Wnb" => "nb·PPᵀ/C, P = randn(MersenneTwister(seed+2), C, C) (replaces the rule's W_nb)",
            "filters" => "box (1+2cos k)/3, binomial (1+cos k)/2 per horizontal axis, applied to the ring",
            "configs" => out, "elapsed_s" => time() - t0))
    end
    println("wrote ", path, @sprintf(" (%.0f s)", time() - t0))
end

main2()
