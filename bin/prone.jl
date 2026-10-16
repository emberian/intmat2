# What predicts the atlas classes, and a Turing-prone rule sampler built from it.
#
#   julia -t 8 --project=. bin/prone.jl [nfresh=2000]
#
# Re-evaluates the 20000 atlas points of bin/atlas.jl (same master seed, frozen
# light), computes interpretable features, fits one-vs-rest L2 logistic regressions
# (standardized features) and single-feature stumps for turing / gridscale / stable,
# then compares turing_prone_rule against random_rule on fresh samples over the same
# physics grid. Writes results/atlas-20261016-prone.json.

include(joinpath(@__DIR__, "atlas.jl"))   # GRID, sample_point, jsonv, ...
using Statistics

const NFRESH = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000

const FEATURES = ["self", "nb", "inp", "bias", "read", "C", "M",
    "log10_tau_dep_over_h", "log10_ero_over_dep", "log10_photo_over_dep", "log10_interaction_length",
    "g1_over_g0", "rho_Wself", "nb_antidiffusion", "Win_s_norm", "Win_L_norm", "Win_n_norm",
    "vdep_dot_h_pos", "vdep_dot_h", "vero_dot_h", "s_star"]

function features(pt, r, ss)
    ts = timescales(pt.p)
    W = Matrix(r.Wself); Wn = Matrix(r.Wnb); Win = Matrix(r.Win)
    S = 1 .- ss.h .^ 2
    g = Intmat2.conductance(pt.p, ss.s, ss.s)
    # growth that −g κ S W_nb can produce at κ = 8: positive part of −min Re eig(S W_nb)
    anti = max(0.0, -minimum(real.(eigvals(S .* Wn)))) * g * 8
    vd = dot(Vector(r.vdep), ss.h)
    [pt.scales.self, pt.scales.nb, pt.scales.inp, pt.scales.bias, pt.scales.read, pt.C, pt.M,
     log10(ts.dep_over_h), log10(ts.ero_over_dep), log10(ts.photo_over_dep), log10(ts.interaction_length),
     ts.g1_over_g0, maximum(abs.(eigvals(W))), anti, norm(Win[:, 2]), norm(Win[:, 4]), norm(Win[:, 1]),
     Float64(vd > 0), vd, dot(Vector(r.vero), ss.h), ss.s]
end

function classify_point(pt, ruler)
    r = ruler(MersenneTwister(pt.seed), pt.C)
    ss = steady_state(r, pt.p; M=pt.M)
    ss.converged || return (cls=:no_steady_state, x=nothing, d=nothing)
    d = dispersion(r, pt.p, ss; light=:frozen, κs=KAPPAS)
    (cls=classify(d), x=features(pt, r, ss), d=d)
end

"""L2-regularized logistic regression by Newton (IRLS). X standardized, intercept first."""
function logistic(X, y; λ=1e-2, iters=50)
    n, k = size(X)
    A = hcat(ones(n), X)
    β = zeros(k + 1)
    for _ in 1:iters
        μ = 1 ./ (1 .+ exp.(-A * β))
        g = A' * (μ .- y) .+ λ .* vcat(0, β[2:end])
        H = A' * (A .* (μ .* (1 .- μ))) + λ * Diagonal(vcat(0, ones(k)))
        Δ = H \ g
        β .-= Δ
        norm(Δ) < 1e-10 && break
    end
    β
end

function auc(score, y)
    o = sortperm(score); r = similar(score); r[o] = 1:length(score)
    npos = sum(y); nneg = length(y) - npos
    (sum(r[y .== 1]) - npos * (npos + 1) / 2) / (npos * nneg)
end

"""Best single threshold on one feature for class y: maximizes the class rate on one side
(with at least 2% of points there). Returns (feature, threshold, side, rate, share)."""
function best_stump(X, y, j)
    v = X[:, j]; best = (rate=0.0, thr=NaN, side=:ge, share=0.0)
    for thr in unique(quantile(v, 0.02:0.02:0.98))
        for side in (:ge, :lt)
            m = side === :ge ? v .>= thr : v .< thr
            sum(m) < 0.02 * length(v) && continue
            rt = mean(y[m])
            rt > best.rate && (best = (rate=rt, thr=thr, side=side, share=mean(m)))
        end
    end
    best
end

function main_prone()
    t0 = time()
    pts = [sample_point(i) for i in 1:NPTS]
    res = Vector{Any}(undef, NPTS)
    Threads.@threads for i in 1:NPTS
        res[i] = classify_point(pts[i], (rng, C) -> random_rule(rng, C, Float64; pts[i].scales...))
    end
    keep = findall(r -> r.x !== nothing, res)
    X = reduce(vcat, (res[i].x' for i in keep))
    cls = [res[i].cls for i in keep]
    μ = mean(X, dims=1); σ = std(X, dims=1); σ[σ .== 0] .= 1
    Z = (X .- μ) ./ σ
    fits = Dict{String,Any}()
    for c in (:turing, :gridscale, :stable)
        y = Float64.(cls .== c)
        β = logistic(Z, y)
        A = auc(hcat(ones(size(Z, 1)), Z) * β, y)
        o = sortperm(abs.(β[2:end]), rev=true)
        @printf("\n%s (base rate %.4f, in-sample AUC %.3f); standardized coefficients:\n", c, mean(y), A)
        for j in o[1:10]
            @printf("  %-26s %+7.3f\n", FEATURES[j], β[j+1])
        end
        stumps = [(FEATURES[j], best_stump(X, y, j)) for j in 1:length(FEATURES)]
        sort!(stumps, by=s -> -s[2].rate)
        for (f, s) in stumps[1:4]
            @printf("  stump %-22s %s %.3g → rate %.4f on %.1f%% of points\n", f, s.side === :ge ? "≥" : "<", s.thr, s.rate, 100s.share)
        end
        fits[String(c)] = Dict("base_rate" => mean(y), "auc_in_sample" => A, "intercept" => β[1],
            "coefficients_standardized" => Dict(FEATURES[j] => β[j+1] for j in 1:length(FEATURES)),
            "stumps" => [Dict("feature" => f, "side" => String(s.side), "threshold" => s.thr, "rate" => s.rate,
                              "share" => s.share) for (f, s) in stumps[1:4]])
    end
    isdefined(Intmat2.Analysis, :turing_prone_rule) || return
    # fresh comparison over the same physics grid, new seeds
    fresh = [sample_point(10^7 + i) for i in 1:NFRESH]
    out = Dict{String,Any}()
    for (name, ruler) in (("random_rule", nothing), ("turing_prone_rule", :prone))
        cl = Vector{Symbol}(undef, NFRESH)
        Threads.@threads for i in 1:NFRESH
            pt = fresh[i]
            f = ruler === nothing ? ((rng, C) -> random_rule(rng, C, Float64; pt.scales...)) :
                                    ((rng, C) -> turing_prone_rule(rng, C, Float64))
            cl[i] = classify_point(pt, f).cls
        end
        frac = Dict(String(c) => count(==(c), cl) / NFRESH for c in unique(cl))
        out[name] = frac
        @printf("\n%s on %d fresh points: %s\n", name, NFRESH,
                join([@sprintf("%s %.4f", k, v) for (k, v) in sort(collect(frac), by=last, rev=true)], ", "))
    end
    path = joinpath(ROOT, "results", "atlas-20261016-prone.json")
    open(path, "w") do io
        jsonv(io, Dict{String,Any}("command" => "julia -t $(Threads.nthreads()) --project=. bin/prone.jl $NFRESH",
            "commit" => strip(read(`git -C $ROOT rev-parse --short HEAD`, String)),
            "atlas_points" => NPTS, "fitted_points" => length(keep), "features" => FEATURES,
            "feature_mean" => vec(μ), "feature_std" => vec(σ), "light" => "frozen",
            "fits" => fits, "fresh_samples" => NFRESH,
            "fresh_physics" => "atlas grid (sample_point(1e7 + i)); random_rule uses the sampled scales, turing_prone_rule its defaults",
            "fresh_class_fractions" => out, "elapsed_s" => time() - t0))
    end
    println("wrote ", path)
end

main_prone()
