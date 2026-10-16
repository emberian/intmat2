# Layered steady state of a deep slab under the atlas Turing recipes: does the class
# of the horizontal dispersion relation change with depth?
#
#   julia --project=. bin/layers.jl [Nz=32] [relax_steps=200000]
#
# Per recipe: layered_steady_state (each layer in local balance under the exact
# optical depth above), the per-layer class (light=:frozen and :local), and for
# recipe 01 the coupled column relaxed with the real step! from that start.

using Intmat2, Random, LinearAlgebra, Printf, TOML
using Intmat2.Analysis
const AN = Intmat2.Analysis
const ROOT = normpath(joinpath(@__DIR__, ".."))
const NZ = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32
const RELAX = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200000

function load(name)
    t = TOML.parsefile(joinpath(ROOT, "recipes", name))
    sc = NamedTuple(Symbol(k) => v for (k, v) in t["rule_scales"])
    p = Physics{Float64}(; (Symbol(k) => Float64(v) for (k, v) in t["physics"])...)
    r = random_rule(MersenneTwister(t["seed"]), t["C"], Float64; sc...)
    r, p, Float64(t["n0"]), t
end

function profile(r, p, layers)
    map(layers) do ss
        ss.converged || return (cls=:no_steady_state, lam=NaN, kap=NaN, cls_local=:no_steady_state)
        d = dispersion(r, p, ss; light=:frozen); dl = dispersion(r, p, ss; light=:local)
        (cls=classify(d), lam=d.lambda_max, kap=d.kappa_star, cls_local=classify(dl))
    end
end

for j in 1:10
    name = @sprintf("atlas-20261016-turing-%02d.toml", j)
    r, p, M, _ = load(name)
    layers = layered_steady_state(r, p, NZ; M)
    pr = profile(r, p, layers)
    @printf("%s  (Nz=%d)\n", name, NZ)
    for k in unique([1, 2, 3, 4, 6, 8, 12, 16, 24, NZ])
        k > length(layers) && continue
        ss = layers[k]; q = pr[k]
        @printf("  layer %2d  τ_above %6.2f  L %.3e  s %.3f n %.3f w %.3f  %-11s λ %+.3e κ* %.3f  (local: %s)\n",
                k, ss.tau_above, ss.L, ss.s, ss.n, ss.w, q.cls, q.lam, q.kap, q.cls_local)
    end
    cls = [q.cls for q in pr]
    @printf("  classes top→bottom: %s\n", join(["$(c)×$(count(==(c), cls))" for c in unique(cls)], ", "))
    begin
        t = @elapsed col = AN.relax_column(r, p, layers; steps=RELAX)
        @printf("  coupled column after %d steps (t=%.0f, %.1f s):\n", RELAX, RELAX * p.dt, t)
        for k in (j == 1 ? unique([1, 2, 3, 4, 6, 8, 12, 16, 24, NZ]) : [1, 2, NZ])
            kk = NZ - k + 1
            @printf("  layer %2d  L %.3e  s %.3f n %.3f w %.3f  (decoupled s %.3f)  |Δh| %.3f\n", k,
                    col.L[1, 1, kk], col.s[1, 1, kk], col.n[1, 1, kk], col.w[1, 1, kk], layers[k].s,
                    norm(col.h[:, 1, 1, kk] .- layers[k].h))
        end
        @printf("  column mass/voxel %.6f (M %.3f)\n", (sum(col.s) + sum(col.n) + sum(col.w)) / NZ, M)
        # per-layer class about the relaxed column (L from the column; own-layer light response)
        cc = map(1:NZ) do k
            kk = NZ - k + 1
            ss = (h=Vector(col.h[:, 1, 1, kk]), s=col.s[1, 1, kk], n=col.n[1, 1, kk], w=col.w[1, 1, kk],
                  L=col.L[1, 1, kk], depth=1, lightc=0.5)
            classify(dispersion(r, p, ss; light=:frozen))
        end
        @printf("  relaxed-column classes top→bottom: %s\n", join(["$(c)×$(count(==(c), cc))" for c in unique(cc)], ", "))
        j == 1 && println("  relaxed per layer: ", join(string.(cc), " "))
        @printf("  dark-layer waste share of column mass: %.3f\n", sum(col.w[1, 1, 1:NZ-3]) / (NZ * M))
    end
end
