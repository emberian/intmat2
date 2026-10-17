#!/usr/bin/env julia
# Run a recipe with a Recorder:  julia --project=env/metal bin/run.jl recipe.toml outdir
#
# Recipe fields (TOML): name, C, seed, backend = "cpu"|"metal", N (+ optional Nz) or dims = [Nx,Ny,Nz],
# steps, capture_every, n0, h_scale, s0, [rule_scales] {self,nb,inp,bias,read} (or an
# explicit `theta` vector of 2C²+8C numbers; `rule_scales.nb_diag` adds that multiple of
# the identity to W_nb), and every Physics field either at top
# level or under [physics]. Optional `fields` (default s,n,w,L,pig) and `h_final = true`
# to also save h in the last capture. `rule_eltype = "Float64"` draws the random rule in
# Float64 (as bin/atlas.jl does) before converting. Missing run fields default to
# N=128 (Nz=N), steps=3000, capture_every=steps÷10, backend="metal", h_scale=0.1; any top-level
# field can be overridden on the command line: run.jl recipe.toml outdir N=64 steps=500.
using Intmat2, TOML, Random, Printf, KernelAbstractions
using Intmat2: Recorder, record!, capture, light!

const RUN_DEFAULTS = Dict{String,Any}("N" => 128, "steps" => 3000,
                                      "backend" => "metal", "h_scale" => 0.1)

function parse_override(kv)
    k, v = split(kv, "="; limit=2)
    k => something(tryparse(Int, v), tryparse(Float64, v), tryparse(Bool, v), String(v))
end

function load_recipe(path, overrides=Pair{String,Any}[])
    r = merge(RUN_DEFAULTS, TOML.parsefile(path), Dict(overrides))
    haskey(r, "dims") && any(first(o) in ("N", "Nz") for o in overrides) && delete!(r, "dims")
    haskey(r, "capture_every") || (r["capture_every"] = max(r["steps"] ÷ 10, 1))
    phys = merge(Dict(k => v for (k, v) in r if Symbol(k) in fieldnames(Physics)), get(r, "physics", Dict()))
    p = Physics{Float32}(; (Symbol(k) => Float32(v) for (k, v) in phys)...)
    C = r["C"]; seed = get(r, "seed", 0)
    rule = if haskey(r, "theta")
        Rule{C,Float32}(Float32.(r["theta"]))
    else
        sc = Dict(get(r, "rule_scales", Dict()))
        diag = Float64(pop!(sc, "nb_diag", 0.0))
        TR = get(r, "rule_eltype", "Float32") == "Float64" ? Float64 : Float32
        θ = collect(vec(random_rule(MersenneTwister(seed), C, TR; (Symbol(k) => Float64(v) for (k, v) in sc)...)))
        for c in 1:C            # nb_diag: W_nb += nb_diag·I, plain diffusion of activity
            θ[C * C + (c - 1) * C + c] += diag
        end
        Rule{C,Float32}(Float32.(θ))
    end
    dims = haskey(r, "dims") ? Tuple(Int.(r["dims"])) : (N = r["N"]; (N, N, get(r, "Nz", N)))
    (; raw=r, p, rule, C, seed, dims)
end

# Metal must be loaded at top level (before `main` runs) so its methods are in the
# world the run executes in; `using` inside a function leaves step! on stale methods.
# `fast = false` in a recipe (or fast=false on the command line) turns off the GPU
# fast-math path. Metal's fast tanh overflows to NaN for large |u| (e.g. the tied
# two-shell recipes with nb1 = 16), so those need it off.
stepf(wd, rc) = haskey(rc.raw, "fast") ? step!(wd, rc.rule, rc.p; fast=Bool(rc.raw["fast"])) : step!(wd, rc.rule, rc.p)

select_backend(name) = name == "cpu" ? CPU() : name == "metal" ? Main.Metal.MetalBackend() : error("unknown backend $name")

function main(recipe_path, outdir, overrides=Pair{String,Any}[])
    rc = load_recipe(recipe_path, overrides)
    r = rc.raw
    be = select_backend(r["backend"])
    mkpath(outdir)
    cp(recipe_path, joinpath(outdir, "recipe.toml"); force=true)
    open(io -> TOML.print(io, Dict(k => v for (k, v) in r if !(v isa Dict))), joinpath(outdir, "run.toml"), "w")
    wd = World(be, rc.rule, rc.dims; n0=get(r, "n0", 1.0), h_scale=get(r, "h_scale", 0.1),
               s0=get(r, "s0", 0.0), seed=rc.seed)
    wd = light!(wd, rc.rule, rc.p)
    fields = Tuple(Symbol.(get(r, "fields", ["s", "n", "w", "L", "pig"])))
    every = get(r, "capture_every", 100); steps = r["steps"]
    rec = Recorder(outdir, rc.rule, rc.p; every, fields,
                   meta=Dict{String,Any}("recipe" => get(r, "name", basename(recipe_path)), "backend" => r["backend"], "note" => get(r, "note", "")))
    m0 = total_mass(wd)
    t0 = time()
    report(wd) = (s = summarize(wd);
        @printf("step %6d  t %8.2f  mean_s %.4f  std_s %.4f  solid>.5 %.3f  n %.4f  w %.4f  |h| %.3f  L_bottom %.4f  dmass %+.2e  %.1f steps/s\n",
                s.step, s.t, s.mean_s, s.std_s, s.solid_frac, s.mean_n, s.mean_w, s.mean_abs_h, s.light_bottom,
                (total_mass(wd) - m0) / m0, s.step / max(time() - t0, 1e-9)); flush(stdout))
    record!(rec, wd); report(wd)
    for _ in 1:steps
        wd = stepf(wd, rc)
        record!(rec, wd) && report(wd)
    end
    record!(rec, wd; force=true)
    if get(r, "h_final", false)
        capture(wd, rc.rule, rc.p, joinpath(outdir, "final"); fields=(fields..., :h))
    end
    wd
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || (println("usage: run.jl recipe.toml outdir [key=value ...]"); exit(2))
    ov = Pair{String,Any}[parse_override(a) for a in ARGS[3:end]]
    load_recipe(ARGS[1], ov).raw["backend"] == "metal" && @eval using Metal
    Base.invokelatest(main, ARGS[1], ARGS[2], ov)   # new world: sees Metal methods
end
