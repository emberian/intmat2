# Rule search: NSGA-II over θ vs. an equal-budget random baseline.
#
#   julia -t 8 --project=env/metal bin/evolve.jl --run g1neg --g1 -0.5 --pop 48 --gens 8 \
#         --modes random,nsga --backend metal
#
# Writes results/evolve-<run>-<mode>.jsonl (one line per evaluated rule: θ, z, σ, parents,
# seeds, physics, eval config, objectives, diagnostics, wall time), results/evolve-<run>.json
# (command, hypervolume and per-objective best per mode, Pareto front) and, for the nsga
# mode, recipes/evolved-<run>-<objective>.toml for the front's best rule on each objective.
using Intmat2, KernelAbstractions, Random, Statistics, Printf
using Intmat2: EvalConfig, evolve, evaluate, hypervolume, nondominated_rank, objvec,
    SELECT_OBJECTIVES, ALL_OBJECTIVES, write_recipe, read_recipe

function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        startswith(args[i], "--") || error("bad arg $(args[i])")
        d[args[i][3:end]] = args[i+1]; i += 2
    end
    d
end
const A = parse_args(ARGS)
arg(k, default) = haskey(A, k) ? (default isa String ? A[k] : parse(typeof(default), A[k])) : default

run = arg("run", "test")
backend_name = arg("backend", "cpu")
be = if backend_name == "metal"
    @eval using Metal
    Metal.MetalBackend()
else
    CPU()
end
C = arg("C", 8)
dims = Tuple(parse.(Int, split(arg("dims", "32,32,24"), ",")))
p = Physics{Float32}(dt=arg("dt", 0.1f0), g0=arg("g0", 0.5f0), g1=arg("g1", -0.5f0),
                     k_dep=arg("k_dep", 0.1f0), k_ero=arg("k_ero", 0.01f0), k_photo=arg("k_photo", 0.05f0),
                     D0=arg("D0", 0.2f0), alpha_s=arg("alpha_s", 0.5f0), alpha_p=arg("alpha_p", 0.1f0),
                     tau_h=arg("tau_h", 1.0f0), L0=arg("L0", 1.0f0))
cfg = EvalConfig(dims=dims, T=arg("T", 1200), cont=arg("cont", 300), seed=arg("world_seed", 1),
                 n0=arg("n0", 1.0), h_scale=arg("h_scale", 0.1), concurrency=arg("concurrency", 4),
                 backend=be)
pop = arg("pop", 48); gens = arg("gens", 8); seed = arg("seed", 1)
modes = Symbol.(split(arg("modes", "random,nsga"), ","))
init = nothing
if haskey(A, "init")      # comma-separated recipe files whose θ seed the first population
    init = [Float64.(vec(read_recipe(f)[1])) for f in split(A["init"], ",")]
end
outdir = joinpath(@__DIR__, "..", "results"); recdir = joinpath(@__DIR__, "..", "recipes")
mkpath(outdir); mkpath(recdir)
cmd = "julia -t $(Threads.nthreads()) --project=$(Base.active_project() |> dirname |> basename) bin/evolve.jl " * join(ARGS, " ")
commit = try readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`) catch; "unknown" end

# --- minimal JSON writer (no extra dependency)
js(x::AbstractString) = "\"" * escape_string(x) * "\""
js(x::Symbol) = js(string(x))
js(x::Bool) = x ? "true" : "false"
js(x::Integer) = string(x)
js(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : "null"
js(x::Nothing) = "null"
js(x::Union{AbstractVector,Tuple}) = "[" * join(js.(x), ",") * "]"
js(x::NamedTuple) = "{" * join((js(string(k)) * ":" * js(v) for (k, v) in pairs(x)), ",") * "}"
js(x::AbstractDict) = "{" * join((js(string(k)) * ":" * js(v) for (k, v) in x), ",") * "}"
js(p::Physics) = js(NamedTuple{fieldnames(Physics)}(Tuple(Float64(getfield(p, f)) for f in fieldnames(Physics))))
cfgnt = (dims=cfg.dims, T=cfg.T, cont=cfg.cont, world_seed=cfg.seed, n0=cfg.n0, h_scale=cfg.h_scale,
         rec_every=cfg.rec_every, probe_every=cfg.probe_every, nprobe=cfg.nprobe,
         reset_scale=cfg.reset_scale, backend=backend_name, C=C)

summary = Dict{String,Any}("command" => cmd, "commit" => commit, "run" => run, "physics" => p,
                           "eval" => cfgnt, "pop" => pop, "gens" => gens, "seed" => seed,
                           "select_objectives" => collect(SELECT_OBJECTIVES), "modes" => Dict{String,Any}())
archives = Dict{Symbol,Any}()
for mode in modes
    path = joinpath(outdir, "evolve-$run-$mode.jsonl")
    io = open(path, "w")
    t0 = time()
    log = function (rec)
        r = rec.res
        line = (run=run, mode=mode, id=rec.id, gen=rec.gen, parents=rec.parents, sigma=rec.sigma,
                es_seed=seed, world_seed=cfg.seed, theta=rec.theta, z=rec.z,
                objectives=NamedTuple{ALL_OBJECTIVES}(Tuple(r[k] for k in ALL_OBJECTIVES)),
                diag=Base.structdiff(r, NamedTuple{ALL_OBJECTIVES}), physics=p, eval=cfgnt,
                batch_wall=rec.batch_wall)
        println(io, js(line)); flush(io)
    end
    arch, P = evolve(C, p, cfg; mode=mode, pop=pop, gens=gens, seed=seed,
                     init=mode == :nsga ? init : nothing, log=log)
    close(io)
    wall = time() - t0
    F = reduce(hcat, [objvec(a.res, SELECT_OBJECTIVES) for a in arch])
    hv = hypervolume(F; rng=MersenneTwister(12345))
    best = Dict(string(o) => maximum(F[m, :]) for (m, o) in enumerate(SELECT_OBJECTIVES))
    front = findall(==(1), nondominated_rank(F))
    archives[mode] = (arch=arch, F=F, front=front)
    summary["modes"][string(mode)] = Dict{String,Any}("evaluations" => length(arch), "wall_s" => wall,
        "hypervolume" => hv, "best" => best, "front_ids" => [arch[i].id for i in front],
        "front_size" => length(front), "archive" => basename(path))
    @printf("%s %s: %d evals in %.0f s, HV %.4f, best %s\n", run, mode, length(arch), wall, hv,
            join(("$k=$(round(v, digits=3))" for (k, v) in best), " "))
    if mode == :nsga
        written = Tuple{String,Int}[]
        for (m, o) in enumerate(SELECT_OBJECTIVES)
            i = front[argmax(F[m, front])]
            a = arch[i]
            any(w -> w[2] == a.id, written) && continue
            rule = Rule{C,Float32}(Float32.(a.theta))
            f = joinpath(recdir, "evolved-$run-$o.toml")
            write_recipe(f, rule, p, cfg; N=collect(cfg.dims), steps=cfg.T, backend=backend_name,
                         note="Pareto-front rule with best $o in $run (archive id $(a.id)); $cmd",
                         objectives=a.res)
            push!(written, (basename(f), a.id))
        end
        summary["modes"]["nsga"]["recipes"] = [w[1] for w in written]
    end
end
# correlations between objectives over everything evaluated in this run
Fall = reduce(hcat, [objvec(a.res, ALL_OBJECTIVES) for k in keys(archives) for a in archives[k].arch])
summary["objective_correlation"] = Dict(string(ALL_OBJECTIVES[i], "~", ALL_OBJECTIVES[j]) =>
    cor(Fall[i, :], Fall[j, :]) for i in 1:length(ALL_OBJECTIVES) for j in i+1:length(ALL_OBJECTIVES))
open(joinpath(outdir, "evolve-$run.json"), "w") do io
    println(io, js(summary))
end
