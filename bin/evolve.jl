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


# --- correlate mode: does lane A's linear analysis predict the measured structure?
#   julia --project=. bin/evolve.jl --correlate results/evolve-a.jsonl,results/evolve-b.jsonl --out name
if haskey(A, "correlate")
    using Intmat2.Analysis: steady_state, dispersion, classify, stable_dt
    numfield(line, key) = (m = match(Regex("\"$key\":(-?[0-9.eE+-]+|null)"), line);
                           m === nothing || m[1] == "null" ? NaN : parse(Float64, m[1]))
    ranks(v) = (o = sortperm(v); r = similar(v, Float64); r[o] = 1:length(v); r)
    spearman(a, b) = cor(ranks(a), ranks(b))
    rows = NamedTuple[]
    for f in split(A["correlate"], ",")
        for line in eachline(f)
            θ = parse.(Float64, split(match(r"\"theta\":\[([^\]]*)\]", line)[1], ","))
            ph = match(r"\"physics\":\{([^}]*)\}", line)[1]
            pd = Dict(Symbol(strip(k, '"')) => parse(Float64, v) for (k, v) in (split(kv, ":") for kv in split(ph, ",")))
            p64 = Physics{Float64}(; pd...)
            C = round(Int, (-8 + sqrt(64 + 8length(θ))) / 4)
            rule = Rule{C,Float64}(θ)
            ss = steady_state(rule, p64; M=numfield(line, "n0"), depth=1)
            d = ss.converged ? dispersion(rule, p64, ss) : nothing
            cls = d === nothing ? :unconverged : classify(d)
            sdt = ss.converged ? Float64(stable_dt(rule, p64; M=numfield(line, "n0"), ss=ss)) : NaN
            push!(rows, (file=basename(f), id=Int(numfield(line, "id")), cls=cls, stable_dt=sdt, dt=p64.dt,
                         lambda_max=d === nothing ? NaN : d.lambda_max, re0=d === nothing ? NaN : d.re0,
                         wavelength=d === nothing ? NaN : d.wavelength,
                         structure=numfield(line, "structure"), structure_top=numfield(line, "structure_top"), lam_meas=numfield(line, "lambda"),
                         struct_frac=numfield(line, "struct_frac")))
        end
    end
    ok = filter(r -> isfinite(r.lambda_max), rows)
    finite_growth = [r.lambda_max - max(r.re0, 0.0) for r in ok]   # finite-k excess over k=0
    S = [r.structure for r in ok]
    bycls = Dict(string(c) => (n=count(r -> r.cls == c, rows),
                               mean_structure=mean([r.structure for r in rows if r.cls == c]),
                               frac_structured=mean([r.structure > 0.1 for r in rows if r.cls == c]))
                 for c in unique(r.cls for r in rows))
    tur = filter(r -> r.cls in (:turing, :wave) && r.structure > 0.1, rows)
    out = (command="julia --project=. bin/evolve.jl " * join(ARGS, " "), n=length(rows), n_analyzed=length(ok),
           spearman_structure_vs_lambda_max=spearman(S, [r.lambda_max for r in ok]),
           spearman_structure_vs_finite_excess=spearman(S, finite_growth),
           spearman_structure_top_vs_lambda_max=(St = [r.structure_top for r in ok]; all(isfinite, St) ? spearman(St, [r.lambda_max for r in ok]) : NaN),
           frac_dt_unstable=mean(r.stable_dt < r.dt for r in ok),
           unstable_dt_ids=[(file=r.file, id=r.id, stable_dt=r.stable_dt, structure=r.structure) for r in ok if r.stable_dt < r.dt],
           by_class=bycls,
           turing_structured_wavelength_pred_vs_meas=[(id=r.id, file=r.file, pred=r.wavelength, meas=r.lam_meas) for r in tur])
    open(joinpath(@__DIR__, "..", "results", "evolve-$(get(A, "out", "atlas"))-correlation.json"), "w") do io
        js(x::AbstractString) = "\"" * escape_string(x) * "\""
        js(x::Symbol) = js(string(x)); js(x::Integer) = string(x)
        js(x::AbstractFloat) = isfinite(x) ? repr(Float64(x)) : "null"
        js(x::Union{AbstractVector,Tuple}) = "[" * join(js.(x), ",") * "]"
        js(x::NamedTuple) = "{" * join((js(string(k)) * ":" * js(v) for (k, v) in pairs(x)), ",") * "}"
        js(x::AbstractDict) = "{" * join((js(string(k)) * ":" * js(v) for (k, v) in x), ",") * "}"
        println(io, js(out))
    end
    for (c, v) in bycls
        @printf("%-12s n=%4d mean structure %.3f  frac>0.1 %.3f\n", c, v.n, v.mean_structure, v.frac_structured)
    end
    @printf("stable_dt < dt for %.3f of analyzed rules\n", out.frac_dt_unstable)
    @printf("spearman(structure, max Re λ) = %.3f   (vs finite-k excess) = %.3f\n",
            out.spearman_structure_vs_lambda_max, out.spearman_structure_vs_finite_excess)
    exit(0)
end

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
# rule prior: --scales self=0.5,nb=0.2,inp=1,bias=0.1,read=2 (random_rule keywords)
scales = haskey(A, "scales") ? (; (Symbol(k) => parse(Float64, v) for (k, v) in (split(kv, "=") for kv in split(A["scales"], ",")))...) : (;)
sampler = nothing
if arg("prior", "gauss") == "prone"     # lane A's Turing-prone sampler (Analysis.turing_prone_rule)
    @eval using Intmat2.Analysis: turing_prone_rule
    sampler = rng -> vec(Base.invokelatest(turing_prone_rule, rng, C, Float64))
end
init = nothing
if haskey(A, "init")      # comma-separated recipe files whose θ seed the first population
    init = [Float64.(vec(read_recipe(f)[1])) for f in split(A["init"], ",")]
end
# --init-turing K: add up to K rules drawn from the prior that lane A's linear analysis
# classifies :turing under this physics (top layer), searched with MersenneTwister(seed+1)
if haskey(A, "init-turing")
    @eval using Intmat2.Analysis: steady_state, dispersion, classify
    K = parse(Int, A["init-turing"]); rng2 = MersenneTwister(seed + 1); p64 = Physics{Float64}(p)
    found = Vector{Float64}[]; tries = 0
    while length(found) < K && tries < 20000
        tries += 1
        rule = random_rule(rng2, C, Float64; scales...)
        ss = Base.invokelatest(steady_state, rule, p64; M=cfg.n0, depth=1)
        ss.converged || continue
        Base.invokelatest(classify, Base.invokelatest(dispersion, rule, p64, ss)) == :turing && push!(found, vec(rule))
    end
    println("init-turing: $(length(found)) turing rules in $tries draws")
    init = vcat(something(init, Vector{Float64}[]), found)
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

# --- eval-recipes mode: score recipe rules, each under its own physics, with this eval config
if haskey(A, "eval-recipes")
    rows = Any[]
    for f in split(A["eval-recipes"], ",")
        rule, pr, d = read_recipe(f)
        r = evaluate([rule], pr, cfg)[1]
        pred = get(d, "predicted", Dict())
        push!(rows, (recipe=basename(f), class=get(pred, "class", ""), lambda_max=get(pred, "lambda_max", NaN),
                     wavelength=get(pred, "wavelength", NaN), physics=pr, theta=Float64.(vec(rule)), res=r))
        @printf("%-36s %-8s pred λ=%5.1f  structure %.3f (λ_meas %5.1f, rms %.3f)  persistence %.3f temporal %.3f memory_h %.3f recovery %.3f mean_s %.3f\n",
                basename(f), get(pred, "class", ""), get(pred, "wavelength", NaN), r.structure, r.lambda, r.struct_rms,
                r.persistence, r.temporal, r.memory_h, r.recovery, r.mean_s)
    end
    open(joinpath(outdir, "evolve-$run.json"), "w") do io
        println(io, js((command=cmd, commit=commit, eval=cfgnt, rows=rows)))
    end
    exit(0)
end

# --- recipe mode: write recipes/evolved-<name>.toml for archive rows, exact θ and physics
#   --recipe results/evolve-g1neg-nsga.jsonl:123:name[,file:id:name...]
if haskey(A, "recipe")
    for spec in split(A["recipe"], ",")
        f, id, name = split(spec, ":")
        line = first(l for l in eachline(f) if occursin("\"id\":$id,", l))
        θ = parse.(Float64, split(match(r"\"theta\":\[([^\]]*)\]", line)[1], ","))
        ph = match(r"\"physics\":\{([^}]*)\}", line)[1]
        pd = Dict(Symbol(strip(k, '"')) => parse(Float32, v) for (k, v) in (split(kv, ":") for kv in split(ph, ",")))
        pr = Physics{Float32}(; pd...)
        Cr = round(Int, (-8 + sqrt(64 + 8length(θ))) / 4)
        obj = Dict{Symbol,Float64}()
        for k in (ALL_OBJECTIVES..., :mean_s, :std_s, :struct_rms, :lambda, :photic)
            m = match(Regex("\"$k\":(-?[0-9.eE+-]+)"), line)
            m === nothing || (obj[k] = parse(Float64, m[1]))
        end
        path = joinpath(recdir, "evolved-$name.toml")
        write_recipe(path, Rule{Cr,Float32}(Float32.(θ)), pr, cfg; N=cfg.dims, steps=cfg.T,
                     backend=backend_name, note="archive $(basename(f)) id $id; eval $(cfg.dims) T=$(cfg.T)",
                     objectives=(; obj...))
        println(path)
    end
    exit(0)
end

# --- classify-prior mode: linear class fractions (lane A Analysis, top layer) of N rules
#     drawn from this prior under this physics, before spending any GPU time on it
if haskey(A, "classify-prior")
    @eval using Intmat2.Analysis: steady_state, dispersion, classify
    nr = parse(Int, A["classify-prior"]); rng = MersenneTwister(seed)
    p64 = Physics{Float64}(p); counts = Dict{Symbol,Int}()
    for _ in 1:nr
        rule = random_rule(rng, C, Float64; scales...)
        ss = Base.invokelatest(steady_state, rule, p64; M=cfg.n0, depth=1)
        c = ss.converged ? Base.invokelatest(classify, Base.invokelatest(dispersion, rule, p64, ss)) : :no_steady_state
        counts[c] = get(counts, c, 0) + 1
    end
    println(run, " ", scales, " ", sort(collect(counts), by=x -> -x[2]))
    open(joinpath(outdir, "evolve-$run-prior.json"), "w") do io
        println(io, js((command=cmd, commit=commit, physics=p, scales=scales, n=nr,
                        fractions=Dict(string(k) => v / nr for (k, v) in counts))))
    end
    exit(0)
end

summary = Dict{String,Any}("command" => cmd, "commit" => commit, "run" => run, "physics" => p,
                           "eval" => cfgnt, "pop" => pop, "gens" => gens, "seed" => seed, "scales" => scales, "prior" => arg("prior", "gauss"),
                           "select_objectives" => collect(SELECT_OBJECTIVES), "modes" => Dict{String,Any}())
archives = Dict{Symbol,Any}()
cache = Dict{Vector{Float64},Any}()
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
                     init=mode == :nsga ? init : nothing, log=log, cache=cache, scales=scales, sampler=sampler)
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
    open(io -> println(io, js(summary)), joinpath(outdir, "evolve-$run.json"), "w")
end
# correlations between objectives over everything evaluated in this run
Fall = reduce(hcat, [objvec(a.res, ALL_OBJECTIVES) for k in keys(archives) for a in archives[k].arch])
summary["objective_correlation"] = Dict(string(ALL_OBJECTIVES[i], "~", ALL_OBJECTIVES[j]) =>
    cor(Fall[i, :], Fall[j, :]) for i in 1:length(ALL_OBJECTIVES) for j in i+1:length(ALL_OBJECTIVES))
open(joinpath(outdir, "evolve-$run.json"), "w") do io
    println(io, js(summary))
end
