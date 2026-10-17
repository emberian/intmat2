# Rule search: NSGA-II over θ vs. an equal-budget random baseline.
#
#   julia -t 8 --project=env/metal bin/evolve.jl --run g1neg --g1 -0.5 --pop 48 --gens 8 \
#         --modes random,nsga --backend metal
#
# Writes results/evolve-<run>-<mode>.jsonl (one line per evaluated rule: θ, z, σ, parents,
# seeds, physics, eval config, objectives, diagnostics, wall time), results/evolve-<run>.json
# (command, hypervolume and per-objective best per mode, Pareto front) and, for the nsga
# mode, recipes/evolved-<run>-<objective>.toml for the front's best rule on each objective.
using Intmat2, KernelAbstractions, Random, Statistics, Printf, LinearAlgebra
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
"Channels C of a θ of length 2C²+8C (one ring) or 3C²+8C (two shells)."
infer_C(n) = something(findfirst(C -> n in (2C * C + 8C, 3C * C + 8C), 1:64), 0)
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
            C = infer_C(length(θ))
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
           turing_lambda_max_quartiles=(lt = [r.lambda_max for r in rows if r.cls == :turing]; isempty(lt) ? Float64[] : quantile(lt, [0.25, 0.5, 0.75])),
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
    println("turing λ_max quartiles: ", out.turing_lambda_max_quartiles)
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
if arg("prior", "gauss") == "tied"     # two-shell tied rules, nb1 ~ U[1,4], beta ~ U[0.3,0.5]
    sampler = rng -> vec(Intmat2.tied_rule(rng, C, Float64; nb1=1 + 3rand(rng), beta=0.3 + 0.2rand(rng)))
    haskey(A, "scales") || (scales = (nb=2.5, nb2=2.5))   # Wnb2 mutated on the Wnb scale
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
    found = Vector{Float64}[]; global tries = 0
    while length(found) < K && tries < 20000
        global tries += 1
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
        cfgr = EvalConfig(dims=cfg.dims, T=cfg.T, cont=cfg.cont, seed=cfg.seed, n0=Float64(get(d, "n0", cfg.n0)),
                          h_scale=Float64(get(d, "h_scale", cfg.h_scale)), backend=cfg.backend)
        r = evaluate([rule], pr, cfgr)[1]
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
        Cr = infer_C(length(θ))
        obj = Dict{Symbol,Float64}()
        for k in (unique((ALL_OBJECTIVES..., Intmat2.LIFE_OBJECTIVES...))..., :mean_s, :std_s, :struct_rms, :lambda, :photic, :n_regions, :event_rate, :stable_dt)
            m = match(Regex("\"$k\":(-?[0-9.eE+-]+)"), line)
            m === nothing || (obj[k] = parse(Float64, m[1]))
        end
        path = joinpath(recdir, "evolved-$name.toml")
        # --t_run T: steps = T/dt of the archived rule (per-rule dt), captures every --cap_dt
        steps_r = haskey(A, "t_run") ? round(Int, arg("t_run", 300.0) / Float64(pr.dt)) : cfg.T
        cap_r = haskey(A, "t_run") ? max(1, round(Int, arg("cap_dt", 2.0) / Float64(pr.dt))) : max(steps_r ÷ 10, 1)
        write_recipe(path, Rule{Cr,Float32}(Float32.(θ)), pr, cfg; N=cfg.dims, steps=steps_r, capture_every=cap_r,
                     backend=arg("recipe_backend", "metal"), note="archive $(basename(f)) id $id; dims $(cfg.dims), $(steps_r) steps at dt $(pr.dt)",
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

# --- hv mode: hypervolume and per-objective best of archives, all rules and only the
# rules passing the explicit-Euler check (ids from evolve-<run>-correlation.json)
#   --hv results/evolve-g1neg-random.jsonl,... --out name
if haskey(A, "hv")
    HVO = arg("objectives", "five") == "life" ? Intmat2.LIFE_SELECT : SELECT_OBJECTIVES
    rows = Any[]
    for f in split(A["hv"], ",")
        runname = match(r"evolve-(.*)-(random|nsga)\.jsonl", basename(f))[1]
        cj = joinpath(dirname(f), "evolve-$runname-correlation.json")
        bad = Set{Int}()
        if isfile(cj)
            for m in eachmatch(Regex("\\{\"file\":\"$(basename(f))\",\"id\":(\\d+)"), read(cj, String))
                push!(bad, parse(Int, m[1]))
            end
        end
        F = Vector{Float64}[]; Fok = Vector{Float64}[]
        for line in eachline(f)
            id = parse(Int, match(r"\"id\":(\d+)", line)[1])
            v = [clamp(parse(Float64, match(Regex("\"$o\":(-?[0-9.eE+-]+)"), line)[1]), 0.0, 1.0) for o in HVO]
            push!(F, v); id in bad || push!(Fok, v)
        end
        M = reduce(hcat, F); Mok = reduce(hcat, Fok)
        row = (archive=basename(f), n=length(F), n_dt_ok=length(Fok),
               hv=hypervolume(M; rng=MersenneTwister(12345)), hv_dt_ok=hypervolume(Mok; rng=MersenneTwister(12345)),
               best=NamedTuple{HVO}(Tuple(maximum(M, dims=2))),
               best_dt_ok=NamedTuple{HVO}(Tuple(maximum(Mok, dims=2))))
        push!(rows, row)
        @printf("%-28s n=%3d ok=%3d  HV %.4f  HV(dt ok) %.4f  best(ok) %s\n", row.archive, row.n, row.n_dt_ok, row.hv, row.hv_dt_ok,
                join(("$k=$(round(v, digits=3))" for (k, v) in pairs(row.best_dt_ok)), " "))
    end
    open(io -> println(io, js((command=cmd, commit=commit, objectives=HVO, rows=rows))),
         joinpath(outdir, "evolve-$(get(A, "out", "hv"))-hv.json"), "w")
    exit(0)
end

# --- tie mode: does evolution keep Wnb2 = −β·Wnb? per generation, median fitted β,
# residual |Wnb2 + β Wnb|/|Wnb2|, min eigenvalue of sym(Wnb), and |Wnb|
#   --tie results/evolve-tied-random.jsonl,results/evolve-tied-nsga.jsonl --out tied
if haskey(A, "tie")
    rows = Any[]
    for f in split(A["tie"], ",")
        bygen = Dict{Int,Vector{NTuple{4,Float64}}}()
        for line in eachline(f)
            θ = parse.(Float64, split(match(r"\"theta\":\[([^\]]*)\]", line)[1], ","))
            g = parse(Int, match(r"\"gen\":(\d+)", line)[1])
            r = Rule{C,Float64}(θ)
            W = Matrix(r.Wnb); W2 = Matrix(r.Wnb2)
            β = -sum(W .* W2) / sum(abs2, W)
            res = norm(W2 + β * W) / max(norm(W2), 1e-12)
            push!(get!(bygen, g, NTuple{4,Float64}[]), (β, res, minimum(eigvals(Symmetric((W + W') / 2))), norm(W)))
        end
        for g in sort(collect(keys(bygen)))
            v = bygen[g]
            row = (archive=basename(f), gen=g, beta=median(first.(v)), resid=median(getindex.(v, 2)),
                   min_eig_Wnb=median(getindex.(v, 3)), norm_Wnb=median(last.(v)))
            push!(rows, row)
            @printf("%-26s gen %d  β %.3f  resid %.3f  min eig sym(Wnb) %+.3f  |Wnb| %.2f\n", row.archive, g, row.beta, row.resid, row.min_eig_Wnb, row.norm_Wnb)
        end
    end
    open(io -> println(io, js((command=cmd, commit=commit, rows=rows))), joinpath(outdir, "evolve-$(get(A, "out", "tie"))-tie.json"), "w")
    exit(0)
end

# --- life mode: region objectives (LIFE_SELECT), per-rule dt, manifold search space
#   --life 1 --space tied|untied|flat --modes random,nsga --n0 0.5 --t_main 100 ...
if haskey(A, "life")
    using Intmat2: LifeConfig, evaluate_life, LIFE_SELECT, LIFE_OBJECTIVES, TiedSpace, with_dt
    lcfg = LifeConfig(dims=dims, t_main=arg("t_main", 100.0), t_cont=arg("t_cont", 25.0), cap_dt=arg("cap_dt", 2.0),
                      n0=arg("n0", 0.5), seed=arg("world_seed", 1), dt_max=arg("dt_max", 0.1),
                      dt_min=arg("dt_min", 0.004), backend=be)
    spname = arg("space", "tied")
    space = spname == "flat" ? nothing :
        TiedSpace(C; untied=spname == "untied", nb1=(arg("nb1_lo", 1.0), arg("nb1_hi", 16.0)),
                  beta=(arg("beta_lo", 0.28), arg("beta_hi", 0.5)))
    # --seed_from file.jsonl:id --seed_n 16 --seed_sigma 0.3: that rule's z plus seed_n−1
    # Gaussian perturbations of it seed the nsga population (untied D block padded with 0)
    init_z = nothing
    if haskey(A, "seed_from")
        f, id = split(A["seed_from"], ":")
        line = first(l for l in eachline(f) if occursin("\"id\":$id,", l))
        z = parse.(Float64, split(match(r"\"z\":\[([^\]]*)\]", line)[1], ","))
        np_ = length(Intmat2.space_scale(space))
        z = vcat(z, zeros(np_ - length(z)))
        zr = MersenneTwister(seed + 2)
        init_z = [i == 1 ? z : z .+ arg("seed_sigma", 0.3) .* randn(zr, length(z)) for i in 1:arg("seed_n", 16)]
    end
    lnt = (dims=lcfg.dims, t_main=lcfg.t_main, t_cont=lcfg.t_cont, cap_dt=lcfg.cap_dt, track_from=lcfg.track_from,
           world_seed=lcfg.seed, n0=lcfg.n0, h_scale=lcfg.h_scale, dt_max=lcfg.dt_max, dt_min=lcfg.dt_min,
           dt_safety=lcfg.dt_safety, min_size=lcfg.min_size, cell=lcfg.cell, backend=backend_name, C=C, space=spname)
    lsum = Dict{String,Any}("command" => cmd, "commit" => commit, "run" => run, "physics_base" => p, "eval" => lnt,
                            "pop" => pop, "gens" => gens, "seed" => seed, "select_objectives" => collect(LIFE_SELECT),
                            "modes" => Dict{String,Any}())
    allarch = Any[]
    for mode in modes
        path = joinpath(outdir, "evolve-$run-$mode.jsonl")
        io = open(path, "w"); t0 = time()
        logf = function (rec)
            r = rec.res
            line = (run=run, mode=mode, id=rec.id, gen=rec.gen, parents=rec.parents, sigma=rec.sigma,
                    es_seed=seed, world_seed=lcfg.seed, theta=rec.theta, z=rec.z,
                    objectives=NamedTuple{LIFE_OBJECTIVES}(Tuple(r[k] for k in LIFE_OBJECTIVES)),
                    diag=Base.structdiff(r, NamedTuple{LIFE_OBJECTIVES}), physics=with_dt(p, r.dt),
                    eval=lnt, batch_wall=rec.batch_wall)
            println(io, js(line)); flush(io)
            if rec.id % pop == 0
                @printf("%s %s gen %d  %.0f s\n", run, mode, rec.gen, time() - t0); flush(stdout)
            end
        end
        arch, P = evolve(C, p, lcfg; mode=mode, pop=pop, gens=gens, seed=seed, log=logf, space=space,
                         eval=evaluate_life, objectives=LIFE_SELECT, init_z=mode == :nsga ? init_z : nothing)
        close(io)
        F = reduce(hcat, [objvec(a.res, LIFE_SELECT) for a in arch])
        front = findall(==(1), nondominated_rank(F))
        hv = hypervolume(F; rng=MersenneTwister(12345))
        lsum["modes"][string(mode)] = Dict{String,Any}("evaluations" => length(arch), "wall_s" => time() - t0,
            "hypervolume" => hv, "best" => Dict(string(o) => maximum(F[m, :]) for (m, o) in enumerate(LIFE_SELECT)),
            "best_turnover_raw" => maximum(a.res.turnover for a in arch),
            "front_ids" => [arch[i].id for i in front], "archive" => basename(path))
        append!(allarch, arch)
        @printf("%s %s: %d evals, HV %.4f, best turnover %.3f\n", run, mode, length(arch), hv,
                maximum(a.res.turnover for a in arch)); flush(stdout)
        open(io -> println(io, js(lsum)), joinpath(outdir, "evolve-$run.json"), "w")
    end
    Fall = reduce(hcat, [Float64[a.res[k] for k in LIFE_OBJECTIVES] for a in allarch])
    lsum["objective_correlation"] = Dict(string(LIFE_OBJECTIVES[i], "~", LIFE_OBJECTIVES[j]) =>
        cor(Fall[i, :], Fall[j, :]) for i in 1:length(LIFE_OBJECTIVES) for j in i+1:length(LIFE_OBJECTIVES))
    open(io -> println(io, js(lsum)), joinpath(outdir, "evolve-$run.json"), "w")
    exit(0)
end

# --- series mode: run recipes on the GPU and keep only region statistics (no captures)
#   --series r1.toml,r2.toml --dims 128,128,48 --t_run 300 --cap_dt 2 --late 150 --out name
# --- identity mode: also capture h; per-region mean activity, network vs fragments
#   --identity r.toml --dims 64,64,24 --t_run 300 --cap_dt 5 --out name
if haskey(A, "series") || haskey(A, "identity")
    using Intmat2: region_mask, label_components, track, life_objectives, region_shapes
    ident_mode = haskey(A, "identity")
    rows = Any[]
    for f in split(ident_mode ? A["identity"] : A["series"], ",")
        rule, pr, d = read_recipe(f)
        Cr = Intmat2.nchannels(rule)
        wd = World(be, rule, dims; n0=Float64(d["n0"]), h_scale=Float64(get(d, "h_scale", 0.1)), seed=Int(get(d, "seed", 1)))
        wd = Intmat2.light!(wd, rule, pr)
        dtr = Float64(pr.dt); every = max(1, round(Int, arg("cap_dt", 2.0) / dtr))
        nsteps = round(Int, arg("t_run", 300.0) / dtr); late = arg("late", 150.0)
        series = Any[]; latecaps = Array{Float64,3}[]; prevlab = nothing
        hstats = Any[]; lineage = Dict{Int,Vector{Any}}(); prevident = Int[]; nextid = 0; prevnet = 0
        t0 = time()
        for st in every:every:nsteps
            wd = Intmat2.run!(wd, rule, pr, every)
            sh = Float64.(Array(wd.s)); t = st * dtr
            lab, sz = label_components(region_mask(sh))
            K = length(sz)
            tr = prevlab === nothing ? nothing : track(prevlab, lab)
            shp = region_shapes(lab, sz)
            push!(series, (t=t, count=K, largest_share=shp.largest_share, isolation=shp.isolation,
                           median_size=shp.median_size, mean_s=mean(sh),
                           births=tr === nothing ? 0 : tr.births, deaths=tr === nothing ? 0 : tr.deaths,
                           splits=tr === nothing ? 0 : tr.splits, merges=tr === nothing ? 0 : tr.merges))
            t >= late - 1e-9 && push!(latecaps, sh)
            if ident_mode && K > 0
                h = Float64.(Array(wd.h))                       # (C, Nx, Ny, Nz)
                net = argmax(sz)
                sums = zeros(Cr, K); zsum = zeros(K)
                for I in CartesianIndices(lab)
                    l = lab[I]; l == 0 && continue
                    sums[:, l] .+= view(h, :, I); zsum[l] += I[3]
                end
                means = sums ./ reshape(sz, 1, :); zc = zsum ./ sz
                frag = [l for l in 1:K if l != net]
                # network voxels in the fragments' depth range: the depth-matched reference
                zlo = isempty(frag) ? 1 : floor(Int, minimum(zc[frag])); zhi = isempty(frag) ? size(lab, 3) : ceil(Int, maximum(zc[frag]))
                nv = [view(h, :, I) for I in CartesianIndices(lab) if lab[I] == net && zlo <= I[3] <= zhi]
                netm = isempty(nv) ? means[:, net] : mean(nv)
                within = isempty(nv) ? NaN : sum(var([v[c] for v in nv]) for c in 1:Cr)
                between = length(frag) >= 2 ? sum(var(means[c, frag]) for c in 1:Cr) : NaN
                dist = [norm(means[:, l] - netm) / sqrt(within) for l in frag]
                # identities: continue via track.inherit; record each lineage's mean h over time
                ident = if tr === nothing
                    nextid += K; collect(nextid-K+1:nextid)
                else
                    [a > 0 ? prevident[a] : (nextid += 1) for a in tr.inherit]
                end
                for l in frag
                    id = ident[l]
                    origin = tr === nothing ? :initial : (tr.inherit[l] > 0 ? :continued : (tr.parent[l] == prevnet ? :split_from_network : tr.parent[l] > 0 ? :split_from_fragment : :birth))
                    v = get!(lineage, id, Any[])
                    isempty(v) && push!(v, origin)
                    push!(v, (t=t, h=means[:, l], z=norm(means[:, l] - netm) / sqrt(within)))
                end
                prevident = ident; prevnet = net
                push!(hstats, (t=t, n_fragments=length(frag), within_network_var=within, between_fragment_var=between,
                               ratio=between / within, median_fragment_distance=isempty(dist) ? NaN : median(dist),
                               frac_fragments_beyond_2sd=isempty(dist) ? NaN : mean(dist .> 2)))
            end
            prevlab = lab
        end
        lf = life_objectives(latecaps, arg("cap_dt", 2.0))
        row = Dict{String,Any}("recipe" => basename(f), "dims" => dims, "dt" => dtr, "t_run" => arg("t_run", 300.0),
                               "late_from" => late, "late" => lf, "series" => series, "wall_s" => time() - t0)
        if ident_mode
            # lineages living ≥ 3 captures: self-similarity of the mean-h signature (first vs last)
            # against same-time similarity between different fragments
            cosv(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-12)
            long = [(v[1], v[2:end]) for v in values(lineage) if length(v) >= 4]
            selfsim = [cosv(L[1].h, L[end].h) for (_, L) in long]
            zfirst = [L[1].z for (_, L) in long]; zlast = [L[end].z for (_, L) in long]
            bytime = Dict{Float64,Vector{Vector{Float64}}}()
            for v in values(lineage), e in v[2:end]
                push!(get!(bytime, e.t, Vector{Float64}[]), e.h)
            end
            cross = Float64[]
            for (_, hs) in bytime, i in 1:min(length(hs), 40), j in i+1:min(length(hs), 40)
                push!(cross, cosv(hs[i], hs[j]))
            end
            origins = Dict(string(o) => count(x -> x[1] == o, long) for o in unique(first.(long)))
            row["h"] = Dict{String,Any}("per_capture" => hstats, "n_long_lineages" => length(long),
                "long_lineage_origins" => origins,
                "selfsim_first_last" => isempty(selfsim) ? Float64[] : quantile(selfsim, [0.25, 0.5, 0.75]),
                "cross_fragment_sim_same_time" => isempty(cross) ? Float64[] : quantile(cross, [0.25, 0.5, 0.75]),
                "z_vs_network_first" => isempty(zfirst) ? Float64[] : quantile(zfirst, [0.25, 0.5, 0.75]),
                "z_vs_network_last" => isempty(zlast) ? Float64[] : quantile(zlast, [0.25, 0.5, 0.75]))
        end
        push!(rows, row)
        @printf("%s: late count %.0f, largest share %.2f, isolation %.2f, events b%d d%d s%d m%d, rate %.4f, fragmentation %.3f turnover %.3f (%.0f s)\n",
                basename(f), lf.n_regions, lf.largest_share, lf.isolation, lf.births, lf.deaths, lf.splits, lf.merges,
                lf.event_rate, lf.fragmentation, lf.turnover, time() - t0)
        ident_mode && println("  h: ", row["h"])
    end
    open(io -> println(io, js((command=cmd, commit=commit, rows=rows))),
         joinpath(outdir, "evolve-$(ident_mode ? "identity" : "series")-$(get(A, "out", run)).json"), "w")
    exit(0)
end

# --- track mode: regions of a bin/run.jl capture series (needs field s)
#   --track outputs/d/<rundir> --out name [--cap_dt 2]
if haskey(A, "track")
    using Intmat2: load_capture, region_mask, label_components, track, life_objectives
    rd = A["track"]
    frames = sort(filter(f -> isdir(joinpath(rd, f)) && all(isdigit, f), readdir(rd)))
    caps = [Float64.(load_capture(joinpath(rd, f)).fields[:s]) for f in frames]
    cap_dt = arg("cap_dt", 2.0)
    labs = [label_components(region_mask(c))[1] for c in caps]
    series = Any[]
    for t in eachindex(labs)
        tr = t == 1 ? (births=0, deaths=0, splits=0, merges=0) : track(labs[t-1], labs[t])
        push!(series, (frame=frames[t], count=Int(maximum(labs[t]; init=Int32(0))), mean_s=mean(caps[t]),
                       births=tr.births, deaths=tr.deaths, splits=tr.splits, merges=tr.merges))
    end
    half = (length(caps) ÷ 2 + 1):length(caps)
    lf_all = life_objectives(caps, cap_dt); lf_half = life_objectives(caps[half], cap_dt)
    open(io -> println(io, js((command=cmd, rundir=rd, cap_dt=cap_dt, dims=size(caps[1]), whole=lf_all,
                               second_half=lf_half, series=series))),
         joinpath(outdir, "evolve-track-$(get(A, "out", basename(rd))).json"), "w")
    println("whole: ", lf_all); println("second half: ", lf_half)
    for x in series[1:max(1, length(series) ÷ 15):end]
        @printf("%s count %4d  b %3d d %3d s %3d m %3d  mean_s %.3f\n", x.frame, x.count, x.births, x.deaths, x.splits, x.merges, x.mean_s)
    end
    exit(0)
end

summary = Dict{String,Any}("command" => cmd, "commit" => commit, "run" => run, "physics" => p,
                           "eval" => cfgnt, "pop" => pop, "gens" => gens, "seed" => seed, "scales" => scales, "prior" => arg("prior", "gauss"), "evaluator" => arg("evaluator", "tasks"),
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
                     init=mode == :nsga ? init : nothing, log=log, cache=cache, scales=scales, sampler=sampler,
                     eval=arg("evaluator", "tasks") == "batch" ? Intmat2.evaluate_batch : evaluate)
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
