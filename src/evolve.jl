# Objectives and rule search (lane D). See docs/evolution.md.

using TOML
#
# A rule is scored by one recorded run from a fixed initial condition plus three short
# continuations (control, activity reset, solid erased). Every objective is a scalar
# where larger is "more of the property"; the selection set lives in [0, 1].
# Search is NSGA-II-style (non-dominated sort + crowding) over self-adaptive Gaussian
# mutations in block-normalized θ space. No weighted sums.

"""
    EvalConfig(; dims, T, cont, seed, n0, h_scale, rec_every, probe_every, nprobe,
               reset_scale, erase_radius, concurrency, backend)

Standard initial condition and measurement schedule. `T` steps of the main run,
`cont` steps for each continuation. `backend` is a KernelAbstractions backend.
"""
Base.@kwdef struct EvalConfig
    dims::NTuple{3,Int} = (32, 32, 24)
    T::Int = 1200
    cont::Int = 300
    seed::Int = 1
    n0::Float64 = 1.0
    h_scale::Float64 = 0.1
    rec_every::Int = 20
    probe_every::Int = 5
    nprobe::Int = 16
    reset_scale::Float64 = 0.01
    erase_radius::Float64 = 0.0      # 0 → min(Nx, Ny)/6
    concurrency::Int = 4
    backend::Any = KA.CPU()
end

"Objectives used for Pareto selection (all in [0,1], larger is better)."
const SELECT_OBJECTIVES = (:persistence, :structure, :temporal, :memory_h, :recovery)
"All objectives recorded per evaluation."
const ALL_OBJECTIVES = (:persistence, :structure, :temporal, :memory_s, :memory_h, :recovery, :light)

# ----------------------------------------------------------------------------- objectives

"""
    persistence(trace; band=(0.05, 0.6), width=0.05) -> [0, 1]

Mean over the last quarter of the record of `exp(-d/width)`, `d` the distance of
`mean_s` from `band`, times `min(1, std_s(end) / max std_s over the last half)` so a
collapsing texture scores lower. 1 = solid present, unsaturated, texture held.
"""
function persistence(trace; band=(0.05, 0.6), width=0.05)
    nr = length(trace)
    q = max(1, nr - nr ÷ 4 + 1):nr
    bandscore(m) = exp(-max(band[1] - m, m - band[2], 0.0) / width)
    b = mean(bandscore(trace[i].mean_s) for i in q)
    half = max(1, nr ÷ 2):nr
    smax = maximum(trace[i].std_s for i in half)
    held = smax > 1e-9 ? min(1.0, trace[end].std_s / smax) : 0.0
    b * held
end

# Lateral DFT matrices are cached per size (host, Float64).
dftmat(N) = [cis(-2π * (a - 1) * (b - 1) / N) for a in 1:N, b in 1:N]
wavenumber(i, N) = 2π * (i - 1 <= N ÷ 2 ? i - 1 : i - 1 - N) / N

"""
    structure(s::Array{<:Real,3}; kband=(2π/16, 2π/4), amp=0.05)
        -> (score, frac, rms, lambda)

Lateral (x–y, per z-layer, layer mean removed) power spectrum of the solid. `frac` is
the share of power with |k| in `kband`; white noise gives `f0` = the band's share of
modes, so `score = clamp((frac−f0)/(1−f0), 0, 1) · min(1, rms/amp)` ∈ [0, 1]. `rms` is
the lateral rms deviation, `lambda` the dominant wavelength (cells). Vertical
stratification (the canopy gradient) is excluded on purpose.
"""
function structure(s::AbstractArray{<:Real,3}; kband=(2π / 16, 2π / 4), amp=0.05)
    Nx, Ny, Nz = size(s)
    Fx = dftmat(Nx); Fy = dftmat(Ny)
    P = zeros(Nx, Ny)
    for k in 1:Nz
        S = Float64.(s[:, :, k])
        S .-= mean(S)
        P .+= abs2.(Fx * S * transpose(Fy))
    end
    P[1, 1] = 0.0
    tot = sum(P)
    inband = 0.0; nband = 0; best = 0.0; λ = 0.0
    for j in 1:Ny, i in 1:Nx
        (i == 1 && j == 1) && continue
        kk = hypot(wavenumber(i, Nx), wavenumber(j, Ny))
        if kband[1] <= kk <= kband[2]
            inband += P[i, j]; nband += 1
        end
        if P[i, j] > best
            best = P[i, j]; λ = 2π / kk
        end
    end
    rms = sqrt(tot / (Nx * Ny)^2 / Nz)   # Parseval: Σ_k|X|² = NxNy Σ_x|x|²
    tot <= 1e-20 && return (score=0.0, frac=0.0, rms=0.0, lambda=0.0)
    frac = inband / tot
    f0 = nband / (Nx * Ny - 1)
    score = clamp((frac - f0) / (1 - f0), 0.0, 1.0) * min(1.0, rms / amp)
    (score=score, frac=frac, rms=rms, lambda=λ)
end

"""
    temporal(probes::Array{<:Real,3}; amp=0.02, periodic_H=0.35)
        -> (score, fixed, periodic)

`probes` is `(C, P, M)`: C channels of P voxels at M equally spaced times. Each series
is linearly detrended and Hann-windowed; its normalized spectral entropy
`H = −Σ p log p / log(M/2)` is 1 for flat (broadband) spectra and near 0 for one line.
Series score `H · min(1, std/amp)`; a fixed point (std < 1e-4) scores 0. `score` is
the mean ∈ [0, 1]; `fixed`, `periodic` are the fractions of series that are fixed or
have `H < periodic_H` (line spectra: limit cycles).
"""
function temporal(probes::AbstractArray{<:Real,3}; amp=0.02, periodic_H=0.35)
    C, P, M = size(probes)
    M < 8 && return (score=0.0, fixed=1.0, periodic=0.0)
    t = collect(1.0:M); tc = t .- mean(t)
    win = [0.5 - 0.5 * cos(2π * (m - 1) / (M - 1)) for m in 1:M]
    F = dftmat(M)[2:M÷2+1, :]
    tot = 0.0; nfix = 0; nper = 0
    for q in 1:P, c in 1:C
        x = Float64.(probes[c, q, :])
        x .-= mean(x)
        x .-= tc .* (sum(tc .* x) / sum(abs2, tc))
        sd = std(x)
        if !(sd > 1e-4)
            nfix += 1; continue
        end
        pw = abs2.(F * (x .* win))
        pw ./= sum(pw)
        H = -sum(v > 0 ? v * log(v) : 0.0 for v in pw) / log(length(pw))
        H < periodic_H && (nper += 1)
        tot += H * min(1.0, sd / amp)
    end
    n = C * P
    (score=tot / n, fixed=nfix / n, periodic=nper / n)
end

function corr0(a, b)
    a = vec(Float64.(a)); b = vec(Float64.(b))
    sa = std(a); sb = std(b)
    (sa > 1e-9 && sb > 1e-9) ? cor(a, b) : 0.0
end

"""
    memory(s_ctrl, s_pert, h_ctrl, h_pert) -> (memory_s, memory_h)

Pearson correlations ∈ [−1, 1] between the control continuation and the continuation
whose activity was reset to small noise (material kept). `memory_s`: is the material
robust to forgetting the activity. `memory_h`: does the activity re-form from the
material (0 if either field is uniform).
"""
memory(s_ctrl, s_pert, h_ctrl, h_pert) = (memory_s=corr0(s_ctrl, s_pert), memory_h=corr0(h_ctrl, h_pert))

"""
    recovery(s_ctrl, s_erased, mask, s_before) -> [−1, 1]

The wound: inside `mask` the solid was removed (its mass returned to `n`) and the
activity reset to small noise. With `m, σ` the mean and std of `s` inside the mask
after the continuation, `1 − (|m_e − m_c| + |σ_e − σ_c|) / (m_c + σ_c)`, clipped:
1 = the wound regrew the control's amount *and* texture (phase-free, so a shifted
stripe still counts), 0 = it stayed empty, < 0 = overshoot. 0 when the region held
(almost) no solid before the cut.
"""
function recovery(s_ctrl, s_er, mask, s_before)
    nm = count(mask)
    (nm < 2 || sum(s_before[mask]) / nm < 1e-3) && return 0.0
    c = s_ctrl[mask]; e = s_er[mask]
    mc, me, σc, σe = mean(c), mean(e), std(c), std(e)
    mc + σc < 1e-9 && return 0.0
    clamp(1 - (abs(me - mc) + abs(σe - σc)) / (mc + σc), -1.0, 1.0)
end

"""
    light_budget(L, L0; lit=0.1) -> (score, photic)

`photic` = mean over columns of the fraction of the depth with `L > lit·L0` (final
state). `score = 4x(1−x)` with `x = photic` ∈ [0, 1]: 0 = blacked out below the top
cell or a transparent slab, 1 = a canopy that lights half the depth.
"""
function light_budget(L::AbstractArray{<:Real,3}, L0; lit=0.1)
    Nx, Ny, Nz = size(L)
    x = count(>(lit * L0), L) / (Nx * Ny * Nz)
    (score=4x * (1 - x), photic=x)
end

# ----------------------------------------------------------------------------- one evaluation

@kernel function probe_kernel!(buf, @Const(h), @Const(idx), r, C)
    c, q = @index(Global, NTuple)
    @inbounds buf[c, q, r] = h[c+C*(idx[q]-1)]
end

"""
    record_run(rule, p, cfg) -> (world, trace, probes)

Run `cfg.T` steps from the standard initial condition, recording `summarize` every
`rec_every` steps and `nprobe` voxels' activity every `probe_every` steps over the
last `T/2` steps (gathered on device, copied once).
"""
function record_run(rule::Rule{C,T}, p::Physics{T}, cfg::EvalConfig) where {C,T}
    be = cfg.backend
    wd = World(be, rule, cfg.dims; n0=cfg.n0, h_scale=cfg.h_scale, seed=cfg.seed)
    wd = light!(wd, rule, p)
    nv = prod(cfg.dims)
    prng = MersenneTwister(cfg.seed + 7919)
    idx = adapt(be, Int32.(rand(prng, 1:nv, cfg.nprobe)))
    t0 = cfg.T - cfg.T ÷ 2
    M = (cfg.T - t0) ÷ cfg.probe_every
    buf = KA.zeros(be, T, C, cfg.nprobe, max(M, 1))
    pk = probe_kernel!(be)
    trace = typeof(summarize(wd))[]
    r = 0
    for st in 1:cfg.T
        wd = step!(wd, rule, p)
        st % cfg.rec_every == 0 && push!(trace, summarize(wd))
        if st > t0 && (st - t0) % cfg.probe_every == 0 && r < M
            r += 1
            pk(buf, wd.h, idx, r, C; ndrange=(C, cfg.nprobe))
        end
    end
    KA.synchronize(be)
    (wd, trace, Array{Float64}(Array(buf)))
end

function erase_mask(s::AbstractArray{<:Real,3}, radius)
    Nx, Ny, Nz = size(s)
    R = radius > 0 ? radius : min(Nx, Ny) / 6
    layer = [mean(view(s, :, :, k)) for k in 1:Nz]
    kc = maximum(layer) > 1e-6 ? argmax(layer) : (Nz + 1) ÷ 2
    c = ((Nx + 1) / 2, (Ny + 1) / 2, kc)
    [(i - c[1])^2 + (j - c[2])^2 + (k - c[3])^2 <= R^2 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
end

const BAD = (persistence=0.0, structure=0.0, temporal=0.0, memory_s=0.0, memory_h=0.0,
             recovery=0.0, light=0.0)

"""
    evaluate_one(rule, p, cfg) -> NamedTuple

All objectives (see `ALL_OBJECTIVES`) plus diagnostics for one rule. Non-finite
states score `BAD` with `finite=false`.
"""
function evaluate_one(rule::Rule{C,T}, p::Physics{T}, cfg::EvalConfig) where {C,T}
    t0 = time()
    wd, trace, probes = record_run(rule, p, cfg)
    s_end = Array{Float64}(Array(wd.s))
    fin = all(isfinite, s_end) && all(isfinite, probes)
    # three continuations from the same state
    ctrl = copy(wd)
    pert = copy(wd)
    rng = MersenneTwister(cfg.seed + 104729)
    copyto!(pert.h, T.(cfg.reset_scale .* randn(rng, size(wd.h))))
    pert = light!(pert, rule, p)
    er = copy(wd)
    mask = erase_mask(s_end, cfg.erase_radius)
    nh = Array(er.n); sh = Array(er.s); hh = Array(er.h)
    nh[mask] .+= sh[mask]; sh[mask] .= zero(T)
    noise = T.(cfg.reset_scale .* randn(rng, size(hh)))
    for I in findall(mask), c in 1:C
        hh[c, I] = noise[c, I]
    end
    copyto!(er.n, nh); copyto!(er.s, sh); copyto!(er.h, hh)
    er = light!(er, rule, p)
    ctrl = run!(ctrl, rule, p, cfg.cont)
    pert = run!(pert, rule, p, cfg.cont)
    er = run!(er, rule, p, cfg.cont)
    sc = Array{Float64}(Array(ctrl.s)); hc = Array(ctrl.h)
    sp = Array{Float64}(Array(pert.s)); hp = Array(pert.h)
    se = Array{Float64}(Array(er.s))
    fin &= all(isfinite, sc) && all(isfinite, hc) && all(isfinite, hp) && all(isfinite, se)
    st = structure(s_end)
    sttop = structure(view(s_end, :, :, size(s_end, 3):size(s_end, 3)))   # the lit top layer alone
    tm = temporal(probes)
    mem = memory(sc, sp, hc, hp)
    lb = light_budget(Array{Float64}(Array(wd.L)), Float64(p.L0))
    obj = fin ? (persistence=persistence(trace), structure=st.score, temporal=tm.score,
                 memory_s=mem.memory_s, memory_h=mem.memory_h,
                 recovery=recovery(sc, se, mask, s_end), light=lb.score) : BAD
    # explicit-Euler check (lane A): a rule whose linearization needs dt < p.dt is
    # scored on a numerical artefact; `objvec` treats it as infeasible.
    sdt = try
        Float64(Analysis.stable_dt(rule, p; M=cfg.n0))
    catch
        NaN
    end
    last = trace[end]
    diag = (stable_dt=sdt, dt_ok=!(sdt < Float64(p.dt)), mean_s=last.mean_s, std_s=last.std_s, solid_frac=last.solid_frac, mean_n=last.mean_n,
            mean_w=last.mean_w, light_bottom=last.light_bottom, mean_abs_h=last.mean_abs_h,
            std_h=last.std_h, mass_drift=last.mass - trace[1].mass, struct_frac=st.frac,
            struct_rms=st.rms, lambda=st.lambda, structure_top=sttop.score, lambda_top=sttop.lambda, t_fixed=tm.fixed, t_periodic=tm.periodic, photic=lb.photic,
            h_autocorr=corr0(Array(wd.h), hc), finite=fin, wall=time() - t0)
    merge(map(x -> isfinite(x) ? x : 0.0, obj), diag)
end

"""
    evaluate(rules, p, cfg) -> Vector{NamedTuple}

Score a population. The default runs `cfg.concurrency` independent worlds at once as
Julia tasks (each task drives its own device queue). Pluggable: any function with this
signature can replace it in `evolve`.
"""
function evaluate(rules::AbstractVector{<:Rule}, p::Physics, cfg::EvalConfig)
    out = Vector{Any}(undef, length(rules))
    sem = Base.Semaphore(cfg.concurrency)
    @sync for i in eachindex(rules)
        Threads.@spawn Base.acquire(sem) do
            out[i] = evaluate_one(rules[i], p, cfg)
        end
    end
    [o for o in out]
end

# ----------------------------------------------------------------------------- Pareto machinery

"`a` dominates `b` (maximization)."
dominates(a, b) = all(a .>= b) && any(a .> b)

"""Non-dominated sorting of columns of `F` (objectives × points). Returns rank (1 = front)."""
function nondominated_rank(F::AbstractMatrix)
    n = size(F, 2)
    S = [Int[] for _ in 1:n]; nd = zeros(Int, n); rank = zeros(Int, n)
    for i in 1:n, j in 1:n
        i == j && continue
        if dominates(view(F, :, i), view(F, :, j))
            push!(S[i], j)
        elseif dominates(view(F, :, j), view(F, :, i))
            nd[i] += 1
        end
    end
    front = findall(==(0), nd); r = 1
    while !isempty(front)
        rank[front] .= r
        nxt = Int[]
        for i in front, j in S[i]
            nd[j] -= 1
            nd[j] == 0 && push!(nxt, j)
        end
        front = nxt; r += 1
    end
    rank
end

"""Crowding distance of the points `idx` (one front) in `F`."""
function crowding(F::AbstractMatrix, idx::AbstractVector{Int})
    d = zeros(length(idx))
    length(idx) <= 2 && return fill(Inf, length(idx))
    for m in 1:size(F, 1)
        v = F[m, idx]; o = sortperm(v)
        d[o[1]] = d[o[end]] = Inf
        span = v[o[end]] - v[o[1]]
        span <= 0 && continue
        for t in 2:length(o)-1
            d[o[t]] += (v[o[t+1]] - v[o[t-1]]) / span
        end
    end
    d
end

"""Select `k` columns of `F` by rank then crowding (NSGA-II environmental selection)."""
function nsga_select(F::AbstractMatrix, k::Integer)
    rank = nondominated_rank(F)
    chosen = Int[]; crowd = zeros(size(F, 2))
    for r in 1:maximum(rank)
        idx = findall(==(r), rank)
        cd = crowding(F, idx); crowd[idx] .= cd
        if length(chosen) + length(idx) <= k
            append!(chosen, idx)
        else
            append!(chosen, idx[sortperm(cd, rev=true)][1:k-length(chosen)])
            break
        end
    end
    chosen, rank, crowd
end

"""
    hypervolume(F; ref=0, nsamples=200_000, rng) -> Float64

Monte Carlo hypervolume of the points (columns of `F`, objectives in [0,1]) dominated
over the reference point 0, as a fraction of the unit cube. Same `rng` seed ⇒ same
samples, so two sets compared with one seed share their noise.
"""
function hypervolume(F::AbstractMatrix; nsamples=200_000, rng=MersenneTwister(0))
    front = F[:, nondominated_rank(F) .== 1]
    m = size(F, 1); hit = 0; u = zeros(m)
    for _ in 1:nsamples
        rand!(rng, u)
        for j in axes(front, 2)
            if all(front[q, j] >= u[q] for q in 1:m)
                hit += 1; break
            end
        end
    end
    hit / nsamples
end

# ----------------------------------------------------------------------------- search

"""Per-parameter scale of `random_rule` (θ = scale ⊙ z, z ~ N(0,1) is a random rule)."""
function rule_scale(C; self=0.8, nb=0.8, inp=0.5, bias=0.1, read=0.5)
    sc = 1 / sqrt(C)
    vcat(fill(self * sc, C * C), fill(nb * sc, C * C), fill(inp, 4C), fill(bias, C), fill(read, 3C))
end
"""Block boundaries of θ (Wself, Wnb, Win, b, vdep, vero, vpig) for block crossover."""
rule_blocks(C) = (o = cumsum([0, C * C, C * C, 4C, C, C, C, C]); [o[i]+1:o[i+1] for i in 1:7])

"""Selection vector in [0,1]; all zeros (dominated by every feasible rule) when the
rule fails the explicit-Euler check `dt_ok` (absent in old archives ⇒ feasible)."""
objvec(r::NamedTuple, objs) = get(r, :dt_ok, true) ? Float64[clamp(getfield(r, o), 0.0, 1.0) for o in objs] :
                                                    zeros(length(objs))

"""
    evolve(C, p, cfg; mode=:nsga, pop=48, gens=8, seed=1, sigma0=0.3, init=nothing,
           eval=evaluate, objectives=SELECT_OBJECTIVES, log=(rec)->nothing)

`mode=:nsga`: (μ+λ) NSGA-II over z = θ ./ rule_scale(C): binary tournament on
(rank, crowding), block-uniform crossover with probability 0.5, self-adaptive
log-normal step `σ' = σ exp(0.2 N)`, `z' = z + σ' N`. `mode=:random`: every evaluation
is a fresh `random_rule` draw (z ~ N(0,1)) — the equal-budget baseline. `scales`
(keywords of `random_rule`: self, nb, inp, bias, read) sets the prior θ = scale ⊙ z.
`sampler(rng) -> θ` replaces the Gaussian prior for the first population and for
`:random` (e.g. `Analysis.turing_prone_rule`); `scales` still normalizes mutation steps. `init`: optional
vector of θ to seed the first population (remaining slots are random). Every evaluated
individual is passed to `log` as a record. `cache` (θ → result) is consulted before
evaluating and filled after, so runs sharing a seed share their identical first
population; cached individuals still count toward the evaluation budget. Returns `(archive, final_population_indices)`.
"""
function evolve(C::Integer, p::Physics{T}, cfg::EvalConfig; mode=:nsga, pop=48, gens=8, seed=1,
                sigma0=0.3, init=nothing, eval=evaluate, objectives=SELECT_OBJECTIVES,
                log=rec -> nothing, cache=Dict{Vector{Float64},Any}(), scales=(;), sampler=nothing) where {T}
    rng = MersenneTwister(seed)
    scale = rule_scale(C; scales...); blocks = rule_blocks(C); np = length(scale)
    draw() = sampler === nothing ? randn(rng, np) : Float64.(sampler(rng)) ./ scale
    archive = NamedTuple[]
    function run_batch!(zs, sigmas, parents, gen)
        rules = [Rule{C,T}(T.(scale .* z)) for z in zs]
        zs = [Float64.(vec(r)) ./ scale for r in rules]    # z of the rule actually run
        t0 = time()
        keys_ = [Float64.(vec(r)) for r in rules]
        todo = findall(k -> !haskey(cache, k), keys_)
        fresh = isempty(todo) ? [] : eval(rules[todo], p, cfg)
        for (j, i) in enumerate(todo)
            cache[keys_[i]] = fresh[j]
        end
        res = [cache[k] for k in keys_]
        for (i, r) in enumerate(res)
            rec = (id=length(archive) + 1, gen=gen, mode=mode, parents=parents[i], sigma=sigmas[i],
                   theta=Float64.(vec(rules[i])), z=zs[i], res=r, batch_wall=time() - t0)
            push!(archive, rec); log(rec)
        end
        length(archive)-length(res)+1:length(archive)
    end
    z0 = [draw() for _ in 1:pop]
    if init !== nothing
        for (i, θ) in enumerate(init[1:min(end, pop)])
            z0[i] = Float64.(θ) ./ scale
        end
    end
    P = collect(run_batch!(z0, fill(sigma0, pop), fill(Int[], pop), 0))
    for g in 1:gens-1
        if mode == :random
            run_batch!([draw() for _ in 1:pop], fill(0.0, pop), fill(Int[], pop), g)
            continue
        end
        FP = reduce(hcat, [objvec(archive[i].res, objectives) for i in P])
        rankP = nondominated_rank(FP)
        crowdP = zeros(length(P))
        for r in unique(rankP)
            idx = findall(==(r), rankP); crowdP[idx] = crowding(FP, idx)
        end
        better(a, b) = rankP[a] < rankP[b] || (rankP[a] == rankP[b] && crowdP[a] > crowdP[b])
        tour() = (a = rand(rng, 1:length(P)); b = rand(rng, 1:length(P)); better(a, b) ? a : b)
        zs = Vector{Vector{Float64}}(); sig = Float64[]; par = Vector{Vector{Int}}()
        for _ in 1:pop
            a = tour(); b = tour()
            A = archive[P[a]]; B = archive[P[b]]
            z = copy(A.z); σ = A.sigma; pa = [A.id]
            if rand(rng) < 0.5
                for bl in blocks
                    rand(rng) < 0.5 && (z[bl] = B.z[bl])
                end
                σ = sqrt(A.sigma * B.sigma); push!(pa, B.id)
            end
            σ = clamp(σ * exp(0.2 * randn(rng)), 0.02, 1.0)
            z .+= σ .* randn(rng, np)
            push!(zs, z); push!(sig, σ); push!(par, pa)
        end
        Q = collect(run_batch!(zs, sig, par, g))
        U = vcat(P, Q)
        FU = reduce(hcat, [objvec(archive[i].res, objectives) for i in U])
        chosen, _, _ = nsga_select(FU, pop)
        P = U[chosen]
    end
    archive, P
end

# ----------------------------------------------------------------------------- recipes

"""
    write_recipe(path, rule, p, cfg; N, steps, capture_every, backend, note, objectives)

TOML recipe with the explicit `theta` (exact rule), `rule_scales`, every `Physics` field,
initial condition and run length. `N` is an integer or a 3-tuple.
"""
function write_recipe(path, rule::Rule{C}, p::Physics, cfg::EvalConfig; N=cfg.dims, steps=cfg.T,
                      capture_every=max(steps ÷ 10, 1), backend="metal", note="", objectives=nothing) where {C}
    d = Dict{String,Any}(
        "name" => splitext(basename(path))[1], "s0" => 0.0, "C" => C, "theta" => Float64.(vec(rule)),
        "rule_scales" => Dict("self" => 0.8, "nb" => 0.8, "inp" => 0.5, "bias" => 0.1, "read" => 0.5),
        "physics" => Dict(string(f) => Float64(getfield(p, f)) for f in fieldnames(Physics)),
        "n0" => cfg.n0, "h_scale" => cfg.h_scale, "seed" => cfg.seed,
        (N isa Integer ? "N" : "dims") => (N isa Integer ? N : collect(N)), "steps" => steps, "capture_every" => capture_every,
        "backend" => backend, "note" => note)
    objectives === nothing || (d["objectives"] = Dict(string(k) => Float64(v) for (k, v) in pairs(objectives)
                                                       if v isa Real && !(v isa Bool)))
    open(io -> TOML.print(io, d; sorted=true), path, "w")
    path
end

"""Read the rule and physics back from a recipe written by `write_recipe` (or with the same keys)."""
function read_recipe(path; T=Float32)
    d = TOML.parsefile(path)
    C = d["C"]
    rule = if haskey(d, "theta")
        Rule{C,T}(T.(d["theta"]))
    else    # seed-constructed (lane A atlas recipes): random_rule(MersenneTwister(seed), C, Float64; rule_scales...)
        sc = Dict(Symbol(k) => v for (k, v) in get(d, "rule_scales", Dict()))
        convert(Rule{C,T}, random_rule(MersenneTwister(d["seed"]), C, Float64; sc...))
    end
    ph = d["physics"]
    p = Physics{T}(; (f => T(ph[string(f)]) for f in fieldnames(Physics) if haskey(ph, string(f)))...)
    rule, p, d
end
