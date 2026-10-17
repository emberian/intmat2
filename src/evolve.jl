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
    ctrl, pert, er, mask = perturbations(wd, rule, p, cfg)
    ctrl = run!(ctrl, rule, p, cfg.cont)
    pert = run!(pert, rule, p, cfg.cont)
    er = run!(er, rule, p, cfg.cont)
    score_rule(rule, p, cfg, trace, probes, Array(wd.s), Array(wd.h), Array(wd.L),
               Array(ctrl.s), Array(ctrl.h), Array(pert.s), Array(pert.h), Array(er.s), mask, t0)
end

"""The three continuation starts from `wd`: control, activity reset, wound (+ its mask)."""
function perturbations(wd::World{C,T}, rule, p, cfg) where {C,T}
    s_end = Array{Float64}(Array(wd.s))
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
    ctrl, pert, er, mask
end

"""All objectives and diagnostics from host copies of the recorded and continued fields."""
function score_rule(rule, p, cfg, trace, probes, s_end, h_end, L_end, sc, hc, sp, hp, se, mask, t0)
    s_end = Float64.(s_end); sc = Float64.(sc); sp = Float64.(sp); se = Float64.(se)
    fin = all(isfinite, s_end) && all(isfinite, probes) && all(isfinite, sc) && all(isfinite, hc) &&
          all(isfinite, hp) && all(isfinite, se)
    st = structure(s_end)
    sttop = structure(view(s_end, :, :, size(s_end, 3):size(s_end, 3)))   # the lit top layer alone
    tm = temporal(probes)
    mem = memory(sc, sp, hc, hp)
    lb = light_budget(Float64.(L_end), Float64(p.L0))
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
    diag = (stable_dt=sdt, dt_ok=!(sdt < Float64(p.dt)), mean_s=last.mean_s, std_s=last.std_s,
            solid_frac=last.solid_frac, mean_n=last.mean_n, mean_w=last.mean_w,
            light_bottom=last.light_bottom, mean_abs_h=last.mean_abs_h, std_h=last.std_h,
            mass_drift=last.mass - trace[1].mass, struct_frac=st.frac, struct_rms=st.rms,
            lambda=st.lambda, structure_top=sttop.score, lambda_top=sttop.lambda, t_fixed=tm.fixed,
            t_periodic=tm.periodic, photic=lb.photic, h_autocorr=corr0(h_end, hc), finite=fin,
            wall=time() - t0)
    merge(map(x -> isfinite(x) ? x : 0.0, obj), diag)
end

@kernel function probe_batch_kernel!(buf, @Const(h), @Const(idx), r, C, nv)
    c, q, b = @index(Global, NTuple)
    @inbounds buf[c, q, r, b] = h[c+C*(idx[q]-1)+C*nv*(b-1)]
end

"""
    evaluate_batch(rules, p, cfg) -> Vector{NamedTuple}

Same measurements as `evaluate_one`, with all rules stepped together in one
`WorldBatch` (one launch per step), the three continuations in a second batch of 3B
worlds. Identical initial condition for every world (`cfg.seed`).
"""
function evaluate_batch(rules::AbstractVector{<:Rule{C,T}}, p::Physics{T}, cfg::EvalConfig) where {C,T}
    t0 = time()
    be = cfg.backend; B = length(rules); nv = prod(cfg.dims)
    wb = WorldBatch(be, C, cfg.dims, B; T=T, seeds=fill(cfg.seed, B), n0=cfg.n0, h_scale=cfg.h_scale)
    θ = device_rules(be, rules)
    wb = light!(wb, θ, p)
    prng = MersenneTwister(cfg.seed + 7919)
    idx = adapt(be, Int32.(rand(prng, 1:nv, cfg.nprobe)))
    tp = cfg.T - cfg.T ÷ 2
    M = (cfg.T - tp) ÷ cfg.probe_every
    buf = KA.zeros(be, T, C, cfg.nprobe, max(M, 1), B)
    pk = probe_batch_kernel!(be)
    traces = [NamedTuple[] for _ in 1:B]
    r = 0
    for st in 1:cfg.T
        wb = step!(wb, θ, p; sync=false)
        if st % cfg.rec_every == 0
            KA.synchronize(be)
            for (b, sm) in enumerate(summarize(wb))
                push!(traces[b], sm)
            end
        end
        if st > tp && (st - tp) % cfg.probe_every == 0 && r < M
            r += 1
            pk(buf, wb.h, idx, r, C, nv; ndrange=(C, cfg.nprobe, B))
        end
    end
    KA.synchronize(be)
    probes = Array{Float64}(Array(buf))
    starts = [perturbations(world(wb, b), rules[b], p, cfg) for b in 1:B]
    cb = WorldBatch(be, [w for k in 1:3 for w in (st[k] for st in starts)])
    cb = run!(cb, device_rules(be, vcat(rules, rules, rules)), p, cfg.cont)
    hS = Array(wb.s); hH = Array(wb.h); hL = Array(wb.L); cS = Array(cb.s); cH = Array(cb.h)
    [score_rule(rules[b], p, cfg, traces[b], probes[:, :, :, b], hS[:, :, :, b], hH[:, :, :, :, b],
                hL[:, :, :, b], cS[:, :, :, b], cH[:, :, :, :, b], cS[:, :, :, B+b], cH[:, :, :, :, B+b],
                cS[:, :, :, 2B+b], starts[b][4], t0) for b in 1:B]
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

"""Per-parameter scale of `random_rule` (θ = scale ⊙ z, z ~ N(0,1) is a random rule).
With `nb2 = 0` (default) the search space is the one-ring layout (2C²+8C, `Wnb2 = 0`);
`nb2 > 0` appends the second-shell block (3C²+8C)."""
function rule_scale(C; self=0.8, nb=0.8, inp=0.5, bias=0.1, read=0.5, nb2=0.0)
    sc = 1 / sqrt(C)
    v = vcat(fill(self * sc, C * C), fill(nb * sc, C * C), fill(inp, 4C), fill(bias, C), fill(read, 3C))
    nb2 > 0 ? vcat(v, fill(nb2 * sc, C * C)) : v
end
"""Block boundaries of θ (Wself, Wnb, Win, b, vdep, vero, vpig[, Wnb2]) for block crossover."""
function rule_blocks(C, np=2C * C + 8C)
    o = cumsum([0, C * C, C * C, 4C, C, C, C, C])
    bl = [o[i]+1:o[i+1] for i in 1:7]
    np > o[end] ? push!(bl, o[end]+1:np) : bl
end

"""Search coordinates of θ: the first `np` entries over `scale`; a nonzero entry beyond
`np` (a second shell the search does not include) is an error, not a silent truncation."""
function θ_to_z(θ, scale)
    np = length(scale)
    all(iszero, view(θ, np+1:length(θ))) ||
        throw(ArgumentError("θ has a nonzero Wnb2 block; search it with scales nb2 > 0"))
    Float64.(θ[1:np]) ./ scale
end

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
function evolve(C::Integer, p::Physics{T}, cfg; mode=:nsga, pop=48, gens=8, seed=1,
                sigma0=0.3, init=nothing, eval=evaluate, objectives=SELECT_OBJECTIVES,
                log=rec -> nothing, cache=Dict{Vector{Float64},Any}(), scales=(;), sampler=nothing,
                space=nothing) where {T}
    rng = MersenneTwister(seed)
    if space === nothing           # z = θ ./ scale in the flat rule layout
        scale = rule_scale(C; scales...); np = length(scale); blocks = rule_blocks(C, np)
        draw = () -> sampler === nothing ? randn(rng, np) : θ_to_z(sampler(rng), scale)
        decode = z -> scale .* z
        fix! = z -> z
    else                           # z = u ./ scale in a structured space (e.g. TiedSpace)
        init === nothing || throw(ArgumentError("init θ cannot be mapped into a structured space"))
        scale = space_scale(space); np = length(scale); blocks = space_blocks(space)
        draw = () -> space_prior(space, rng) ./ scale
        decode = z -> space_theta(space, scale .* z)
        fix! = z -> space_clamp!(space, z, scale)
    end
    archive = NamedTuple[]
    function run_batch!(zs, sigmas, parents, gen)
        rules = [Rule{C,T}(T.(decode(z))) for z in zs]
        space === nothing && (zs = [θ_to_z(vec(r), scale) for r in rules])    # z of the rule actually run
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
            z0[i] = θ_to_z(θ, scale)
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
            fix!(z)
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
        "physics" => Dict(string(f) => parse(Float64, string(getfield(p, f))) for f in fieldnames(Physics)),
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

# ============================================================================= two-shell manifold

"""
    TiedSpace(C; untied=false, self=0.8, inp=0.5, bias=0.1, read=0.5,
              nb1=(1.0, 16.0), beta=(0.28, 0.5), beta_clamp=(0.2, 0.6), dscale=0.3)

Search coordinates on the tied two-shell manifold:
`u = [Wself (C²) | P (C²) | log nb1 | β | Win (4C) | b | vdep | vero | vpig (C each) | D (C², untied only)]`,
mapped to a rule by `Wnb = nb1·PPᵀ/C` (positive semidefinite) and
`Wnb2 = −β·Wnb + nb1·D/C`. `D` starts at 0; with `untied = true` mutation may move it
(step `dscale`, in units of the entries of `PPᵀ`), so selection, not mutation, decides
whether to leave the manifold. Prior: `P ~ N(0,1)`, `nb1` log-uniform on `nb1`,
`β` uniform on `beta`; the other blocks as `tied_rule`/`random_rule`. β is clamped to
`beta_clamp` after every mutation. Mutation step scales: Wself `self/√C`, P 1, log nb1
0.3, β 0.05, the rest their prior scales.
"""
Base.@kwdef struct TiedSpace
    C::Int
    untied::Bool = false
    self::Float64 = 0.8
    inp::Float64 = 0.5
    bias::Float64 = 0.1
    read::Float64 = 0.5
    nb1::NTuple{2,Float64} = (1.0, 16.0)
    beta::NTuple{2,Float64} = (0.28, 0.5)
    beta_clamp::NTuple{2,Float64} = (0.2, 0.6)
    dscale::Float64 = 0.3
end
TiedSpace(C::Integer; kw...) = TiedSpace(; C=Int(C), kw...)

function space_scale(sp::TiedSpace)
    C = sp.C; sc = 1 / sqrt(C)
    v = vcat(fill(sp.self * sc, C * C), ones(C * C), [0.3, 0.05], fill(sp.inp, 4C), fill(sp.bias, C),
             fill(sp.read, 3C))
    sp.untied ? vcat(v, fill(sp.dscale, C * C)) : v
end

function space_blocks(sp::TiedSpace)
    C = sp.C
    o = cumsum([0, C * C, C * C, 2, 4C, C, C, C, C])
    bl = [o[i]+1:o[i+1] for i in 1:8]
    sp.untied ? push!(bl, o[end]+1:o[end]+C*C) : bl
end

function space_prior(sp::TiedSpace, rng)
    C = sp.C; sc = 1 / sqrt(C)
    lognb1 = log(sp.nb1[1]) + rand(rng) * (log(sp.nb1[2]) - log(sp.nb1[1]))
    β = sp.beta[1] + rand(rng) * (sp.beta[2] - sp.beta[1])
    u = vcat(randn(rng, C * C) .* (sp.self * sc), randn(rng, C * C), [lognb1, β],
             randn(rng, 4C) .* sp.inp, randn(rng, C) .* sp.bias, randn(rng, 3C) .* sp.read)
    sp.untied ? vcat(u, zeros(C * C)) : u
end

"""Index of β in `u`."""
beta_index(sp::TiedSpace) = 2sp.C * sp.C + 2

function space_clamp!(sp::TiedSpace, z, scale)
    i = beta_index(sp)
    z[i] = clamp(z[i] * scale[i], sp.beta_clamp...) / scale[i]
    z
end

"""Flat rule vector (`vec(rule)` layout, 3C²+8C) of manifold coordinates `u`."""
function space_theta(sp::TiedSpace, u)
    C = sp.C; o = 0
    take(n) = (v = u[o+1:o+n]; o += n; v)
    Wself = take(C * C); P = reshape(take(C * C), C, C)
    nb1 = exp(u[o+1]); β = clamp(u[o+2], sp.beta_clamp...); o += 2
    rest = take(4C + 4C)                     # Win | b | vdep | vero | vpig
    Wnb = (nb1 / C) .* (P * P')
    Wnb2 = -β .* Wnb
    sp.untied && (Wnb2 .+= (nb1 / C) .* reshape(take(C * C), C, C))
    vcat(Wself, vec(Wnb), rest, vec(Wnb2))
end

# ============================================================================= regions: life-shaped objectives

"""
    label_components(mask; min_size=8) -> (labels::Array{Int32,3}, sizes)

6-connected components of `mask`, periodic in x and y, walls in z (union–find).
Components smaller than `min_size` voxels are dropped (label 0). Labels are 1..K.
"""
function label_components(mask::AbstractArray{Bool,3}; min_size::Integer=8)
    Nx, Ny, Nz = size(mask)
    LI = LinearIndices(mask)
    parent = collect(1:length(mask))
    find(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    unite(a, b) = (ra = find(a); rb = find(b); ra != rb && (parent[max(ra, rb)] = min(ra, rb)))
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        mask[i, j, k] || continue
        q = LI[i, j, k]
        ii = i == Nx ? 1 : i + 1; mask[ii, j, k] && unite(q, LI[ii, j, k])
        jj = j == Ny ? 1 : j + 1; mask[i, jj, k] && unite(q, LI[i, jj, k])
        k < Nz && mask[i, j, k+1] && unite(q, LI[i, j, k+1])
    end
    cnt = Dict{Int,Int}()
    @inbounds for q in eachindex(mask)
        mask[q] && (r = find(q); cnt[r] = get(cnt, r, 0) + 1)
    end
    ids = Dict{Int,Int32}(); sizes = Int[]
    for r in sort!(collect(keys(cnt)))
        cnt[r] >= min_size || continue
        push!(sizes, cnt[r]); ids[r] = Int32(length(sizes))
    end
    labels = zeros(Int32, size(mask))
    @inbounds for q in eachindex(mask)
        mask[q] && (labels[q] = get(ids, find(q), Int32(0)))
    end
    labels, sizes
end

"""Solid above the midpoint of its 10th and 90th percentiles; empty if that spread < `contrast`."""
function region_mask(s::AbstractArray{<:Real,3}; contrast=0.02)
    q10, q90 = quantile(vec(s), (0.1, 0.9))
    q90 - q10 < contrast ? falses(size(s)) : s .> (q10 + q90) / 2
end

"""
    track(prev, next) -> (births, deaths, splits, merges, inherit)

Components of consecutive captures are linked when they share a voxel. A new component
with no predecessor is a birth, an old one with no successor a death; an old one with k ≥ 2
successors counts k−1 splits, a new one with k ≥ 2 predecessors k−1 merges.
`inherit[j]` is the old label that new component j continues (mutual largest overlap),
or 0 (a new identity).
"""
function track(prev::AbstractArray{Int32,3}, next::AbstractArray{Int32,3})
    Kp = maximum(prev; init=Int32(0)); Kn = maximum(next; init=Int32(0))
    ov = Dict{Tuple{Int32,Int32},Int}()
    @inbounds for q in eachindex(prev)
        a = prev[q]; b = next[q]
        (a > 0 && b > 0) && (ov[(a, b)] = get(ov, (a, b), 0) + 1)
    end
    nsucc = zeros(Int, Kp); npred = zeros(Int, Kn)
    bests = zeros(Int32, Kp); bestsv = zeros(Int, Kp); bestp = zeros(Int32, Kn); bestpv = zeros(Int, Kn)
    for ((a, b), v) in ov
        nsucc[a] += 1; npred[b] += 1
        v > bestsv[a] && (bestsv[a] = v; bests[a] = b)
        v > bestpv[b] && (bestpv[b] = v; bestp[b] = a)
    end
    inherit = [bestp[j] > 0 && bests[bestp[j]] == j ? bestp[j] : Int32(0) for j in 1:Kn]
    (births=count(==(0), npred), deaths=count(==(0), nsucc),
     splits=sum(max.(nsucc .- 1, 0); init=0), merges=sum(max.(npred .- 1, 0); init=0), inherit=inherit)
end

"""
    life_objectives(caps, cap_dt; min_size=8, cell=8, rate_target=0.1) -> NamedTuple

From a time series of solid fields `caps` (equally spaced by `cap_dt`), regions = 6-connected
components (periodic x,y) of `region_mask`, ≥ `min_size` voxels, tracked by overlap.
- `population` = `4x(1−x)`, x = clamp(mean count / (volume/cell³), 0, 1): one global
  lattice (x ≈ 0) and dust (x ≥ 1) both score 0, peak at half the target count.
- `turnover` = `4y(1−y)`, y = clamp(r / (2·rate_target), 0, 1), r = (births + deaths +
  splits + merges) per region per unit time: frozen and wholesale replacement score 0.
- `lifetime_spread` = normalized entropy of identity lifetimes over log2 bins of `cap_dt`.
- `localization` = `(1 − PR)·min(1, rms/0.05)` on the last field, PR the participation
  ratio of the lateral deviation energy `(s − layer mean)²`; a sinusoidal lattice has
  PR ≈ 0.51, blobs filling a fraction f have PR ≈ f.
"""
function life_objectives(caps::AbstractVector{<:AbstractArray{<:Real,3}}, cap_dt;
                         min_size=8, cell=8, rate_target=0.1)
    labs = [label_components(region_mask(c); min_size)[1] for c in caps]
    counts = [Int(maximum(l; init=Int32(0))) for l in labs]
    nb = nd = nsp = nm = 0
    first_seen = Dict{Int,Int}(); last_seen = Dict{Int,Int}()
    ident = [Int[] for _ in labs]; nextid = 0
    ident[1] = collect(1:counts[1]); nextid = counts[1]
    for i in 1:counts[1]; first_seen[i] = 1; last_seen[i] = 1; end
    for t in 2:length(labs)
        tr = track(labs[t-1], labs[t])
        nb += tr.births; nd += tr.deaths; nsp += tr.splits; nm += tr.merges
        ident[t] = map(tr.inherit) do a
            a > 0 ? ident[t-1][a] : (nextid += 1)
        end
        for id in ident[t]
            haskey(first_seen, id) || (first_seen[id] = t)
            last_seen[id] = t
        end
    end
    Nx, Ny, Nz = size(caps[end])
    target = Nx * Ny * Nz / cell^3
    μc = mean(counts)
    x = clamp(μc / target, 0.0, 1.0)
    span = (length(caps) - 1) * cap_dt
    events = nb + nd + nsp + nm
    r = (μc > 0 && span > 0) ? events / (μc * span) : 0.0
    y = clamp(r / (2rate_target), 0.0, 1.0)
    lt = [(last_seen[id] - first_seen[id] + 1) * cap_dt for id in keys(first_seen)]
    nbin = max(2, ceil(Int, log2(length(caps))) + 1)
    spread = if length(lt) >= 2
        h = zeros(nbin)
        for l in lt
            h[clamp(floor(Int, log2(l / cap_dt)) + 1, 1, nbin)] += 1
        end
        h ./= sum(h)
        -sum(v > 0 ? v * log(v) : 0.0 for v in h) / log(nbin)
    else
        0.0
    end
    s = Float64.(caps[end])
    d = similar(s)
    for k in 1:Nz
        sl = view(s, :, :, k); d[:, :, k] .= abs2.(sl .- mean(sl))
    end
    sd = sum(d)
    PR = sd > 1e-20 ? sd^2 / (length(d) * sum(abs2, d)) : 1.0
    rms = sqrt(sd / length(d))
    (population=4x * (1 - x), turnover=4y * (1 - y), lifetime_spread=spread,
     localization=(1 - PR) * min(1.0, rms / 0.05),
     n_regions=μc, n_regions_last=counts[end], births=nb, deaths=nd, splits=nsp, merges=nm,
     event_rate=r, participation=PR, n_identities=length(lt))
end

"Selection set for region-shaped behaviour."
const LIFE_SELECT = (:localization, :population, :turnover, :temporal, :recovery)
"All objectives recorded by `evaluate_life`."
const LIFE_OBJECTIVES = (:localization, :population, :turnover, :lifetime_spread, :temporal,
                         :memory_h, :recovery, :persistence, :structure)

"""
    LifeConfig(; dims, t_main, t_cont, cap_dt, probe_dt, track_from, seed, n0, h_scale, nprobe,
               reset_scale, erase_radius, dt_max, dt_min, dt_safety, min_size, cell, backend)

Measurement schedule in physical time. Each rule gets its own step
`dt = clamp(dt_safety·stable_dt, dt_min, dt_max)` rounded down to `dt_max·2^-k`
(floor `dt_min`), so fast rules cost more steps instead of being infeasible. Captures of
`s` every `cap_dt` (persistence over the whole run; regions tracked from
`track_from·t_main` on), activity probes every `probe_dt` over the last half.
"""
Base.@kwdef struct LifeConfig
    dims::NTuple{3,Int} = (32, 32, 24)
    t_main::Float64 = 100.0
    t_cont::Float64 = 25.0
    cap_dt::Float64 = 2.0
    probe_dt::Float64 = 0.5
    track_from::Float64 = 0.5
    seed::Int = 1
    n0::Float64 = 0.5
    h_scale::Float64 = 0.1
    nprobe::Int = 16
    reset_scale::Float64 = 0.01
    erase_radius::Float64 = 0.0
    dt_max::Float64 = 0.1
    dt_min::Float64 = 0.004
    dt_safety::Float64 = 0.8
    min_size::Int = 8
    cell::Int = 8
    backend::Any = KA.CPU()
end

with_dt(p::Physics{T}, dt) where {T} =
    Physics{T}(; (f => (f === :dt ? T(dt) : getfield(p, f)) for f in fieldnames(Physics))...)

"""Step for a rule: `dt_safety·stable_dt` rounded down to `dt_max·2^-k`, within [dt_min, dt_max]."""
function choose_dt(sdt, cfg::LifeConfig)
    isfinite(sdt) || return max(cfg.dt_max / 8, cfg.dt_min)
    d = clamp(cfg.dt_safety * sdt, cfg.dt_min, cfg.dt_max)
    b = cfg.dt_max
    while b > d && b / 2 >= cfg.dt_min
        b /= 2
    end
    b > d ? cfg.dt_min : b
end

"""
    evaluate_life(rules, p, cfg::LifeConfig) -> Vector{NamedTuple}

`LIFE_OBJECTIVES` plus diagnostics (`dt`, `stable_dt`, region counts and events, …). Rules
sharing a step run together in one `WorldBatch`.
"""
function evaluate_life(rules::AbstractVector{<:Rule{C,T}}, p::Physics{T}, cfg::LifeConfig) where {C,T}
    sdts = Vector{Float64}(undef, length(rules))
    Threads.@threads for i in eachindex(rules)
        sdts[i] = try
            Float64(Analysis.stable_dt(rules[i], p; M=cfg.n0))
        catch
            NaN
        end
    end
    dts = [choose_dt(s, cfg) for s in sdts]
    out = Vector{Any}(undef, length(rules))
    for d in unique(dts)
        idx = findall(==(d), dts)
        res = life_group(rules[idx], with_dt(p, d), cfg, sdts[idx])
        out[idx] = res
    end
    [o for o in out]
end

function life_group(rules::AbstractVector{<:Rule{C,T}}, p::Physics{T}, cfg::LifeConfig, sdts) where {C,T}
    t0 = time()
    be = cfg.backend; B = length(rules); nv = prod(cfg.dims); dt = Float64(p.dt)
    wb = WorldBatch(be, C, cfg.dims, B; T=T, seeds=fill(cfg.seed, B), n0=cfg.n0, h_scale=cfg.h_scale)
    θ = device_rules(be, rules)
    wb = light!(wb, θ, p)
    m0 = total_mass(wb)
    nmain = round(Int, cfg.t_main / dt)
    capevery = max(1, round(Int, cfg.cap_dt / dt))
    pevery = max(1, round(Int, cfg.probe_dt / dt))
    tp = nmain - nmain ÷ 2
    M = (nmain - tp) ÷ pevery
    prng = MersenneTwister(cfg.seed + 7919)
    idx = adapt(be, Int32.(rand(prng, 1:nv, cfg.nprobe)))
    buf = KA.zeros(be, T, C, cfg.nprobe, max(M, 1), B)
    pk = probe_batch_kernel!(be)
    caps = Array{T,4}[]; lbot = Vector{Float64}[]; captimes = Float64[]
    r = 0
    for st in 1:nmain
        wb = step!(wb, θ, p; sync=false)
        if st % capevery == 0
            KA.synchronize(be)
            push!(caps, Array(wb.s))
            push!(lbot, vec(mean(Float64.(Array(view(wb.L, :, :, 1, :))); dims=(1, 2))))
            push!(captimes, st * dt)
        end
        if st > tp && (st - tp) % pevery == 0 && r < M
            r += 1
            pk(buf, wb.h, idx, r, C, nv; ndrange=(C, cfg.nprobe, B))
        end
    end
    KA.synchronize(be)
    probes = Array{Float64}(Array(buf))
    starts = [perturbations(world(wb, b), rules[b], p, cfg) for b in 1:B]
    cb = WorldBatch(be, [w for k in 1:3 for w in (st[k] for st in starts)])
    cb = run!(cb, device_rules(be, vcat(rules, rules, rules)), p, round(Int, cfg.t_cont / dt))
    m1 = total_mass(wb)
    hH = Array(wb.h); cS = Array(cb.s); cH = Array(cb.h)
    track_idx = findall(>=(cfg.track_from * cfg.t_main - 1e-9), captimes)
    map(1:B) do b
        sb = [c[:, :, :, b] for c in caps]
        trace = [(mean_s=mean(c), std_s=std(c; corrected=false), light_bottom=lb[b])
                 for (c, lb) in zip(sb, lbot)]
        lf = life_objectives(sb[track_idx], cfg.cap_dt; min_size=cfg.min_size, cell=cfg.cell)
        s_end = Float64.(sb[end])
        sc = Float64.(cS[:, :, :, b]); sp = cS[:, :, :, B+b]; se = Float64.(cS[:, :, :, 2B+b])
        hc = cH[:, :, :, :, b]; hp = cH[:, :, :, :, B+b]
        fin = all(isfinite, s_end) && all(isfinite, sc) && all(isfinite, hc) && all(isfinite, hp)
        tm = temporal(probes[:, :, :, b])
        st = structure(s_end)
        obj = fin ? (localization=lf.localization, population=lf.population, turnover=lf.turnover,
                     lifetime_spread=lf.lifetime_spread, temporal=tm.score,
                     memory_h=corr0(hc, hp), recovery=recovery(sc, se, starts[b][4], s_end),
                     persistence=persistence(trace), structure=st.score) :
              NamedTuple{LIFE_OBJECTIVES}(ntuple(_ -> 0.0, length(LIFE_OBJECTIVES)))
        diag = (dt=dt, stable_dt=sdts[b], dt_ok=!(sdts[b] < dt), mean_s=trace[end].mean_s,
                std_s=trace[end].std_s, light_bottom=trace[end].light_bottom,
                mass_drift=(m1[b] - m0[b]) / nv, lambda=st.lambda, n_regions=lf.n_regions,
                n_regions_last=lf.n_regions_last, births=lf.births, deaths=lf.deaths, splits=lf.splits,
                merges=lf.merges, event_rate=lf.event_rate, participation=lf.participation,
                n_identities=lf.n_identities, t_fixed=tm.fixed, t_periodic=tm.periodic,
                h_autocorr=corr0(hH[:, :, :, :, b], hc), finite=fin, wall=time() - t0)
        merge(map(x -> isfinite(x) ? x : 0.0, obj), diag)
    end
end
