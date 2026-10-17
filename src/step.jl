# One complete step is two passes:
#   update: h, s, n, w, L → h2, s2, n2, w2, and the absorbance A (written into L)
#           per voxel, in this order: activity (h2 from old h, s, n, w, L; coupling
#           over the ring at distance 1 through Wnb and the axis shell at distance 2
#           through Wnb2, which reads only h and s there),
#           transport (diffusion of old n, w through old s), material (dep, ero, photo
#           on the transported n, w with old s, old L and new h2), absorbance of (s2, h2).
#   light:  A → L, one work-item per (i, j) column marching down from the lit top face.
# then the buffers swap. The update writes only the voxel it owns; L is read and
# overwritten at the owning voxel only, so the in-place absorbance is race-free.
#
# The per-voxel bodies work on linear offsets, so the same code serves a single World
# and a WorldBatch (B worlds along a trailing dimension, see batch.jl).

@inline softplus(x::T) where {T} = x > T(20) ? x : log1p(exp(x))
@inline sigmoid(x::T) where {T} = one(T) / (one(T) + exp(-x))

# Math mode of the update kernel: Val(false) is the precise reference (bitwise equal to
# the original four kernels); Val(true) uses the backends' fast tanh/exp/log, which on
# Apple GPUs halves the step and moves the 64³×50-step gate by < 1e-6
# (docs/performance.md). Default: precise on the CPU, fast on GPUs.
@inline act_tanh(::Val{false}, x) = tanh(x)
@inline act_tanh(::Val{true}, x) = Base.FastMath.tanh_fast(x)
@inline softplus(::Val{false}, x) = softplus(x)
@inline softplus(::Val{true}, x::T) where {T} =
    x > T(20) ? x : Base.FastMath.log_fast(one(T) + Base.FastMath.exp_fast(x))
@inline sigmoid(::Val{false}, x) = sigmoid(x)
@inline sigmoid(::Val{true}, x::T) where {T} = one(T) / (one(T) + Base.FastMath.exp_fast(-x))

fast_math_default(be) = !(be isa KA.CPU)

# Indexing. Every kernel launches over a *linear* range and works in Int32 offsets:
# KernelAbstractions' Cartesian `@index(Global, NTuple)` costs a 64-bit div/mod chain
# per work-item, which on Apple GPUs cut a plain 256³ copy from ~300 to ~50 GB/s
# (docs/performance.md). `q` is the 0-based linear voxel offset (batch-major last), the
# scalar fields are read at `q+1` and channel c of h at `C*q + c`. Callers guarantee
# `C*Nx*Ny*Nz*B < typemax(Int32)`.

@inline function loadh(h, ::Val{C}, q::Int32) where {C}
    o = Int32(C) * q
    SVector{C}(ntuple(c -> @inbounds(h[o + Int32(c)]), Val(C)))
end

# Cartesian form, for callers outside the step (e.g. render.jl).
@inline function loadh(h, ::Val{C}, i, j, k) where {C}
    SVector{C}(ntuple(c -> @inbounds(h[c, i, j, k]), Val(C)))
end

@inline conductance(p::Physics{T}, sx, sy) where {T} =
    clamp(p.g0 + p.g1 * (sx + sy) / 2, zero(T), one(T))

@inline mobility(p::Physics{T}, s) where {T} = p.D0 * (one(T) - s)^2

@inline function face_D(p::Physics{T}, sx, sy) where {T}
    Dx = mobility(p, sx); Dy = mobility(p, sy)
    d = Dx + Dy
    d > zero(T) ? 2 * Dx * Dy / d : zero(T)
end

# The rule inside the update kernel is read from a flat parameter vector θ in the
# `vec(rule)` layout (Wself | Wnb | Win | b | vdep | vero | vpig, column-major), at the
# point of use. Passing `Rule` by value made the compiler load all 2C²+8C numbers at
# kernel entry and keep them live through the neighbour loop; the register pressure
# capped the pipeline at 384 threads and cost more than all the memory traffic.
@inline θvec(θ, off::Int32, ::Val{C}, ::Type{T}) where {C,T} =
    SVector{C,T}(ntuple(c -> @inbounds(θ[off + Int32(c)]), Val(C)))

# `W * x` for a column-major R×K block of θ at `off`, column by column, with exactly
# the per-row `muladd` chain of StaticArrays' `SMatrix * SVector` (bitwise equal).
@generated function θmatvec(θ, off::Int32, ::Val{R}, x::SVector{K,T}) where {R,K,T}
    ex = Expr(:block)
    acc = [Symbol(:u, r) for r in 1:R]
    for j in 1:K, r in 1:R
        idx = (j - 1) * R + r
        push!(ex.args, j == 1 ?
            :($(acc[r]) = @inbounds(θ[off + Int32($idx)]) * x[1]) :
            :($(acc[r]) = muladd(@inbounds(θ[off + Int32($idx)]), x[$j], $(acc[r]))))
    end
    push!(ex.args, :(SVector{$R,$T}($(acc...))))
    ex
end

# θ[off+1:off+C] ⋅ x as a strict left-to-right fma chain. StaticArrays' `dot` is an
# `@simd` reduction whose association the CPU compiler may change between code paths,
# which broke bitwise equality between a World and the same world inside a batch.
@inline function θdot(θ, off::Int32, x::SVector{C,T}) where {C,T}
    acc = zero(T)
    for c in 1:C
        acc = fma(@inbounds(θ[off + Int32(c)]), x[c], acc)
    end
    acc
end

# i mod N for i ∈ [-2N, 3N) without a division.
@inline function wrap(i::Int32, N::Int32)
    while i < Int32(0); i += N; end
    while i >= N; i -= N; end
    i
end

# Second-shell voxel at offset y: conductance between the two end voxels, h only.
@inline function shell(h, s, p::Physics{T}, ::Val{C}, hx, sx, y::Int32) where {C,T}
    @inbounds conductance(p, sx, s[y + Int32(1)]) .* (loadh(h, Val(C), y) - hx)
end

# Neighbour at offset y: its contribution to the coupling m and to the fluxes fn, fw.
@inline function face(h, s, n, w, p::Physics{T}, ::Val{C}, hx, sx, nx, wx, y::Int32) where {C,T}
    @inbounds begin
        sy = s[y + Int32(1)]
        dm = conductance(p, sx, sy) .* (loadh(h, Val(C), y) - hx)
        D = face_D(p, sx, sy)
        dn = D * (n[y + Int32(1)] - nx)
        dw = D * (w[y + Int32(1)] - wx)
    end
    dm, dn, dw
end

"""
Split a 0-based linear offset into (b, i, j, k), all 0-based, for worlds of
`Nx×Ny×Nz` voxels stacked along a trailing batch dimension.
"""
@inline function voxel_coords(q::Int32, Nx::Int32, Ny::Int32, Nz::Int32)
    NxNy = Nx * Ny
    V = NxNy * Nz
    b = q ÷ V;  r = q - b * V
    k = r ÷ NxNy; r -= k * NxNy
    j = r ÷ Nx;  i = r - j * Nx
    b, i, j, k
end

# The whole per-voxel step for voxel offset q with in-world coordinates (i, j, k).
# θ[θo + 1 : θo + 3C²+8C] is the rule.
@inline function update_voxel!(h2, s2, n2, w2, L, h, s, n, w, θ, θo::Int32, ::Val{C}, fm::Val, p::Physics{T},
                               Nx::Int32, Ny::Int32, Nz::Int32, q::Int32,
                               i::Int32, j::Int32, k::Int32) where {C,T}
    o = Int32(1)
    CC = Int32(C * C)
    NxNy = Nx * Ny
    @inbounds begin
        hx = loadh(h, Val(C), q)
        sx = s[q + o]; nx = n[q + o]; wx = w[q + o]; Lx = L[q + o]
        m = zero(SVector{C,T}); fn = zero(T); fw = zero(T)
        # x neighbours (periodic), then y (periodic), then z (slab: no-flux walls)
        dm, dn, dw = face(h, s, n, w, p, Val(C), hx, sx, nx, wx, i == Nx - o ? q - i : q + o)
        m += dm; fn += dn; fw += dw
        dm, dn, dw = face(h, s, n, w, p, Val(C), hx, sx, nx, wx, i == 0 ? q + Nx - o : q - o)
        m += dm; fn += dn; fw += dw
        dm, dn, dw = face(h, s, n, w, p, Val(C), hx, sx, nx, wx, j == Ny - o ? q - j * Nx : q + Nx)
        m += dm; fn += dn; fw += dw
        dm, dn, dw = face(h, s, n, w, p, Val(C), hx, sx, nx, wx, j == 0 ? q + (Ny - o) * Nx : q - Nx)
        m += dm; fn += dn; fw += dw
        if k < Nz - o
            dm, dn, dw = face(h, s, n, w, p, Val(C), hx, sx, nx, wx, q + NxNy)
            m += dm; fn += dn; fw += dw
        end
        if k > 0
            dm, dn, dw = face(h, s, n, w, p, Val(C), hx, sx, nx, wx, q - NxNy)
            m += dm; fn += dn; fw += dw
        end
        # second shell: the axis voxels at distance 2 (x, y periodic for any N ≥ 1)
        two = Int32(2)
        m2 = shell(h, s, p, Val(C), hx, sx, q + (wrap(i + two, Nx) - i))
        m2 += shell(h, s, p, Val(C), hx, sx, q + (wrap(i - two, Nx) - i))
        m2 += shell(h, s, p, Val(C), hx, sx, q + (wrap(j + two, Ny) - j) * Nx)
        m2 += shell(h, s, p, Val(C), hx, sx, q + (wrap(j - two, Ny) - j) * Nx)
        if k + two < Nz
            m2 += shell(h, s, p, Val(C), hx, sx, q + two * NxNy)
        end
        if k >= two
            m2 += shell(h, s, p, Val(C), hx, sx, q - two * NxNy)
        end
        # activity
        inp = SVector{4,T}(nx, sx, wx, Lx)
        u = θmatvec(θ, θo, Val(C), hx) + θmatvec(θ, θo + CC, Val(C), m) +
            θmatvec(θ, θo + Int32(2) * CC, Val(C), inp) + θvec(θ, θo + Int32(2) * CC + Int32(4C), Val(C), T)
        u += θmatvec(θ, θo + Int32(2) * CC + Int32(8C), Val(C), m2)   # Wnb2 (added last: Wnb2 = 0 is exact)
        hn = hx + (p.dt / p.tau_h) .* (act_tanh.(fm, u) - hx)
        # transport
        nt = nx + p.dt * fn
        wt = wx + p.dt * fw
        # material, on the transported n, w and the new activity
        adep = softplus(fm, θdot(θ, θo + Int32(2) * CC + Int32(5C), hn))
        aero = softplus(fm, θdot(θ, θo + Int32(2) * CC + Int32(6C), hn))
        dep = min(p.k_dep * adep * nt * (one(T) - sx) * p.dt, nt, one(T) - sx)
        ero = min(p.k_ero * aero * sx * p.dt, sx)
        photo = min(p.k_photo * Lx * wt * p.dt, wt)
        sn = sx + dep - ero
        oh = Int32(C) * q
        ntuple(c -> (h2[oh + Int32(c)] = hn[c]; nothing), Val(C))
        s2[q + o] = sn
        n2[q + o] = nt - dep + photo
        w2[q + o] = wt + ero - photo
        L[q + o] = p.alpha_s * sn + p.alpha_p * sigmoid(fm, θdot(θ, θo + Int32(2) * CC + Int32(7C), hn))
    end
end

# March column `col` (0-based over Nx·Ny·B) down from the lit face, turning
# absorbance into light in place.
@inline function light_column!(L, p::Physics{T}, Nx::Int32, Ny::Int32, Nz::Int32, col::Int32) where {T}
    NxNy = Nx * Ny
    b = col ÷ NxNy
    base = col + b * (NxNy * Nz - NxNy) + Int32(1)
    @inbounds begin
        τ = zero(T)
        for k in Nz-Int32(1):Int32(-1):Int32(0)
            x = base + k * NxNy
            a = L[x]
            L[x] = p.L_ambient + p.L0 * exp(-(τ + a / 2))
            τ += a
        end
    end
end

# The update over B ≥ 1 worlds of V = Nx·Ny·Nz voxels stacked along a trailing
# dimension; θ holds Bθ ∈ {1, B} rules of P = 3C²+8C numbers back to back (Bθ = 1:
# all worlds share one rule). Each workgroup first
# copies the (at most two) rules its voxels use into threadgroup memory. The launch
# range is padded to whole workgroups (`unsafe_indices`) so every lane reaches the
# barrier; padded lanes do no work.
@kernel unsafe_indices=true function update_kernel!(h2, s2, n2, w2, L, @Const(h), @Const(s), @Const(n),
        @Const(w), @Const(θ), ::Val{C}, fm::Val, p::Physics{T}, Nx::Int32, Ny::Int32, Nz::Int32,
        B::Int32, Bθ::Int32) where {C,T}
    lθ = @localmem T (2 * (3C * C + 8C),)
    # (values are recomputed after the barrier: KA's CPU backend splits the body there)
    P = Int32(3C * C + 8C)
    q = (@index(Global, Linear) - 1) % Int32
    li = (@index(Local, Linear) - 1) % Int32
    V = Nx * Ny * Nz
    gs = Int32(@groupsize()[1])
    b0 = min((q - li) ÷ V, B - Int32(1))               # world of the group's first voxel
    nθ = min((q - li + gs - Int32(1)) ÷ V, B - Int32(1)) > b0 ? Int32(2) * P : P
    t = li
    while t < nθ
        hi = t >= P
        bt = Bθ == Int32(1) ? Int32(0) : min(b0 + (hi ? Int32(1) : Int32(0)), B - Int32(1))
        @inbounds lθ[t + Int32(1)] = θ[bt * P + (hi ? t - P : t) + Int32(1)]
        t += gs
    end
    @synchronize
    P = Int32(3C * C + 8C)
    q = (@index(Global, Linear) - 1) % Int32
    li = (@index(Local, Linear) - 1) % Int32
    b0 = min((q - li) ÷ (Nx * Ny * Nz), B - Int32(1))
    if q < Nx * Ny * Nz * B
        b, i, j, k = voxel_coords(q, Nx, Ny, Nz)
        update_voxel!(h2, s2, n2, w2, L, h, s, n, w, lθ, (b - b0) * P, Val(C), fm, p,
                      Nx, Ny, Nz, q, i, j, k)
    end
end

@kernel function light_kernel!(L, p::Physics{T}, Nx::Int32, Ny::Int32, Nz::Int32) where {T}
    col = (@index(Global, Linear) - 1) % Int32
    light_column!(L, p, Nx, Ny, Nz, col)
end

# Absorbance of the current state for B worlds (θ with Bθ ∈ {1, B} columns).
@kernel function absorbance_kernel!(L, @Const(s), @Const(h), @Const(θ), ::Val{C}, p::Physics{T},
                                    V::Int32, Bθ::Int32) where {C,T}
    q = (@index(Global, Linear) - 1) % Int32
    b = Bθ == Int32(1) ? Int32(0) : q ÷ V
    off = b * Int32(3C * C + 8C) + Int32(2C * C + 7C)
    @inbounds L[q + Int32(1)] = p.alpha_s * s[q + Int32(1)] +
        p.alpha_p * sigmoid(θdot(θ, off, loadh(h, Val(C), q)))
end

"""Workgroup sizes of the update and light passes (see docs/performance.md)."""
update_workgroup(be) = 32
light_workgroup(be) = 64

# Device copies of rules, cached per backend so that `step!(wd, rule, p)` uploads θ
# once per distinct rule. A kernel launched with `sync=false` may still be reading an
# entry, so the cache is only ever cleared after synchronizing the backend (dropping
# the last reference to an in-flight array lets the GC free it under the kernel).
const _θ_cache = Dict{Any,Any}()
const _θ_lock = ReentrantLock()
const _θ_cache_limit = 4096

function cached_upload(make, be, key)
    lock(_θ_lock) do
        k = (typeof(be), key)
        hit = get(_θ_cache, k, nothing)
        hit !== nothing && return hit
        if length(_θ_cache) >= _θ_cache_limit
            KA.synchronize(be)
            empty!(_θ_cache)
        end
        _θ_cache[k] = make()
    end
end

"""Device-resident `vec(rule)` on backend `be` (cached per rule)."""
device_theta(be, rule::Rule{C,T}) where {C,T} = cached_upload(be, rule) do
    θ = KA.allocate(be, T, nparams(rule))
    copyto!(θ, collect(vec(rule)))
    θ
end

function launch_update!(be, h2, s2, n2, w2, L, h, s, n, w, θ, ::Val{C}, fm::Val, p,
                        X::Int32, Y::Int32, Z::Int32, B::Int32, Bθ::Int32=Int32(1)) where {C}
    wg = update_workgroup(be)
    total = Int(X) * Int(Y) * Int(Z) * Int(B)
    # Flat vectors: World and WorldBatch then share one kernel specialization.
    v = vec
    update_kernel!(be, wg)(v(h2), v(s2), v(n2), v(w2), v(L), v(h), v(s), v(n), v(w), v(θ), Val(C), fm, p,
                           X, Y, Z, B, Bθ; ndrange=cld(total, wg) * wg)
end

function check_int32(C, dims, B=1)
    C * prod(dims) * B < typemax(Int32) ||
        throw(ArgumentError("C·Nx·Ny·Nz·B = $(C*prod(dims)*B) exceeds Int32 offsets"))
end

"""
    step!(wd::World, rule::Rule, p::Physics; fast=(backend is a GPU)) -> World

Advance one `p.dt`. Returns the world with swapped buffers; the argument is
stale afterwards. Synchronizes the backend unless `sync=false` (launches queue in
order, so consecutive unsynchronized steps are safe; a Metal sync costs ~1 ms, which
dominates grids below ~128³). `fast=false` gives the precise math
that is bitwise equal to the original four-kernel step; `fast=true` (the GPU default)
uses fast tanh/exp/log.
"""
function step!(wd::World{C,T}, rule::Rule{C,T}, p::Physics{T};
               fast::Bool=fast_math_default(backend(wd)), sync::Bool=true) where {C,T}
    be = backend(wd)
    Nx, Ny, Nz = size(wd)
    check_int32(C, (Nx, Ny, Nz))
    X, Y, Z = Int32(Nx), Int32(Ny), Int32(Nz)
    θ = device_theta(be, rule)
    launch_update!(be, wd.h2, wd.s2, wd.n2, wd.w2, wd.L, wd.h, wd.s, wd.n, wd.w, θ, Val(C),
                   Val(fast), p, X, Y, Z, Int32(1))
    light_kernel!(be, light_workgroup(be))(vec(wd.L), p, X, Y, Z; ndrange=Nx * Ny)
    sync && KA.synchronize(be)
    World{C,T,typeof(wd.h),typeof(wd.s)}(wd.h2, wd.s2, wd.n2, wd.w2, wd.L,
        wd.h, wd.s, wd.n, wd.w, wd.t + p.dt, wd.step + 1)
end

# Absorbance then the column march, for B worlds; flat vectors as in `launch_update!`.
function launch_light!(be, L, s, h, θ, ::Val{C}, p, X::Int32, Y::Int32, Z::Int32, B::Int32,
                       Bθ::Int32) where {C}
    V = X * Y * Z
    absorbance_kernel!(be, update_workgroup(be))(vec(L), vec(s), vec(h), vec(θ), Val(C), p, V, Bθ;
                                                 ndrange=Int(V) * Int(B))
    light_kernel!(be, light_workgroup(be))(vec(L), p, X, Y, Z; ndrange=Int(X) * Int(Y) * Int(B))
end

"""Recompute `L` from the current `s` and `h` (used after construction or edits)."""
function light!(wd::World{C,T}, rule::Rule{C,T}, p::Physics{T}) where {C,T}
    be = backend(wd)
    Nx, Ny, Nz = size(wd)
    check_int32(C, (Nx, Ny, Nz))
    launch_light!(be, wd.L, wd.s, wd.h, device_theta(be, rule), Val(C), p,
                  Int32(Nx), Int32(Ny), Int32(Nz), Int32(1), Int32(1))
    KA.synchronize(be)
    wd
end

"""Run `nsteps` steps, synchronizing once at the end. Returns the final world."""
function run!(wd::World, rule::Rule, p::Physics, nsteps::Integer; kw...)
    for _ in 1:nsteps
        wd = step!(wd, rule, p; kw..., sync=false)
    end
    KA.synchronize(backend(wd))
    wd
end
