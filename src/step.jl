# One complete step is two passes:
#   update: h, s, n, w, L → h2, s2, n2, w2, and the absorbance A (written into L)
#           per voxel, in this order: activity (h2 from old h, s, n, w, L),
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

"""Light absorbed by one cell: solid plus pigment read out of the activity (same
expression as the last line of `update_voxel!`)."""
@inline absorbance(rule::Rule{C,T}, p::Physics{T}, sx, hx) where {C,T} =
    p.alpha_s * sx + p.alpha_p * sigmoid(dot(rule.vpig, hx))

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
# θ[θo + 1 : θo + 2C²+8C] is the rule.
@inline function update_voxel!(h2, s2, n2, w2, L, h, s, n, w, θ, θo::Int32, ::Val{C}, p::Physics{T},
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
        # activity
        inp = SVector{4,T}(nx, sx, wx, Lx)
        u = θmatvec(θ, θo, Val(C), hx) + θmatvec(θ, θo + CC, Val(C), m) +
            θmatvec(θ, θo + Int32(2) * CC, Val(C), inp) + θvec(θ, θo + Int32(2) * CC + Int32(4C), Val(C), T)
        hn = hx + (p.dt / p.tau_h) .* (tanh.(u) - hx)
        # transport
        nt = nx + p.dt * fn
        wt = wx + p.dt * fw
        # material, on the transported n, w and the new activity
        adep = softplus(dot(θvec(θ, θo + Int32(2) * CC + Int32(5C), Val(C), T), hn))
        aero = softplus(dot(θvec(θ, θo + Int32(2) * CC + Int32(6C), Val(C), T), hn))
        dep = min(p.k_dep * adep * nt * (one(T) - sx) * p.dt, nt, one(T) - sx)
        ero = min(p.k_ero * aero * sx * p.dt, sx)
        photo = min(p.k_photo * Lx * wt * p.dt, wt)
        sn = sx + dep - ero
        oh = Int32(C) * q
        ntuple(c -> (h2[oh + Int32(c)] = hn[c]; nothing), Val(C))
        s2[q + o] = sn
        n2[q + o] = nt - dep + photo
        w2[q + o] = wt + ero - photo
        L[q + o] = p.alpha_s * sn + p.alpha_p * sigmoid(dot(θvec(θ, θo + Int32(2) * CC + Int32(7C), Val(C), T), hn))
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
            L[x] = p.L0 * exp(-(τ + a / 2))
            τ += a
        end
    end
end

@kernel function update_kernel!(h2, s2, n2, w2, L, @Const(h), @Const(s), @Const(n), @Const(w),
                                @Const(θ), ::Val{C}, p::Physics{T}, Nx::Int32, Ny::Int32, Nz::Int32) where {C,T}
    q = (@index(Global, Linear) - 1) % Int32
    _, i, j, k = voxel_coords(q, Nx, Ny, Nz)
    update_voxel!(h2, s2, n2, w2, L, h, s, n, w, θ, Int32(0), Val(C), p, Nx, Ny, Nz, q, i, j, k)
end

@kernel function light_kernel!(L, p::Physics{T}, Nx::Int32, Ny::Int32, Nz::Int32) where {T}
    col = (@index(Global, Linear) - 1) % Int32
    light_column!(L, p, Nx, Ny, Nz, col)
end

@kernel function absorbance_kernel!(L, @Const(s), @Const(h), rule::Rule{C,T}, p::Physics{T}) where {C,T}
    q = (@index(Global, Linear) - 1) % Int32
    @inbounds L[q + Int32(1)] = absorbance(rule, p, s[q + Int32(1)], loadh(h, Val(C), q))
end

"""Workgroup sizes of the update and light passes (see docs/performance.md)."""
update_workgroup(be) = 256
light_workgroup(be) = 64

# One-entry cache of the device copy of the last rule stepped on each backend type, so
# `step!(wd, rule, p)` in a loop uploads θ once. Entries are replaced, never mutated,
# so a task holding an old array is unaffected by another task's miss.
const _θ_cache = Dict{DataType,Any}()
const _θ_lock = ReentrantLock()

"""Device-resident `vec(rule)` on backend `be` (cached for the last rule)."""
function device_theta(be, rule::Rule{C,T}) where {C,T}
    lock(_θ_lock) do
        hit = get(_θ_cache, typeof(be), nothing)
        hit !== nothing && hit[1] === rule && return hit[2]
        θ = KA.allocate(be, T, nparams(rule))
        copyto!(θ, collect(vec(rule)))
        _θ_cache[typeof(be)] = (rule, θ)
        θ
    end
end

function check_int32(C, dims, B=1)
    C * prod(dims) * B < typemax(Int32) ||
        throw(ArgumentError("C·Nx·Ny·Nz·B = $(C*prod(dims)*B) exceeds Int32 offsets"))
end

"""
    step!(wd::World, rule::Rule, p::Physics) -> World

Advance one `p.dt`. Returns the world with swapped buffers; the argument is
stale afterwards. Synchronizes the backend.
"""
function step!(wd::World{C,T}, rule::Rule{C,T}, p::Physics{T}) where {C,T}
    be = backend(wd)
    Nx, Ny, Nz = size(wd)
    check_int32(C, (Nx, Ny, Nz))
    X, Y, Z = Int32(Nx), Int32(Ny), Int32(Nz)
    θ = device_theta(be, rule)
    update_kernel!(be, update_workgroup(be))(wd.h2, wd.s2, wd.n2, wd.w2, wd.L, wd.h, wd.s, wd.n, wd.w,
                                             θ, Val(C), p, X, Y, Z; ndrange=Nx * Ny * Nz)
    light_kernel!(be, light_workgroup(be))(wd.L, p, X, Y, Z; ndrange=Nx * Ny)
    KA.synchronize(be)
    World{C,T,typeof(wd.h),typeof(wd.s)}(wd.h2, wd.s2, wd.n2, wd.w2, wd.L,
        wd.h, wd.s, wd.n, wd.w, wd.t + p.dt, wd.step + 1)
end

"""Recompute `L` from the current `s` and `h` (used after construction or edits)."""
function light!(wd::World{C,T}, rule::Rule{C,T}, p::Physics{T}) where {C,T}
    be = backend(wd)
    Nx, Ny, Nz = size(wd)
    check_int32(C, (Nx, Ny, Nz))
    absorbance_kernel!(be, update_workgroup(be))(wd.L, wd.s, wd.h, rule, p; ndrange=Nx * Ny * Nz)
    light_kernel!(be, light_workgroup(be))(wd.L, p, Int32(Nx), Int32(Ny), Int32(Nz); ndrange=Nx * Ny)
    KA.synchronize(be)
    wd
end

"""Run `nsteps` steps. Returns the final world."""
function run!(wd::World, rule::Rule, p::Physics, nsteps::Integer)
    for _ in 1:nsteps
        wd = step!(wd, rule, p)
    end
    wd
end
