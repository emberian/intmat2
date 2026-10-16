# The four kernels. One complete step is:
#   activity:  h, s, n, w, L  → h2
#   transport: n, w, s        → n2, w2        (diffusion only)
#   material:  h2, s, n2, w2, L → s2, n2, w2  (in place on the new buffers)
#   light:     s2, h2         → L
# then the buffers swap. Every kernel writes only the voxel it owns.

@inline softplus(x::T) where {T} = x > T(20) ? x : log1p(exp(x))
@inline sigmoid(x::T) where {T} = one(T) / (one(T) + exp(-x))

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

@kernel function activity_kernel!(h2, @Const(h), @Const(s), @Const(n), @Const(w), @Const(L),
                                  rule::Rule{C,T}, p::Physics{T}, Nx, Ny, Nz) where {C,T}
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        hx = loadh(h, Val(C), i, j, k)
        sx = s[i, j, k]
        m = zero(SVector{C,T})
        # x neighbours (periodic)
        ii = i == Nx ? 1 : i + 1
        m += conductance(p, sx, s[ii, j, k]) .* (loadh(h, Val(C), ii, j, k) - hx)
        ii = i == 1 ? Nx : i - 1
        m += conductance(p, sx, s[ii, j, k]) .* (loadh(h, Val(C), ii, j, k) - hx)
        # y neighbours (periodic)
        jj = j == Ny ? 1 : j + 1
        m += conductance(p, sx, s[i, jj, k]) .* (loadh(h, Val(C), i, jj, k) - hx)
        jj = j == 1 ? Ny : j - 1
        m += conductance(p, sx, s[i, jj, k]) .* (loadh(h, Val(C), i, jj, k) - hx)
        # z neighbours (slab: no coupling through the walls)
        if k < Nz
            m += conductance(p, sx, s[i, j, k+1]) .* (loadh(h, Val(C), i, j, k + 1) - hx)
        end
        if k > 1
            m += conductance(p, sx, s[i, j, k-1]) .* (loadh(h, Val(C), i, j, k - 1) - hx)
        end
        inp = SVector{4,T}(n[i, j, k], sx, w[i, j, k], L[i, j, k])
        u = rule.Wself * hx + rule.Wnb * m + rule.Win * inp + rule.b
        hn = hx + (p.dt / p.tau_h) .* (tanh.(u) - hx)
        for c in 1:C
            h2[c, i, j, k] = hn[c]
        end
    end
end

@kernel function transport_kernel!(n2, w2, @Const(n), @Const(w), @Const(s),
                                   p::Physics{T}, Nx, Ny, Nz) where {T}
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        sx = s[i, j, k]; nx = n[i, j, k]; wx = w[i, j, k]
        fn = zero(T); fw = zero(T)
        ii = i == Nx ? 1 : i + 1
        D = face_D(p, sx, s[ii, j, k]); fn += D * (n[ii, j, k] - nx); fw += D * (w[ii, j, k] - wx)
        ii = i == 1 ? Nx : i - 1
        D = face_D(p, sx, s[ii, j, k]); fn += D * (n[ii, j, k] - nx); fw += D * (w[ii, j, k] - wx)
        jj = j == Ny ? 1 : j + 1
        D = face_D(p, sx, s[i, jj, k]); fn += D * (n[i, jj, k] - nx); fw += D * (w[i, jj, k] - wx)
        jj = j == 1 ? Ny : j - 1
        D = face_D(p, sx, s[i, jj, k]); fn += D * (n[i, jj, k] - nx); fw += D * (w[i, jj, k] - wx)
        if k < Nz
            D = face_D(p, sx, s[i, j, k+1]); fn += D * (n[i, j, k+1] - nx); fw += D * (w[i, j, k+1] - wx)
        end
        if k > 1
            D = face_D(p, sx, s[i, j, k-1]); fn += D * (n[i, j, k-1] - nx); fw += D * (w[i, j, k-1] - wx)
        end
        n2[i, j, k] = nx + p.dt * fn
        w2[i, j, k] = wx + p.dt * fw
    end
end

@kernel function material_kernel!(s2, n2, w2, @Const(h2), @Const(s), @Const(L),
                                  rule::Rule{C,T}, p::Physics{T}) where {C,T}
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        hx = loadh(h2, Val(C), i, j, k)
        sx = s[i, j, k]; nx = n2[i, j, k]; wx = w2[i, j, k]; Lx = L[i, j, k]
        adep = softplus(dot(rule.vdep, hx))
        aero = softplus(dot(rule.vero, hx))
        dep = min(p.k_dep * adep * nx * (one(T) - sx) * p.dt, nx, one(T) - sx)
        ero = min(p.k_ero * aero * sx * p.dt, sx)
        photo = min(p.k_photo * Lx * wx * p.dt, wx)
        s2[i, j, k] = sx + dep - ero
        n2[i, j, k] = nx - dep + photo
        w2[i, j, k] = wx + ero - photo
    end
end

# One work-item per (i, j) column, marching down from the lit top face.
@kernel function light_kernel!(L, @Const(s), @Const(h), rule::Rule{C,T}, p::Physics{T}, Nz) where {C,T}
    i, j = @index(Global, NTuple)
    @inbounds begin
        τ = zero(T)
        for k in Nz:-1:1
            hx = loadh(h, Val(C), i, j, k)
            a = p.alpha_s * s[i, j, k] + p.alpha_p * sigmoid(dot(rule.vpig, hx))
            L[i, j, k] = p.L0 * exp(-(τ + a / 2))
            τ += a
        end
    end
end

"""
    step!(wd::World, rule::Rule, p::Physics) -> World

Advance one `p.dt`. Returns the world with swapped buffers; the argument is
stale afterwards. Synchronizes the backend.
"""
function step!(wd::World{C,T}, rule::Rule{C,T}, p::Physics{T}) where {C,T}
    be = backend(wd)
    Nx, Ny, Nz = size(wd)
    activity_kernel!(be)(wd.h2, wd.h, wd.s, wd.n, wd.w, wd.L, rule, p, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    transport_kernel!(be)(wd.n2, wd.w2, wd.n, wd.w, wd.s, p, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    material_kernel!(be)(wd.s2, wd.n2, wd.w2, wd.h2, wd.s, wd.L, rule, p; ndrange=(Nx, Ny, Nz))
    light_kernel!(be)(wd.L, wd.s2, wd.h2, rule, p, Nz; ndrange=(Nx, Ny))
    KA.synchronize(be)
    World{C,T,typeof(wd.h),typeof(wd.s)}(wd.h2, wd.s2, wd.n2, wd.w2, wd.L,
        wd.h, wd.s, wd.n, wd.w, wd.t + p.dt, wd.step + 1)
end

"""Recompute `L` from the current `s` and `h` (used after construction or edits)."""
function light!(wd::World{C,T}, rule::Rule{C,T}, p::Physics{T}) where {C,T}
    be = backend(wd)
    Nx, Ny, Nz = size(wd)
    light_kernel!(be)(wd.L, wd.s, wd.h, rule, p, Nz; ndrange=(Nx, Ny))
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
