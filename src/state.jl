"""
    World{C,T,A4,A3}

All state of one specimen. `h` is channel-first `(C, Nx, Ny, Nz)`; the scalar
fields are `(Nx, Ny, Nz)`. `L` is derived (recomputed by `step!`). The `*2`
fields are the write buffers of the double-buffered step; `step!` returns a
`World` with the roles swapped, so always use its return value.

Periodic in x and y. Slab in z: light enters at `k = Nz`, transport and coupling
see no-flux walls at `k = 1` and `k = Nz`.
"""
struct World{C,T,A4<:AbstractArray{T,4},A3<:AbstractArray{T,3}}
    h::A4
    s::A3
    n::A3
    w::A3
    L::A3
    h2::A4
    s2::A3
    n2::A3
    w2::A3
    t::T
    step::Int
end

channels(::World{C}) where {C} = C
Base.size(wd::World) = size(wd.s)
Base.eltype(::World{C,T}) where {C,T} = T
backend(wd::World) = KA.get_backend(wd.s)

"""
    World(be, rule_or_C, (Nx,Ny,Nz); T=Float32, n0=1.0, h_scale=0.1, s0=0.0, seed=0)

Allocate on backend `be` (`CPU()`, `MetalBackend()`, `ROCBackend()`, …). Initial
nutrient is uniform `n0`, activity is `h_scale·randn`, solid is `s0·rand`,
waste is zero, so total mass per voxel is `n0 + mean(s)`. The initial state is
generated on the CPU with `seed` and copied, so it is backend-independent.
"""
function World(be, C::Integer, dims::NTuple{3,Integer}; T::Type=Float32,
               n0=1.0, h_scale=0.1, s0=0.0, seed::Integer=0)
    rng = MersenneTwister(seed)
    Nx, Ny, Nz = dims
    h0 = T.(h_scale .* randn(rng, C, Nx, Ny, Nz))
    s0a = T.(s0 .* rand(rng, Nx, Ny, Nz))
    n0a = fill(T(n0), Nx, Ny, Nz)
    w0a = zeros(T, Nx, Ny, Nz)
    L0a = zeros(T, Nx, Ny, Nz)
    up(a) = adapt(be, a)
    World{C,T,typeof(up(h0)),typeof(up(s0a))}(up(h0), up(s0a), up(n0a), up(w0a), up(L0a),
        up(similar(h0)), up(similar(s0a)), up(similar(n0a)), up(similar(w0a)), zero(T), 0)
end
World(be, r::Rule{C,T}, dims::NTuple{3,Integer}; kw...) where {C,T} = World(be, C, dims; T=T, kw...)

"""Copy a world to another backend (e.g. `adapt(CPU(), w)` to inspect GPU state)."""
function Adapt.adapt_structure(to, wd::World{C,T}) where {C,T}
    f(a) = adapt(to, a)
    h = f(wd.h); s = f(wd.s)
    World{C,T,typeof(h),typeof(s)}(h, s, f(wd.n), f(wd.w), f(wd.L),
        f(wd.h2), f(wd.s2), f(wd.n2), f(wd.w2), wd.t, wd.step)
end

"""Exact deep copy on the same backend."""
Base.copy(wd::World{C,T,A4,A3}) where {C,T,A4,A3} =
    World{C,T,A4,A3}(copy(wd.h), copy(wd.s), copy(wd.n), copy(wd.w), copy(wd.L),
                     copy(wd.h2), copy(wd.s2), copy(wd.n2), copy(wd.w2), wd.t, wd.step)
