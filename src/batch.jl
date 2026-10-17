# Independent-world batching: B worlds of the same size in one allocation, stepped by
# one launch of the same update kernel as `World` (B along a trailing dimension).
#
# API (exported: WorldBatch, world, device_rules)
#
#   WorldBatch(be, C, dims::NTuple{3}, B; T=Float32, seeds=1:B, n0=1.0, h_scale=0.1, s0=0.0)
#       World b starts bitwise identical to `World(be, C, dims; T, n0, h_scale, s0, seed=seeds[b])`.
#   WorldBatch(be, worlds::AbstractVector{<:World})          # stack existing worlds (copies)
#   WorldBatch(be, rule::Rule{C,T}, dims, B; kw...)          # C and T from a rule
#
#   step!(wb, rules, p::Physics; fast=(GPU), sync=true) -> WorldBatch   (argument is stale)
#   light!(wb, rules, p) -> wb
#   run!(wb, rules, p, nsteps; fast) -> WorldBatch                     (one sync at the end)
#       `rules` is one `Rule{C,T}` shared by all worlds, a `Vector{Rule{C,T}}` of length B
#       (world b uses rules[b]), or a device matrix from `device_rules`.
#   device_rules(be, rules::AbstractVector{<:Rule}) -> P×B device matrix (P = 3C²+8C)
#       Upload once and reuse when the same rules are stepped many times. Passing a
#       `Vector{Rule}` to `step!` goes through a cache keyed on the rules' values (hashing
#       B rules per step; pass the matrix in hot loops).
#
#   world(wb, b) -> World            a copy of world b on the same backend
#   total_mass(wb) -> Vector{Float64}   per-world Σ(n+s+w)
#   summarize(wb) -> Vector{NamedTuple} per-world `summarize`
#   size(wb) == (Nx, Ny, Nz); nworlds(wb) == B
#
# Fields: h (C,Nx,Ny,Nz,B); s n w L (Nx,Ny,Nz,B); h2 s2 n2 w2 write buffers; t; step.

export WorldBatch, world, nworlds, device_rules

struct WorldBatch{C,T,A5<:AbstractArray{T,5},A4<:AbstractArray{T,4}}
    h::A5
    s::A4
    n::A4
    w::A4
    L::A4
    h2::A5
    s2::A4
    n2::A4
    w2::A4
    t::T
    step::Int
end

Base.size(wb::WorldBatch) = size(wb.s)[1:3]
nworlds(wb::WorldBatch) = size(wb.s, 4)
channels(::WorldBatch{C}) where {C} = C
Base.eltype(::WorldBatch{C,T}) where {C,T} = T
backend(wb::WorldBatch) = KA.get_backend(wb.s)

function WorldBatch(be, worlds::AbstractVector{<:World{C,T}}) where {C,T}
    isempty(worlds) && throw(ArgumentError("WorldBatch needs at least one world"))
    dims = size(first(worlds))
    all(w -> size(w) == dims, worlds) || throw(ArgumentError("all worlds must have the same size"))
    B = length(worlds)
    check_int32(C, dims, B)
    host(a) = Array(a)
    stack5(f) = adapt(be, cat((reshape(host(f(w)), size(f(w))..., 1) for w in worlds)...; dims=5))
    stack4(f) = adapt(be, cat((reshape(host(f(w)), size(f(w))..., 1) for w in worlds)...; dims=4))
    h = stack5(w -> w.h); s = stack4(w -> w.s)
    WorldBatch{C,T,typeof(h),typeof(s)}(h, s, stack4(w -> w.n), stack4(w -> w.w), stack4(w -> w.L),
        similar(h), similar(s), similar(s), similar(s), first(worlds).t, first(worlds).step)
end

function WorldBatch(be, C::Integer, dims::NTuple{3,Integer}, B::Integer; T::Type=Float32,
                    seeds=1:B, kw...)
    length(seeds) == B || throw(ArgumentError("need $B seeds, got $(length(seeds))"))
    WorldBatch(be, [World(KA.CPU(), C, dims; T=T, seed=sd, kw...) for sd in seeds])
end
WorldBatch(be, r::Rule{C,T}, dims::NTuple{3,Integer}, B::Integer; kw...) where {C,T} =
    WorldBatch(be, C, dims, B; T=T, kw...)

"""Copy of world `b` of a batch, as a `World` on the same backend."""
function world(wb::WorldBatch{C,T}, b::Integer) where {C,T}
    f(a) = copy(selectdim(a, ndims(a), b))
    h = f(wb.h); s = f(wb.s)
    World{C,T,typeof(h),typeof(s)}(h, s, f(wb.n), f(wb.w), f(wb.L),
        f(wb.h2), f(wb.s2), f(wb.n2), f(wb.w2), wb.t, wb.step)
end

function Adapt.adapt_structure(to, wb::WorldBatch{C,T}) where {C,T}
    f(a) = adapt(to, a)
    h = f(wb.h); s = f(wb.s)
    WorldBatch{C,T,typeof(h),typeof(s)}(h, s, f(wb.n), f(wb.w), f(wb.L),
        f(wb.h2), f(wb.s2), f(wb.n2), f(wb.w2), wb.t, wb.step)
end

Base.copy(wb::WorldBatch{C,T,A5,A4}) where {C,T,A5,A4} =
    WorldBatch{C,T,A5,A4}(copy(wb.h), copy(wb.s), copy(wb.n), copy(wb.w), copy(wb.L),
                          copy(wb.h2), copy(wb.s2), copy(wb.n2), copy(wb.w2), wb.t, wb.step)

total_mass(wb::WorldBatch) =
    vec(Float64.(Array(sum(wb.n; dims=(1, 2, 3)))) .+ Float64.(Array(sum(wb.s; dims=(1, 2, 3)))) .+
        Float64.(Array(sum(wb.w; dims=(1, 2, 3)))))
summarize(wb::WorldBatch) = [summarize(world(adapt(KA.CPU(), wb), b)) for b in 1:nworlds(wb)]

# ---- rules on the device -------------------------------------------------------------

"""Upload rules as a P×B device matrix (column b = `vec(rules[b])`)."""
function device_rules(be, rules::AbstractVector{<:Rule{C,T}}) where {C,T}
    θ = KA.allocate(be, T, nparams(first(rules)), length(rules))
    copyto!(θ, reduce(hcat, [collect(vec(r)) for r in rules]))
    θ
end

# Keyed on the rules' values (a copy), so mutating the caller's vector cannot alias.
cached_device_rules(be, rules::AbstractVector{<:Rule}) =
    cached_upload(() -> device_rules(be, rules), be, Tuple(rules))

# A rule bank for B worlds: θ with B columns, or one column shared by all (B_θ = 1).
# Returns (θ, Bθ, shell2): shell2 is false only when every rule is known to have Wnb2 = 0
# (a pre-uploaded matrix is assumed to use the shell).
bank(be, wb::WorldBatch{C,T}, r::Rule{C,T}) where {C,T} = (device_theta(be, r), Int32(1), has_shell(r))
function bank(be, wb::WorldBatch{C,T}, rules::AbstractVector{<:Rule{C,T}}) where {C,T}
    length(rules) == nworlds(wb) || throw(ArgumentError("need $(nworlds(wb)) rules, got $(length(rules))"))
    (cached_device_rules(be, rules), Int32(nworlds(wb)), any(has_shell, rules))
end
function bank(be, wb::WorldBatch{C,T}, θ::AbstractMatrix{T}) where {C,T}
    size(θ, 1) == 3C * C + 8C || throw(ArgumentError("θ must have $(3C*C+8C) rows (vec(rule))"))
    size(θ, 2) in (1, nworlds(wb)) || throw(ArgumentError("θ must have 1 or $(nworlds(wb)) columns"))
    (θ, Int32(size(θ, 2)), true)
end

# ---- stepping ------------------------------------------------------------------------

function step!(wb::WorldBatch{C,T}, rules, p::Physics{T};
               fast::Bool=fast_math_default(backend(wb)), sync::Bool=true) where {C,T}
    be = backend(wb)
    Nx, Ny, Nz = size(wb); B = nworlds(wb)
    check_int32(C, (Nx, Ny, Nz), B)
    X, Y, Z = Int32(Nx), Int32(Ny), Int32(Nz)
    θ, Bθ, shell2 = bank(be, wb, rules)
    launch_update!(be, wb.h2, wb.s2, wb.n2, wb.w2, wb.L, wb.h, wb.s, wb.n, wb.w, θ, Val(C),
                   Val(fast), p, X, Y, Z, Int32(B), Bθ, shell2)
    light_kernel!(be, light_workgroup(be))(vec(wb.L), p, X, Y, Z; ndrange=Nx * Ny * B)
    sync && KA.synchronize(be)
    WorldBatch{C,T,typeof(wb.h),typeof(wb.s)}(wb.h2, wb.s2, wb.n2, wb.w2, wb.L,
        wb.h, wb.s, wb.n, wb.w, wb.t + p.dt, wb.step + 1)
end

function run!(wb::WorldBatch, rules, p::Physics, nsteps::Integer; kw...)
    for _ in 1:nsteps
        wb = step!(wb, rules, p; kw..., sync=false)
    end
    KA.synchronize(backend(wb))
    wb
end

function light!(wb::WorldBatch{C,T}, rules, p::Physics{T}) where {C,T}
    be = backend(wb)
    Nx, Ny, Nz = size(wb); B = nworlds(wb)
    check_int32(C, (Nx, Ny, Nz), B)
    θ, Bθ, _ = bank(be, wb, rules)
    launch_light!(be, wb.L, wb.s, wb.h, θ, Val(C), p, Int32(Nx), Int32(Ny), Int32(Nz), Int32(B), Bθ)
    KA.synchronize(be)
    wb
end
