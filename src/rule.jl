"""
    Rule{C,T}

The learnable local rule θ shared by every voxel. `C` activity channels, four
material inputs `[n, s, w, L]`. It is `isbits` (StaticArrays) so it is passed to
GPU kernels by value.

Fields: `Wself`, `Wnb` (C×C), `Win` (C×4), `b`, `vdep`, `vero`, `vpig` (C).
`nparams(Rule{C}) == 2C² + 8C`. `vec(rule)` flattens, `Rule{C,T}(θ)` unflattens,
in the field order above, column-major within matrices.
"""
struct Rule{C,T,CC,C4}
    Wself::SMatrix{C,C,T,CC}
    Wnb::SMatrix{C,C,T,CC}
    Win::SMatrix{C,4,T,C4}
    b::SVector{C,T}
    vdep::SVector{C,T}
    vero::SVector{C,T}
    vpig::SVector{C,T}
end

nparams(::Type{<:Rule{C}}) where {C} = 2C * C + 8C
nparams(::Rule{C}) where {C} = 2C * C + 8C
nchannels(::Type{<:Rule{C}}) where {C} = C
nchannels(::Rule{C}) where {C} = C

function Rule{C,T}(θ::AbstractVector) where {C,T}
    length(θ) == 2C * C + 8C || throw(ArgumentError("Rule{$C} needs $(2C*C+8C) parameters, got $(length(θ))"))
    o = 0
    take(n) = (v = view(θ, o+1:o+n); o += n; v)
    Wself = SMatrix{C,C,T}(take(C * C))
    Wnb = SMatrix{C,C,T}(take(C * C))
    Win = SMatrix{C,4,T}(take(4C))
    b = SVector{C,T}(take(C))
    vdep = SVector{C,T}(take(C))
    vero = SVector{C,T}(take(C))
    vpig = SVector{C,T}(take(C))
    Rule{C,T,C * C,4C}(Wself, Wnb, Win, b, vdep, vero, vpig)
end

Base.vec(r::Rule{C,T}) where {C,T} =
    vcat(vec(r.Wself), vec(r.Wnb), vec(r.Win), r.b, r.vdep, r.vero, r.vpig)

Base.convert(::Type{Rule{C,T}}, r::Rule{C}) where {C,T} = Rule{C,T}(T.(vec(r)))

"""
    random_rule(rng, C, T=Float32; self=0.8, nb=0.8, inp=0.5, bias=0.1, read=0.5)

A random rule. Recurrent weights are scaled by `1/√C` so the spectral radius is
`O(scale)`; `self ≈ 1` sits near the edge of the fixed-point's stability, which
is where the atlas finds the interesting regimes.
"""
function random_rule(rng::AbstractRNG, C::Integer, ::Type{T}=Float32;
                     self=0.8, nb=0.8, inp=0.5, bias=0.1, read=0.5) where {T}
    sc = 1 / sqrt(C)
    θ = vcat(randn(rng, C * C) .* (self * sc),
             randn(rng, C * C) .* (nb * sc),
             randn(rng, 4C) .* inp,
             randn(rng, C) .* bias,
             randn(rng, C) .* read,
             randn(rng, C) .* read,
             randn(rng, C) .* read)
    Rule{C,T}(T.(θ))
end
random_rule(C::Integer, T::Type=Float32; kw...) = random_rule(Random.default_rng(), C, T; kw...)

"""
    Physics{T}

Timescales and material coefficients. Grid spacing is 1. `6 D0 dt ≤ 1` and
`dt ≤ tau_h` are necessary, not sufficient: the explicit neighbour coupling and
weakly damped oscillatory modes also bound `dt`. Use
`Intmat2.Analysis.stable_dt(rule, p; M)` for the actual bound (docs/analysis.md).

Absorption defaults are per cell and deliberately small: at `alpha_s = 0.05` a
column of half-solid has an e-folding depth of 40 voxels, so a 128-deep slab is
translucent. The layered analysis showed that `alpha_s = 2` turns everything below
the top three layers into a dark waste store and removes the predicted patterns.
"""
Base.@kwdef struct Physics{T}
    dt::T = 0.05
    tau_h::T = 1.0      # activity relaxation time
    g0::T = 0.5         # neighbour conductance with no solid
    g1::T = -0.5        # conductance change per unit solid (negative: solid insulates)
    k_dep::T = 0.1      # deposition rate
    k_ero::T = 0.01     # erosion rate
    k_photo::T = 0.05   # light-driven waste → nutrient
    D0::T = 0.2         # diffusivity of n and w with no solid
    alpha_s::T = 0.05   # light absorption per unit solid per cell (translucent matter)
    alpha_p::T = 0.1    # light absorption per unit pigment per cell
    L0::T = 1.0         # illumination at the top face
end

Physics{T}(p::Physics) where {T} = Physics{T}(; (f => T(getfield(p, f)) for f in fieldnames(Physics))...)

"""Design coordinates of a `Physics` (see DESIGN.md)."""
function timescales(p::Physics)
    τdep = 1 / p.k_dep
    (tau_h=p.tau_h, tau_dep=τdep, tau_ero=1 / p.k_ero, tau_photo=1 / (p.k_photo * p.L0),
     tau_diff=1 / p.D0, interaction_length=sqrt(p.D0 * τdep),
     dep_over_h=τdep / p.tau_h, ero_over_dep=p.k_dep / p.k_ero,
     photo_over_dep=p.k_dep / (p.k_photo * p.L0), g1_over_g0=p.g1 / p.g0)
end
