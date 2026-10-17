"""
    Intmat2.Analysis

Homogeneous steady state, linearization and dispersion relation of the one-field
system, and its classification. See `docs/analysis.md`.

State ordering everywhere in this module is `z = [h₁…h_C, n, s, w]` (the order of
`W_in`'s first three columns). Perturbations are horizontal Fourier modes uniform
in z; every k-dependence enters through `κ = 2Σᵢ(1 − cos kᵢ)` over the two
horizontal axes, so the dispersion relation is a function of `κ ∈ [0, 8]`.
"""
module Analysis

using LinearAlgebra
using Random
using KernelAbstractions
using ..Intmat2: Rule, Physics, nchannels, World, step!, light!, run!, softplus, sigmoid, conductance, mobility

export steady_state, jacobian, stepmap, dispersion, classify, kappa, axis_wavelength, stable_dt,
       turing_prone_rule, layered_steady_state, Pencil2, dispersion2, kappa2, kappa_iso, kappa_iso_planar

f64(x::AbstractArray) = Float64.(Array(x))

# ------------------------------------------------------------------ steady state

"""Depth factor `c` in `L = L0 exp(−c a)`: layer `depth` (1 = top) of a laterally
and vertically uniform slab sees `depth − 1` full layers plus half its own."""
depthfactor(depth::Real) = depth - 0.5

lightat(p, c, s, pig) = p.L0 * exp(-c * (p.alpha_s * s + p.alpha_p * pig))

"""
Material balance for fixed `h`: the unique `s ∈ (0,1)` with
`n + s + w = M`, `k_dep A_d n (1−s) = k_ero A_e s = k_photo L(s) w`.
The left side `s + e s/(k_photo L(s)) + e s/(a(1−s))` is strictly increasing on
`[0,1)` from 0 to ∞, so bisection is exact.
"""
function material_balance(p, a, e, M, Lfun)
    a > 0 && e > 0 && p.k_photo > 0 || return nothing
    f(s) = s + e * s / (p.k_photo * Lfun(s)) + e * s / (a * (1 - s)) - M
    lo, hi = 0.0, 1.0
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        f(mid) > 0 ? (hi = mid) : (lo = mid)
    end
    s = (lo + hi) / 2
    L = Lfun(s)
    n = e * s / (a * (1 - s))
    w = e * s / (p.k_photo * L)
    (s=s, n=n, w=w, L=L)
end

function material_of_h(rule, p, h, M, depth, Lfix)
    depth, Lamb = depth isa Tuple ? depth : (depth, 0.0)
    vdep = f64(rule.vdep); vero = f64(rule.vero); vpig = f64(rule.vpig)
    a = p.k_dep * softplus(dot(vdep, h))
    e = p.k_ero * softplus(dot(vero, h))
    pig = sigmoid(dot(vpig, h))
    c = depthfactor(depth)
    Lfun = Lfix === nothing ? (s -> Lamb + lightat(p, c, s, pig)) :
           Lfix isa Tuple ? (s -> Lamb + p.L0 * exp(-(Lfix[2] + (p.alpha_s * s + p.alpha_p * pig) / 2))) :
           (s -> Float64(Lfix))
    material_balance(p, a, e, M, Lfun)
end

function residual_h(rule, p, h, M, depth, Lfix)
    x = material_of_h(rule, p, h, M, depth, Lfix)
    x === nothing && return fill(NaN, length(h)), x
    Win = f64(rule.Win)
    u = f64(rule.Wself) * h + Win * [x.n, x.s, x.w, x.L] + f64(rule.b)
    tanh.(u) .- h, x
end

"""
    steady_state(rule, p; M=1.0, depth=1, L=nothing, h0=zeros(C), tol=1e-13, maxiter=200)

Spatially uniform fixed point `(h*, n*, s*, w*, L*)` at mass per voxel `M = n+s+w`.
Light is self-consistent at layer `depth` of a uniform slab,
`L* = L0 exp(−(depth−½)(α_s s* + α_p σ(v_pig·h*)))`, unless `L` fixes it, or
`tau_above = τ` places the layer under optical depth `τ`: `L* = L0 exp(−(τ + a*/2))`
(and `light=:local` then perturbs only the layer's own absorbance). `L_ambient`
(default `p.L_ambient`) adds a uniform, unattenuated term to `L*`.
Activity: damped Newton (finite-difference Jacobian, backtracking) on
`G(h) = tanh(W_self h + W_in [n,s,w,L](h) + b) − h` from `h0` (default 0), with the
material solved exactly for each `h`. Several fixed points may exist; this returns
the one Newton reaches from `h0`.

If Newton stalls, it alternates relaxation `h ← h + 0.2 G` with Newton restarts
(`method = :relax_newton`). Returns a NamedTuple with `h, n, s, w, L, M, depth,
converged, iters, residual, clamped, method` (`clamped` is true when a `min` in `material_kernel!` would bind at
`p.dt`, i.e. the discrete stepper's fixed point differs from this one).
"""
function steady_state(rule::Rule{C}, p::Physics; M=1.0, depth=1, L=nothing, tau_above=nothing,
                      L_ambient=nothing, h0=zeros(C), tol=1e-13, maxiter=200) where {C}
    p = Physics{Float64}(p)
    L_ambient === nothing && (L_ambient = p.L_ambient)
    lightc = depthfactor(depth)
    Lamb = Float64(L_ambient)
    depth0 = depth
    depth = (depth, Lamb)
    if tau_above !== nothing
        L = (:above, Float64(tau_above)); lightc = 0.5
    end
    r = newton_h(rule, p, Float64.(h0), M, depth, L, tol, maxiter)
    method = :newton
    if !r.converged
        # Newton stalled in a local minimum of |G|: relax h ← h + β G with the
        # material slaved (the activity's own dynamics), restarting Newton.
        h = Float64.(h0); iters = r.iters
        for round in 1:20
            for _ in 1:200
                G, _ = residual_h(rule, p, h, M, depth, L)
                all(isfinite, G) || break
                h = clamp.(h .+ 0.2 .* G, -1 + 1e-12, 1 - 1e-12)
            end
            r = newton_h(rule, p, h, M, depth, L, tol, maxiter)
            iters += 200 + r.iters
            r.converged && break
            h = r.h
        end
        r = merge(r, (iters=iters,))
        method = :relax_newton
    end
    h, x, nG, iters, converged = r.h, r.x, r.nG, r.iters, r.converged
    # degenerate saturation (s → 1 to machine precision, or h pinned at ±1) is not
    # a usable linearization point
    if x !== nothing && (!all(isfinite, (x.n, x.s, x.w, x.L)) || 1 - x.s < 1e-9 ||
                         any(>=(1 - 1e-10), abs.(h)))
        converged = false
    end
    clamped = false
    if x !== nothing
        Ad = softplus(dot(f64(rule.vdep), h)); Ae = softplus(dot(f64(rule.vero), h))
        dep = p.k_dep * Ad * x.n * (1 - x.s) * p.dt
        ero = p.k_ero * Ae * x.s * p.dt
        photo = p.k_photo * x.L * x.w * p.dt
        clamped = dep >= min(x.n, 1 - x.s) || ero >= x.s || photo >= x.w
    end
    depth = depth0
    x === nothing && return (h=h, n=NaN, s=NaN, w=NaN, L=NaN, M=Float64(M), depth=depth,
                             converged=false, iters=iters, residual=nG, clamped=false, method=method,
                             lightc=lightc, absorbance=NaN, L_ambient=Lamb)
    a = p.alpha_s * x.s + p.alpha_p * sigmoid(dot(f64(rule.vpig), h))
    (h=h, n=x.n, s=x.s, w=x.w, L=x.L, M=Float64(M), depth=depth,
     converged=converged, iters=iters, residual=nG, clamped=clamped, method=method,
     lightc=lightc, absorbance=a, L_ambient=Lamb)
end

function newton_h(rule::Rule{C}, p, h, M, depth, L, tol, maxiter) where {C}
    G, x = residual_h(rule, p, h, M, depth, L)
    nG = norm(G)
    iters = 0
    J = zeros(C, C)
    while iters < maxiter && isfinite(nG) && nG > tol
        iters += 1
        for j in 1:C
            δ = 1e-7 * max(1.0, abs(h[j]))
            hp = copy(h); hp[j] += δ
            Gp, _ = residual_h(rule, p, hp, M, depth, L)
            J[:, j] .= (Gp .- G) ./ δ
        end
        Δ = try
            -(J \ G)
        catch
            G  # singular: relaxation step
        end
        t = 1.0
        accepted = false
        while t > 1e-6
            hn = clamp.(h .+ t .* Δ, -1 + 1e-12, 1 - 1e-12)
            Gn, xn = residual_h(rule, p, hn, M, depth, L)
            if isfinite(norm(Gn)) && norm(Gn) < (1 - 1e-4 * t) * nG
                h, G, x, nG = hn, Gn, xn, norm(Gn)
                accepted = true
                break
            end
            t /= 2
        end
        accepted || break
    end
    converged = isfinite(nG) && nG <= max(tol, 1e-10)
    (h=h, x=x, nG=nG, iters=iters, converged=converged)
end

# ------------------------------------------------------------------ linearization

"""Pieces shared by the continuous Jacobian and the discrete step map."""
function linear_parts(rule::Rule{C}, p, ss, light::Symbol) where {C}
    h = ss.h
    S = 1 .- h .^ 2                                # tanh'(u*) since h* = tanh(u*)
    vdep = f64(rule.vdep); vero = f64(rule.vero); vpig = f64(rule.vpig)
    Ad = softplus(dot(vdep, h)); Ae = softplus(dot(vero, h))
    σd = sigmoid(dot(vdep, h)); σe = sigmoid(dot(vero, h)); pig = sigmoid(dot(vpig, h))
    n, s, w, L = ss.n, ss.s, ss.w, ss.L
    N = C + 3
    # δL = Λ·δz. :frozen → 0. :local → the voxel's column above is perturbed
    # coherently with it (exact for Nz = 1 and for k_z = 0 modes at the top layer).
    Λ = zeros(N)
    if light === :local
        c = hasproperty(ss, :lightc) ? ss.lightc : depthfactor(ss.depth)
        Lat = L - (hasproperty(ss, :L_ambient) ? ss.L_ambient : 0.0)   # attenuated part
        Λ[1:C] .= -Lat * c * p.alpha_p * pig * (1 - pig) .* vpig
        Λ[C+2] = -Lat * c * p.alpha_s
    elseif light !== :frozen
        throw(ArgumentError("light must be :frozen or :local"))
    end
    # rows of the material fluxes w.r.t. z = [h; n; s; w], and their L-derivative
    dep = zeros(N); ero = zeros(N); pho = zeros(N)
    dep[1:C] .= p.k_dep * σd * n * (1 - s) .* vdep
    dep[C+1] = p.k_dep * Ad * (1 - s)
    dep[C+2] = -p.k_dep * Ad * n
    ero[1:C] .= p.k_ero * σe * s .* vero
    ero[C+2] = p.k_ero * Ae
    pho[C+3] = p.k_photo * L
    phoL = p.k_photo * w
    g = conductance(p, s, s)                       # g* (δg·(h_y−h_x) vanishes: h_y−h_x = 0)
    D = mobility(p, s)                             # D(s*) (δD·∇n vanishes: ∇n = 0)
    (S=S, Λ=Λ, dep=dep, ero=ero, pho=pho, phoL=phoL, g=g, D=D,
     Wself=f64(rule.Wself), Wnb=f64(rule.Wnb), Wnb2=f64(rule.Wnb2), Win=f64(rule.Win))
end

"""
    jacobian(rule, p, ss, κ; κ2=0, light=:frozen) -> Matrix (C+3)×(C+3)

Continuous-time Jacobian (dt → 0 limit of `step!`) about `ss` for a Fourier mode
with coupling symbols `κ` (ring) and `κ2` (shell) and transport symbol `κD`
(default `κ`). For `p.stencil == 0` they are `κ = κD = 2Σ(1−cos kᵢ)` and
`κ2 = κ(2k)`; for `p.stencil == 1` the ring is `kappa_iso(k)` and the shell
`kappa_iso(2k)` while transport keeps the 6-face `κD = 2Σ(1−cos kᵢ)`
(`ring_symbol`, `shell_symbol`). Coupling `→ −g*(κ W_nb + κ2 W_nb2) δh`; transport
`→ −D(s*) κD`. `light=:frozen` sets δL = 0; `:local` lets L respond to the voxel's
own δs, δpig through the depth factor (exact for a one-layer world).
"""
function jacobian(rule::Rule{C}, p::Physics, ss, κ::Real; κ2::Real=0.0, κD::Real=κ,
                  light::Symbol=:frozen) where {C}
    p = Physics{Float64}(p)
    q = linear_parts(rule, p, ss, light)
    N = C + 3
    J = zeros(N, N)
    SW = q.S .* q.Win
    J[1:C, 1:C] .= (q.S .* (q.Wself .- q.g * κ .* q.Wnb .- q.g * κ2 .* q.Wnb2) - I) ./ p.tau_h
    J[1:C, C+1:C+3] .= SW[:, 1:3] ./ p.tau_h
    J[1:C, :] .+= (SW[:, 4] * q.Λ') ./ p.tau_h
    pho = q.pho .+ q.phoL .* q.Λ
    J[C+1, :] .= -q.dep .+ pho
    J[C+2, :] .= q.dep .- q.ero
    J[C+3, :] .= q.ero .- pho
    J[C+1, C+1] -= q.D * κD
    J[C+3, C+3] -= q.D * κD
    J
end

"""
    stepmap(rule, p, ss, κ; κ2=0, light=:frozen) -> Matrix (C+3)×(C+3)

Exact linearization of one `step!` (activity → transport → material, in that order,
material seeing the new `h` and the transported `n, w`, both seeing last step's `L`)
for the same Fourier mode. `stepmap = I + dt·jacobian + O(dt²)`.
"""
function stepmap(rule::Rule{C}, p::Physics, ss, κ::Real; κ2::Real=0.0, κD::Real=κ,
                 light::Symbol=:frozen) where {C}
    p = Physics{Float64}(p)
    q = linear_parts(rule, p, ss, light)
    N = C + 3
    a = p.dt / p.tau_h
    T = 1 - p.dt * q.D * κD
    # B1: z → y = [h', ñ, s, w̃]
    B1 = Matrix{Float64}(I, N, N)
    SW = q.S .* q.Win
    B1[1:C, 1:C] .= (1 - a) * I + a .* (q.S .* (q.Wself .- q.g * κ .* q.Wnb .- q.g * κ2 .* q.Wnb2))
    B1[1:C, C+1:C+3] .= a .* SW[:, 1:3]
    B1[1:C, :] .+= a .* (SW[:, 4] * q.Λ')
    B1[C+1, C+1] = T
    B1[C+3, C+3] = T
    # B2: y → z', fluxes evaluated at y (material sees h', ñ, s, w̃); P: δL column
    B2 = Matrix{Float64}(I, N, N)
    B2[C+1, :] .+= p.dt .* (-q.dep .+ q.pho)
    B2[C+2, :] .+= p.dt .* (q.dep .- q.ero)
    B2[C+3, :] .+= p.dt .* (q.ero .- q.pho)
    P = zeros(N)
    P[C+1] = p.dt * q.phoL
    P[C+3] = -p.dt * q.phoL
    B2 * B1 .+ P * q.Λ'
end

# ------------------------------------------------------------------ dispersion

"""Orthonormal basis of the mass-conserving subspace δn + δs + δw = 0."""
function massfree_basis(C)
    N = C + 3
    e = zeros(N); e[C+1:C+3] .= 1 / sqrt(3)
    Q = nullspace(e')
    Q
end

"""
    Pencil(rule, p, ss; light=:frozen)

Both linearizations are affine in the two symbols: `J = J0 − κ JK − κ2 J2` and
`A = A0 − κ AK − κ2 A2` (continuous Jacobian and one-step map). Built once per
steady state. `twoscale` is true when the rule's `W_nb2` is nonzero; then the
dispersion relation is no longer a function of κ alone.
"""
struct Pencil
    J0::Matrix{Float64}; JK::Matrix{Float64}; J2::Matrix{Float64}; JD::Matrix{Float64}
    A0::Matrix{Float64}; AK::Matrix{Float64}; A2::Matrix{Float64}; AD::Matrix{Float64}
    Q::Matrix{Float64}
    dt::Float64
    twoscale::Bool
    stencil::Int
    planar::Bool
end
function Pencil(rule::Rule{C}, p::Physics, ss; light::Symbol=:frozen, planar::Bool=false) where {C}
    J0 = jacobian(rule, p, ss, 0.0; light); A0 = stepmap(rule, p, ss, 0.0; light)
    JK = J0 - jacobian(rule, p, ss, 1.0; κD=0.0, light); AK = A0 - stepmap(rule, p, ss, 1.0; κD=0.0, light)
    JD = J0 - jacobian(rule, p, ss, 0.0; κD=1.0, light); AD = A0 - stepmap(rule, p, ss, 0.0; κD=1.0, light)
    two = any(!iszero, rule.Wnb2)
    J2 = two ? J0 - jacobian(rule, p, ss, 0.0; κ2=1.0, light) : zeros(size(J0))
    A2 = two ? A0 - stepmap(rule, p, ss, 0.0; κ2=1.0, light) : zeros(size(A0))
    Pencil(J0, JK, J2, JD, A0, AK, A2, AD, massfree_basis(C), Float64(p.dt), two, Int(stencil_of(p)), planar)
end
stencil_of(p::Physics) = hasproperty(p, :stencil) ? Int(p.stencil) : 0

"""19-point isotropic ring symbol (axis ×1/3, edge diagonals ×1/6), `≈ |k|²` at small k."""
kappa_iso(kx, ky, kz=0.0) = (2 / 3) * ((1 - cos(kx)) + (1 - cos(ky)) + (1 - cos(kz))) +
    (2 / 3) * ((1 - cos(kx) * cos(ky)) + (1 - cos(kx) * cos(kz)) + (1 - cos(ky) * cos(kz)))

"""Coupling symbols of `p.stencil` (0: axis ring and axis shell; 1: 19-point ring and
the 19-point stencil dilated by 2). Transport always uses `kappa`."""
ring_symbol(st::Integer, k...) = st == 0 ? kappa(k...) : kappa_iso(k...)
shell_symbol(st::Integer, k...) = ring_symbol(st, (2 .* k)...)

"""
19-point ring restricted to one z-layer (`N×N×1` worlds, or any voxel whose z
neighbours are cut by the walls): the step keeps the weights 1/3 and 1/6 and drops
the missing neighbours, whose terms `1 − cos(kₓ)` etc. do not vanish for `k_z = 0`.
`(2/3)[(1−cos kx) + (1−cos ky) + (1−cos kx cos ky)]`, `≈ (2/3)|k|²` and anisotropic
(the isotropic 2D 9-point stencil needs axis:diagonal = 4:1, this is 2:1).
"""
kappa_iso_planar(kx, ky, kz=0.0) = (2 / 3) * ((1 - cos(kx)) + (1 - cos(ky)) + (1 - cos(kx) * cos(ky)))

"""Symbol triple (ring, shell, transport) of a wavevector for this pencil. `planar`
pencils use the one-layer form of the isotropic stencil."""
function symbols(P, k...)
    if P.stencil != 0 && P.planar
        k2 = 2 .* k
        (kappa_iso_planar(k...), kappa_iso_planar(k2...), kappa(k...))
    else
        (ring_symbol(P.stencil, k...), shell_symbol(P.stencil, k...), kappa(k...))
    end
end

"""Dominant eigenvalue (max real part) of `J(κ)`. At κ = 0 the exact zero mode of
mass conservation is removed by restricting to the invariant subspace δn+δs+δw=0."""
function dominant(P::Pencil, κ, κ2=0.0, κD=κ)
    J = κ == 0 && κ2 == 0 && κD == 0 ? P.Q' * P.J0 * P.Q : P.J0 .- κ .* P.JK .- κ2 .* P.J2 .- κD .* P.JD
    λ = eigvals!(J)
    λ[argmax(real.(λ))]
end

"""Dominant multiplier (max modulus) of the one-step map at κ (mass mode removed at 0)."""
function dominant_map(P::Pencil, κ, κ2=0.0, κD=κ)
    A = κ == 0 && κ2 == 0 && κD == 0 ? P.Q' * P.A0 * P.Q : P.A0 .- κ .* P.AK .- κ2 .* P.A2 .- κD .* P.AD
    μ = eigvals!(A)
    μ[argmax(abs.(μ))]
end

"""Growth rate `log|μ_max|/dt` of the one-step map at κ."""
map_rate(P::Pencil, κ, κ2=0.0, κD=κ) = log(abs(dominant_map(P, κ, κ2, κD))) / P.dt

"""Ring symbol κ of a mode `(kx, ky[, kz])` (radians per voxel)."""
kappa(kx, ky) = 2(1 - cos(kx)) + 2(1 - cos(ky))
kappa(kx, ky, kz) = kappa(kx, ky) + 2(1 - cos(kz))
"""Second-shell (axis, distance 2) symbol `κ2(k) = κ(2k)`."""
kappa2(kx, ky) = kappa(2kx, 2ky)
kappa2(kx, ky, kz) = kappa(2kx, 2ky, 2kz)

"""Wavelength in voxels of the axis-aligned mode with this κ (κ ≤ 4); for κ > 4
only diagonal modes exist and the diagonal wavelength is returned."""
function axis_wavelength(κ)
    κ <= 0 && return Inf
    κ <= 4 && return 2π / (2asin(sqrt(κ) / 2))
    kx = 2asin(sqrt(min(κ / 2, 4.0)) / 2)
    2π / (sqrt(2) * kx)
end

const DEFAULT_KAPPAS = [8.0 * (j / 160)^2 for j in 1:160]

"""
    dispersion(rule, p, ss; κs, light=:frozen, refine=true)

`Re λ_max(κ)` and `Im` of that eigenvalue over `κs` (plus κ = 0), the location of
the maximum (golden-section refined between grid neighbours), and the matching
discrete-map growth rate `log|μ_max|/dt`. Returns a NamedTuple:
`κs, re, im, re0, im0, band_lo, band_hi` (first and last unstable grid κ), `lambda_max, kappa_star, kstar, wavelength, imag_star,
map_rate_max, map_rate0, map_rate_star`.
"""
function dispersion(rule::Rule, p::Physics, ss; κs=DEFAULT_KAPPAS, light::Symbol=:frozen,
                    refine::Bool=true, nk::Integer=48, planar::Bool=false)
    dispersion(Pencil(rule, p, ss; light, planar); κs, refine, nk)
end
function dispersion(P::Pencil; κs=DEFAULT_KAPPAS, refine::Bool=true, nk::Integer=48)
    (P.twoscale || P.stencil != 0) && return dispersion_k(P; nk, refine)
    λ0 = dominant(P, 0.0)
    λs = [dominant(P, κ) for κ in κs]
    re = real.(λs); im = imag.(λs)
    j = argmax(re)
    κstar, λstar = κs[j], λs[j]
    if refine && 1 < j < length(κs)
        a, b = κs[j-1], κs[j+1]
        φ = (sqrt(5) - 1) / 2
        f(κ) = real(dominant(P, κ))
        x1 = b - φ * (b - a); x2 = a + φ * (b - a); f1 = f(x1); f2 = f(x2)
        for _ in 1:30
            if f1 > f2
                b, x2, f2 = x2, x1, f1; x1 = b - φ * (b - a); f1 = f(x1)
            else
                a, x1, f1 = x1, x2, f2; x2 = a + φ * (b - a); f2 = f(x2)
            end
        end
        κr = (a + b) / 2
        λr = dominant(P, κr)
        real(λr) > real(λstar) && ((κstar, λstar) = (κr, λr))
    end
    mrates = [map_rate(P, κ) for κ in κs]
    m0 = map_rate(P, 0.0)
    kst = κstar <= 4 ? 2asin(sqrt(κstar) / 2) : sqrt(2) * 2asin(sqrt(min(κstar / 2, 4.0)) / 2)
    U = findall(>(1e-9), re)
    band_lo = isempty(U) ? NaN : κs[first(U)]
    band_hi = isempty(U) ? NaN : κs[last(U)]
    (κs=collect(κs), re=re, im=im, re0=real(λ0), im0=imag(λ0), band_lo=band_lo, band_hi=band_hi,
     lambda_max=real(λstar), kappa_star=κstar, kstar=kst, wavelength=axis_wavelength(κstar),
     imag_star=imag(λstar), map_rate_max=max(maximum(mrates), m0), map_rate0=m0,
     map_rate_star=map_rate(P, κstar))
end

"""
    dispersion_k(P; nk=48, refine=true)

Dispersion for a two-shell rule or an isotropic stencil: `Re λ_max` over horizontal wavevectors `0 ≤ ky ≤ kx ≤ π`
(step `π/nk`; the symbols are symmetric under kx ↔ ky and sign flips), refined on a
4× finer patch around the maximum. Same fields as the one-ring result; `κs, re, im`
are the axis sweep `(k, 0)` with `κs = κ(k, 0)`, `kappa_star = κ(k*)`,
`kstar = |k*|`, `wavelength = 2π/|k*|`, and `band_lo, band_hi` the extreme κ of
unstable grid modes.
"""
function dispersion_k(P::Pencil; nk::Integer=48, refine::Bool=true)
    λ0 = dominant(P, 0.0)
    best = (re=-Inf, λ=0.0im, kx=0.0, ky=0.0)
    band_lo, band_hi = Inf, -Inf
    mmax = map_rate(P, 0.0)
    ev(kx, ky) = dominant(P, symbols(P, kx, ky)...)
    for i in 0:nk, j in 0:i
        (i == 0 && j == 0) && continue
        kx = π * i / nk; ky = π * j / nk
        λ = ev(kx, ky)
        real(λ) > best.re && (best = (re=real(λ), λ=λ, kx=kx, ky=ky))
        if real(λ) > 1e-9
            κ = kappa(kx, ky); band_lo = min(band_lo, κ); band_hi = max(band_hi, κ)
        end
        mmax = max(mmax, map_rate(P, symbols(P, kx, ky)...))
    end
    if refine
        h = π / nk
        for a in -4:4, b in -4:4
            kx = clamp(best.kx + a * h / 4, 0, π); ky = clamp(best.ky + b * h / 4, 0, π)
            (kx == 0 && ky == 0) && continue
            λ = ev(kx, ky)
            real(λ) > best.re && (best = (re=real(λ), λ=λ, kx=kx, ky=ky))
        end
    end
    ks = [π * i / nk for i in 1:nk]
    ax = [ev(k, 0.0) for k in ks]
    kk = hypot(best.kx, best.ky)
    (κs=[kappa(k, 0.0) for k in ks], re=real.(ax), im=imag.(ax), re0=real(λ0), im0=imag(λ0),
     band_lo=isfinite(band_lo) ? band_lo : NaN, band_hi=isfinite(band_hi) ? band_hi : NaN,
     lambda_max=best.re, kappa_star=kappa(best.kx, best.ky), kstar=kk, kx=best.kx, ky=best.ky,
     wavelength=2π / kk, imag_star=imag(best.λ), map_rate_max=mmax, map_rate0=map_rate(P, 0.0),
     map_rate_star=map_rate(P, symbols(P, best.kx, best.ky)...))
end

"""
    classify(d; tol=1e-9) -> Symbol

From a `dispersion` result:
- `:stable`       max Re λ ≤ tol everywhere (κ = 0 included, mass mode excluded)
- `:gridscale`    κ = 0 stable, fastest κ* > 4 (no axis mode longer than 2 voxels:
                  checkerboard, usually `W_nb` anti-diffusion); checked before the next two
- `:turing`       κ = 0 stable, finite-κ unstable, dominant real
- `:wave`         κ = 0 stable, finite-κ unstable, dominant complex (wave instability)
- `:hopf`         κ = 0 unstable with complex dominant, and no finite κ grows faster
- `:homogeneous`  κ = 0 unstable with real dominant, and no finite κ grows faster
- `:mixed`        κ = 0 unstable, but a finite κ grows faster
"""
function classify(d; tol=1e-9)
    unstable0 = d.re0 > tol
    finite = d.lambda_max > tol
    osc(im) = abs(im) > 1e-7
    if !unstable0 && !finite
        :stable
    elseif !unstable0
        d.kappa_star > 4 ? :gridscale : osc(d.imag_star) ? :wave : :turing
    elseif d.lambda_max > d.re0 * (1 + 1e-3) + tol
        :mixed
    else
        osc(d.im0) ? :hopf : :homogeneous
    end
end

"""
    layered_steady_state(rule, p, Nz; M=1.0) -> Vector (index 1 = top layer)

Per-layer steady states of a laterally uniform slab, marched down from the lit
face: layer `j` sits under the exact optical depth `τ_j = Σ_{i<j} a_i*` of the
layers above (`a* = α_s s* + α_p σ(v_pig·h*)`), each at mass `M`. Vertical
coupling of `h` and vertical transport of `n, w` between layers are neglected
(each layer is in local balance); `relax_column` gives the coupled column.
Each layer is solved from `h0` = the layer above's `h*` (top: 0).
"""
function layered_steady_state(rule::Rule{C}, p::Physics, Nz::Integer; M=1.0) where {C}
    out = []
    τ = 0.0
    h0 = zeros(C)
    for j in 1:Nz
        ss = steady_state(rule, p; M, tau_above=τ, h0=h0)
        push!(out, merge(ss, (layer=j, tau_above=τ)))
        ss.converged || break
        τ += ss.absorbance
        h0 = ss.h
    end
    out
end

"""
    relax_column(rule, p, layers; steps) -> World

Run the real `step!` on a `1×1×Nz` column (periodic x, y of size 1, so only the
vertical coupling and transport act), initialized from `layered_steady_state`
(top layer at `k = Nz`), for `steps` steps. Returns the final world.
"""
function relax_column(rule::Rule{C}, p::Physics, layers; steps::Integer) where {C}
    Nz = length(layers)
    wd = World(KernelAbstractions.CPU(), rule, (1, 1, Nz); T=Float64, h_scale=0.0)
    for (j, ss) in enumerate(layers)
        k = Nz - j + 1
        wd.h[:, 1, 1, k] .= ss.h
        wd.s[1, 1, k] = ss.s; wd.n[1, 1, k] = ss.n; wd.w[1, 1, k] = ss.w
    end
    wd = light!(wd, rule, Physics{Float64}(p))
    run!(wd, rule, Physics{Float64}(p), steps)
end

# ------------------------------------------------------------------ coupling variants (research path)
# `Pencil` handles the adopted two-shell rule; `Pencil2` keeps the ring filters and
# W_nb / W_nb2 overrides used by bin/atlas2.jl.

"""
    Pencil2(rule, p, ss; Wnb=rule.Wnb, Wnb2=rule.Wnb2, light=:frozen)

Linearization for a proposed two-scale coupling
`m = Σ_{|d|=1} g (h_y − h_x)` (weight `W_nb`) `+ Σ_{d = ±2 eᵢ} g (h_y − h_x)` (weight
`W_nb2`), optionally with the ring applied to a filtered `h̄` (see `dispersion2`).
`J(k) = J0 − κ1(k) F(k) K1 − κ2(k) K2 − κ1(k) KD` with
`K1 = S g* W_nb/τ_h`, `K2 = S g* W_nb2/τ_h`, `KD = D(s*)` on n, w.
Both shells use the same conductance `g(s_x, s_y)`; at the uniform state that is `g*`.
"""
struct Pencil2
    J0::Matrix{Float64}; K1::Matrix{Float64}; K2::Matrix{Float64}; KD::Matrix{Float64}
    Q::Matrix{Float64}
end
function Pencil2(rule::Rule{C}, p::Physics, ss; Wnb=nothing, Wnb2=nothing, light::Symbol=:frozen) where {C}
    p = Physics{Float64}(p)
    q = linear_parts(rule, p, ss, light)
    N = C + 3
    J0 = jacobian(rule, p, ss, 0.0; light)
    K1 = zeros(N, N); K2 = zeros(N, N); KD = zeros(N, N)
    K1[1:C, 1:C] .= q.S .* (q.g .* (Wnb === nothing ? q.Wnb : Matrix{Float64}(Wnb))) ./ p.tau_h
    K2[1:C, 1:C] .= q.S .* (q.g .* (Wnb2 === nothing ? q.Wnb2 : Matrix{Float64}(Wnb2))) ./ p.tau_h
    KD[C+1, C+1] = q.D; KD[C+3, C+3] = q.D
    Pencil2(J0, K1, K2, KD, massfree_basis(C))
end

"""Per-axis filter symbols: `:none` 1, `:box` (1 + 2cos k)/3 (3-point mean; negative for
k > 2π/3), `:binomial` (1 + cos k)/2 (¼,½,¼; ≥ 0, zero at k = π)."""
filtersym(f::Symbol, k) = f === :none ? 1.0 : f === :box ? (1 + 2cos(k)) / 3 :
                          f === :binomial ? (1 + cos(k)) / 2 : throw(ArgumentError("filter $f"))

"""Coupling symbols of a horizontal mode (kx, ky): ring `κ1 = 2Σ(1−cos kᵢ)` and the
axis radius-2 shell `κ2 = 2Σ(1−cos 2kᵢ) = κ1(2k)`."""

function dominant2(P::Pencil2, kx, ky; filter::Symbol=:none)
    if kx == 0 && ky == 0
        λ = eigvals!(P.Q' * P.J0 * P.Q)
    else
        κ1 = kappa(kx, ky); κ2 = kappa2(kx, ky)
        F = filtersym(filter, kx) * filtersym(filter, ky)
        λ = eigvals!(P.J0 .- (κ1 * F) .* P.K1 .- κ2 .* P.K2 .- κ1 .* P.KD)
    end
    λ[argmax(real.(λ))]
end

"""
    dispersion2(P::Pencil2; filter=:none, nk=24)

Max `Re λ` over the horizontal wavevector grid `0 ≤ ky ≤ kx ≤ π` in steps of `π/nk`
(the symbols are symmetric under kx ↔ ky and sign flips). Returns the fields
`classify` needs (`kappa_star` is the ring symbol κ1 at the fastest k) plus `kx, ky`
and `wavelength = 2π/|k*|`.
"""
function dispersion2(P::Pencil2; filter::Symbol=:none, nk::Integer=24)
    λ0 = dominant2(P, 0.0, 0.0)
    best = (re=-Inf, λ=0.0im, kx=0.0, ky=0.0)
    for i in 0:nk, j in 0:i
        (i == 0 && j == 0) && continue
        kx = π * i / nk; ky = π * j / nk
        λ = dominant2(P, kx, ky; filter)
        real(λ) > best.re && (best = (re=real(λ), λ=λ, kx=kx, ky=ky))
    end
    kk = hypot(best.kx, best.ky)
    (re0=real(λ0), im0=imag(λ0), lambda_max=best.re, imag_star=imag(best.λ),
     kappa_star=kappa(best.kx, best.ky), kx=best.kx, ky=best.ky, wavelength=2π / kk)
end

# ------------------------------------------------------------------ Turing-prone rules

"""
    turing_prone_rule(rng, C, T=Float32; self=1.0, nb=0.4, inp=2.0, light_in=0.25,
                      bias=0.5, read=3.0, align=true)

A random rule drawn from the region the atlas fit (`bin/prone.jl`,
`results/atlas-20261016-prone.json`) associates with Turing-type instability:
- `W_nb = nb·Q Qᵀ/C` is positive semidefinite, so `−g κ W_nb` damps every channel:
  no grid-scale anti-diffusion (the main gridscale predictor);
- strong material inputs (`inp`) but a weak light column (`light_in·inp`);
- strong read-outs (`read`), and with `align` the deposition read-out is aligned and
  the erosion read-out anti-aligned with a reference activity
  `h_ref = tanh(W_in [0.5, 0.3, 0.1, 0.3] + b)`, so that at a nearby steady state
  deposition is on and erosion is off (`v_dep·h* > 0`, `v_ero·h* < 0`, high `s*`).
Low mass per voxel (`M ≈ 0.5`) is the strongest physics-side predictor; it is not
part of the rule.
"""
function turing_prone_rule(rng::AbstractRNG, C::Integer, ::Type{T}=Float32; self=1.0, nb=0.4, inp=2.0,
                           light_in=0.25, bias=0.5, read=3.0, align=true) where {T}
    sc = 1 / sqrt(C)
    Wself = randn(rng, C, C) .* (self * sc)
    Q = randn(rng, C, C)
    Wnb = nb .* (Q * Q') ./ C
    Win = randn(rng, C, 4) .* inp
    Win[:, 4] .*= light_in
    b = randn(rng, C) .* bias
    vdep = randn(rng, C) .* read; vero = randn(rng, C) .* read; vpig = randn(rng, C) .* read
    if align
        href = tanh.(Win * [0.5, 0.3, 0.1, 0.3] .+ b)
        vdep = abs.(vdep) .* sign.(href)
        vero = -abs.(vero) .* sign.(href)
    end
    Rule{C,T}(T.(vcat(vec(Wself), vec(Wnb), vec(Win), b, vdep, vero, vpig)))
end

# ------------------------------------------------------------------ time step bound

withdt(p::Physics{T}, dt) where {T} =
    Physics{T}(; (f => (f === :dt ? T(dt) : getfield(p, f)) for f in fieldnames(Physics))...)

"""
    stable_dt(rule, p; M=1.0, depth=1, light=:local, κmax=12, nκ=241) -> T

Largest `dt ≤ min(τ_h, 1/(6 D0))` for which the exact one-step map about the
homogeneous steady state does not grow faster than the continuous system, at every
`κ ∈ [0, κmax]` (12 = the 3D maximum; for a two-shell rule, every 3D wavevector on a
grid of step `π/nk3`):
`log|μ_max(κ)|/dt ≤ 1.5 max(Re λ(κ), 0) + atol`, `atol = 1e-3/τ_h` by default.
For a linearly stable steady state this is spectral radius ≤ e^{atol·dt} ≈ 1 at every
κ; for an unstable one (Turing, Hopf) it admits the physical growth and rejects
numerical growth. `atol` absorbs the O(dt) shift of marginal modes at band edges
(e.g. Re λ = 8e-6 → map rate 2e-4 at dt 0.2); growth slower than `atol` is not
resolved by this bound. Bisection over `dt` (40 halvings); returns 0 if even `dt_max/1000` fails.
"""
function stable_dt(rule::Rule, p::Physics{T}; M=1.0, depth=1, ss=nothing, light::Symbol=:local,
                   κmax=12.0, nκ=241, nk3=16, atol=1e-3 / p.tau_h) where {T}
    p64 = Physics{Float64}(p)
    ss === nothing && (ss = steady_state(rule, p64; M, depth))
    dmax = p64.D0 > 0 ? min(p64.tau_h, 1 / (6 * p64.D0)) : p64.tau_h
    # symbol pairs (κ, κ2) to check: the κ line for one ring; for two shells, the 3D
    # wavevector grid 0 ≤ kz ≤ ky ≤ kx ≤ π (step π/nk3), since κ alone no longer fixes κ2
    st = stencil_of(p64)
    syms = if any(!iszero, rule.Wnb2) || st != 0
        [(ring_symbol(st, ks...), shell_symbol(st, ks...), kappa(ks...))
         for i in 0:nk3 for j in 0:i for l in 0:j for ks in ((π * i / nk3, π * j / nk3, π * l / nk3),)]
    else
        [(κ, 0.0, κ) for κ in range(0, κmax, length=nκ)]
    end
    Pc = Pencil(rule, p64, ss; light)
    allowed = [max(real(dominant(Pc, sy...)), 0.0) for sy in syms]
    function ok(dt)
        P = Pencil(rule, withdt(p64, dt), ss; light)
        all(map_rate(P, sy...) <= 1.5 * a + atol for (sy, a) in zip(syms, allowed))
    end
    ok(dmax) && return T(dmax)
    lo, hi = dmax / 1000, dmax
    ok(lo) || return zero(T)
    for _ in 1:40
        mid = (lo + hi) / 2
        ok(mid) ? (lo = mid) : (hi = mid)
    end
    T(lo)
end

# ------------------------------------------------------------------ validation helpers

"""Complex 2D DFT of a real matrix by dense matrix products (no FFT dependency;
fine for the ≤ 128² validation grids)."""
function dft2d(A::AbstractMatrix)
    Nx, Ny = size(A)
    Ex = [cis(-2π * (a - 1) * (x - 1) / Nx) for a in 1:Nx, x in 1:Nx]
    Ey = [cis(-2π * (b - 1) * (y - 1) / Ny) for b in 1:Ny, y in 1:Ny]
    Ex * Matrix{Float64}(A) * transpose(Ey)
end

"""Power spectrum of a real 2D array (mean removed)."""
power2d(A::AbstractMatrix) = abs2.(dft2d(A .- sum(A) / length(A)))

"""
    mode_amplitudes(wd, ss; layer=1) -> Matrix

Per horizontal Fourier mode, the Euclidean norm over all `C+3` fields of the DFT
of the deviation from the steady state `ss` in z-layer `layer`.
"""
function mode_amplitudes(wd, ss; layer=1)
    C = size(wd.h, 1)
    P = abs2.(dft2d(view(wd.s, :, :, layer) .- ss.s)) .+
        abs2.(dft2d(view(wd.n, :, :, layer) .- ss.n)) .+
        abs2.(dft2d(view(wd.w, :, :, layer) .- ss.w))
    for c in 1:C
        P .+= abs2.(dft2d(view(wd.h, c, :, :, layer) .- ss.h[c]))
    end
    sqrt.(P)
end

"""κ of each lattice mode of an `Nx×Ny` grid, as a matrix aligned with `power2d`."""
lattice_kappa(Nx, Ny) = [kappa(2π * (a - 1) / Nx, 2π * (b - 1) / Ny) for a in 1:Nx, b in 1:Ny]
lattice_kappa2(Nx, Ny) = [kappa2(2π * (a - 1) / Nx, 2π * (b - 1) / Ny) for a in 1:Nx, b in 1:Ny]
"""Predicted one-step growth rate of every lattice mode of an `Nx×Ny` grid."""
lattice_rates(P::Pencil, Nx, Ny) =
    [map_rate(P, symbols(P, 2π * (a - 1) / Nx, 2π * (b - 1) / Ny)...) for a in 1:Nx, b in 1:Ny]

"""Fill a CPU Float64 world (h, s, n, w) with the steady state."""
function set_uniform!(wd, ss)
    C = size(wd.h, 1)
    for c in 1:C
        wd.h[c, :, :, :] .= ss.h[c]
    end
    wd.s .= ss.s; wd.n .= ss.n; wd.w .= ss.w
    wd
end

"""
    measure_mode_rates(rule, p, ss; N=48, t1, t2, amp=1e-6, seed=1)

Run the real `step!` on a CPU Float64 `N×N×1` world (one layer: light is exactly
local and z-coupling is absent, so the `light=:local` linearization is exact) from
`ss` plus zero-mean Gaussian noise of size `amp` in every field. Returns
`(rates, amp1, amp2, world, norms)`: per-mode growth rates `log(a(t2)/a(t1))/(t2−t1)`
of the all-field mode amplitudes, the amplitudes themselves, the final world, and
the deviation norm after every step.
"""
function measure_mode_rates(rule::Rule{C}, p::Physics, ss; N=48, t1, t2, amp=1e-6, seed=1) where {C}
    p = Physics{Float64}(p)
    wd = World(KernelAbstractions.CPU(), rule, (N, N, 1); T=Float64, h_scale=0.0)
    set_uniform!(wd, ss)
    rng = Random.MersenneTwister(seed)
    noise(dims) = (x = amp .* randn(rng, dims...); x .- sum(x) / length(x))
    for c in 1:C
        wd.h[c, :, :, :] .+= noise((N, N, 1))
    end
    wd.s .+= noise((N, N, 1)); wd.n .+= noise((N, N, 1)); wd.w .+= noise((N, N, 1))
    wd = light!(wd, rule, p)
    dev(wd) = sqrt(sum(abs2, wd.s .- ss.s) + sum(abs2, wd.n .- ss.n) + sum(abs2, wd.w .- ss.w) +
                   sum(abs2, wd.h .- reshape(ss.h, C, 1, 1, 1)))
    n1 = round(Int, t1 / p.dt); n2 = round(Int, t2 / p.dt)
    norms = [dev(wd)]
    a1 = nothing
    for i in 1:n2
        wd = step!(wd, rule, p)
        push!(norms, dev(wd))
        i == n1 && (a1 = mode_amplitudes(wd, ss))
    end
    a2 = mode_amplitudes(wd, ss)
    (rates=log.(a2 ./ a1) ./ ((n2 - n1) * p.dt), amp1=a1, amp2=a2, world=wd, norms=norms)
end

end # module Analysis
