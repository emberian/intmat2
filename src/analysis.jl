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
using ..Intmat2: Rule, Physics, World, step!, light!, run!, softplus, sigmoid, conductance, mobility

export steady_state, jacobian, stepmap, dispersion, classify, kappa, axis_wavelength

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
    vdep = f64(rule.vdep); vero = f64(rule.vero); vpig = f64(rule.vpig)
    a = p.k_dep * softplus(dot(vdep, h))
    e = p.k_ero * softplus(dot(vero, h))
    pig = sigmoid(dot(vpig, h))
    c = depthfactor(depth)
    Lfun = Lfix === nothing ? (s -> lightat(p, c, s, pig)) : (s -> Float64(Lfix))
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
`L* = L0 exp(−(depth−½)(α_s s* + α_p σ(v_pig·h*)))`, unless `L` fixes it.
Activity: damped Newton (finite-difference Jacobian, backtracking) on
`G(h) = tanh(W_self h + W_in [n,s,w,L](h) + b) − h` from `h0` (default 0), with the
material solved exactly for each `h`. Several fixed points may exist; this returns
the one Newton reaches from `h0`.

If Newton stalls, it alternates relaxation `h ← h + 0.2 G` with Newton restarts
(`method = :relax_newton`). Returns a NamedTuple with `h, n, s, w, L, M, depth,
converged, iters, residual, clamped, method` (`clamped` is true when a `min` in `material_kernel!` would bind at
`p.dt`, i.e. the discrete stepper's fixed point differs from this one).
"""
function steady_state(rule::Rule{C}, p::Physics; M=1.0, depth=1, L=nothing,
                      h0=zeros(C), tol=1e-13, maxiter=200) where {C}
    p = Physics{Float64}(p)
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
    clamped = false
    if x !== nothing
        Ad = softplus(dot(f64(rule.vdep), h)); Ae = softplus(dot(f64(rule.vero), h))
        dep = p.k_dep * Ad * x.n * (1 - x.s) * p.dt
        ero = p.k_ero * Ae * x.s * p.dt
        photo = p.k_photo * x.L * x.w * p.dt
        clamped = dep >= min(x.n, 1 - x.s) || ero >= x.s || photo >= x.w
    end
    x === nothing && return (h=h, n=NaN, s=NaN, w=NaN, L=NaN, M=Float64(M), depth=depth,
                             converged=false, iters=iters, residual=nG, clamped=false, method=method)
    (h=h, n=x.n, s=x.s, w=x.w, L=x.L, M=Float64(M), depth=depth,
     converged=converged, iters=iters, residual=nG, clamped=clamped, method=method)
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
        c = depthfactor(ss.depth)
        Λ[1:C] .= -L * c * p.alpha_p * pig * (1 - pig) .* vpig
        Λ[C+2] = -L * c * p.alpha_s
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
     Wself=f64(rule.Wself), Wnb=f64(rule.Wnb), Win=f64(rule.Win))
end

"""
    jacobian(rule, p, ss, κ; light=:frozen) -> Matrix (C+3)×(C+3)

Continuous-time Jacobian (dt → 0 limit of `step!`) about `ss` for a horizontal
Fourier mode with `κ = 2Σ(1−cos kᵢ)`. Coupling `m → −g* κ δh`; transport
`→ −D(s*) κ`. `light=:frozen` sets δL = 0; `:local` lets L respond to the voxel's
own δs, δpig through the depth factor (exact for a one-layer world).
"""
function jacobian(rule::Rule{C}, p::Physics, ss, κ::Real; light::Symbol=:frozen) where {C}
    p = Physics{Float64}(p)
    q = linear_parts(rule, p, ss, light)
    N = C + 3
    J = zeros(N, N)
    SW = q.S .* q.Win
    J[1:C, 1:C] .= (q.S .* (q.Wself .- q.g * κ .* q.Wnb) - I) ./ p.tau_h
    J[1:C, C+1:C+3] .= SW[:, 1:3] ./ p.tau_h
    J[1:C, :] .+= (SW[:, 4] * q.Λ') ./ p.tau_h
    pho = q.pho .+ q.phoL .* q.Λ
    J[C+1, :] .= -q.dep .+ pho
    J[C+2, :] .= q.dep .- q.ero
    J[C+3, :] .= q.ero .- pho
    J[C+1, C+1] -= q.D * κ
    J[C+3, C+3] -= q.D * κ
    J
end

"""
    stepmap(rule, p, ss, κ; light=:frozen) -> Matrix (C+3)×(C+3)

Exact linearization of one `step!` (activity → transport → material, in that order,
material seeing the new `h` and the transported `n, w`, both seeing last step's `L`)
for the same Fourier mode. `stepmap = I + dt·jacobian + O(dt²)`.
"""
function stepmap(rule::Rule{C}, p::Physics, ss, κ::Real; light::Symbol=:frozen) where {C}
    p = Physics{Float64}(p)
    q = linear_parts(rule, p, ss, light)
    N = C + 3
    a = p.dt / p.tau_h
    T = 1 - p.dt * q.D * κ
    # B1: z → y = [h', ñ, s, w̃]
    B1 = Matrix{Float64}(I, N, N)
    SW = q.S .* q.Win
    B1[1:C, 1:C] .= (1 - a) * I + a .* (q.S .* (q.Wself .- q.g * κ .* q.Wnb))
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

Both linearizations are affine in κ: `J(κ) = J0 − κ JK` and `A(κ) = A0 − κ AK`
(continuous Jacobian and one-step map). Built once per steady state.
"""
struct Pencil
    J0::Matrix{Float64}; JK::Matrix{Float64}
    A0::Matrix{Float64}; AK::Matrix{Float64}
    Q::Matrix{Float64}
    dt::Float64
end
function Pencil(rule::Rule{C}, p::Physics, ss; light::Symbol=:frozen) where {C}
    J0 = jacobian(rule, p, ss, 0.0; light); J1 = jacobian(rule, p, ss, 1.0; light)
    A0 = stepmap(rule, p, ss, 0.0; light); A1 = stepmap(rule, p, ss, 1.0; light)
    Pencil(J0, J0 - J1, A0, A0 - A1, massfree_basis(C), Float64(p.dt))
end

"""Dominant eigenvalue (max real part) of `J(κ)`. At κ = 0 the exact zero mode of
mass conservation is removed by restricting to the invariant subspace δn+δs+δw=0."""
function dominant(P::Pencil, κ)
    J = κ == 0 ? P.Q' * P.J0 * P.Q : P.J0 .- κ .* P.JK
    λ = eigvals!(J)
    λ[argmax(real.(λ))]
end

"""Dominant multiplier (max modulus) of the one-step map at κ (mass mode removed at 0)."""
function dominant_map(P::Pencil, κ)
    A = κ == 0 ? P.Q' * P.A0 * P.Q : P.A0 .- κ .* P.AK
    μ = eigvals!(A)
    μ[argmax(abs.(μ))]
end

"""Growth rate `log|μ_max|/dt` of the one-step map at κ."""
map_rate(P::Pencil, κ) = log(abs(dominant_map(P, κ))) / P.dt

"""κ of a lattice mode `(kx, ky)` (radians per voxel)."""
kappa(kx, ky) = 2(1 - cos(kx)) + 2(1 - cos(ky))

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
                    refine::Bool=true)
    dispersion(Pencil(rule, p, ss; light); κs, refine)
end
function dispersion(P::Pencil; κs=DEFAULT_KAPPAS, refine::Bool=true)
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
