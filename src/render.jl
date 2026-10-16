# A volume ray marcher whose only light source is the simulation's own light field L.
#
# The slab is a finite block [0,Nx]×[0,Ny]×[0,Nz] in voxel units (z up, k=Nz is the lit
# top, voxel centres at i-½). Along each camera ray, per unit length:
#   extinction  σ_c(x) = density·(α_s·s + α_p·p̃·pig_absorb_c)
#   scattering  β_c(x) = density·(α_s·s·solid_c + α_p·p̃·pigment_c)
#   in-scatter  β_c(x)·(L(x) + ambient)      L trilinear from the simulation
# with p̃ = max(pig − pig_floor, 0)/(1 − pig_floor). Radiance is integrated front to
# back with exact per-segment attenuation, then exposure, a shoulder v/(1+shoulder·v),
# and gamma. `prefilter=true` blurs s and pig with a 3³ binomial kernel first: it
# removes single-cell (checkerboard) alternation the camera cannot resolve, which
# otherwise aliases into moiré.

export Optics, Camera, Clip, camera, render, render_world, write_png

"""
    Optics(; alpha_s, alpha_p, density, solid, pigment, pigment_absorb, pig_floor,
             background, ambient, exposure, shoulder, step, gamma)

Optical parameters of a photograph. `alpha_s`/`alpha_p` should be the run's Physics
values (`Optics(p::Physics; kw...)` copies them); `density` scales the camera-path
extinction only (1 = the light the material itself sees). Colours are RGB albedos.
"""
struct Optics
    alpha_s::Float32
    alpha_p::Float32
    density::Float32
    solid::SVector{3,Float32}
    pigment::SVector{3,Float32}
    pigment_absorb::SVector{3,Float32}
    pig_floor::Float32
    background::SVector{3,Float32}
    ambient::Float32
    exposure::Float32
    shoulder::Float32
    step::Float32
    gamma::Float32
end
function Optics(; alpha_s=2.0, alpha_p=2.0, density=1.0, solid=(0.90, 0.89, 0.86),
                pigment=(0.95, 0.42, 0.16), pigment_absorb=(0.35, 1.0, 1.4), pig_floor=0.0,
                background=(0.012, 0.012, 0.014), ambient=0.0, exposure=1.0, shoulder=0.25, step=0.5, gamma=2.2)
    v3(x) = SVector{3,Float32}(x...)
    Optics(alpha_s, alpha_p, density, v3(solid), v3(pigment), v3(pigment_absorb), pig_floor,
           v3(background), ambient, exposure, shoulder, step, gamma)
end
Optics(p::Physics; kw...) = Optics(; alpha_s=Float64(p.alpha_s), alpha_p=Float64(p.alpha_p), kw...)

"""Pinhole or orthographic camera. Build with `camera(pos, target; ...)`."""
struct Camera
    pos::SVector{3,Float32}
    fwd::SVector{3,Float32}
    right::SVector{3,Float32}
    up::SVector{3,Float32}
    tanhalf::Float32     # perspective: tan(fov/2); ortho: half-height in voxels
    ortho::Bool
end

"""
    camera(pos, target; up=(0,0,1), fov=35, ortho=false, half_height=nothing)

`fov` is the vertical field of view in degrees. With `ortho=true` the view is
parallel and `half_height` (voxels) sets the framing.
"""
function camera(pos, target; up=(0, 0, 1), fov=35, ortho=false, half_height=nothing)
    P = SVector{3,Float64}(pos...); Tg = SVector{3,Float64}(target...)
    f = normalize(Tg - P)
    r = normalize(cross(f, SVector{3,Float64}(up...)))
    u = cross(r, f)
    th = ortho ? something(half_height, norm(Tg - P) * tand(fov / 2)) : tand(fov / 2)
    Camera(SVector{3,Float32}(P), SVector{3,Float32}(f), SVector{3,Float32}(r), SVector{3,Float32}(u), th, ortho)
end

"""
    Clip(; lo=(0,0,0), hi=(Inf,Inf,Inf), normal=(0,0,0), offset=0)

Keep points inside the box `lo ≤ x ≤ hi` (voxel units) and with `normal·x ≤ offset`.
A zero normal disables the plane.
"""
struct Clip
    lo::SVector{3,Float32}
    hi::SVector{3,Float32}
    normal::SVector{3,Float32}
    offset::Float32
end
Clip(; lo=(0, 0, 0), hi=(Inf, Inf, Inf), normal=(0, 0, 0), offset=0) =
    Clip(SVector{3,Float32}(lo...), SVector{3,Float32}(min.(hi, 1f6)...), SVector{3,Float32}(normal...), offset)

@inline function trilerp(a, x::T, y::T, z::T, Nx, Ny, Nz) where {T}
    fx = clamp(x - T(0.5), zero(T), T(Nx - 1))
    fy = clamp(y - T(0.5), zero(T), T(Ny - 1))
    fz = clamp(z - T(0.5), zero(T), T(Nz - 1))
    i0 = unsafe_trunc(Int32, fx); j0 = unsafe_trunc(Int32, fy); k0 = unsafe_trunc(Int32, fz)
    tx = fx - T(i0); ty = fy - T(j0); tz = fz - T(k0)
    i1 = min(i0 + Int32(1), Int32(Nx - 1)); j1 = min(j0 + Int32(1), Int32(Ny - 1)); k1 = min(k0 + Int32(1), Int32(Nz - 1))
    i0 += Int32(1); i1 += Int32(1); j0 += Int32(1); j1 += Int32(1); k0 += Int32(1); k1 += Int32(1)
    @inbounds begin
        c00 = a[i0, j0, k0] * (one(T) - tx) + a[i1, j0, k0] * tx
        c10 = a[i0, j1, k0] * (one(T) - tx) + a[i1, j1, k0] * tx
        c01 = a[i0, j0, k1] * (one(T) - tx) + a[i1, j0, k1] * tx
        c11 = a[i0, j1, k1] * (one(T) - tx) + a[i1, j1, k1] * tx
    end
    c0 = c00 * (one(T) - ty) + c10 * ty
    c1 = c01 * (one(T) - ty) + c11 * ty
    c0 * (one(T) - tz) + c1 * tz
end

@inline function hash01(x::UInt32)
    x = (x ⊻ (x >> UInt32(16))) * 0x7feb352d
    x = (x ⊻ (x >> UInt32(15))) * 0x846ca68b
    x = x ⊻ (x >> UInt32(16))
    Float32(x >> UInt32(8)) * Float32(1 / 16777216)
end

@kernel function march_kernel!(img, @Const(s), @Const(pig), @Const(L), cam::Camera, opt::Optics, clip::Clip,
                               W, H, Nx, Ny, Nz)
    px, py = @index(Global, NTuple)
    T = Float32
    @inbounds begin
        aspect = T(W) / T(H)
        sx = (T(2) * (T(px) - T(0.5)) / T(W) - one(T)) * aspect
        sy = one(T) - T(2) * (T(py) - T(0.5)) / T(H)
        if cam.ortho
            o = cam.pos + (sx * cam.tanhalf) * cam.right + (sy * cam.tanhalf) * cam.up
            d = cam.fwd
        else
            o = cam.pos
            d = normalize(cam.fwd + (sx * cam.tanhalf) * cam.right + (sy * cam.tanhalf) * cam.up)
        end
        # slab box ∩ clip box
        blo = max.(clip.lo, SVector{3,T}(0, 0, 0))
        bhi = min.(clip.hi, SVector{3,T}(Nx, Ny, Nz))
        inv = SVector{3,T}(one(T) / d[1], one(T) / d[2], one(T) / d[3])
        ta = (blo - o) .* inv; tb = (bhi - o) .* inv
        t0 = max(max(min(ta[1], tb[1]), min(ta[2], tb[2])), max(min(ta[3], tb[3]), zero(T)))
        t1 = min(min(max(ta[1], tb[1]), max(ta[2], tb[2])), max(ta[3], tb[3]))
        acc = SVector{3,T}(0, 0, 0)
        tr = SVector{3,T}(1, 1, 1)
        if t1 > t0
            ds = opt.step
            t = t0 + ds * hash01(UInt32(px) * UInt32(9781) + UInt32(py) * UInt32(6271) + UInt32(17))
            pf = opt.pig_floor
            while t < t1
                x = o + t * d
                if dot(clip.normal, x) <= clip.offset
                    sv = trilerp(s, x[1], x[2], x[3], Nx, Ny, Nz)
                    pv = trilerp(pig, x[1], x[2], x[3], Nx, Ny, Nz)
                    Lv = trilerp(L, x[1], x[2], x[3], Nx, Ny, Nz)
                    pe = max(pv - pf, zero(T)) / (one(T) - pf)
                    as = opt.density * opt.alpha_s * sv
                    ap = opt.density * opt.alpha_p * pe
                    σ = as .+ ap .* opt.pigment_absorb
                    β = as .* opt.solid .+ ap .* opt.pigment
                    att = exp.(-σ .* ds)
                    # exact integral of β·Lin·e^{-σ u} over the segment
                    frac = ifelse.(σ .> T(1e-6), (one(T) .- att) ./ max.(σ, T(1e-6)), ds)
                    acc = acc .+ tr .* β .* frac .* (Lv + opt.ambient)
                    tr = tr .* att
                    if max(tr[1], tr[2], tr[3]) < T(1e-3)
                        break
                    end
                end
                t += ds
            end
        end
        for c in 1:3
            v = opt.exposure * acc[c]
            v = v / (one(T) + opt.shoulder * v)      # soft shoulder, linear near 0
            v = v + tr[c] * opt.background[c]
            img[c, px, py] = clamp(v, zero(T), one(T))^(one(T) / opt.gamma)
        end
    end
end

"""
    render(s, pig, L, cam, optics; width=800, height=width, clip=Clip(), backend=get_backend(s))

Ray-march the volume. `s`, `pig`, `L` are `(Nx,Ny,Nz)` Float32 arrays on one
backend. Returns a host `Array{Float32,3}` of size `(3, width, height)` in [0,1]
(row 1 is the top of the image).
"""
function render(s::AbstractArray{Float32,3}, pig::AbstractArray{Float32,3}, L::AbstractArray{Float32,3},
                cam::Camera, opt::Optics; width::Integer=800, height::Integer=width, clip::Clip=Clip(),
                prefilter::Bool=false)
    be = KA.get_backend(s)
    if prefilter
        s = binomial3(s); pig = binomial3(pig)
    end
    img = KA.allocate(be, Float32, 3, width, height)
    Nx, Ny, Nz = size(s)
    march_kernel!(be, (16, 16))(img, s, pig, L, cam, opt, clip, Int32(width), Int32(height),
                                Int32(Nx), Int32(Ny), Int32(Nz); ndrange=(width, height))
    KA.synchronize(be)
    Array(img)
end

@kernel function binomial3_kernel!(out, @Const(a), Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)
    T = eltype(out)
    acc = zero(T); wsum = zero(T)
    @inbounds for dk in -1:1, dj in -1:1, di in -1:1
        ii = i + di; jj = j + dj; kk = k + dk
        if 1 <= ii <= Nx && 1 <= jj <= Ny && 1 <= kk <= Nz
            w = T((2 - abs(di)) * (2 - abs(dj)) * (2 - abs(dk)))
            acc += w * a[ii, jj, kk]; wsum += w
        end
    end
    @inbounds out[i, j, k] = acc / wsum
end

"""3³ binomial ([1,2,1]⊗³) blur, renormalized at the block faces."""
function binomial3(a::AbstractArray{T,3}) where {T}
    be = KA.get_backend(a)
    out = similar(a)
    binomial3_kernel!(be)(out, a, size(a)...; ndrange=size(a))
    KA.synchronize(be)
    out
end

@kernel function pig_kernel!(pig, @Const(h), rule::Rule{C,T}) where {C,T}
    i, j, k = @index(Global, NTuple)
    @inbounds pig[i, j, k] = sigmoid(dot(rule.vpig, loadh(h, Val(C), i, j, k)))
end

"""Render a live `World` on its own backend (pigment computed on the device)."""
function render_world(wd::World{C,T}, rule::Rule{C,T}, cam::Camera, opt::Optics; kw...) where {C,T}
    be = backend(wd)
    pig = KA.allocate(be, T, size(wd)...)
    pig_kernel!(be)(pig, wd.h, rule; ndrange=size(wd))
    KA.synchronize(be)
    render(wd.s, pig, wd.L, cam, opt; kw...)
end

# ---------------------------------------------------------------- PNG (stdlib only)

const CRC_TABLE = let t = Vector{UInt32}(undef, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = (c & 1) != 0 ? 0xedb88320 ⊻ (c >> 1) : c >> 1
        end
        t[n+1] = c
    end
    t
end
function crc32(data)
    c = 0xffffffff
    for b in data
        c = CRC_TABLE[((c ⊻ b) & 0xff)+1] ⊻ (c >> 8)
    end
    c ⊻ 0xffffffff
end
function png_chunk(io, tag, data)
    write(io, hton(UInt32(length(data))))
    td = vcat(Vector{UInt8}(tag), data)
    write(io, td); write(io, hton(crc32(td)))
end

"""
    write_png(path, img)

Write a `(3,W,H)` Float32 image in [0,1] as an 8-bit RGB PNG (stored deflate, no
dependencies; convert to WebP for publishing).
"""
function write_png(path::AbstractString, img::AbstractArray{<:Real,3})
    _, W, H = size(img)
    raw = Vector{UInt8}(undef, H * (1 + 3W))
    o = 0
    for y in 1:H
        raw[o+=1] = 0x00
        for x in 1:W, c in 1:3
            raw[o+=1] = round(UInt8, clamp(Float32(img[c, x, y]), 0f0, 1f0) * 255)
        end
    end
    z = UInt8[0x78, 0x01]
    a, b = UInt32(1), UInt32(0)
    for v in raw
        a = (a + v) % UInt32(65521); b = (b + a) % UInt32(65521)
    end
    for st in 1:65535:length(raw)
        en = min(st + 65534, length(raw))
        n = en - st + 1
        push!(z, en == length(raw) ? 0x01 : 0x00, n % UInt8, (n >> 8) % UInt8, ~n % UInt8, (~n >> 8) % UInt8)
        append!(z, view(raw, st:en))
    end
    append!(z, reinterpret(UInt8, [hton((b << 16) | a)]))
    open(path, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        ihdr = vcat(reinterpret(UInt8, [hton(UInt32(W)), hton(UInt32(H))]), UInt8[8, 2, 0, 0, 0])
        png_chunk(io, "IHDR", ihdr)
        png_chunk(io, "IDAT", z)
        png_chunk(io, "IEND", UInt8[])
    end
    path
end

"""
    photo_setup(o::AbstractDict, dims, p::Physics) -> (; opt, cam, clip, width, height)

Interpret a parsed optics.toml (see bin/photograph.jl): `[optics]` Optics fields
(`alpha_s`/`alpha_p` default to `p`), `[camera]` pos/target as fractions of `dims`,
`up`, `fov`, `ortho`, `half_height` (fraction of Nz), `[clip]` lo/hi (fractions),
`normal`, `offset` (fraction of the extent along `normal`).
"""
function photo_setup(o::AbstractDict, dims, p::Physics)
    D = Float64.(collect(dims))
    tup(v) = v isa AbstractVector ? Tuple(Float64.(v)) : Float64(v)
    opt = Optics(p; (Symbol(k) => tup(v) for (k, v) in get(o, "optics", Dict()))...)
    c = get(o, "camera", Dict())
    pos = Tuple(Float64.(get(c, "pos", [0.5, -1.2, 1.6])) .* D)
    tgt = Tuple(Float64.(get(c, "target", [0.5, 0.5, 0.45])) .* D)
    hh = haskey(c, "half_height") ? c["half_height"] * D[3] : nothing
    cam = camera(pos, tgt; up=Tuple(get(c, "up", [0, 0, 1])), fov=get(c, "fov", 35),
                 ortho=get(c, "ortho", false), half_height=hh)
    k = get(o, "clip", Dict())
    nrm = Float64.(get(k, "normal", [0, 0, 0]))
    clip = Clip(; lo=Tuple(Float64.(get(k, "lo", [0, 0, 0])) .* D), hi=Tuple(Float64.(get(k, "hi", [1, 1, 1])) .* D),
                normal=Tuple(nrm), offset=get(k, "offset", 0.0) * sum(abs.(nrm) .* D))
    w = get(o, "width", 1600)
    (; opt, cam, clip, width=w, height=get(o, "height", w), prefilter=get(o, "prefilter", false))
end
