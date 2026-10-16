#!/usr/bin/env julia
# Photograph a capture with its own light field:
#   julia --project=env/metal bin/photograph.jl capturedir optics.toml out.png
#   julia --project=env/metal bin/photograph.jl --series rundir optics.toml framesdir
#
# optics.toml: top-level width, height, backend ("cpu"|"metal");
#   [optics]  any Optics field (alpha_s/alpha_p default to the capture's Physics)
#   [camera]  pos, target as FRACTIONS of (Nx,Ny,Nz); fov (deg); ortho; half_height (fraction of Nz)
#   [clip]    lo, hi (fractions), normal, offset (fraction of the slab diagonal along normal)
using Intmat2, TOML, KernelAbstractions, Printf
using Intmat2: load_capture, Optics, camera, Clip, render, write_png, MiniJSON

# Metal must be loaded at top level (before `main` runs) so its methods are in the
# world the run executes in; `using` inside a function leaves step! on stale methods.
select_backend(name) = name == "cpu" ? CPU() : name == "metal" ? Main.Metal.MetalBackend() : error("unknown backend $name")

function setup(o, cap)
    dims = Float64.(cap.manifest["dims"])
    oc = get(o, "optics", Dict())
    kw = Dict{Symbol,Any}(Symbol(k) => (v isa Vector ? Tuple(Float64.(v)) : Float64(v)) for (k, v) in oc)
    opt = Optics(cap.physics; kw...)
    c = get(o, "camera", Dict())
    pos = Tuple(Float64.(get(c, "pos", [0.5, -1.2, 1.6])) .* dims)
    tgt = Tuple(Float64.(get(c, "target", [0.5, 0.5, 0.45])) .* dims)
    hh = haskey(c, "half_height") ? c["half_height"] * dims[3] : nothing
    cam = camera(pos, tgt; fov=get(c, "fov", 35), ortho=get(c, "ortho", false), half_height=hh)
    k = get(o, "clip", Dict())
    lo = Tuple(Float64.(get(k, "lo", [0, 0, 0])) .* dims)
    hi = Tuple(Float64.(get(k, "hi", [1, 1, 1])) .* dims)
    nrm = Float64.(get(k, "normal", [0, 0, 0]))
    off = get(k, "offset", 0.0) * sum(abs.(nrm) .* dims)
    (; opt, cam, clip=Clip(; lo, hi, normal=Tuple(nrm), offset=off))
end

function shoot(be, capdir, o, out)
    cap = load_capture(capdir)
    st = setup(o, cap)
    up(a) = Intmat2.adapt(be, a)
    f = cap.fields
    t0 = time()
    img = render(up(f[:s]), up(f[:pig]), up(f[:L]), st.cam, st.opt;
                 width=get(o, "width", 1600), height=get(o, "height", get(o, "width", 1600)), clip=st.clip)
    write_png(out, img)
    @printf("%s  step %d  %.2fs  mean %.3f\n", out, cap.manifest["step"], time() - t0, sum(img) / length(img))
end

function main(args)
    if args[1] == "--series"
        rundir, optf, framedir = args[2:4]
        o = TOML.parsefile(optf); be = select_backend(get(o, "backend", "metal"))
        mkpath(framedir)
        series = MiniJSON.parsejson(read(joinpath(rundir, "series.json"), String))
        for (i, fr) in enumerate(series["frames"])
            shoot(be, joinpath(rundir, fr["dir"]), o, joinpath(framedir, @sprintf("frame%04d.png", i)))
        end
    else
        capdir, optf, out = args
        o = TOML.parsefile(optf)
        shoot(select_backend(get(o, "backend", "metal")), capdir, o, out)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    get(TOML.parsefile(ARGS[end-1]), "backend", "metal") == "metal" && @eval using Metal
    Base.invokelatest(main, ARGS)
end
