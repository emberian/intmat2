#!/usr/bin/env julia
# Photograph a capture with its own light field:
#   julia --project=env/metal bin/photograph.jl capturedir optics.toml out.png
#   julia --project=env/metal bin/photograph.jl --series rundir optics.toml framesdir
#   julia --project=env/metal bin/photograph.jl --many capturedir outdir a.toml b.toml ...
#
# optics.toml: top-level width, height, backend ("cpu"|"metal"), prefilter (bool);
#   [optics]  any Optics field (alpha_s/alpha_p default to the capture's Physics);
#             pig_floor may be "q0.9", a quantile of the capture's pigment
#   [camera]  pos, target as FRACTIONS of (Nx,Ny,Nz); up; fov (deg); ortho; half_height (fraction of Nz)
#   [clip]    lo, hi (fractions), normal, offset (fraction of the slab diagonal along normal)
using Intmat2, TOML, KernelAbstractions, Printf
using Intmat2: load_capture, render, write_png, MiniJSON

# Metal must be loaded at top level (before `main` runs) so its methods are in the
# world the run executes in; `using` inside a function leaves step! on stale methods.
select_backend(name) = name == "cpu" ? CPU() : name == "metal" ? Main.Metal.MetalBackend() : error("unknown backend $name")

# `pig_floor = "q0.9"` means the 0.9 quantile of this capture's pigment field; `s_floor`
# takes the same form for solid.
# "mid" means (q0.1 + q0.9)/2, the threshold the region tracker in src/evolve.jl uses.
function resolve_quantiles(o, fields)
    o2 = deepcopy(o)
    op = get(o2, "optics", Dict())
    for (key, f) in (("pig_floor", :pig), ("s_floor", :s))
        v = get(op, key, 0.0)
        v isa AbstractString || continue
        sorted = sort(vec(fields[f]))
        qt(q) = Float64(sorted[clamp(round(Int, q * length(sorted)), 1, length(sorted))])
        op[key] = v == "mid" ? (qt(0.1) + qt(0.9)) / 2 : qt(parse(Float64, lstrip(v, 'q')))
    end
    o2
end

function shoot(be, capdir, o, out)
    cap = load_capture(capdir)
    o = resolve_quantiles(o, cap.fields)
    st = Intmat2.photo_setup(o, cap.manifest["dims"], cap.physics)
    up(a) = Intmat2.adapt(be, a)
    f = cap.fields
    t0 = time()
    img = render(up(f[:s]), up(f[:pig]), up(f[:L]), st.cam, st.opt; width=st.width, height=st.height, clip=st.clip,
                 prefilter=st.prefilter)
    write_png(out, img)
    @printf("%s  step %d  %.2fs  mean %.3f\n", out, cap.manifest["step"], time() - t0, sum(img) / length(img))
end

function main(args)
    if args[1] == "--many"
        capdir, outdir = args[2:3]; mkpath(outdir)
        be = select_backend(get(TOML.parsefile(args[4]), "backend", "metal"))
        for optf in args[4:end]
            shoot(be, capdir, TOML.parsefile(optf), joinpath(outdir, splitext(basename(optf))[1] * ".png"))
        end
    elseif args[1] == "--series"
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
    get(TOML.parsefile(ARGS[1] == "--many" ? ARGS[4] : ARGS[end-1]), "backend", "metal") == "metal" && @eval using Metal
    Base.invokelatest(main, ARGS)
end
