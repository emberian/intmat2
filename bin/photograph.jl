#!/usr/bin/env julia
# Photograph a capture with its own light field:
#   julia --project=env/metal bin/photograph.jl capturedir optics.toml out.png
#   julia --project=env/metal bin/photograph.jl --series rundir optics.toml framesdir
#
# optics.toml: top-level width, height, backend ("cpu"|"metal"), prefilter (bool);
#   [optics]  any Optics field (alpha_s/alpha_p default to the capture's Physics)
#   [camera]  pos, target as FRACTIONS of (Nx,Ny,Nz); up; fov (deg); ortho; half_height (fraction of Nz)
#   [clip]    lo, hi (fractions), normal, offset (fraction of the slab diagonal along normal)
using Intmat2, TOML, KernelAbstractions, Printf
using Intmat2: load_capture, render, write_png, MiniJSON

# Metal must be loaded at top level (before `main` runs) so its methods are in the
# world the run executes in; `using` inside a function leaves step! on stale methods.
select_backend(name) = name == "cpu" ? CPU() : name == "metal" ? Main.Metal.MetalBackend() : error("unknown backend $name")

function shoot(be, capdir, o, out)
    cap = load_capture(capdir)
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
