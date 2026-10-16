#!/usr/bin/env julia
# Film a run without storing captures: run the recipe on its backend and render the
# live world every `every` steps with optics.toml, then encode with ffmpeg if present.
#   julia --project=env/metal bin/film.jl recipe.toml optics.toml outdir every=10 [key=value ...]
# Extra key=value pairs override recipe fields as in bin/run.jl. Optional
# `orbit=deg` turns the camera that many degrees about the vertical axis over the film.
include(joinpath(@__DIR__, "run.jl"))
using Intmat2: render_world, write_png, photo_setup, camera

function film(recipe, optf, outdir, overrides)
    kv = Dict(overrides)
    every = Int(pop!(kv, "every", 10)); orbit = Float64(pop!(kv, "orbit", 0.0))
    rc = load_recipe(recipe, collect(Pair{String,Any}, kv)); r = rc.raw
    be = select_backend(r["backend"])
    o = TOML.parsefile(optf)
    mkpath(outdir)
    cp(recipe, joinpath(outdir, "recipe.toml"); force=true); cp(optf, joinpath(outdir, "optics.toml"); force=true)
    wd = World(be, rc.rule, rc.dims; n0=get(r, "n0", 1.0), h_scale=r["h_scale"], s0=get(r, "s0", 0.0), seed=rc.seed)
    wd = light!(wd, rc.rule, rc.p)
    st = photo_setup(o, rc.dims, rc.p)
    centre = collect(rc.dims) ./ 2
    nframes = r["steps"] ÷ every
    log = open(joinpath(outdir, "film.log"), "w")
    for f in 0:nframes
        cam = st.cam
        if orbit != 0
            θ = deg2rad(orbit) * f / max(nframes, 1)
            rel = Float64.(st.cam.pos) .- centre
            rp = (cos(θ) * rel[1] - sin(θ) * rel[2], sin(θ) * rel[1] + cos(θ) * rel[2], rel[3])
            tg = Float64.(st.cam.pos) .+ Float64.(st.cam.fwd)   # direction preserved relative to orbit
            rt = tg .- centre
            tr = (cos(θ) * rt[1] - sin(θ) * rt[2], sin(θ) * rt[1] + cos(θ) * rt[2], rt[3])
            cam = camera(Tuple(centre .+ rp), Tuple(centre .+ tr); fov=get(get(o, "camera", Dict()), "fov", 35))
        end
        img = render_world(wd, rc.rule, cam, st.opt; width=st.width, height=st.height, clip=st.clip, prefilter=st.prefilter)
        write_png(joinpath(outdir, @sprintf("frame%05d.png", f)), img)
        s = summarize(wd)
        @printf(log, "%d %d %.5f %.5f %.5f %.5f\n", f, s.step, s.mean_s, s.std_s, s.light_bottom, s.mean_w); flush(log)
        f % 20 == 0 && @printf("frame %d step %d mean_s %.4f light_bottom %.4f\n", f, s.step, s.mean_s, s.light_bottom)
        f == nframes && break
        for _ in 1:every
            wd = step!(wd, rc.rule, rc.p)
        end
    end
    close(log)
    if Sys.which("ffmpeg") !== nothing
        run(`ffmpeg -loglevel error -y -framerate 30 -i $(joinpath(outdir, "frame%05d.png")) -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart $(joinpath(outdir, "film.mp4"))`)
        println("wrote ", joinpath(outdir, "film.mp4"))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    ov = Pair{String,Any}[parse_override(a) for a in ARGS[4:end]]
    load_recipe(ARGS[1], filter(p -> !(first(p) in ("every", "orbit")), ov)).raw["backend"] == "metal" && @eval using Metal
    Base.invokelatest(film, ARGS[1], ARGS[2], ARGS[3], ov)
end
