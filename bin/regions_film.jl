#!/usr/bin/env julia
# julia --project=. bin/regions_film.jl rundir outdir   (then ffmpeg -framerate 12 -i outdir/frame%04d.png ...)
# Works on solid-only captures (no L or h needed): it shows the tracker mask, not light.
# Diagnostic film: mid-depth slab (z 21..28) seen from above. Grey = the largest connected
# component of the tracker's mask (s > (q10+q90)/2), orange = every other component (>= 8 voxels).
using Intmat2, Printf
rundir, out = ARGS; mkpath(out)
E = Intmat2
series = E.MiniJSON.parsejson(read(joinpath(rundir, "series.json"), String))
zs = 21:28; up = 4
open(joinpath(out, "counts.txt"), "w") do io
for (n, fr) in enumerate(series["frames"])
    s = E.load_capture(joinpath(rundir, fr["dir"])).fields[:s]
    m = E.region_mask(s)
    lab, sizes = E.label_components(m; min_size=8)
    big = isempty(sizes) ? Int32(0) : Int32(argmax(sizes))
    Nx, Ny, _ = size(s)
    img = zeros(Float32, 3, Nx * up, Ny * up)
    nfrag = 0
    for j in 1:Ny, i in 1:Nx
        net = 0; frag = false
        for k in zs
            l = lab[i, j, k]
            l == 0 && continue
            l == big ? (net += 1) : (frag = true)
        end
        c = frag ? (0.95f0, 0.45f0, 0.15f0) : net > 0 ? (0.25f0 + 0.6f0 * net / length(zs)) .* (1f0, 1f0, 0.97f0) : (0.03f0, 0.03f0, 0.035f0)
        for b in 1:up, a in 1:up, ch in 1:3
            img[ch, (i-1)*up+a, (Ny-j)*up+b] = c[ch]
        end
    end
    E.write_png(joinpath(out, @sprintf("frame%04d.png", n)), img)
    @printf(io, "%d %.1f %d\n", n, fr["t"], length(sizes))
end
end
