#!/usr/bin/env julia
# julia --project=env/metal bin/regions_photo.jl capturedir outdir optics.toml ...
# Lit by the simulation's own L; the solid shown is the tracker's mask (s > (q10+q90)/2);
# the network (largest component) is pale neutral, every other component is tinted.
# The tint is a label, not pigment.
using Intmat2, Metal, TOML
E = Intmat2
cap, out = ARGS[1], ARGS[2]; optfs = ARGS[3:end]
c = E.load_capture(cap); s = c.fields[:s]; L = c.fields[:L]
m = E.region_mask(s); lab, sizes = E.label_components(m; min_size=8)
big = Int32(argmax(sizes))
sd = Float32.(s .* m)
frag = Float32.((lab .!= 0) .& (lab .!= big))
mkpath(out)
for f in optfs
    o = TOML.parsefile(f)
    st = E.photo_setup(o, size(s), c.physics)
    up(a) = E.adapt(Metal.MetalBackend(), a)
    img = E.render(up(sd), up(frag), up(L), st.cam, st.opt; width=st.width, height=st.height, clip=st.clip)
    E.write_png(joinpath(out, splitext(basename(f))[1] * ".png"), img)
    println(f)
end
