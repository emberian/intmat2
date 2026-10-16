"""Total conserved mass `Σ (n + s + w)`. Computed in Float64 on the host-side reduction."""
total_mass(wd::World) = Float64(sum(wd.n)) + Float64(sum(wd.s)) + Float64(sum(wd.w))

"""
    summarize(wd) -> NamedTuple

Cheap whole-world statistics: mass, mean/std of each scalar field, mean |h|,
fraction of voxels with solid above 0.5, light reaching the bottom face.
"""
function summarize(wd::World{C,T}) where {C,T}
    Nx, Ny, Nz = size(wd)
    nv = Nx * Ny * Nz
    μs = Float64(sum(wd.s)) / nv
    σs = sqrt(max(Float64(sum(abs2, wd.s)) / nv - μs^2, 0.0))
    μn = Float64(sum(wd.n)) / nv
    μw = Float64(sum(wd.w)) / nv
    μh = Float64(sum(abs, wd.h)) / (nv * C)
    σh = sqrt(max(Float64(sum(abs2, wd.h)) / (nv * C) - (Float64(sum(wd.h)) / (nv * C))^2, 0.0))
    solid_frac = Float64(count(>(T(0.5)), wd.s)) / nv
    Lbottom = Float64(sum(view(wd.L, :, :, 1))) / (Nx * Ny)
    (t=Float64(wd.t), step=wd.step, mass=μn + μs + μw, mean_s=μs, std_s=σs, mean_n=μn,
     mean_w=μw, mean_abs_h=μh, std_h=σh, solid_frac=solid_frac, light_bottom=Lbottom)
end

function Base.show(io::IO, ::MIME"text/plain", wd::World{C,T}) where {C,T}
    Nx, Ny, Nz = size(wd)
    s = summarize(wd)
    @printf(io, "World{%d,%s} %d×%d×%d on %s  t=%.3f step=%d\n", C, T, Nx, Ny, Nz,
            nameof(typeof(backend(wd))), s.t, s.step)
    @printf(io, "  mass/voxel %.6f  s %.4f±%.4f (frac>0.5 %.3f)  n %.4f  w %.4f  |h| %.4f±%.4f  L_bottom %.4f",
            s.mass, s.mean_s, s.std_s, s.solid_frac, s.mean_n, s.mean_w, s.mean_abs_h, s.std_h, s.light_bottom)
end
