# Alternative rule passing for batches, measured against the device rules matrix: the B
# rules as a by-value kernel argument (NTuple of flattened SVectors), staged into
# threadgroup memory by dynamic tuple indexing. Not used by src/batch.jl.
# usage: julia --project=env/metal bench/batch_tuple.jl [metal|rocm]
be_name = get(ARGS, 1, "metal")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
using Intmat2: world, nworlds, device_rules, update_voxel!, voxel_coords, update_workgroup, light_kernel!, light_workgroup
using Intmat2.StaticArrays
@kernel unsafe_indices=true function tuple_update!(h2, s2, n2, w2, L, @Const(h), @Const(s), @Const(n), @Const(w),
        θs, ::Val{C}, fm::Val, p::Physics{T}, Nx::Int32, Ny::Int32, Nz::Int32, B::Int32) where {C,T}
    lθ = @localmem T (2 * (3C * C + 8C),)
    P = Int32(3C * C + 8C)
    q = (@index(Global, Linear) - 1) % Int32
    li = (@index(Local, Linear) - 1) % Int32
    V = Nx * Ny * Nz
    gs = Int32(@groupsize()[1])
    b0 = min((q - li) ÷ V, B - Int32(1))
    t = li
    while t < Int32(2) * P
        hi = t >= P
        bt = min(b0 + (hi ? Int32(1) : Int32(0)), B - Int32(1))
        @inbounds lθ[t + Int32(1)] = θs[bt + Int32(1)][(hi ? t - P : t) + Int32(1)]
        t += gs
    end
    @synchronize
    P = Int32(3C * C + 8C)
    q = (@index(Global, Linear) - 1) % Int32
    li = (@index(Local, Linear) - 1) % Int32
    b0 = min((q - li) ÷ (Nx * Ny * Nz), B - Int32(1))
    if q < Nx * Ny * Nz * B
        b, i, j, k = voxel_coords(q, Nx, Ny, Nz)
        update_voxel!(h2, s2, n2, w2, L, h, s, n, w, lθ, (b - b0) * P, Val(C), fm, true, p, Nx, Ny, Nz, q, i, j, k)
    end
end
function tstep!(wb, θs, p)
    N = Int32(size(wb)[1]); B = nworlds(wb); wg = update_workgroup(be); tot = Int(N)^3 * B
    v = vec
    tuple_update!(be, wg)(v(wb.h2), v(wb.s2), v(wb.n2), v(wb.w2), v(wb.L), v(wb.h), v(wb.s), v(wb.n), v(wb.w),
                          θs, Val(8), Val(true), p, N, N, N, Int32(B); ndrange=cld(tot, wg) * wg)
    light_kernel!(be, light_workgroup(be))(vec(wb.L), p, N, N, N; ndrange=Int(N)^2 * B)
    typeof(wb)(wb.h2, wb.s2, wb.n2, wb.w2, wb.L, wb.h, wb.s, wb.n, wb.w, wb.t + p.dt, wb.step + 1)
end
p = gate_physics()
for N in (32,), B in (4, 16, 64)
    rs = [random_rule(MersenneTwister(100 + b), 8, Float32; self=1.0, nb=1.0) for b in 1:B]
    θ = device_rules(be, rs); θs = Tuple(SVector{256,Float32}(collect(vec(r))) for r in rs)
    wa = light!(WorldBatch(be, 8, (N, N, N), B; seeds=1:B, h_scale=0.3, s0=0.3), θ, p); wt = copy(wa)
    n = max(20, round(Int, 3e8 / (N^3 * B)))
    try
        wt = tstep!(wt, θs, p); wa = step!(wa, θ, p)
        same = Array(wt.h) == Array(wa.h)
        bm = 0.0; bt = 0.0
        for _ in 1:3
            t = @elapsed (wa = run!(wa, θ, p, n)); bm = max(bm, B * n / t)
            t = @elapsed (for _ in 1:n; wt = tstep!(wt, θs, p); end; KA.synchronize(be)); bt = max(bt, B * n / t)
        end
        @printf("N=%d B=%-3d rules matrix %9.1f  rules tuple %9.1f worlds·steps/s  (tuple bitwise=%s)\n", N, B, bm, bt, same)
    catch e
        @printf("N=%d B=%-3d rules tuple failed: %s\n", N, B, first(sprint(showerror, e), 160))
    end
end
