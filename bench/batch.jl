# WorldBatch on a GPU: equality with worlds stepped alone, then aggregate worlds·steps/s
# for B ∈ {1,4,16,64} at 32³ and 64³, with per-world rules (device matrix) and a
# shared rule, against B separate Worlds stepped one after another.
# usage: julia --project=env/metal bench/batch.jl [metal|rocm] [out.json]
be_name = get(ARGS, 1, "metal")
if be_name == "metal"; using Metal; be = MetalBackend()
else; using AMDGPU; be = ROCBackend(); end
include(joinpath(@__DIR__, "common.jl"))
using Intmat2: world, nworlds, device_rules
out = get(ARGS, 2, "")
p = gate_physics()
rules_for(B) = [random_rule(MersenneTwister(100 + b), 8, Float32; self=1.0, nb=1.0) for b in 1:B]

# equality: batch of 5 (odd grid, groups straddle worlds) vs 5 worlds alone, 30 steps
let B = 5, dims = (24, 20, 18), rs = rules_for(5)
    wb = run!(light!(WorldBatch(be, 8, dims, B; seeds=1:B, h_scale=0.3, s0=0.3), rs, p), rs, p, 30)
    worst = 0.0; bitwise = true
    for b in 1:B
        w = run!(light!(World(be, 8, dims; seed=b, h_scale=0.3, s0=0.3), rs[b], p), rs[b], p, 30)
        a = KA.adapt(CPU(), world(wb, b)); c = KA.adapt(CPU(), w)
        for f in (:h, :s, :n, :w, :L)
            worst = max(worst, maximum(abs.(getfield(a, f) .- getfield(c, f))))
            bitwise &= getfield(a, f) == getfield(c, f)
        end
    end
    @printf("batch vs alone (%s, B=5, 24x20x18, 30 steps): max|Δ| = %.2e bitwise=%s\n", be_name, worst, bitwise)
end

rows = []
for N in (32, 64), B in (1, 4, 16, 64)
    rs = rules_for(B); θ = device_rules(be, rs)
    n = max(20, round(Int, 3e8 / (N^3 * B)))
    wb = run!(light!(WorldBatch(be, 8, (N, N, N), B; seeds=1:B, h_scale=0.3, s0=0.3), θ, p), θ, p, 3)
    best_m = 0.0; best_s = 0.0
    for _ in 1:3
        t = @elapsed (wb = run!(wb, θ, p, n)); best_m = max(best_m, B * n / t)
        t = @elapsed (wb = run!(wb, rs[1], p, n)); best_s = max(best_s, B * n / t)
    end
    # B separate worlds, stepped round-robin without per-step sync
    ws = [run!(light!(World(be, 8, (N, N, N); seed=b, h_scale=0.3, s0=0.3), rs[b], p), rs[b], p, 3) for b in 1:min(B, 16)]
    nsep = max(5, n ÷ 4)
    t = @elapsed begin
        for _ in 1:nsep, b in eachindex(ws)
            ws[b] = step!(ws[b], rs[b], p; sync=false)
        end
        KA.synchronize(be)
    end
    sep = length(ws) * nsep / t
    wb = nothing; ws = nothing; GC.gc()
    @printf("N=%-3d B=%-3d batch(rules matrix) %9.1f  batch(shared rule) %9.1f  separate worlds %9.1f  worlds·steps/s\n",
            N, B, best_m, best_s, sep)
    push!(rows, (N=N, B=B, matrix=round(best_m; digits=1), shared=round(best_s; digits=1), separate=round(sep; digits=1)))
end
if !isempty(out)
    open(out, "w") do io
        println(io, "{\"command\": \"julia --project=env/", be_name == "metal" ? "metal" : "amd", " bench/batch.jl ", join(ARGS, " "),
                "\", \"dtype\": \"Float32\", \"C\": 8, \"unit\": \"worlds*steps/s, best of 3, fast math\",")
        println(io, " \"rows\": [", join(["{\"N\": $(r.N), \"B\": $(r.B), \"batch_rules_matrix\": $(r.matrix), \"batch_shared_rule\": $(r.shared), \"separate_worlds\": $(r.separate)}" for r in rows], ",\n   "), "]}")
    end
end
