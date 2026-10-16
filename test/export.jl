using Intmat2: capture, load_capture, Recorder, record!, MiniJSON

@testset "export round trip" begin
    C = 4
    r = random_rule(MersenneTwister(11), C)
    p = Physics{Float32}(alpha_p=0.3f0)
    w = Intmat2.run!(Intmat2.light!(World(CPU(), r, (8, 6, 4); h_scale=0.3, seed=2), r, p), r, p, 20)
    dir = mktempdir()
    man = capture(w, r, p, dir; fields=(:s, :n, :w, :L, :pig, :h), compact=((1, (:s, :pig)), (2, (:s, :pig, :L, :n, :w))))
    for k in ("dims", "C", "dtype", "t", "step", "rule", "physics", "total_mass", "summary", "commit", "compact")
        @test haskey(man, k)
    end
    c = load_capture(dir)
    @test c.fields[:s] == w.s && c.fields[:L] == w.L && c.fields[:n] == w.n && c.fields[:w] == w.w
    @test c.fields[:h] == w.h
    @test c.fields[:pig] ≈ Intmat2.pigment_field(w.h, r)
    @test c.rule == r
    @test c.physics == p
    @test c.manifest["step"] == 20 && c.manifest["total_mass"] ≈ total_mass(w)
    @test filesize(joinpath(dir, "u8", "ds2", "s.u8")) == 4 * 3 * 2
    @test MiniJSON.parsejson(MiniJSON.json(Dict("a" => [1, 2.5, nothing], "b" => "x\"y"))) ==
          Dict("a" => Any[1, 2.5, nothing], "b" => "x\"y")
    rec = Recorder(joinpath(dir, "series"), r, p; every=5)
    for _ in 1:10
        record!(rec, w)
        w = step!(w, r, p)
    end
    sj = MiniJSON.parsejson(read(joinpath(dir, "series", "series.json"), String))
    @test [f["step"] for f in sj["frames"]] == [20, 25]
end
