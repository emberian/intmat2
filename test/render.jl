using Intmat2: Optics, Camera, Clip, camera, render, render_world, write_png

@testset "renderer" begin
    N = 16
    z = zeros(Float32, N, N, N)
    opt = Optics(; alpha_s=2.0, alpha_p=2.0, background=(0.1, 0.2, 0.3), gamma=1.0)
    cam = camera((N / 2, -3N, N / 2 + 10), (N / 2, N / 2, N / 2); fov=40)
    img = render(z, z, ones(Float32, N, N, N), cam, opt; width=24, height=20)
    @test size(img) == (3, 24, 20)
    @test all(img[1, :, :] .≈ 0.1f0) && all(img[3, :, :] .≈ 0.3f0)

    # Opaque slab, light from the simulation's own column march.
    C = 2
    r = Rule{C,Float32}(zeros(Float32, 2C * C + 8C))
    p = Physics{Float32}(alpha_s=2.0f0, alpha_p=0.0f0)
    w = World(CPU(), r, (N, N, N); s0=0.0, h_scale=0.0)
    w.s .= 1
    w = Intmat2.light!(w, r, p)
    side = camera((N / 2, -4N, N / 2), (N / 2, N / 2, N / 2); ortho=true, half_height=N / 2 + 1)
    img = render_world(w, r, side, Optics(p; background=(0, 0, 0)); width=32, height=32)
    @test all(isfinite, img) && all(0 .<= img .<= 1)
    top = img[:, 8:24, 3]; bottom = img[:, 8:24, 30]
    @test minimum(top) > 0.2
    @test maximum(bottom) < 0.02
    # a cut plane removes the lit top; the exposed interior is unlit
    img2 = render_world(w, r, side, Optics(p; background=(0, 0, 0)); width=32, height=32, clip=Clip(normal=(0, 0, 1), offset=N / 2))
    @test maximum(img2[:, 8:24, 3]) == 0 && maximum(img2[:, 8:24, 18]) < 0.02
    f = tempname() * ".png"
    write_png(f, img)
    @test read(f)[2:4] == UInt8['P', 'N', 'G'] && filesize(f) > 3 * 32 * 32
end
