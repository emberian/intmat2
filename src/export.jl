# Captures: raw little-endian fields + manifest.json + UInt8 compact copies for the browser.
#
#   dir/manifest.json        dims, C, t, step, rule, physics, mass, summary, commit, files
#   dir/<f>.f32              Float32 (Nx,Ny,Nz) for s n w L pig; h is (C,Nx,Ny,Nz)
#   dir/u8/ds<k>/<f>.u8      box-averaged by k, quantized to 0..255 over [0, hi]
#
# A `Recorder` writes one capture per `every` steps into dir/NNNNNN/ and keeps
# dir/series.json (the frame list with summaries) current.

export capture, load_capture, Recorder, record!, pigment_field

# ---------------------------------------------------------------- minimal JSON

module MiniJSON
export json, parsejson

json(x) = (io = IOBuffer(); write_json(io, x, 0); String(take!(io)))

write_json(io, ::Nothing, d) = print(io, "null")
write_json(io, x::Bool, d) = print(io, x ? "true" : "false")
write_json(io, x::Integer, d) = print(io, x)
write_json(io, x::AbstractFloat, d) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
write_json(io, x::Union{AbstractString,Symbol}, d) = escape(io, string(x))
function escape(io, s)
    print(io, '"')
    for c in s
        c == '"' ? print(io, "\\\"") : c == '\\' ? print(io, "\\\\") : c == '\n' ? print(io, "\\n") :
        c < ' ' ? print(io, "\\u", string(UInt16(c), base=16, pad=4)) : print(io, c)
    end
    print(io, '"')
end
function write_json(io, x::Union{AbstractVector,Tuple}, d)
    flat = all(v -> v isa Union{Number,Nothing,AbstractString}, x)
    print(io, '[')
    for (i, v) in enumerate(x)
        i > 1 && print(io, flat ? ", " : ",")
        flat || print(io, '\n', "  "^(d + 1))
        write_json(io, v, d + 1)
    end
    flat || isempty(x) || print(io, '\n', "  "^d)
    print(io, ']')
end
function write_pairs(io, ps, d)
    print(io, '{')
    for (i, (k, v)) in enumerate(ps)
        i > 1 && print(io, ',')
        print(io, '\n', "  "^(d + 1)); escape(io, string(k)); print(io, ": ")
        write_json(io, v, d + 1)
    end
    isempty(ps) || print(io, '\n', "  "^d)
    print(io, '}')
end
write_json(io, x::AbstractDict, d) = write_pairs(io, collect(pairs(x)), d)
write_json(io, x::NamedTuple, d) = write_pairs(io, collect(pairs(x)), d)

function parsejson(s::AbstractString)
    s = String(s); i = Ref(1)
    v = pvalue(s, i); skipws(s, i)
    i[] <= ncodeunits(s) && error("trailing JSON at $(i[])")
    v
end
function skipws(s, i)
    while i[] <= ncodeunits(s) && s[i[]] in (' ', '\n', '\r', '\t')
        i[] += 1
    end
end
function pvalue(s, i)
    skipws(s, i); c = s[i[]]
    if c == '{'
        d = Dict{String,Any}(); i[] += 1; skipws(s, i)
        s[i[]] == '}' && (i[] += 1; return d)
        while true
            skipws(s, i); k = pstring(s, i); skipws(s, i)
            s[i[]] == ':' || error("expected :"); i[] += 1
            d[k] = pvalue(s, i); skipws(s, i)
            s[i[]] == ',' ? (i[] += 1) : s[i[]] == '}' ? (i[] += 1; return d) : error("expected , or }")
        end
    elseif c == '['
        a = Any[]; i[] += 1; skipws(s, i)
        s[i[]] == ']' && (i[] += 1; return a)
        while true
            push!(a, pvalue(s, i)); skipws(s, i)
            s[i[]] == ',' ? (i[] += 1) : s[i[]] == ']' ? (i[] += 1; return a) : error("expected , or ]")
        end
    elseif c == '"'
        return pstring(s, i)
    elseif startswith(SubString(s, i[]), "true")
        i[] += 4; return true
    elseif startswith(SubString(s, i[]), "false")
        i[] += 5; return false
    elseif startswith(SubString(s, i[]), "null")
        i[] += 4; return nothing
    else
        j = i[]
        while i[] <= ncodeunits(s) && s[i[]] in "+-0123456789.eE"
            i[] += 1
        end
        t = s[j:i[]-1]
        return occursin(r"[.eE]", t) ? parse(Float64, t) : parse(Int, t)
    end
end
function pstring(s, i)
    s[i[]] == '"' || error("expected string at $(i[])"); i[] += 1
    io = IOBuffer()
    while (c = s[i[]]) != '"'
        if c == '\\'
            i[] += 1; e = s[i[]]
            if e == 'u'
                write(io, Char(parse(UInt16, s[i[]+1:i[]+4], base=16))); i[] += 4
            else
                write(io, e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e)
            end
        else
            write(io, c)
        end
        i[] = nextind(s, i[])
    end
    i[] += 1
    String(take!(io))
end
end # MiniJSON
using .MiniJSON

# ---------------------------------------------------------------- helpers

"""`pigment_field(h, rule)`: σ(v_pig·h) per voxel on the host, as the light kernel sees it."""
function pigment_field(h::AbstractArray{T,4}, rule::Rule{C}) where {T,C}
    hh = Array(h)
    v = Float64.(rule.vpig)
    pig = Array{Float32}(undef, size(hh)[2:4])
    @inbounds for I in CartesianIndices(pig)
        a = 0.0
        for c in 1:C
            a += v[c] * hh[c, I]
        end
        pig[I] = Float32(1 / (1 + exp(-a)))
    end
    pig
end

function producing_commit()
    root = pkgdir(@__MODULE__)
    try
        sha = readchomp(`git -C $root rev-parse HEAD`)
        dirty = !isempty(readchomp(`git -C $root status --porcelain --untracked-files=no`))
        dirty ? sha * "-dirty" : sha
    catch
        "unknown"
    end
end

function write_raw(path, a::AbstractArray{Float32})
    open(path, "w") do io
        write(io, ENDIAN_BOM == 0x04030201 ? a : htol.(a))
    end
end
function read_raw(path, dims)
    a = Array{Float32}(undef, dims...)
    open(io -> read!(io, a), path)
    ENDIAN_BOM == 0x04030201 ? a : ltoh.(a)
end

function downsample(a::Array{Float32,3}, k::Int)
    k == 1 && return a
    Nx, Ny, Nz = size(a) .÷ k
    out = zeros(Float32, Nx, Ny, Nz)
    @inbounds for kk in 1:Nz, jj in 1:Ny, ii in 1:Nx
        acc = 0f0
        for c in 0:k-1, b in 0:k-1, q in 0:k-1
            acc += a[(ii-1)*k+q+1, (jj-1)*k+b+1, (kk-1)*k+c+1]
        end
        out[ii, jj, kk] = acc / k^3
    end
    out
end

quantize(a, hi) = UInt8[round(UInt8, clamp(x / hi, 0f0, 1f0) * 255) for x in a]

physics_dict(p::Physics) = Dict(string(f) => Float64(getfield(p, f)) for f in fieldnames(Physics))

# ---------------------------------------------------------------- capture

"""
    capture(wd, rule, p, dir; fields=(:s,:n,:w,:L,:pig), compact=((1,(:s,:pig)), (2,ALL5), (4,ALL5)), meta=Dict())

Write the world (any backend; copied to the host first) as raw Float32 fields plus
`manifest.json`. `:pig` is σ(v_pig·h) computed with `rule`; add `:h` for the full
activity state (C× larger). `compact` lists `(downsample, fields)` UInt8 copies.
Returns the manifest.
"""
function capture(wd::World{C}, rule::Rule{C}, p::Physics, dir::AbstractString;
                 fields=(:s, :n, :w, :L, :pig),
                 compact=((1, (:s, :pig)), (2, (:s, :pig, :L, :n, :w)), (4, (:s, :pig, :L, :n, :w))),
                 meta=Dict{String,Any}()) where {C}
    mkpath(dir)
    host = Dict{Symbol,Array{Float32}}()
    get_field(f) = get!(host, f) do
        f === :pig ? pigment_field(wd.h, rule) :
        f === :h ? Float32.(Array(wd.h)) : Float32.(Array(getfield(wd, f)))
    end
    files = Dict{String,Any}()
    for f in fields
        a = get_field(f)
        write_raw(joinpath(dir, "$f.f32"), a)
        files[string(f)] = Dict("file" => "$f.f32", "dims" => collect(size(a)), "dtype" => "float32le")
    end
    hi_of(f) = f in (:s, :pig) ? 1.0f0 : f === :L ? Float32(p.L0) : max(maximum(get_field(f)), eps(Float32))
    comp = Any[]
    for (k, fs) in compact
        sub = joinpath("u8", "ds$k"); mkpath(joinpath(dir, sub))
        entry = Dict{String,Any}("ds" => k, "dims" => collect(size(wd) .÷ k), "fields" => Dict{String,Any}())
        for f in fs
            hi = hi_of(f)
            write(joinpath(dir, sub, "$f.u8"), quantize(downsample(get_field(f), k), hi))
            entry["fields"][string(f)] = Dict("file" => "$sub/$f.u8", "lo" => 0.0, "hi" => Float64(hi))
        end
        push!(comp, entry)
    end
    sm = summarize(wd)
    man = Dict{String,Any}(
        "format" => "intmat2-capture-1",
        "dims" => collect(size(wd)), "C" => C, "dtype" => "float32le", "order" => "column-major (x fastest); h is (C,Nx,Ny,Nz); k=Nz is the lit top",
        "t" => Float64(wd.t), "step" => wd.step,
        "rule" => Float64.(vec(rule)), "physics" => physics_dict(p),
        "total_mass" => total_mass(wd), "summary" => sm,
        "commit" => producing_commit(), "files" => files, "compact" => comp)
    merge!(man, meta)
    write(joinpath(dir, "manifest.json"), json(man))
    man
end

"""
    load_capture(dir) -> (; manifest, fields::Dict{Symbol,Array{Float32}}, rule, physics)

Read a capture written by `capture`. `rule` is a `Rule{C,Float32}`, `physics` a `Physics{Float32}`.
"""
function load_capture(dir::AbstractString)
    man = parsejson(read(joinpath(dir, "manifest.json"), String))
    fields = Dict{Symbol,Array{Float32}}()
    for (f, d) in man["files"]
        fields[Symbol(f)] = read_raw(joinpath(dir, d["file"]), Int.(d["dims"]))
    end
    C = man["C"]
    rule = Rule{C,Float32}(Float32.(man["rule"]))
    physics = Physics{Float32}(; (Symbol(k) => Float32(v) for (k, v) in man["physics"])...)
    (; manifest=man, fields, rule, physics)
end

# ---------------------------------------------------------------- recorder

"""
    Recorder(dir, rule, p; every=100, fields=(:s,:n,:w,:L,:pig), meta=Dict())

`record!(rec, wd)` captures when `wd.step % every == 0` into `dir/NNNNNN/` and
rewrites `dir/series.json`. Returns `true` if it captured.
"""
mutable struct Recorder{R,P}
    dir::String
    rule::R
    physics::P
    every::Int
    fields::Tuple
    meta::Dict{String,Any}
    frames::Vector{Any}
end
Recorder(dir, rule, p; every=100, fields=(:s, :n, :w, :L, :pig), meta=Dict{String,Any}()) =
    (mkpath(dir); Recorder(String(dir), rule, p, every, Tuple(fields), Dict{String,Any}(meta), Any[]))

function record!(rec::Recorder, wd::World; force=false)
    (force || wd.step % rec.every == 0) || return false
    any(f -> f["step"] == wd.step, rec.frames) && return false
    name = @sprintf("%06d", wd.step)
    man = capture(wd, rec.rule, rec.physics, joinpath(rec.dir, name); fields=rec.fields)
    push!(rec.frames, Dict{String,Any}("step" => wd.step, "t" => Float64(wd.t), "dir" => name,
                                       "summary" => man["summary"], "total_mass" => man["total_mass"]))
    series = Dict{String,Any}("format" => "intmat2-series-1", "dims" => collect(size(wd)),
                              "C" => channels(wd), "rule" => Float64.(vec(rec.rule)),
                              "physics" => physics_dict(rec.physics), "every" => rec.every,
                              "commit" => man["commit"], "frames" => rec.frames)
    merge!(series, rec.meta)
    write(joinpath(rec.dir, "series.json"), json(series))
    true
end

# ---------------------------------------------------------------- browser volumes

"""
    viewer_volume(rundir, outdir; ds=1, lo=(1,1,1), extent=nothing, max_frames=16,
                  fields=(:s,:pig,:L,:n,:w), series_fields=(:s,:pig,:L))

Stage a Recorder run for `web/viewer.html`: a box `lo .+ (0:extent-1)` (default the
whole slab), box-averaged by `ds`, quantized to UInt8. Every chosen frame carries
`series_fields`; the last carries all `fields`. Writes `outdir/index.json` and
`outdir/fNNN_<field>.u8`. Each field's `hi` is its maximum over the staged frames
(`s`, `pig` use 1, `L` uses `L0`).
"""
function viewer_volume(rundir, outdir; ds=1, lo=(1, 1, 1), extent=nothing, max_frames=16,
                       fields=(:s, :pig, :L, :n, :w), series_fields=(:s, :pig, :L))
    series = parsejson(read(joinpath(rundir, "series.json"), String))
    frames = series["frames"]
    idx = unique(round.(Int, range(1, length(frames), length=min(max_frames, length(frames)))))
    dims = Int.(series["dims"])
    ext = something(extent, Tuple(dims .- lo .+ 1))
    ext = ext isa Integer ? (ext, ext, ext) : Tuple(ext)
    rng = ntuple(d -> lo[d]:lo[d]+ext[d]-1, 3)
    L0 = Float64(series["physics"]["L0"])
    mkpath(outdir)
    loaded = Dict{Tuple{Int,Symbol},Array{Float32,3}}()
    for (n, i) in enumerate(idx)
        fs = n == length(idx) ? fields : series_fields
        c = load_capture(joinpath(rundir, frames[i]["dir"]))
        for f in fs
            loaded[(n, f)] = downsample(c.fields[f][rng...], ds)
        end
    end
    hi = Dict(f => (f in (:s, :pig) ? 1.0 : f === :L ? L0 :
                    maximum(maximum(a) for ((_, g), a) in loaded if g == f; init=1e-6)) for f in fields)
    out = Any[]
    for (n, i) in enumerate(idx)
        fr = frames[i]
        entry = Dict{String,Any}("step" => fr["step"], "t" => fr["t"], "summary" => fr["summary"], "fields" => Dict{String,Any}())
        for f in fields
            haskey(loaded, (n, f)) || continue
            name = @sprintf("f%03d_%s.u8", n, f)
            write(joinpath(outdir, name), quantize(loaded[(n, f)], Float32(hi[f])))
            entry["fields"][string(f)] = name
        end
        push!(out, entry)
    end
    vdims = collect(size(first(values(loaded))))
    write(joinpath(outdir, "index.json"), json(Dict{String,Any}(
        "format" => "intmat2-viewer-1", "dims" => vdims, "ds" => ds, "source_dims" => dims,
        "lo" => collect(lo), "extent" => collect(ext), "hi" => Dict(string(k) => v for (k, v) in hi),
        "physics" => series["physics"], "commit" => series["commit"], "frames" => out)))
    outdir
end
