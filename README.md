# Interior 2

**Evolving matter as one continuous field.** Every voxel carries recurrent
activity, deposited solid, nutrient, waste and light. Activity deposits solid out
of nutrient; solid changes who the activity can talk to, blocks transport and
shades what lies beneath; light turns waste back into nutrient. Mass is conserved.
A 184-number local rule, shared by every voxel, decides what grows. There are no
agents: a cell is whatever region the dynamics make.

[Design](DESIGN.md) · [Working here](AGENTS.md) · Predecessor: [intmat](https://github.com/emberian/intmat) (archive)

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl                 # conservation, bounds, oracle
julia --project=env/metal -e 'using Pkg; Pkg.instantiate()'   # Apple GPU
```

```julia
using Intmat2, Metal
r = random_rule(8); p = Physics{Float32}()
w = World(MetalBackend(), r, (128,128,128))
w = Intmat2.light!(w, r, p)
for _ in 1:1000; w = step!(w, r, p); end
summarize(w)
```
