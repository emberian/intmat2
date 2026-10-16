"""
Interior 2: evolving matter as one continuous field.

See DESIGN.md. The package is four KernelAbstractions kernels (`step!`), a rule
struct (`Rule`), a physics struct (`Physics`), a world (`World`), and the tools
that analyze, observe, render and search them.
"""
module Intmat2

using KernelAbstractions
using Adapt
using StaticArrays
using LinearAlgebra
using Random
using Statistics
using Printf

const KA = KernelAbstractions

export Rule, Physics, World, step!, nparams, random_rule, summarize, total_mass

include("rule.jl")
include("state.jl")
include("step.jl")
include("observe.jl")
include("analysis.jl")   # steady state, dispersion, atlas
include("export.jl")     # captures and manifests
include("render.jl")     # GPU ray marcher using the in-sim light
include("batch.jl")      # independent-world batching
include("evolve.jl")     # objectives and rule search

end # module
