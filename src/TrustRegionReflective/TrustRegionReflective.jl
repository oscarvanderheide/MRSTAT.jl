module TrustRegionReflective

# Registered modules
using Memento
using LinearAlgebra
using StaticArrays
using Statistics
using TickTock

@kwdef struct SolverOptions{T,I,B}
    min_ratio::T = 0.05
    max_iter_trf::I = 20
    max_iter_steihaug::I = 20
    tol_steihaug::T = 0.1
    init_scale_radius::T = 0.1
    save_every_iter::B = false
    use_C::B = true
end

mutable struct SolverOutput
    x # parameters
    f # cost
    r # residual vector
    t # elapsed time (s)
end

include("utils.jl")
include("solver.jl")
include("steihaug.jl")

end # module
