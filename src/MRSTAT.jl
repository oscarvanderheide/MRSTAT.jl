module MRSTAT

using BlochSimulators
using StaticArrays, LinearAlgebra, Statistics, StructArrays
using LinearMaps
using ComputationalResources
using ImagePhantoms
using QMRIColors

include("TrustRegionReflective/TrustRegionReflective.jl")
include("DerivativeOperations/DerivativeOperations.jl")

using .TrustRegionReflective
using .DerivativeOperations

include("utils/make_phantom.jl")
include("utils/objective.jl")
include("utils/simulation_data.jl")

const NUM_COILS = 1

function mrstat_recon(
    raw_data::AbstractArray{T},
    sequence::BlochSimulator,
    coordinates,
    coil_sensitivities::AbstractArray{T},
    trajectory::CartesianTrajectory2D,
    transmit_field::AbstractArray;
    x0=T₁T₂ρˣρʸ(log(1.0), log(0.100), 1.0, 0.0),
    LB=T₁T₂ρˣρʸ(log(0.1), log(0.001), -Inf, -Inf),
    UB=T₁T₂ρˣρʸ(log(7.0), log(3.000), Inf, Inf),
    trf_options=TrustRegionReflective.SolverOptions(),
    intermediate_plots = false
) where {T<:Complex}

    # Repeat the initial guess and bounds for each voxel
    # Note: x0, LB and UB are in log space for T₁ and T₂
    num_voxels = length(coordinates)
    @assert length(transmit_field) == num_voxels
    x0 = repeat(x0', num_voxels) |> f32
    LB = repeat(LB', num_voxels) |> f32
    UB = repeat(UB', num_voxels) |> f32

    # Check coil sensitivities
    @assert all(Cᵢ -> !iszero(sum(Cᵢ)), coil_sensitivities)

    resource = CUDALibs()
    objfun = (x, mode) -> objective(x, resource, mode, raw_data, sequence, coordinates, coil_sensitivities, trajectory, transmit_field) 

    plotfun(x, figtitle) = nothing 


    output = TrustRegionReflective.solver(objfun, vec(x0), vec(LB), vec(UB), trf_options, plotfun)

    return output
end

function example_mrstat_recon()
    simulation_data = generate_simulation_data()
    return mrstat_recon(simulation_data...)
end

end # module MRSTAT