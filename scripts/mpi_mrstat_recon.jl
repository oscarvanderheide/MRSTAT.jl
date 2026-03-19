#!/usr/bin/env julia
# ============================================================================
#  MPI-based MRSTAT Reconstruction
#
#  Launch via SLURM:
#    sbatch scripts/slurm/run_mpi_single_node.sh
#    sbatch scripts/slurm/run_mpi_multi_node.sh
# ============================================================================

using MPI
MPI.Init()

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activate mrstat_main project

using CUDA
using BlochSimulators
using BlochSimulators: f32, gpu
using MRSTAT
using MRSTAT: NUM_COILS, TrustRegionReflective, DerivativeOperations,
              optim_to_physical_pars, plot_T₁T₂ρ
using MRSTAT.DerivativeOperations: simulate_derivatives, Jv, Jᴴv
using ComputationalResources: AbstractResource, CUDALibs
using LinearAlgebra, LinearMaps, StaticArrays, StructArrays

# Load MPI utilities (plain includes, shares MRSTAT namespace)
include(joinpath(@__DIR__, "..", "src", "mpi", "MPIResources.jl"))
include(joinpath(@__DIR__, "..", "src", "mpi", "mpi_objective.jl"))

# ========================= MPI Setup ========================================

res    = MPICUDALibs()
comm   = res.comm
rank   = res.rank
nranks = res.nranks

rank == 0 && println("=== MPI MRSTAT Reconstruction ===")
rank == 0 && println("    Ranks: $nranks")
println("    Rank $rank → GPU $(res.local_gpu_id) ($(CUDA.name(CUDA.device())))")
MPI.Barrier(comm)

# ========================= Data Generation ==================================
# All ranks generate identical data (deterministic), then partition per-voxel arrays.

rank == 0 && println("\nGenerating simulation data …")
using Random; Random.seed!(42)   # fixed seed so all ranks generate identical phantom
raw_data, sequence, coords_full, coils_full, trajectory =
    MRSTAT.generate_simulation_data()

total_nvox = length(coords_full)
N = isqrt(total_nvox)

rank == 0 && println("    Image: $N × $N ($total_nvox voxels)")

# ========================= Voxel Partitioning ===============================

vr = voxel_partition(total_nvox, rank, nranks)
local_nvox = length(vr)
println("    Rank $rank → voxels $(first(vr)):$(last(vr)) ($local_nvox voxels)")

local_coords = partition_structarray(coords_full, vr)
local_coils  = gpu(f32(Array(coils_full)[vr, :]))
local_tx     = ones(Float32, local_nvox)   # uniform transmit field

MPI.Barrier(comm)
rank == 0 && println("Data partitioned.\n")

# ========================= Optimization =====================================

x0_per = Float32[log(1.0), log(0.100), 1.0, 0.0]
LB_per = Float32[log(0.1), log(0.001), -Inf, -Inf]
UB_per = Float32[log(7.0), log(3.000),  Inf,  Inf]

x0 = repeat(x0_per', total_nvox) |> vec
LB = repeat(LB_per', total_nvox) |> vec
UB = repeat(UB_per', total_nvox) |> vec

transmit_field_full = ones(Float32, total_nvox)

plotfun(x, figtitle) = if is_root(res)
    physical = MRSTAT.optim_to_physical_pars(x, transmit_field_full)
    MRSTAT.plot_T₁T₂ρ(physical, N, N, figtitle)
end

plotfun(x0, "Initial Guess")

objfun = (x, mode) -> mpi_objective(
    x, res, mode,
    raw_data, sequence, local_coords, local_coils,
    trajectory, local_tx, vr, total_nvox,
)

trf_opts = TrustRegionReflective.SolverOptions(;
    max_iter_trf      = 20,
    max_iter_steihaug = 20,
    tol_steihaug      = 0.1,
)

# ========================= Run ==============================================

rank == 0 && println("Starting TRF solver …\n")
MPI.Barrier(comm)

output = TrustRegionReflective.solver(objfun, x0, LB, UB, trf_opts, plotfun)

MPI.Barrier(comm)

if is_root(res)
    println("\n=== Reconstruction complete ===")
    println("    Final cost: $(output.f[end])")
    println("    Iterations: $(size(output.f, 2))")
end

MPI.Finalize()
