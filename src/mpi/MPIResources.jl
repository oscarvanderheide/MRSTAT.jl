# MPI + CUDA resource definitions and utilities
# Included (not a module) — shares MRSTAT's namespace

using MPI

"""
    MPICUDALibs <: AbstractResource

MPI + CUDA resource type. Each MPI rank binds to one GPU.
"""
struct MPICUDALibs <: AbstractResource{Nothing}
    comm::MPI.Comm
    rank::Int
    nranks::Int
    local_gpu_id::Int
end

function MPICUDALibs(comm::MPI.Comm = MPI.COMM_WORLD)
    rank   = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    # mpirun sets OMPI_COMM_WORLD_LOCAL_RANK; srun sets SLURM_LOCALID
    local_rank = parse(Int, get(ENV, "OMPI_COMM_WORLD_LOCAL_RANK",
                                get(ENV, "SLURM_LOCALID", string(rank))))
    local_gpu_id = local_rank % length(CUDA.devices())
    CUDA.device!(local_gpu_id)
    return MPICUDALibs(comm, rank, nranks, local_gpu_id)
end

is_root(r::MPICUDALibs) = r.rank == 0

function voxel_partition(total::Int, rank::Int, nranks::Int)
    base, rem = divrem(total, nranks)
    if rank < rem
        count  = base + 1
        offset = rank * (base + 1)
    else
        count  = base
        offset = rem * (base + 1) + (rank - rem) * base
    end
    return (offset + 1):(offset + count)
end

function partition_structarray(sa, vr::UnitRange)
    x_local = Array(sa.x)[vr]
    y_local = Array(sa.y)[vr]
    z_local = Array(sa.z)[vr]
    return gpu(f32(StructArray{Coordinates{Float32}}((x_local, y_local, z_local))))
end

function extract_local_optimpars(optimpars::AbstractVector, voxel_range::UnitRange)
    total_voxels = length(optimpars) ÷ 4
    M = reshape(optimpars, total_voxels, 4)
    return vec(M[voxel_range, :])
end

function allreduce_sum!(buf::CuArray, comm::MPI.Comm)
    sendbuf = Array(buf)
    recvbuf = similar(sendbuf)
    MPI.Allreduce!(sendbuf, recvbuf, +, comm)
    copyto!(buf, recvbuf)
end

function allreduce_sum(val::T, comm::MPI.Comm) where {T<:Real}
    result = Ref(val)
    MPI.Allreduce!(Ref(val), result, +, comm)
    return result[]
end

function mpi_allgatherv(local_vec::Vector{T}, comm::MPI.Comm) where {T}
    local_count = Int32(length(local_vec))
    counts = MPI.Allgather(local_count, comm)
    total  = sum(counts)
    recv   = Vector{T}(undef, total)
    MPI.Allgatherv!(local_vec, VBuffer(recv, counts), comm)
    return recv
end
