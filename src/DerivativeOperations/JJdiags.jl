function block_diagonal_inverse_approximation(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates)

    dp = J_column_dot_products(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates)


    return H⁻¹_approx
end



"""
    J_column_dot_products(::AbstractResource, echos, ∂echos::NamedTuple, parameters::AbstractVector{<:AbstractTissueProperties}, coil_sensitivities::AbstractMatrix, trajectory::CartesianTrajectory2D, coordinates::AbstractVector{<:Coordinates})

Compute the dot products of each column of the Jacobian matrix J with itself, i.e., ||J[:,i]||² for each column i.
The Jacobian matrix J contains partial derivatives of the signal model w.r.t. tissue parameters in all voxels.
Like in `Jv`, the actual J is not stored explicitly - entries are computed from pre-stored partial derivatives
of magnetization at echo times and on-the-fly computations.

# Arguments
- `resource::AbstractResource`: Not used in this implementation, can be omitted.
- `echos`:          Matrix{Complex} of size (# readouts, # voxels) with phase-encoded
                    magnetization at echo times.
- `∂echos::NamedTuple`: The partial derivatives of the echos.
- `parameters`:     Tissue parameters per voxel.
- `trajectory`:     Cartesian trajectory struct.
- `coil_sensitivities`:     Matrix{Complex} of size (# voxels, # coils) with coil sensitivities.
- `coordinates`:    Vector of Coordinates (e.g. x,y,z values) for all voxels.

# Returns
- `column_norms_squared`: ComponentVector with squared norms of each parameter column across all voxels,
  organized by parameter type (T₁, T₂, B₁, B₀, ρˣ, ρʸ).

# Notes:
- This implementation computes ||J[:,i]||² efficiently without explicitly forming the full Jacobian matrix
- Follows the same parameter scaling as the original Jv function (logarithmic scaling for T₁ and T₂)
- Works on both CPU and GPU through vectorized operations
"""
function J_column_dot_products(
    ::AbstractResource,
    echos::AbstractMatrix{<:Complex},
    ∂echos::NamedTuple,
    parameters::StructVector{<:AbstractTissueProperties},
    trajectory,
    coil_sensitivities::AbstractMatrix{<:Complex},
    coordinates::StructVector{<:Coordinates}
)
    # Load constants (same as in Jv)
    T₁ = parameters.T₁
    T₂ = parameters.T₂
    R₂ = inv.(T₂)
    x = coordinates.x
    ρ = complex.(parameters.ρˣ, parameters.ρʸ)

    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    num_samples_per_readout = trajectory.nsamplesperreadout
    num_readouts = trajectory.nreadouts
    num_coils = size(coil_sensitivities, 2)
    num_voxels = length(parameters)

    # Readout dynamics (same as in Jv)
    θ = Δkₓ * x
    E = @. exp(-Δt * R₂) * exp(im * (Δkₓ * x))

    # Sample point indices relative to echo time
    ns = num_samples_per_readout
    s = -(ns ÷ 2):((ns÷2)-1) |> transpose
    Eˢ = @. E^s

    # Partial derivatives w.r.t T₂
    ∂E∂T₂ = @. Δt * R₂^2 * E
    ∂Eˢ∂T₂ = @. s * (E^(s - 1)) * ∂E∂T₂

    # Initialize output ComponentVector
    result_components = []
    result_names = Symbol[]

    # # Helper function to compute squared norm of a column contribution
    # function compute_column_norm_squared(factor_matrix, echo_matrix)
    #     # factor_matrix: (num_voxels,) or (num_samples_per_readout, num_voxels)
    #     # echo_matrix: (num_readouts, num_voxels)
    #     total = 0.0
    #     for i in 1:num_coils
    #         cᵢ = @view coil_sensitivities[:, i]
    #         if ndims(factor_matrix) == 1
    #             # factor_matrix is per-voxel, echo_matrix varies with readout
    #             weighted_factors = factor_matrix .* cᵢ
    #             contribution = transpose(weighted_factors .* Eˢ) * transpose(echo_matrix)
    #         else
    #             # factor_matrix varies with sample and voxel
    #             contribution = transpose(factor_matrix .* (cᵢ' .* Eˢ)) * transpose(echo_matrix)
    #         end
    #         total += real(sum(abs2, contribution))
    #     end
    #     return total
    # end

    # T₁ derivatives - one column per voxel
    T₁_dots = zeros(eltype(T₁), num_voxels)
    for v in 1:num_voxels
        factor_v = ρ[v] * T₁[v]  # Logarithmic scaling correction
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            weighted_factor = factor_v * cᵢ
            # T₁ contribution: weighted_factor * Eˢ[v,:] * ∂echos.T₁[:,v]
            contribution = transpose(weighted_factor .* Eˢ[:, v]) * ∂echos.T₁[:, v]
            total += real(sum(abs2, contribution))
        end
        T₁_dots[v] = total
    end
    push!(result_components, T₁_dots)
    push!(result_names, :T₁)

    # T₂ derivatives - one column per voxel (has two terms)
    T₂_dots = zeros(eltype(T₂), num_voxels)
    for v in 1:num_voxels
        factor_v = ρ[v] * T₂[v]  # Logarithmic scaling correction
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            weighted_factor = factor_v * cᵢ

            # First term: weighted_factor * ∂Eˢ∂T₂[v,:] * echos[:,v]
            contrib1 = transpose(weighted_factor .* ∂Eˢ∂T₂[:, v]) * echos[:, v]

            # Second term: weighted_factor * Eˢ[v,:] * ∂echos.T₂[:,v]
            contrib2 = transpose(weighted_factor .* Eˢ[:, v]) * ∂echos.T₂[:, v]

            total += real(sum(abs2, contrib1 + contrib2))
        end
        T₂_dots[v] = total
    end
    push!(result_components, T₂_dots)
    push!(result_names, :T₂)

    # B₁ derivatives (optional)
    if :B₁ in propertynames(∂echos)
        B₁_dots = zeros(real(eltype(ρ)), num_voxels)
        for v in 1:num_voxels
            factor_v = ρ[v]
            total = 0.0
            for i in 1:num_coils
                cᵢ = coil_sensitivities[v, i]
                weighted_factor = factor_v * cᵢ
                # B₁ contribution: weighted_factor * Eˢ[v,:] * ∂echos.B₁[:,v]
                contribution = transpose(weighted_factor .* Eˢ[:, v]) * ∂echos.B₁[:, v]
                total += real(sum(abs2, contribution))
            end
            B₁_dots[v] = total
        end
        push!(result_components, B₁_dots)
        push!(result_names, :B₁)
    end

    # B₀ derivatives (optional)
    if :B₀ in propertynames(∂echos)
        B₀_dots = zeros(real(eltype(ρ)), num_voxels)
        for v in 1:num_voxels
            factor_v = ρ[v]
            total = 0.0
            for i in 1:num_coils
                cᵢ = coil_sensitivities[v, i]
                weighted_factor = factor_v * cᵢ

                # First term: weighted_factor * ∂Eˢ∂B₀[v,:] * echos[:,v] (TODO: ∂Eˢ∂B₀ computation)
                # For now, assuming this term exists or is zero

                # Second term: weighted_factor * Eˢ[v,:] * ∂echos.B₀[:,v]
                contribution = transpose(weighted_factor .* Eˢ[:, v]) * ∂echos.B₀[:, v]
                total += real(sum(abs2, contribution))
            end
            B₀_dots[v] = total
        end
        push!(result_components, B₀_dots)
        push!(result_names, :B₀)
    end

    # ρˣ derivatives - one column per voxel
    ρˣ_dots = zeros(real(eltype(ρ)), num_voxels)
    for v in 1:num_voxels
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            # ρˣ contribution: cᵢ * Eˢ[v,:] * echos[:,v]
            contribution = transpose(cᵢ .* Eˢ[:, v]) * echos[:, v]
            total += real(sum(abs2, contribution))
        end
        ρˣ_dots[v] = total
    end
    push!(result_components, ρˣ_dots)
    push!(result_names, :ρˣ)

    # ρʸ derivatives - one column per voxel
    ρʸ_dots = zeros(real(eltype(ρ)), num_voxels)
    for v in 1:num_voxels
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            # ρʸ contribution: im * cᵢ * Eˢ[v,:] * echos[:,v]
            contribution = transpose(im * cᵢ .* Eˢ[:, v]) * echos[:, v]
            total += real(sum(abs2, contribution))
        end
        ρʸ_dots[v] = total
    end
    push!(result_components, ρʸ_dots)
    push!(result_names, :ρʸ)

    # Create ComponentVector
    return ComponentVector(NamedTuple{Tuple(result_names)}(result_components))
end

# Convenience methods
J_column_dot_products(echos::AbstractArray, args...) = J_column_dot_products(CPU1(), echos, args...)
J_column_dot_products(echos::CuArray, args...) = J_column_dot_products(CUDALibs(), echos, args...)



"""
    J_column_dot_products(::AbstractResource, echos, ∂echos::NamedTuple, parameters::AbstractVector{<:AbstractTissueProperties}, coil_sensitivities::AbstractMatrix, trajectory::CartesianTrajectory2D, coordinates::AbstractVector{<:Coordinates})

Compute the dot products of each column of the Jacobian matrix J with itself, i.e., ||J[:,i]||² for each column i.
The Jacobian matrix J contains partial derivatives of the signal model w.r.t. tissue parameters in all voxels.
Like in `Jv`, the actual J is not stored explicitly - entries are computed from pre-stored partial derivatives
of magnetization at echo times and on-the-fly computations.

# Arguments
- `resource::AbstractResource`: Not used in this implementation, can be omitted.
- `echos`:          Matrix{Complex} of size (# readouts, # voxels) with phase-encoded
                    magnetization at echo times.
- `∂echos::NamedTuple`: The partial derivatives of the echos.
- `parameters`:     Tissue parameters per voxel.
- `trajectory`:     Cartesian trajectory struct.
- `coil_sensitivities`:     Matrix{Complex} of size (# voxels, # coils) with coil sensitivities.
- `coordinates`:    Vector of Coordinates (e.g. x,y,z values) for all voxels.

# Returns
- `column_norms_squared`: ComponentVector with squared norms of each parameter column across all voxels,
  organized by parameter type (T₁, T₂, B₁, B₀, ρˣ, ρʸ).

# Notes:
- This implementation computes ||J[:,i]||² efficiently without explicitly forming the full Jacobian matrix
- Follows the same parameter scaling as the original Jv function (logarithmic scaling for T₁ and T₂)
- Works on both CPU and GPU through vectorized operations
"""
function J_column_dot_products(
    ::AbstractResource,
    echos::AbstractMatrix{<:Complex},
    ∂echos::NamedTuple,
    parameters::StructVector{<:AbstractTissueProperties},
    trajectory,
    coil_sensitivities::AbstractMatrix{<:Complex},
    coordinates::StructVector{<:Coordinates}
)
    # Load constants (same as in Jv)
    T₁ = parameters.T₁
    T₂ = parameters.T₂
    R₂ = inv.(T₂)
    x = coordinates.x
    ρ = complex.(parameters.ρˣ, parameters.ρʸ)

    Δt = trajectory.Δt
    Δkₓ = trajectory.Δk_adc
    num_samples_per_readout = trajectory.nsamplesperreadout
    num_readouts = trajectory.nreadouts
    num_coils = size(coil_sensitivities, 2)
    num_voxels = length(parameters)

    # Readout dynamics (same as in Jv)
    θ = Δkₓ * x
    E = @. exp(-Δt * R₂) * exp(im * (Δkₓ * x))

    # Sample point indices relative to echo time
    ns = num_samples_per_readout
    s = -(ns ÷ 2):((ns÷2)-1) |> transpose
    Eˢ = @. E^s

    # Partial derivatives w.r.t T₂
    ∂E∂T₂ = @. Δt * R₂^2 * E
    ∂Eˢ∂T₂ = @. s * (E^(s - 1)) * ∂E∂T₂

    # Initialize output ComponentVector
    result_components = []
    result_names = Symbol[]

    # # Helper function to compute squared norm of a column contribution
    # function compute_column_norm_squared(factor_matrix, echo_matrix)
    #     # factor_matrix: (num_voxels,) or (num_samples_per_readout, num_voxels)
    #     # echo_matrix: (num_readouts, num_voxels)
    #     total = 0.0
    #     for i in 1:num_coils
    #         cᵢ = @view coil_sensitivities[:, i]
    #         if ndims(factor_matrix) == 1
    #             # factor_matrix is per-voxel, echo_matrix varies with readout
    #             weighted_factors = factor_matrix .* cᵢ
    #             contribution = transpose(weighted_factors .* Eˢ) * transpose(echo_matrix)
    #         else
    #             # factor_matrix varies with sample and voxel
    #             contribution = transpose(factor_matrix .* (cᵢ' .* Eˢ)) * transpose(echo_matrix)
    #         end
    #         total += real(sum(abs2, contribution))
    #     end
    #     return total
    # end

    J_T₁ = ∂echos.T₁
    J_T₂ = ∂echos.T₂
    J_ρˣ = echos
    J_ρˣ = im * echos

    dots(A,B) = sum(conj.(A), B, dims=1) |> vec
    T12 = abs2.(T1)
    rho = abs2.(rho)



    JJ_T₁T₁ = dots(J_T₁, J_T₁)
    JJ_T₁T₂ = dots(J_T₁, J_T₂)
    JJ_T₁ρˣ = dots(J_T₁, J_ρˣ)
    JJ_T₁ρʸ = dots(J_T₁, J_ρʸ)
    JJ_T₂T₂ = dots(J_T₂, J_T₂)
    JJ_T₂ρˣ = dots(J_T₂, J_ρˣ)
    JJ_T₂ρʸ = dots(J_T₂, J_ρʸ)
    JJ_ρˣρˣ = dots(J_ρˣ, J_ρˣ)
    JJ_ρˣρʸ = dots(J_ρˣ, J_ρʸ)
    JJ_ρʸρʸ = dots(J_ρʸ, J_ρʸ)

    for c in 1:num_coils

    end



    # sum over coils

    # multiply by nr of points in readout direction



    # T₁ derivatives - one column per voxel
    T₁_norms = zeros(eltype(T₁), num_voxels)
    for v in 1:num_voxels
        factor_v = ρ[v] * T₁[v]  # Logarithmic scaling correction
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            weighted_factor = factor_v * cᵢ
            # T₁ contribution: weighted_factor * Eˢ[v,:] * ∂echos.T₁[:,v]
            contribution = transpose(weighted_factor .* Eˢ[:, v]) * ∂echos.T₁[:, v]
            total += real(sum(abs2, contribution))
        end
        T₁_norms[v] = total
    end
    push!(result_components, T₁_norms)
    push!(result_names, :T₁)

    # T₂ derivatives - one column per voxel (has two terms)
    T₂_norms = zeros(eltype(T₂), num_voxels)
    for v in 1:num_voxels
        factor_v = ρ[v] * T₂[v]  # Logarithmic scaling correction
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            weighted_factor = factor_v * cᵢ

            # First term: weighted_factor * ∂Eˢ∂T₂[v,:] * echos[:,v]
            contrib1 = transpose(weighted_factor .* ∂Eˢ∂T₂[:, v]) * echos[:, v]

            # Second term: weighted_factor * Eˢ[v,:] * ∂echos.T₂[:,v]
            contrib2 = transpose(weighted_factor .* Eˢ[:, v]) * ∂echos.T₂[:, v]

            total += real(sum(abs2, contrib1 + contrib2))
        end
        T₂_norms[v] = total
    end
    push!(result_components, T₂_norms)
    push!(result_names, :T₂)

    # ρˣ derivatives - one column per voxel
    ρˣ_norms = zeros(real(eltype(ρ)), num_voxels)
    for v in 1:num_voxels
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            # ρˣ contribution: cᵢ * Eˢ[v,:] * echos[:,v]
            contribution = transpose(cᵢ .* Eˢ[:, v]) * echos[:, v]
            total += real(sum(abs2, contribution))
        end
        ρˣ_norms[v] = total
    end
    push!(result_components, ρˣ_norms)
    push!(result_names, :ρˣ)

    # ρʸ derivatives - one column per voxel
    ρʸ_norms = zeros(real(eltype(ρ)), num_voxels)
    for v in 1:num_voxels
        total = 0.0
        for i in 1:num_coils
            cᵢ = coil_sensitivities[v, i]
            # ρʸ contribution: im * cᵢ * Eˢ[v,:] * echos[:,v]
            contribution = transpose(im * cᵢ .* Eˢ[:, v]) * echos[:, v]
            total += real(sum(abs2, contribution))
        end
        ρʸ_norms[v] = total
    end
    push!(result_components, ρʸ_norms)
    push!(result_names, :ρʸ)

    # Create ComponentVector
    return ComponentVector(NamedTuple{Tuple(result_names)}(result_components))
end

# Convenience methods
J_column_dot_products(echos::AbstractArray, args...) = J_column_dot_products(CPU1(), echos, args...)
J_column_dot_products(echos::CuArray, args...) = J_column_dot_products(CUDALibs(), echos, args...)
