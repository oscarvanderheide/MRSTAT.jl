function objective(optimpars::Vector{<:Real}, resource, mode, raw_data, sequence, coordinates, coil_sensitivities, trajectory, transmit_field)

    # We compute the residual rᵢ = ||d Σᵢ (dᵢ - M(T₁,T₂,B₁,B₀)*Cᵢ*ρ) f = (1/2) * |r|^2
    # The gradient is computed as g = ℜ(Jᴴr)

    # mode 0 -> compute f and r only mode 1 -> compute f, r and g mode 2 -> compute f, r, g
    # and assemble approximate Hessian

    # Convert optimpars (Vector{<:Real}) to Vector{<:AbstractTissueParameters} to be used in
    # simulations
    parameters = optim_to_physical_pars(optimpars, transmit_field)

    # Convert to single precision and send to gpu device
    parameters = gpu(f32(parameters))

    # Compute magnetization at echo times
    echos = simulate_magnetization(resource, sequence, parameters)

    if mode == 0

        # Apply phase encoding
        phase_encoding!(echos, trajectory, coordinates)

        # Compute signal
        s = magnetization_to_signal(resource, echos, parameters, trajectory, coordinates, coil_sensitivities)

        # Compute residual r
        r = s - raw_data;

        # From array to vector of svectors
        r = reshape(collect(r), :, NUM_COILS)
        r = map(SVector{NUM_COILS}, eachrow(r)) |> gpu

        # Compute cost f
        f = 0.5 * sum(norm.(r).^2);

        return f, r

    elseif mode > 0

        # Compute partial derivatives of magnetization at echo time
        ∂echos = simulate_derivatives(echos, resource, sequence, parameters)

        # Apply phase encoding
        @show typeof(coordinates)
        phase_encoding!(echos, trajectory, coordinates)
        phase_encoding!(∂echos, trajectory, coordinates)

        # Compute signal
        s = magnetization_to_signal(resource, echos, parameters, trajectory, coordinates, coil_sensitivities)

        # Compute residual r
        r = s - raw_data;

        # From array to vector of svectors
        r = reshape(collect(r), :, NUM_COILS)
        r = map(SVector{NUM_COILS}, eachrow(r)) |> gpu

        # Compute cost f
        f = 0.5 * sum(norm.(r).^2);

        # Compute gradient


        coil_sensitivities = collect(coil_sensitivities)
        coil_sensitivities_svec = map(SVector{NUM_COILS}, eachrow(coil_sensitivities)) |> gpu

        g = Jᴴv(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates, r)
        g = StructArray(g)
        g = reduce(vcat,fieldarrays(g))
        g = real.(g)
        g = collect(g)

        mode == 1 && return f, r, g

        # Make Gauss-Newton matrix multiply function
        reJᴴJ(x) = begin
            np = 4 # nr of reconstruction parameters per voxel
            x = reshape(x,:,np)
            x = map(SVector{4}, eachcol(x)...) |> gpu
            y = Jv(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates, x)
            z = Jᴴv(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates, y)
            return real.(reduce(vcat,fieldarrays(StructArray(z))))
        end

        H = LinearMap(
            v -> reJᴴJ(v),
            v -> v, # adjoint operation not used
        length(g),length(g));

        H⁻¹_approx = block_diagonal_inverse_approximation(resource, echos, ∂echos, parameters, coil_sensitivities_svec, trajectory, coordinates)

        return f, r, g, H, H⁻¹_approx
    end
end

function optim_to_physical_pars(optimpars, transmit_field)

    optimpars = reshape(optimpars,:,4)
    T₁ = exp.(optimpars[:,1])
    T₂ = exp.(optimpars[:,2])
    B₁ = vec(transmit_field)
    ρˣ = optimpars[:,3]
    ρʸ = optimpars[:,4]

    return map(T₁T₂B₁ρˣρʸ, T₁, T₂, B₁, ρˣ, ρʸ) |> StructArray
end
