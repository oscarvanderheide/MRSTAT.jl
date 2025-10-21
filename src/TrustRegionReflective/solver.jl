function solver(objective, x0, LB, UB, options::SolverOptions, plotfun)

    # Inspired by trust region reflective from SciPy

    # Load the initial guess
    x = x0;

    println("    Calling f,r,g,H = objective(x,2)")
    f,r,g,H = objective(x,2);

    # # Distance to boundary
    # v, dv = computeDistanceToBoundaries(x, g, LB, UB);
    # # Initial trust radius
    # Δ = 0.1 * norm( x ./ (sqrt.(v) ) );o
    println("Setting initial trust radius")
    Δ = 0.1 * norm(x);
    Δlimit = Δ*1E-10

    iter = 1;
    converged = false;
    t = 0.0;

    state = SolverOutput(x, f, r, t)

    options.save_every_iter && write_to_disk(state)

    while ( (iter < (options.max_iter_trf + 1)) && (!converged) )

        # We determine two scaling fawctors: one from the diagonal of JᴴJ. This one makes parameters with low curvature move faster.
        # The other is related to the distance of parameters to their respective boundaries.
        # It slows down parameters that are close to their boundaries.

        println("ITERATION #$(iter)")
        tick();

        if iter > 1
        println("    Calling f,r,g,H = objective(x,2)")
        f,r,g,H, H⁻¹_approx = objective(x,2);
        end

        println("    f: $(f)", )
        println("    Δ: $(Δ)")

        v, dv = computeDistanceToBoundaries(x, g, LB, UB);

        # Make scaling operator and scale gradient and Hessian
        D = sqrt.(v);
        ĝ = D .* g;
        C = dv .* g;
        if options.use_C  
            Ĥ = x -> (D .* (H * (D.*x))) + (C .* x)
        else
            Ĥ = x -> D .* (H * (D.*x))
        end
        D⁻¹ = inv.(D)

        step_accepted = false
        perform_steihaug = true
        sh_iter = -1

        steps = zeros(length(x), options.max_iter_steihaug)

        while !step_accepted

            # Compute potential step using Steihaug
            # Set preconditioner
            P = y -> D⁻¹ .* (H⁻¹_approx * (D⁻¹ .* y)) ; 
            z0 = zeros(length(ĝ));
            if perform_steihaug
                steps = steihaug(Ĥ, ĝ, Δ, P, options.max_iter_steihaug, options.tol_steihaug, z0)
                ŝ = steps[:,end]
            else
                ŝ = steps[:,sh_iter]
            end

            # ŝ = Krylov.cg(Ĥ, -ĝ, atol = options.tol_steihaug, rtol = options.tol_steihaug, itmax = options.max_iter_steihaug, radius = Δ, verbose = true)[1]

            s = D .* ŝ;
            x_new = x + s;
            # Select best step taking into account feasible region
            θ = max(0.995, 1 - norm(v .* g, Inf));
            @info "Choose step"
            @time step, step_hat, step_value = chooseStep(x, Ĥ, ĝ, s, ŝ, D, Δ, θ, LB, UB);
            x_new = x + step
            # Compute new objective
            println("    Calling f,r = objective(x_new,0)")
            f_new,r_new = objective(x_new, 0);
            # Compute reduction
            actualReduction     = -(f_new - f);
            predictedReduction  = -( (g' * s) + 0.5 * s' * (H * s) );
            modifiedReduction =  -(f_new - f + 0.5 * ŝ' * (C .* ŝ))
            ratio = modifiedReduction / predictedReduction;

            println("   reduction: $(actualReduction)")
            println("   ratio: $(ratio)")

            if (actualReduction > 0 && ratio > 0.1) || Δ < Δlimit
                println("    Step accepted")
                step_accepted = true

                Δ = adjustTrustRadius( ratio, ŝ, Δ, options.min_ratio);
                x = x_new;
                f = f_new;
                r = r_new;

                t += tok();

                state.x = hcat(state.x, x)
                state.f = hcat(state.f, f)
                state.r = hcat(state.r, r)
                state.t = hcat(state.t, t)

                perform_steihaug = true
            else
                println("    Find a smaller step (reduction: $(actualReduction)")

                # sometimes this part keeps on iterating forever, need add
                # a counter and have some maximum nr of tries

                sh_iter = size(steps,2)

                while sh_iter >= size(steps,2)
                    Δ = 0.5 * Δ;
                    println("   Trust radius reduced to: $(Δ)")
                    sh_iter = findlast( norm.(eachcol(steps)) .<= Δ)

                    if sh_iter === nothing
                        sh_iter = 1
                        break
                    end

                end
                if sh_iter == 1
                    perform_steihaug = true
                else
                    perform_steihaug = false
                end
            end
        end # Step accepted

        iter += 1;

        plotfun(x, "Iteration: $iter")
    end

    return state
end
