using LinearAlgebra, Random

# Set a seed for reproducibility
Random.seed!(42)


# HELPER FUNCTIONS
#

function generate_symmetric_block_diagonal_matrix(block_dim::Int, n::Int)
    blocks = Matrix{Diagonal{Float64}}(undef, block_dim, block_dim)
    for i in 1:block_dim
        for j in i:block_dim
            random_diag_entries = rand(n) .* 9 .+ 1
            block = Diagonal(random_diag_entries)
            blocks[i, j] = block
            if i != j
                blocks[j, i] = block
            end
        end
    end
    return blocks
end

function form_full_matrix(blocks::Matrix{<:AbstractMatrix})
    block_dim = size(blocks, 1)
    return hvcat(Tuple(block_dim for _ in 1:block_dim), blocks...)
end


# GENERAL NxN BLOCK INVERSE (with the fix)
#

function get_minor(blocks::Matrix, i::Int, j::Int)
    return blocks[1:end .!= i, 1:end .!= j]
end

function block_determinant(blocks::Matrix{<:Diagonal})
    m = size(blocks, 1)
    if m == 1
        return blocks[1, 1]
    end
    n = size(blocks[1, 1], 1)
    det_B = Diagonal(zeros(n))
    for j in 1:m
        sign = (-1)^(1 + j)
        minor = get_minor(blocks, 1, j)
        det_B += sign * blocks[1, j] * block_determinant(minor)
    end
    return det_B
end

function block_adjugate(blocks::Matrix{<:Diagonal})
    m = size(blocks, 1)
    adj_B = similar(blocks)
    for i in 1:m
        for j in 1:m
            sign = (-1)^(i + j)
            minor = get_minor(blocks, i, j)
            adj_B[j, i] = sign * block_determinant(minor)
        end
    end
    return adj_B
end

function block_inverse_adjugate(blocks::Matrix{<:Diagonal})
    det_B = block_determinant(blocks)
    if isapprox(det(det_B), 0.0)
        error("Block determinant is singular. The block matrix is not invertible.")
    end
    det_B_inv = inv(det_B)
    adj_B = block_adjugate(blocks)

    # CORRECTED LINE: Use `map` to apply the multiplication to each block.
    return map(adj_block -> det_B_inv * adj_block, adj_B)
end


# MAIN TEST SCRIPT for 4x4 SYMMETRIC CASE
#

function main_4x4_symmetric_test()
    # --- Parameters ---
    n = 5       # Size of each diagonal block (e.g., 5x5)
    block_dim = 4 # We are testing the 4x4 case

    println("="^60)
    println("Testing 4x4 Symmetric Block Matrix Inverse (Adjugate Method)")
    println("="^60)

    blocks_4x4 = generate_symmetric_block_diagonal_matrix(block_dim, n)
    X_4x4 = form_full_matrix(blocks_4x4)

    println("Generated a 4x4 symmetric block matrix.")
    println("Full matrix size: $(size(X_4x4, 1))x$(size(X_4x4, 2))")
    println("Is the original full matrix symmetric? ", issymmetric(X_4x4))
    println("-"^60)

    println("Calculating inverse using the general adjugate method...")
    inv_blocks_4x4 = block_inverse_adjugate(blocks_4x4)
    X_inv_calculated = form_full_matrix(inv_blocks_4x4)
    println("Calculation complete.")
    println("-"^60)

    println("Verifying results:")

    identity_matrix = I(block_dim * n)
    product = X_4x4 * X_inv_calculated
    println("  1. Verification (X * X_inv ≈ I): ", isapprox(product, identity_matrix))

    println("  2. Symmetry Check (Is the inverse symmetric?): ", issymmetric(X_inv_calculated))

    X_inv_true = inv(X_4x4)
    println("  3. Comparison (Our_Inv ≈ Julia's_inv): ", isapprox(X_inv_calculated, X_inv_true))
    println("="^60)
end

# Run the test
main_4x4_symmetric_test()
