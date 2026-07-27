@testset "expectation_value" begin
    # Cross-check MPSKit's generic `expectation_value` against a brute-force contraction
    # of the purified density operator, for every code path it can dispatch through:
    # single-site, two-site (adjacent and non-adjacent), and a FiniteMPOHamiltonian.
    P, K, Vb = ℂ^2, ℂ^2, ℂ^2
    ψ = FiniteLPTN(3, P, K, Vb)
    A1, A2, A3 = ψ[1], ψ[2], ψ[3]  # getindex picks whichever of AL/AC/AR is cached

    @tensor Λ[l1 p1 k1 p2 k2 p3 k3 l2] := A1[l1 p1 k1; b1] * A2[b1 p2 k2; b2] * A3[b2 p3 k3; l2]
    @tensor trace_rho = Λ[l1 p1 k1 p2 k2 p3 k3 l2] * conj(Λ[l1 p1 k1 p2 k2 p3 k3 l2])
    @test trace_rho ≈ lptn_trace(ψ)

    O1 = randn(ComplexF64, P, P)
    O2 = randn(ComplexF64, P, P)

    @testset "single-site" begin
        val = expectation_value(ψ, 1 => O1)
        @tensor num = conj(Λ[l1 p1p k1 p2 k2 p3 k3 l2]) * O1[p1p; p1] * Λ[l1 p1 k1 p2 k2 p3 k3 l2]
        @test val ≈ num / trace_rho
    end

    @testset "two-site (adjacent)" begin
        val = expectation_value(ψ, (1, 2) => O1 ⊗ O2)
        @tensor num = conj(Λ[l1 p1p k1 p2p k2 p3 k3 l2]) * O1[p1p; p1] * O2[p2p; p2] *
            Λ[l1 p1 k1 p2 k2 p3 k3 l2]
        @test val ≈ num / trace_rho
    end

    @testset "two-site (non-adjacent)" begin
        val = expectation_value(ψ, (1, 3) => O1 ⊗ O2)
        @tensor num = conj(Λ[l1 p1p k1 p2 k2 p3p k3 l2]) * O1[p1p; p1] * O2[p3p; p3] *
            Λ[l1 p1 k1 p2 k2 p3 k3 l2]
        @test val ≈ num / trace_rho
    end

    @testset "FiniteMPOHamiltonian" begin
        H = FiniteMPOHamiltonian([P, P, P], 1 => O1, 2 => O2)
        val = expectation_value(ψ, H)
        @tensor num1 = conj(Λ[l1 p1p k1 p2 k2 p3 k3 l2]) * O1[p1p; p1] * Λ[l1 p1 k1 p2 k2 p3 k3 l2]
        @tensor num2 = conj(Λ[l1 p1 k1 p2p k2 p3 k3 l2]) * O2[p2p; p2] * Λ[l1 p1 k1 p2 k2 p3 k3 l2]
        @test val ≈ (num1 + num2) / trace_rho
    end
end
