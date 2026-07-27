@testset "approximate" begin
    # `approximate!`'s DMRG/DMRG2 algorithms are built on AC_projection/AC2_projection,
    # which reuse the same MPO_AC_Hamiltonian/MPO_AC2_Hamiltonian machinery already
    # validated for GenericMPSTensor{S,3} elsewhere (expectation_values.jl,
    # mpskit_assumptions.jl). Applying the identity FiniteMPO to a FiniteLPTN and fitting a
    # same-bond-dimension ansatz to it should recover the original state essentially
    # exactly, since no truncation is actually needed.
    P, K, Vb = ℂ^2, ℂ^2, ℂ^4
    ϕ = FiniteLPTN(4, P, K, Vb)
    ψ0 = FiniteLPTN(4, P, K, Vb)

    # a plain identity FiniteMPO needs trivial bond legs added to a bare physical-space
    # identity map before it can be used as an MPOTensor
    Iden = FiniteMPO(fill(MPSKit.add_util_leg(id(P)), 4))

    alg = DMRG2(; trscheme = truncrank(dim(Vb)), tol = 1.0e-12, maxiter = 20, verbosity = 0)
    ψf, = approximate(ψ0, (Iden, ϕ), alg)

    @test dot(ψf, ϕ) ≈ 1 atol = 1.0e-8
end
