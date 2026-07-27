@testset "timestep / changebonds" begin
    # MPSKit's TDVP/TDVP2/OptimalExpand machinery is not written with LPTN in mind, but
    # relies on the same GenericMPSTensor{S,3}-specific "density matrix" fallbacks
    # (MPO_AC_Hamiltonian, MPO_AC2_Hamiltonian, TransferMatrix) already checked in
    # mpskit_assumptions.jl and expectation_values.jl. These tests confirm the physical
    # consequence: real-time evolution and bond expansion conserve Tr[ρ], and imaginary-time
    # evolution from infinite temperature reproduces the exact thermal state.

    P, K, Vb = ℂ^2, ℂ^2, ℂ^2
    O0 = randn(ComplexF64, P, P)
    O = O0 + O0'  # a genuinely Hermitian local operator: non-Hermitian H does not
    # conserve Tr[ρ] even in exact evolution, so this matters for the test to be meaningful

    @testset "TDVP" begin
        ψ = FiniteLPTN(4, P, K, Vb)
        H = FiniteMPOHamiltonian(fill(P, 4), 1 => O, 2 => O, 3 => O)
        ψ2, = timestep(ψ, H, 0.0, 0.01, TDVP())
        @test lptn_trace(ψ2) ≈ lptn_trace(ψ)
    end

    @testset "TDVP2" begin
        ψ = FiniteLPTN(4, P, K, Vb)
        H = FiniteMPOHamiltonian(fill(P, 4), 1 => O, 2 => O, 3 => O)
        alg = TDVP2(; trscheme = truncrank(4))
        ψ2, = timestep(ψ, H, 0.0, 0.01, alg)
        @test lptn_trace(ψ2) ≈ lptn_trace(ψ)
    end

    @testset "OptimalExpand" begin
        ψ = FiniteLPTN(4, P, K, Vb)
        H = FiniteMPOHamiltonian(fill(P, 4), 1 => O, 2 => O, 3 => O)
        envs = environments(ψ, H)

        bonddims_before = [dim(space(ψ.AL[i], numind(ψ.AL[i]))) for i in 1:3]
        alg = OptimalExpand(; trscheme = truncrank(2))
        ψ2, = MPSKit.changebonds(ψ, H, alg, envs)
        bonddims_after = [dim(space(ψ2.AL[i], numind(ψ2.AL[i]))) for i in 1:3]

        @test all(bonddims_after .> bonddims_before)
        @test lptn_trace(ψ2) ≈ lptn_trace(ψ)
    end

    @testset "finite temperature" begin
        # Standard purification-of-thermal-state construction: start from an
        # infinite-temperature LPTN (Kraus space = physical space, tensor = identity map,
        # trivial bonds), then apply imaginary-time evolution for β/2. This should give
        # ρ = e^{-βH}/Z exactly, since evolving only the physical leg (Kraus untouched)
        # gives ρ = e^{-βH/2} ρ₀ e^{-βH/2} = e^{-βH} for ρ₀ = 1 (maximally mixed).
        Vtriv = oneunit(P)
        A = zeros(ComplexF64, Vtriv ⊗ P ⊗ P ← Vtriv)
        for p in 1:dim(P)
            A[1, p, p, 1] = 1.0
        end
        ψ = FiniteMPS([A])
        H = FiniteMPOHamiltonian([P], 1 => O)

        β = 0.7
        ψ2, = timestep(ψ, H, 0.0, β / 2, TDVP(); imaginary_evolution = true)

        Λ = ψ2[1]
        @tensor rho[p1; p2] := Λ[1 p1 k; 4] * conj(Λ[1 p2 k; 4])
        rho = rho / tr(rho)

        exact = exp(-β * convert(Array, O))
        exact = exact / tr(exact)

        @test convert(Array, rho) ≈ exact
    end
end
