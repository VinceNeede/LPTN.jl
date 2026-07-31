@testset "FiniteLindbladian" begin
    P = ℂ^2
    d = dim(P)
    O1 = randn(ComplexF64, P, P)
    H = FiniteMPOHamiltonian([P, P, P], 1 => O1)

    L1 = randn(ComplexF64, P, P)
    L2 = randn(ComplexF64, P, P)

    @testset "construction" begin
        ℒ = FiniteLindbladian(H, 1 => L1, 1 => L2, 3 => L1)
        @test jump_operators(ℒ, 1) == [L1, L2]
        @test jump_operators(ℒ, 2) == []
        @test jump_operators(ℒ, 3) == [L1]
        @test ℒ.hamiltonian === H

        Pwrong = ℂ^3
        Lwrong = randn(ComplexF64, Pwrong, Pwrong)
        @test_throws ArgumentError FiniteLindbladian(H, 1 => Lwrong)
    end

    @testset "dissipator_matrices" begin
        # Cross-check against a brute-force dense computation of
        # 𝒟(ρ) = Σ_k L_k ρ L_k† - ½{L_k†L_k, ρ}, using the vectorization convention
        # vec(ρ)[k + (l-1)d] = ρ[k,l] (so that Dm * vec(ρ) == vec(𝒟(ρ))).
        ρ0 = randn(ComplexF64, d, d)
        ρ0 = ρ0 * ρ0'
        ρ0 ./= tr(ρ0)

        L1m, L2m = convert(Array, L1), convert(Array, L2)
        dissipator(ρ, Ls) = sum(Ls) do Lm
            return Lm * ρ * Lm' - 0.5 * (Lm' * Lm * ρ + ρ * Lm' * Lm)
        end

        @testset "single jump operator" begin
            Ds = dissipator_matrices([L1])
            @test length(Ds) == 1
            Dm = reshape(convert(Array, only(Ds)), d^2, d^2)
            expected = dissipator(ρ0, [L1m])
            @test reshape(Dm * vec(ρ0), d, d) ≈ expected
        end

        @testset "multiple jump operators" begin
            Ds = dissipator_matrices([L1, L2])
            @test length(Ds) == 2
            Dm = sum(D -> reshape(convert(Array, D), d^2, d^2), Ds)
            expected = dissipator(ρ0, [L1m, L2m])
            @test reshape(Dm * vec(ρ0), d, d) ≈ expected
        end

        @test_throws ArgumentError dissipator_matrices(typeof(L1)[])
    end

    @testset "kraus_operators" begin
        # Cross-check the bundled Kraus isometry K : P⊗Kaux ← P against a brute-force
        # dense application of exp(dt*D) itself (built via `dissipator_matrices` reshaped
        # to a dense d²×d² matrix, as in the `dissipator_matrices` testset above), for
        # both a single and multiple jump operators, and verify the channel is exactly
        # trace-preserving (Σ_i K_i†K_i = 1, i.e. contracting K with its adjoint over
        # both the physical output leg and Kaux gives the identity on the input leg) for
        # a `dt` that isn't small.
        ρ0 = randn(ComplexF64, d, d)
        ρ0 = ρ0 * ρ0'
        ρ0 ./= tr(ρ0)
        dt = 0.3

        for Ls in ([L1], [L1, L2])
            D = sum(dissipator_matrices(Ls))
            K = kraus_operators(D, dt)
            Km = convert(Array, K)  # (d, n_kraus, d) = (out, Kaux, in)
            n_kraus = size(Km, 2)

            @test sum(m -> Km[:, m, :]' * Km[:, m, :], 1:n_kraus) ≈ convert(Array, id(P))

            Dm = reshape(convert(Array, D), d^2, d^2)
            expected = reshape(exp(dt .* Dm) * vec(ρ0), d, d)
            viaKraus = sum(m -> Km[:, m, :] * ρ0 * Km[:, m, :]', 1:n_kraus)
            @test viaKraus ≈ expected
        end
    end

    @testset "apply_kraus" begin
        Vb, K0 = ℂ^2, ℂ^2
        N = 3
        ψ = FiniteLPTN(N, P, K0, Vb)

        D = sum(dissipator_matrices([L1]))
        K = kraus_operators(D, 0.3)
        Km = convert(Array, K)  # (d, n_kraus, d) = (out, Kaux, in)
        n_kraus = size(Km, 2)

        @testset "single site: preserves the reduced map on the virtual legs" begin
            # Since K is an isometry, contracting the channeled tensor with its own
            # conjugate over the physical+Kraus legs must give the same result as for
            # the original tensor (the whole point of a CPTP map: it doesn't leak
            # probability into the virtual legs).
            A = ψ[1]
            A′ = apply_kraus(A, K)
            @test krausspace(A′) == fuse(krausspace(A), codomain(K)[2])
            @tensor lhs[vl vlp; vr vrp] := A[vl p k; vr] * conj(A[vlp p k; vrp])
            @tensor rhs[vl vlp; vr vrp] := A′[vl p k; vr] * conj(A′[vlp p k; vrp])
            @test lhs ≈ rhs
        end

        @testset "whole chain vs brute force" begin
            # Cross-check against a brute-force dense computation of the channel applied
            # at every site of a 3-site chain: build the full purified density operator
            # from the original chain, apply the (dense) Kraus operators classically at
            # each site, and compare Tr[ρ] and a probe expectation value against the
            # LPTN-side result.
            A1, A2, A3 = ψ[1], ψ[2], ψ[3]
            @tensor Λ[l1 p1 k1 p2 k2 p3 k3 l2] := A1[l1 p1 k1; b1] * A2[b1 p2 k2; b2] *
                A3[b2 p3 k3; l2]
            @tensor rho_full[p1 p2 p3; p1p p2p p3p] := conj(Λ[l1 p1p k1 p2p k2 p3p k3 l2]) *
                Λ[l1 p1 k1 p2 k2 p3 k3 l2]
            rho = reshape(convert(Array, rho_full), d^3, d^3)

            Id = convert(Array, id(P))
            siteop(op, n) = kron((i == n ? op : Id for i in 1:3)...)

            rho2 = zeros(ComplexF64, d^3, d^3)
            for i1 in 1:n_kraus, i2 in 1:n_kraus, i3 in 1:n_kraus
                Kfull = siteop(Km[:, i1, :], 1) * siteop(Km[:, i2, :], 2) *
                    siteop(Km[:, i3, :], 3)
                rho2 .+= Kfull * rho * Kfull'
            end
            @test tr(rho2) ≈ tr(rho)

            O = randn(ComplexF64, P, P)
            expected = tr(siteop(convert(Array, O), 2) * rho2) / tr(rho2)

            trace_before = lptn_trace(ψ)
            ψ2 = apply_kraus!(ψ, [K, K, K])
            @test ψ2 === ψ  # mutates and returns the same object, doesn't allocate a new one
            @test lptn_trace(ψ) ≈ trace_before
            @test expectation_value(ψ, 2 => O) ≈ expected
        end

        @test_throws ArgumentError apply_kraus(ψ[1], kraus_operators(sum(dissipator_matrices([randn(ComplexF64, ℂ^3, ℂ^3)])), 0.3))
        @test_throws ArgumentError apply_kraus!(ψ, [K, K])
    end
end
