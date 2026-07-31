"""
    FiniteLindbladian{H,O}

A finite-chain Lindblad generator `ℒ = -i[H, ·] + 𝒟`, split into a coherent part `H`
(a `MPSKit.FiniteMPOHamiltonian`, evolved separately via `TDVP`/`TDVP2`) and a purely
on-site dissipative part: for each site `n`, zero or more jump operators `L_k : P ← P`
(`P` the physical space at site `n`), contributing
`𝒟_n(ρ) = Σ_k L_k ρ L_k† - ½{L_k†L_k, ρ}` to the dissipator.

Long-range (multi-site) jump operators are not supported.
"""
struct FiniteLindbladian{H, O}
    hamiltonian::H
    jumps::Vector{Vector{O}}
end

"""
    FiniteLindbladian(H::MPSKit.FiniteMPOHamiltonian, jumps::Pair{<:Integer}...)

Construct a [`FiniteLindbladian`](@ref) from a Hamiltonian `H` and a list of
`site => L` pairs, one per jump operator (a site may appear zero, one, or several
times). Each `L` must map the physical space at that site to itself.
"""
function FiniteLindbladian(H::MPSKit.FiniteMPOHamiltonian, jumps::Pair{<:Integer}...)
    N = length(H)
    O = isempty(jumps) ? Any : mapreduce(jump -> typeof(last(jump)), typejoin, jumps)
    jumpvec = [O[] for _ in 1:N]
    for (n, L) in jumps
        P = physicalspace(H, n)
        domain(L) == codomain(L) == ProductSpace(P) ||
            throw(ArgumentError("jump operator at site $n does not map the physical space $P to itself"))
        push!(jumpvec[n], L)
    end
    return FiniteLindbladian{typeof(H), O}(H, jumpvec)
end

"""
    jump_operators(ℒ::FiniteLindbladian, n::Integer)

Return the on-site jump operators `{L_k}` acting at site `n` of `ℒ`.
"""
jump_operators(ℒ::FiniteLindbladian, n::Integer) = ℒ.jumps[n]

"""
    dissipator_matrices(Ls)

For a set of on-site jump operators `Ls = {L_k}` (each `L_k : P ← P`), return the `N`
maps on the vectorized (doubled) space `P ⊗ P' ← P ⊗ P'` representing each jump
operator's contribution to the vectorized dissipator
`𝒟(ρ) = Σ_k L_k ρ L_k† - ½{L_k†L_k, ρ}`: `ρ` itself is vectorized as an element of
`P ⊗ P'`, with the first leg the ket index and the second (dual) leg the bra index.
Each returned map is a genuine `TensorMap` endomorphism (`codomain == domain`), built via
`@tensor` contractions that let TensorKit itself track which legs are conjugated, rather
than converting `L` to a plain dense matrix and reshuffling indices by hand — so this
stays correct for any (including graded/non-abelian) physical space, not just a trivial
`ComplexSpace`. Being genuine `TensorMap`s, summing them and calling `exp` on the result
(to get the dissipator's propagator over a timestep `dt`) works directly, with no
conversion to a plain array needed.
"""
function dissipator_matrices(Ls::AbstractVector{<:AbstractTensorMap})
    isempty(Ls) && throw(ArgumentError("need at least one jump operator"))
    P = only(codomain(first(Ls)))
    𝟙 = id(P)
    return map(Ls) do L
        LdL = L' * L
        @tensor D[k, l; i, j] := L[k, i] * conj(L[l, j]) -
            0.5 * LdL[k, i] * conj(𝟙[l, j]) -
            0.5 * 𝟙[k, i] * conj(LdL[l, j])
        return D
    end
end

"""
    kraus_operators(D::AbstractTensorMap, dt::Real; trunc = notrunc())

Given a (summed) on-site dissipator generator `D : P⊗P' ← P⊗P'` (as returned by
[`dissipator_matrices`](@ref), summed over a site's jump operators) and a timestep `dt`,
return a single isometry-like map `K : P⊗Kaux ← P` bundling the Kraus operators of the
*exact* CPTP channel `ρ ↦ Σ_i K_i ρ K_i† = exp(dt·D)(ρ)` into one extra ("Kraus") leg
`Kaux`: slicing `K` at a fixed `Kaux` index gives one `K_i : P ← P`, and
`Σ_i K_i†K_i = 1`. Returning a single tensor with a new leg (rather than a `Vector` of
separate `P ← P` operators) is what lets this be attached directly to an `LPTNTensor`'s
Kraus leg (by fusing `Kaux` with the existing one), rather than requiring a further
unpacking step.

Computed via the Choi–Jamiołkowski isomorphism: `U = exp(dt·D)` is regrouped from a
superoperator on `P⊗P'` into the Choi matrix `C` on `P⊗P'` (swapping `U`'s second
codomain leg with its first domain leg), which is Hermitian and positive-semidefinite
exactly because `exp(dt·D)` is completely positive for any `dt ≥ 0` (the defining
property of a GKSL generator). `C`'s eigendecomposition `C = Σ_i λ_i vᵢvᵢ†` (via
`eigh_trunc`, so `trunc` controls Kraus-rank truncation, e.g. dropping the numerical
noise below `C`'s true rank) gives the Kraus operators as `√λᵢ vᵢ`, reshaped into the
extra `Kaux` leg of `K` rather than being sliced out into a `Vector`. Unlike a
first-order `K_0 = 1 - dt(iH + ½ΣL†L)`, `K_k = √dt·L_k` expansion, this is exactly
trace-preserving for any `dt`, not just to `O(dt)`.
"""
function kraus_operators(D::AbstractTensorMap, dt::Real; trunc = notrunc())
    U = exp(dt * D)
    C = permute(U, ((1, 3), (2, 4)))
    Λ, V = eigh_trunc(C; trunc)
    VΛ = V * sqrt(Λ)
    return permute(VΛ, ((1, 3), (2,)))
end

"""
    apply_kraus(A::LPTNTensor, K::AbstractTensorMap)

Apply a bundled Kraus isometry `K : P⊗Kaux ← P` (as returned by [`kraus_operators`](@ref))
to the physical leg of a single [`LPTNTensor`](@ref) `A`, growing its Kraus leg from
`Kold` to `fuse(Kold, Kaux)`. Because `K` is an isometry (`K'*K == id(P)`, guaranteed by
`kraus_operators`), this preserves whatever left/right canonical form `A` was already in
(contracting the result with its own conjugate over the physical+Kraus legs gives the
same reduced map on the virtual legs as for `A` itself).
"""
function apply_kraus(A::LPTNTensor, K::AbstractTensorMap)
    P = lptn_physicalspace(A)
    Kold = krausspace(A)
    codomain(K)[1] == domain(K)[1] == P ||
        throw(ArgumentError("Kraus isometry does not act on the physical space $P of A"))
    Kaux = codomain(K)[2]
    F = MPSKit.fuser(scalartype(A), Kold, Kaux)
    @tensor A′[vl p′ knew; vr] := K[p′ kaux; p] * A[vl p kold; vr] * F[knew; kold kaux]
    return A′
end

"""
    apply_kraus!(ψ::FiniteLPTN, Ks::AbstractVector{<:AbstractTensorMap})

Apply one bundled Kraus isometry per site (`Ks[n]` acting at site `n`, as returned by
[`kraus_operators`](@ref)) to every site of the chain `ψ`, growing each site's Kraus leg
accordingly, and update `ψ` in place (returning it).
"""
function apply_kraus!(ψ::FiniteLPTN, Ks::AbstractVector{<:AbstractTensorMap})
    length(Ks) == length(ψ) ||
        throw(ArgumentError("need one Kraus isometry per site, got $(length(Ks)) for $(length(ψ)) sites"))

    for n in eachindex(ψ)
        ψ.AC[n] = apply_kraus(ψ.AC[n], Ks[n])
    end
    return ψ
end
