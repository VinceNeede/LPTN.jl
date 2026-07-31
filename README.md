# LPTN.jl

A Julia package for **locally purified tensor networks (LPTN)**, built on top of
[TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) and
[MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl).

An LPTN represents a mixed state (density operator) `ρ` on a chain of physical sites as a
purification: a matrix product state `Λ` over the enlarged Hilbert space
`H_phys ⊗ H_kraus`, with `ρ = Tr_kraus[|Λ⟩⟨Λ|]`. Each site tensor of `Λ` therefore carries
one extra "Kraus" (purification) leg on top of the usual physical and left/right virtual
legs of an MPS tensor.

**Using MPSKit means several of its functions already work on `FiniteLPTN`.**
`LPTNTensor` and `FiniteLPTN` are defined as thin aliases over `MPSKit.GenericMPSTensor`/
`FiniteMPS`, so MPSKit's own machinery — `expectation_value`, real/imaginary-time evolution
(`TDVP`/`TDVP2`), bond expansion (`OptimalExpand`), and state compression
(`approximate`/`DMRG`/`DMRG2`) — already applies to `FiniteLPTN` unchanged. Each piece was
traced through its actual dispatch and cross-checked numerically against brute-force
contraction (see `test/`) rather than assumed to work by analogy.

Dissipative dynamics are handled by `FiniteLindbladian` (a `FiniteMPOHamiltonian` plus
on-site jump operators): its dissipator is exponentiated exactly into a bundled Kraus
isometry (via the Choi–Jamiołkowski isomorphism, not a first-order `O(dt)` expansion),
which `apply_kraus!` applies to a `FiniteLPTN` in place, growing each site's Kraus leg.
See the Quick example below and `test/lindbladian.jl` for the brute-force cross-checks.

**Status:** early-stage / work in progress. This README doubles as the package's
documentation until there is enough content to warrant splitting it out.

## Installation

This package is not registered. From the Julia REPL:

```julia
using Pkg
Pkg.develop(url="https://github.com/VinceNeede/LPTN.jl")
```

## Quick example

```julia
using LPTN, TensorKit, MPSKit  # FiniteLPTN is a FiniteMPS, so its whole API is MPSKit's

# a chain of 5 sites, each with physical space ℂ^2, Kraus space ℂ^2,
# and virtual bond dimension up to 3
ψ = FiniteLPTN(5, ℂ^2, ℂ^2, ℂ^3)

lptn_trace(ψ)                      # Tr[ρ] -- note this is norm(ψ)^2, not norm(ψ)

O0 = randn(ComplexF64, ℂ^2, ℂ^2)
O = O0 + O0'                       # Hermitian, so Tr[ρ] is conserved under real-time evolution
expectation_value(ψ, 3 => O)       # Tr[ρ O] / Tr[ρ] at site 3
expectation_value(ψ, (2, 4) => O ⊗ O)  # also works for non-adjacent multi-site operators

H = FiniteMPOHamiltonian(fill(ℂ^2, 5), i => O for i in 1:5)
envs = environments(ψ, H)          # reuse across repeated calls, e.g. across a sweep
expectation_value(ψ, H, envs)      # works for a full FiniteMPOHamiltonian / FiniteMPO too

ψ2, envs2 = timestep(ψ, H, 0.0, 0.01, TDVP())      # real-time evolution
ψ3, envs3 = timestep(ψ, H, 0.0, 0.01, TDVP2(; trscheme = truncrank(4)))  # two-site, with truncation

# finite temperature: starting from an infinite-temperature purification (identity map
# between physical and Kraus space) and evolving in imaginary time by β/2 gives ρ = e^{-βH}/Z
# exactly -- see test/timestep.jl for the full worked-out recipe

# dissipative dynamics: a Lindbladian is H plus on-site jump operators (site => L pairs)
L = randn(ComplexF64, ℂ^2, ℂ^2)
ℒ = FiniteLindbladian(H, (n => L for n in 1:5)...)   # same jump operator at every site

dt = 0.01
Ks = [kraus_operators(sum(dissipator_matrices(jump_operators(ℒ, n))), dt) for n in 1:5]
apply_kraus!(ψ, Ks)   # grows every site's Kraus leg in place; Tr[ρ] is preserved exactly
```

## API overview

- [`LPTN.LPTNTensor`](src/tensors.jl): a single-site LPTN tensor, `Vₗ ⊗ P ⊗ K ← Vᵣ`.
  Defined as `MPSKit.GenericMPSTensor{S,3}` — not a new type, so it inherits any generic
  MPSKit machinery written for that type for free. Build one directly with plain
  TensorKit constructors, e.g. `rand(Vₗ ⊗ P ⊗ K ← Vᵣ)`.
- [`lptn_physicalspace`](src/tensors.jl), [`krausspace`](src/tensors.jl),
  [`combinedspace`](src/tensors.jl): space accessors, defined both on a single
  `LPTNTensor` and on a whole `FiniteLPTN` chain (site accessors read whatever gauge is
  already cached, without forcing a gauge transform).
- [`FiniteLPTN`](src/states.jl): a finite chain of `LPTNTensor`s. An alias for
  `MPSKit.FiniteMPS{<:LPTNTensor{S}, <:MPSKit.MPSBondTensor{S}}` — again not a new type,
  so the whole chain reuses `FiniteMPS`'s lazily computed left/right/center gauges as-is.
- [`lptn_trace`](src/states.jl): `Tr[ρ]` for a `FiniteLPTN` chain.
- [`FiniteLindbladian`](src/lindbladian.jl): a `FiniteMPOHamiltonian` plus on-site jump
  operators (`site => L` pairs, [`jump_operators`](src/lindbladian.jl) to read them back).
- [`dissipator_matrices`](src/lindbladian.jl): turns a site's jump operators into
  `TensorMap`s on the vectorized (doubled) space `P⊗P' ← P⊗P'`; sum them for that site's
  full dissipator generator.
- [`kraus_operators`](src/lindbladian.jl): exponentiates a (summed) dissipator generator
  over a timestep `dt` into a single bundled Kraus isometry `P⊗Kaux ← P`, via the
  Choi–Jamiołkowski isomorphism — exactly trace-preserving for any `dt`, not just to
  first order.
- [`apply_kraus`](src/lindbladian.jl)/[`apply_kraus!`](src/lindbladian.jl): apply a
  bundled Kraus isometry to a single `LPTNTensor`, or one per site to a whole `FiniteLPTN`
  in place, growing each site's Kraus leg.

None of MPSKit's own API is re-exported here — re-exporting an ever-growing, arbitrary
subset would just be a maintenance liability against a fast-moving dependency. Add
`using MPSKit` alongside `using LPTN` for any of the following:

- `expectation_value`: `Tr[ρ O] / Tr[ρ]`. Works for a local operator (single-site,
  two-site, or arbitrary/non-adjacent multi-site), a `FiniteMPO`, or a full
  `FiniteMPOHamiltonian`, cross-checked against brute-force contraction in
  `test/expectation_values.jl`. `environments`, which these build on to avoid
  recomputing the same boundary contractions repeatedly (e.g. across a sweep), is
  reused the same way.
- `timestep`/`TDVP`/`TDVP2`: real- and imaginary-time evolution under a
  `FiniteMPOHamiltonian`. Imaginary time from an infinite-temperature purification gives
  finite-temperature states; see `test/timestep.jl` for the full recipe.
- `changebonds`/`OptimalExpand`: grows the bond dimension while leaving the state itself
  unchanged (verified: `Tr[ρ]` is preserved exactly), which is what makes it safe to
  combine with single-site TDVP (which cannot grow bond dimension on its own).
- `approximate`/`DMRG`/`DMRG2`: variationally fits a (possibly smaller-bond-dimension)
  `FiniteLPTN` to approximate `O * ϕ` for a `FiniteMPO`/`FiniteMPOHamiltonian` `O` and
  state `ϕ`; see `test/approximate.jl`.

## Testing

```julia
using Pkg
Pkg.test("LPTN")
```

Tests are split by topic under `test/`:
- `tensors.jl` — `LPTNTensor` construction and space accessors.
- `states.jl` — `FiniteLPTN` construction, site accessors, and `lptn_trace` (cross-checked
  against a brute-force contraction of the purified density operator).
- `expectation_values.jl` — `expectation_value` for single-site, two-site
  (adjacent/non-adjacent), and `FiniteMPOHamiltonian` operators, each cross-checked
  against a brute-force contraction of the purified density operator.
- `timestep.jl` — `Tr[ρ]` conservation under `TDVP`/`TDVP2` and bond growth under
  `OptimalExpand` (all with a genuinely Hermitian Hamiltonian — a non-Hermitian one
  would not conserve `Tr[ρ]` even under exact evolution, so the check would be
  meaningless), and finite-temperature imaginary-time evolution cross-checked against
  direct matrix exponentiation of `e^{-βH}`.
- `approximate.jl` — `approximate`/`DMRG2` recovers a `FiniteLPTN` from the identity
  `FiniteMPO` applied to it, at matching bond dimension.
- `lindbladian.jl` — `FiniteLindbladian` construction, `dissipator_matrices` and
  `kraus_operators` cross-checked against brute-force dense computation of the Lindblad
  dissipator/exact channel, and `apply_kraus`/`apply_kraus!` cross-checked against a
  brute-force contraction of the purified density operator with the channel applied
  classically at each site.
- `mpskit_assumptions.jl` — pins down the specific MPSKit behaviors this package depends
  on but does not own (e.g. `FiniteMPS`'s genericity over site-tensor rank), so a future
  MPSKit release that changes them fails loudly here rather than silently corrupting
  results downstream.

## Roadmap

- [x] Time evolution (real/imaginary time via `TDVP`/`TDVP2`, bond growth via
      `OptimalExpand`), finite temperature, and state compression via
      `approximate`/`DMRG`/`DMRG2` — all reused unchanged from MPSKit
- [x] `FiniteLindbladian` (Hamiltonian + on-site jump operators), exact Kraus operators
      via Choi-matrix exponentiation, and `apply_kraus!` to grow a `FiniteLPTN`'s Kraus
      legs under a per-site channel
- [ ] Trotterizing coherent (`H`, via `TDVP`) and dissipative (`apply_kraus!`) evolution
      together into a single repeated-timestep loop — steady-state search is out of
      scope for now
- [ ] Truncation/compression of the Kraus bond
- [ ] CI
