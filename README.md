# LPTN.jl

A Julia package for **locally purified tensor networks (LPTN)**, built on top of
[TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl) and
[MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl).

An LPTN represents a mixed state (density operator) `ρ` on a chain of physical sites as a
purification: a matrix product state `Λ` over the enlarged Hilbert space
`H_phys ⊗ H_kraus`, with `ρ = Tr_kraus[|Λ⟩⟨Λ|]`. Each site tensor of `Λ` therefore carries
one extra "Kraus" (purification) leg on top of the usual physical and left/right virtual
legs of an MPS tensor.

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
using LPTN, TensorKit
using MPSKit: FiniteMPOHamiltonian  # LPTN re-exports expectation_value/environments, but
                                    # constructing a Hamiltonian is still an MPSKit-level task

# a chain of 5 sites, each with physical space ℂ^2, Kraus space ℂ^2,
# and virtual bond dimension up to 3
ψ = FiniteLPTN(5, ℂ^2, ℂ^2, ℂ^3)

lptn_trace(ψ)                      # Tr[ρ], not norm(ψ) -- see "Design notes" below

O = randn(ComplexF64, ℂ^2, ℂ^2)
expectation_value(ψ, 3 => O)       # Tr[ρ O] / Tr[ρ] at site 3
expectation_value(ψ, (2, 4) => O ⊗ O)  # also works for non-adjacent multi-site operators

H = FiniteMPOHamiltonian(fill(ℂ^2, 5), i => O for i in 1:5)
envs = environments(ψ, H)          # reuse across repeated calls, e.g. across a sweep
expectation_value(ψ, H, envs)      # works for a full FiniteMPOHamiltonian / FiniteMPO too
```

## API overview

- [`LPTN.LPTNTensor`](src/tensors.jl): a single-site LPTN tensor, `Vₗ ⊗ P ⊗ K ← Vᵣ`.
  Defined as `MPSKit.GenericMPSTensor{S,3}` — not a new type, so it inherits any generic
  MPSKit machinery written for that type for free.
- [`LPTN.LPTNMapSpace`](src/tensors.jl): describes the space of an `LPTNTensor`; use
  `rand`/`randn`/`zeros` on one to build a tensor of that shape.
- [`lptn_physicalspace`](src/tensors.jl), [`krausspace`](src/tensors.jl),
  [`combinedspace`](src/tensors.jl): space accessors, defined both on a single
  `LPTNTensor` and on a whole `FiniteLPTN` chain (site accessors read whatever gauge is
  already cached, without forcing a gauge transform).
- [`FiniteLPTN`](src/states.jl): a finite chain of `LPTNTensor`s. An alias for
  `MPSKit.FiniteMPS{<:LPTNTensor{S}, <:MPSKit.MPSBondTensor{S}}` — again not a new type,
  so the whole chain reuses `FiniteMPS`'s lazily computed left/right/center gauges as-is.
- [`lptn_trace`](src/states.jl): `Tr[ρ]` for a `FiniteLPTN` chain.
- `expectation_value` (from MPSKit, re-exported): `Tr[ρ O] / Tr[ρ]`. Works for a local
  operator (single-site, two-site, or arbitrary/non-adjacent multi-site), a `FiniteMPO`,
  or a full `FiniteMPOHamiltonian` — all without any LPTN-specific code, see "Design
  notes" below. `MPSKit.environments`, which these build on to avoid recomputing the
  same boundary contractions repeatedly (e.g. across a sweep), is reused the same way.

## Design notes

**Why `LPTNTensor`/`FiniteLPTN` are aliases, not new types.** `MPSKit.GenericMPSTensor{S,N}`
and `MPSKit.FiniteMPS{A,B}` are already generic over the site-tensor rank `N` — this isn't
incidental, MPSKit's own transfer-matrix code (`transfer_left`/`transfer_right` in
`transfermatrix/transfer.jl`) has methods for `GenericMPSTensor{S,3}` explicitly labeled
*"Matrix Product Density Operators"* / *"density matrix transfer"*, and PEPSKit.jl reuses
the same pattern for its boundary MPS (with `N` varying per algorithm). So rather than
reimplementing gauge machinery or writing LPTN-specific transfer matrices, `LPTNTensor` and
`FiniteLPTN` are defined as aliases matching the existing generic types, and all of the
canonicalization, `expectation_value`, etc. machinery is inherited unchanged.

One consequence: since these are aliases rather than distinct types, defining new methods
on them is fine (the function is ours), but *overloading* an existing Base/MPSKit function
(e.g. `norm`, `tr`) directly on `FiniteLPTN` would be type piracy — the same tensor shape
is used by unrelated code (e.g. PEPSKit's boundary MPS), so it isn't "our" type to
redefine behavior on. This is why `Tr[ρ]` is exposed as the new function `lptn_trace`
rather than by overloading `norm`.

**`Tr[ρ] = norm(ψ)^2`, not `norm(ψ)`.** Tracing the purified density operator sums `|Λ|²`
over *both* physical and Kraus indices — the squared 2-norm of `ψ` as a plain vector, not
the (unsquared) MPS norm. `lptn_trace` is implemented as `norm(ψ)^2` rather than
`dot(ψ, ψ)`, since `MPSKit.FiniteMPS`'s `norm` is an O(1) read of the gauge-center tensor,
while `dot` does a full O(N) chain contraction that gives the same answer more slowly.

**Why `expectation_value` needs no LPTN-specific code.** For an operator `O` acting only
on the physical factor at a site (identity on that site's Kraus leg and everywhere else),
the purification identity gives `Tr[ρO] = ⟨Λ|(O⊗1_kraus)|Λ⟩`. MPSKit's canonical-gauge
expectation-value algorithm computes exactly `⟨Λ|(O_i ⊗ 1_{everything else it doesn't
touch})|Λ⟩ / ⟨Λ|Λ⟩` for any `GenericMPSTensor`-based chain, regardless of what the "extra"
legs represent. These coincide as soon as the Kraus leg is among the legs the operator
doesn't touch — which it always is here, since `O` is only ever given the physical space.

This holds not just for the single/two-site and arbitrary-multi-site `Pair` forms, but
also for a full `FiniteMPO`/`FiniteMPOHamiltonian`: their `expectation_value` builds
`MPSKit.environments` (the `GL`/`GR` boundary tensors) by growing them one site at a
time via `MPSKit.transfer_left`/`transfer_right`, which is exactly the same
`GenericMPSTensor{S,3}`-specific "density matrix transfer" machinery — so environments
built this way are just as valid for an LPTN chain as for a plain MPS, without change.

## Testing

```julia
using Pkg
Pkg.test("LPTN")
```

Tests are split by topic under `test/`:
- `tensors.jl` — `LPTNTensor`/`LPTNMapSpace` construction and space accessors.
- `states.jl` — `FiniteLPTN` construction, site accessors, and `lptn_trace` (cross-checked
  against a brute-force contraction of the purified density operator).
- `expectation_values.jl` — `expectation_value` for single-site, two-site
  (adjacent/non-adjacent), and `FiniteMPOHamiltonian` operators, each cross-checked
  against a brute-force contraction of the purified density operator.
- `mpskit_assumptions.jl` — pins down the specific MPSKit behaviors this package depends
  on but does not own (e.g. `FiniteMPS`'s genericity over site-tensor rank), so a future
  MPSKit release that changes them fails loudly here rather than silently corrupting
  results downstream.

## Roadmap

- [ ] Applying a quantum channel (Kraus operators) to an `LPTN` tensor
- [ ] Truncation/compression of the Kraus bond
- [ ] Time evolution (dissipative dynamics)
- [ ] CI
