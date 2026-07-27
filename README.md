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
using LPTN, TensorKit, MPSKit  # FiniteLPTN is a FiniteMPS, so its whole API is MPSKit's

# a chain of 5 sites, each with physical space ℂ^2, Kraus space ℂ^2,
# and virtual bond dimension up to 3
ψ = FiniteLPTN(5, ℂ^2, ℂ^2, ℂ^3)

lptn_trace(ψ)                      # Tr[ρ], not norm(ψ) -- see "Design notes" below

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

None of MPSKit's own API (`expectation_value`, `environments`, `timestep`/`TDVP`/`TDVP2`,
`changebonds`/`OptimalExpand`, ...) is re-exported here — since `FiniteLPTN` *is* a
`FiniteMPS`, essentially all of it applies already (see "Design notes"), and re-exporting
an ever-growing, arbitrary subset would just be a maintenance liability against a
fast-moving dependency. `using MPSKit` alongside `using LPTN` for any of it:

- `expectation_value`: `Tr[ρ O] / Tr[ρ]`. Works for a local operator (single-site,
  two-site, or arbitrary/non-adjacent multi-site), a `FiniteMPO`, or a full
  `FiniteMPOHamiltonian` — all without any LPTN-specific code, see "Design notes" below.
  `environments`, which these build on to avoid recomputing the same boundary
  contractions repeatedly (e.g. across a sweep), is reused the same way.
- `timestep`/`TDVP`/`TDVP2`: real- and imaginary-time evolution under a
  `FiniteMPOHamiltonian`, again with no LPTN-specific code. Imaginary time from an
  infinite-temperature purification gives finite-temperature states; see "Design notes"
  below and `test/timestep.jl` for the full recipe.
- `changebonds`/`OptimalExpand`: grows the bond dimension while leaving the state itself
  unchanged (verified: `Tr[ρ]` is preserved exactly), which is what makes it safe to
  combine with single-site TDVP (which cannot grow bond dimension on its own).
- `approximate`/`DMRG`/`DMRG2`: variationally fits a (possibly smaller-bond-dimension)
  `FiniteLPTN` to approximate `O * ϕ` for a `FiniteMPO`/`FiniteMPOHamiltonian` `O` and
  state `ϕ` — again with no LPTN-specific code, see "Design notes" below.

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

**Time evolution and bond expansion need no LPTN-specific code either.** TDVP's effective
single-site Hamiltonian action has two implementations in MPSKit: a Jordan-block-optimized
one hardcoded to rank-2 (`MPSTensor`) tensors, and a fully generic fallback
(`MPO_AC_Hamiltonian`) with an explicit `GenericMPSTensor{<:Any,3}` method. Since
`FiniteLPTN`'s tensors don't match the Jordan-restricted signature, Julia dispatches to the
generic fallback automatically — slower (it can't exploit the Hamiltonian's sparse Jordan
structure), but correct, applying the Hamiltonian only to the physical leg and leaving the
Kraus leg untouched, exactly like `expectation_value`. TDVP2's two-site update has a matching
rank-`(3,3)` fallback in `MPO_AC2_Hamiltonian`, and `OptimalExpand`'s bond-growing step
builds entirely on this same `MPO_AC_Hamiltonian`/`TransferMatrix` machinery plus rank-agnostic
QR/LQ/null-space utilities. All three were verified numerically (not just by source
inspection): real- and imaginary-time TDVP/TDVP2 conserve `Tr[ρ]` to machine precision for a
Hermitian Hamiltonian, `OptimalExpand` grows bond dimension while leaving `Tr[ρ]` exactly
unchanged, and imaginary-time evolution from an infinite-temperature purification reproduces
`e^{-βH}/Z` to machine precision against direct matrix exponentiation.

**`approximate` needs no LPTN-specific code either.** `approximate!`'s `DMRG`/`DMRG2`
algorithms are built on `AC_projection`/`AC2_projection`, which are themselves thin
wrappers around the same `MPO_AC_Hamiltonian`/`MPO_AC2_Hamiltonian` machinery already
covered above — so the same rank-`3`/rank-`(3,3)` fallbacks apply here too. Verified by
fitting a same-bond-dimension `FiniteLPTN` to the identity `FiniteMPO` applied to another
LPTN chain: `DMRG2` converges in 2 iterations and recovers the original state essentially
exactly (`dot(ψ_fit, ϕ) ≈ 1`), as expected since no truncation is actually needed at
matching bond dimension.

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
- `timestep.jl` — `Tr[ρ]` conservation under `TDVP`/`TDVP2` and bond growth under
  `OptimalExpand` (all with a genuinely Hermitian Hamiltonian — a non-Hermitian one
  would not conserve `Tr[ρ]` even under exact evolution, so the check would be
  meaningless), and finite-temperature imaginary-time evolution cross-checked against
  direct matrix exponentiation of `e^{-βH}`.
- `approximate.jl` — `approximate`/`DMRG2` recovers a `FiniteLPTN` from the identity
  `FiniteMPO` applied to it, at matching bond dimension.
- `mpskit_assumptions.jl` — pins down the specific MPSKit behaviors this package depends
  on but does not own (e.g. `FiniteMPS`'s genericity over site-tensor rank), so a future
  MPSKit release that changes them fails loudly here rather than silently corrupting
  results downstream.

## Roadmap

- [x] Time evolution (real/imaginary time via `TDVP`/`TDVP2`, bond growth via
      `OptimalExpand`), finite temperature, and state compression via
      `approximate`/`DMRG`/`DMRG2` — all reused unchanged from MPSKit
- [ ] Constructing a dissipative (Lindbladian) generator acting on the physical+Kraus
      legs — the shared prerequisite for either option below
- [ ] Either: Trotterized application of local Kraus/channel operators (repeated small
      time steps, similar to what `TDVP` already does for a Hamiltonian), or: a direct
      steady-state search via non-Hermitian DMRG, targeting the eigenvalue with the
      largest real part (`:LR`) of the generator, since a physical Lindbladian's unique
      steady state sits at exactly `λ = 0`. MPSKit's DMRG currently hardcodes the `:SR`
      selector (and defaults its eigensolver to `ishermitian=true`, though that part is
      already configurable elsewhere in MPSKit), so this route would need either a small
      upstream change or a local workaround.
- [ ] Truncation/compression of the Kraus bond
- [ ] CI
