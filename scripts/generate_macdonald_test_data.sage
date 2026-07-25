#!/usr/bin/env sage
"""
Generate ground-truth Macdonald P-basis structure constants (ordinary,
top-degree Littlewood-Richardson coefficients: P_lambda * P_mu = sum_nu
c^nu_{lambda,mu} P_nu, so |nu| = |lambda| + |mu|) for all unordered pairs
of factor partitions lambda, mu with |lambda|, |mu| <= MAX_FACTOR_SIZE.

The monic P basis (coefficient of m_lambda in P_lambda is 1) is used
because it is canonical and convention-independent -- Sage's own "J" basis
uses a per-partition normalization that need not match the HookPrime/Hook
based J normalization this repo will use, so P is the safe cross-check
target. (Verified: for Jack, Sage's P-basis products match this repo's
`coefficient --normalization p` output exactly, and combining Sage's
P-basis values with this repo's own hook-factor formulas reproduces the
already-trusted jackStructureConstants6.m J-basis values exactly too.)

Sage's SymmetricFunctions(...).macdonald() basis only implements the
*ordinary* (non-shifted) Macdonald symmetric functions, so this script can
only produce the top-degree slice (|nu| = |lambda| + |mu|) of the more
general shifted/binomial-type structure constants that the Rust tool will
eventually compute. It is still a solid ground-truth source for that slice.

Usage: sage generate_macdonald_test_data.sage
"""

R = QQ['q', 't']
q, t = R.gens()
F = R.fraction_field()

Sym = SymmetricFunctions(F)
Mac = Sym.macdonald(q=q, t=t)
# Use the monic P basis: coefficient of m_lambda in P_lambda is 1, so this
# is a canonical, convention-independent normalization (unlike Sage's own
# "J" basis, whose overall per-partition scalar need not match the
# HookPrime/Hook-based J normalization this repo uses -- verified this by
# cross-checking Sage's Jack P-basis products against the already-trusted
# jackStructureConstants6.m J-basis data via the repo's own hook formulas).
P = Mac.P()

MAX_FACTOR_SIZE = 3

def factor_partitions(max_size):
    out = []
    for n in range(1, max_size + 1):
        out.extend(Partitions(n))
    return out

def format_partition(p):
    return "{" + ",".join(str(x) for x in p) + "}"

def format_rational_function(expr):
    # expr is an element of F = Frac(QQ[q,t]); print as q*t style, matching
    # the existing repo's `a^k` -> `q^i*t^j` convention as closely as
    # possible, using Sage's own factored string form.
    expr = expr.factor() if expr != 0 else expr
    return str(expr).replace(" ", "")

def main():
    factors = factor_partitions(MAX_FACTOR_SIZE)
    pairs = []
    for i, lam in enumerate(factors):
        for mu in factors[: i + 1]:
            pairs.append((lam, mu))

    records = []
    for lam, mu in pairs:
        prod = P(Partition(list(lam))) * P(Partition(list(mu)))
        prod = prod.map_coefficients(lambda c: F(c))
        for nu, coeff in prod:
            coeff = F(coeff)
            records.append((tuple(lam), tuple(mu), tuple(nu), coeff))

    out_path = "macdonaldPStructureConstants3.m"
    with open(out_path, "w") as f:
        for lam, mu, nu, coeff in records:
            value_str = format_rational_function(coeff)
            f.write(
                "MacdonaldPStructureConstant[{{{}, {}, {}}}]:={};\n".format(
                    format_partition(lam), format_partition(mu), format_partition(nu), value_str
                )
            )

    jsonl_path = "macdonaldPStructureConstants3.jsonl"
    import json
    with open(jsonl_path, "w") as f:
        for lam, mu, nu, coeff in records:
            f.write(json.dumps({
                "lambda": [int(x) for x in lam],
                "mu": [int(x) for x in mu],
                "nu": [int(x) for x in nu],
                "value": str(coeff),
            }) + "\n")

    print("wrote {} records to {} and {}".format(len(records), out_path, jsonl_path))

main()
