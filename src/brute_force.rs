//! A completely independent "brute force" computation of ordinary Macdonald
//! P-basis structure constants, for cross-checking [`crate::macdonald`]'s
//! interpolation/shifted-theory-based algorithms. Shares no code with them
//! beyond basic partition combinatorics and the `Poly2`/`RationalFunction2`
//! number types -- reuses none of `adjacent_binomial_coefficient`,
//! `normalizing_factor`, `shifted_macdonald_evaluate`, etc.
//!
//! Strategy: work entirely in the power-sum basis, where multiplication is
//! trivial (`p_lambda * p_mu = p_{lambda merged with mu}`) and the Macdonald
//! inner product is diagonal (`<p_lambda, p_mu> = delta * z_lambda(q,t)`).
//! Build each P_lambda via Gram-Schmidt starting from m_lambda's power-sum
//! expansion (obtained via plain rational linear algebra -- evaluate both
//! bases at several integer points and solve the resulting system exactly
//! over Q -- rather than recalling/re-deriving a combinatorial transition
//! formula from memory). Then:
//!
//!   c^nu_{lambda,mu} = <P_lambda * P_mu, Q_nu> = <P_lambda * P_mu, P_nu> / <P_nu, P_nu>
//!
//! using Q_nu = P_nu / <P_nu,P_nu> (from <P_nu,Q_nu> = 1).

use std::collections::HashMap;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use crate::macdonald::{hook_c_product_qt, Poly2, RationalFunction2};
use crate::{conjugate_partition, integer_partitions, partition_size, trim_partition, Partition};

type Rat = Ratio<BigInt>;

/// Standard next-permutation-in-lexicographic-order; returns false (leaving
/// `a` unchanged) once `a` is the final (descending) permutation.
fn next_permutation(a: &mut [usize]) -> bool {
    let n = a.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    while i > 0 && a[i - 1] >= a[i] {
        i -= 1;
    }
    if i == 0 {
        return false;
    }
    let mut j = n - 1;
    while a[j] <= a[i - 1] {
        j -= 1;
    }
    a.swap(i - 1, j);
    a[i..].reverse();
    true
}

/// All distinct permutations of a multiset, via repeated next-permutation
/// starting from ascending order.
fn distinct_permutations(items: &[usize]) -> Vec<Vec<usize>> {
    let mut a = items.to_vec();
    a.sort_unstable();
    let mut result = vec![a.clone()];
    while next_permutation(&mut a) {
        result.push(a.clone());
    }
    result
}

/// p_kappa(x) = prod_i (sum_j x_j^kappa_i), plain integer evaluation.
fn evaluate_power_sum(kappa: &[usize], x: &[BigInt]) -> BigInt {
    let mut result = BigInt::one();
    for &k in kappa {
        let mut s = BigInt::zero();
        for xi in x {
            s += num_traits::pow(xi.clone(), k);
        }
        result *= s;
    }
    result
}

/// m_lambda(x) = sum over distinct permutations of lambda (zero-padded to
/// len(x)) of the corresponding monomial, plain integer evaluation.
fn evaluate_monomial(lambda: &[usize], x: &[BigInt]) -> BigInt {
    let n = x.len();
    let mut exps = lambda.to_vec();
    exps.resize(n, 0);
    let mut total = BigInt::zero();
    for perm in distinct_permutations(&exps) {
        let mut term = BigInt::one();
        for (xi, &e) in x.iter().zip(perm.iter()) {
            term *= num_traits::pow(xi.clone(), e);
        }
        total += term;
    }
    total
}

/// A deterministic pseudo-random point in n-space (small positive
/// integers), keyed by `seed` and `point_index` so distinct (seed,
/// point_index) pairs give unrelated points -- unlike a single
/// affine-in-point_index family, which provably fails to span enough
/// dimensions once p(n) grows past n+1 (see `compute_m_to_p_transition`).
/// Plain splitmix64, not for cryptographic use, just for well-mixed
/// deterministic integers.
fn random_point(n: usize, seed: u64, point_index: usize) -> Vec<BigInt> {
    let mut state = 0x9E3779B97F4A7C15u64
        ^ seed.wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (point_index as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            BigInt::from(2 + (z % 97))
        })
        .collect()
}

/// Gauss-Jordan elimination on the augmented system [a | b], reducing `a`
/// to the identity and `b` to the solution (a p x p system with p x p RHS,
/// solving a * result = b_original for `result`, left in `b`). Returns
/// false (leaving `a`/`b` partially modified) if the system is singular,
/// rather than panicking, since the caller retries with different
/// evaluation points in that case.
fn gauss_jordan_solve(a: &mut [Vec<Rat>], b: &mut [Vec<Rat>]) -> bool {
    let p = a.len();
    for col in 0..p {
        let mut pivot_row = col;
        while pivot_row < p && a[pivot_row][col].is_zero() {
            pivot_row += 1;
        }
        if pivot_row >= p {
            return false;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot_val = a[col][col].clone();
        for j in 0..p {
            a[col][j] = std::mem::replace(&mut a[col][j], Rat::zero()) / pivot_val.clone();
        }
        for j in 0..b[col].len() {
            b[col][j] = std::mem::replace(&mut b[col][j], Rat::zero()) / pivot_val.clone();
        }

        for row in 0..p {
            if row == col {
                continue;
            }
            let factor = a[row][col].clone();
            if factor.is_zero() {
                continue;
            }
            for j in 0..p {
                let sub = factor.clone() * a[col][j].clone();
                a[row][j] -= sub;
            }
            for j in 0..b[row].len() {
                let sub = factor.clone() * b[col][j].clone();
                b[row][j] -= sub;
            }
        }
    }
    true
}

/// For every partition lambda of n, its expansion (in some target basis
/// indexed the same way, here always partitions of n) f_lambda = sum_kappa
/// c_kappa p_kappa, as an exact rational (q,t-independent) map kappa ->
/// c_kappa, where f is given by `evaluate` (e.g. the monomial or Schur
/// basis). Computed via plain linear algebra (evaluate both f and the
/// power-sum basis at several integer points, solve exactly) rather than a
/// recalled combinatorial transition formula.
fn compute_basis_to_p_transition(
    n: usize,
    evaluate: impl Fn(&[usize], &[BigInt]) -> BigInt,
) -> HashMap<Partition, HashMap<Partition, Rat>> {
    if n == 0 {
        let mut result = HashMap::new();
        result.insert(Vec::new(), HashMap::from([(Vec::new(), Rat::one())]));
        return result;
    }

    let partitions = integer_partitions(n);
    let p = partitions.len();

    // Evaluation points must NOT all lie in one affine-in-k family (e.g.
    // x^(k) = base + stride*k for a single scalar k): every entry of the
    // resulting matrix row is then a degree-<=n polynomial in k, a space of
    // dimension only n+1, so once p(n) > n+1 the columns are provably
    // linearly dependent no matter what stride/base is chosen (this is
    // exactly what made every stride from 7 to 1006 fail at n=4). Use a
    // deterministic PRNG instead, so each point is unrelated to the others,
    // with a retry loop (reseeded) as a safety net against genuine
    // (non-structural) unlucky draws.
    let b_matrix;
    let mut seed = 0u64;
    loop {
        let points: Vec<Vec<BigInt>> = (0..p).map(|k| random_point(n, seed, k)).collect();

        let mut a_matrix: Vec<Vec<Rat>> = points
            .iter()
            .map(|pt| {
                partitions
                    .iter()
                    .map(|kappa| Rat::from_integer(evaluate_power_sum(kappa, pt)))
                    .collect()
            })
            .collect();

        let mut candidate_b: Vec<Vec<Rat>> = points
            .iter()
            .map(|pt| {
                partitions
                    .iter()
                    .map(|lambda| Rat::from_integer(evaluate(lambda, pt)))
                    .collect()
            })
            .collect();

        if gauss_jordan_solve(&mut a_matrix, &mut candidate_b) {
            b_matrix = candidate_b;
            break;
        }
        seed += 1;
        assert!(
            seed < 1000,
            "could not find non-singular evaluation points for degree {n} after 1000 attempts"
        );
    }

    // b_matrix[k][i] is now the coefficient of p_{partitions[k]} in f_{partitions[i]}.
    let mut result = HashMap::new();
    for (i, lambda) in partitions.iter().enumerate() {
        let mut coeffs = HashMap::new();
        for (k, kappa) in partitions.iter().enumerate() {
            let c = b_matrix[k][i].clone();
            if !c.is_zero() {
                coeffs.insert(kappa.clone(), c);
            }
        }
        result.insert(lambda.clone(), coeffs);
    }
    result
}

fn compute_m_to_p_transition(n: usize) -> HashMap<Partition, HashMap<Partition, Rat>> {
    compute_basis_to_p_transition(n, evaluate_monomial)
}

/// s_lambda(x) = sum over SSYT of shape lambda with entries in 1..=len(x)
/// of the corresponding monomial, plain integer evaluation (reusing the
/// same tableau enumerator as the Jack/Macdonald evaluation code).
fn evaluate_schur(lambda: &[usize], x: &[BigInt]) -> BigInt {
    let n = x.len();
    let mut total = BigInt::zero();
    for tableau in crate::semistandard_tableaux(lambda, n) {
        let mut counts = vec![0u32; n + 1];
        for row in &tableau {
            for &entry in row {
                counts[entry] += 1;
            }
        }
        let mut term = BigInt::one();
        for (value, &count) in counts.iter().enumerate().skip(1) {
            if count > 0 {
                term *= num_traits::pow(x[value - 1].clone(), count as usize);
            }
        }
        total += term;
    }
    total
}

fn compute_s_to_p_transition(n: usize) -> HashMap<Partition, HashMap<Partition, Rat>> {
    compute_basis_to_p_transition(n, evaluate_schur)
}

/// h_r(x) = complete homogeneous symmetric polynomial of degree r, via the
/// standard DP for the coefficients of prod_i 1/(1-x_i*z) (dp[j] after
/// processing x_1..x_i is h_j(x_1,...,x_i)).
fn evaluate_homogeneous_single(r: usize, x: &[BigInt]) -> BigInt {
    let mut dp = vec![BigInt::zero(); r + 1];
    dp[0] = BigInt::one();
    for xi in x {
        for j in 1..=r {
            let contribution = xi * &dp[j - 1];
            dp[j] += contribution;
        }
    }
    dp[r].clone()
}

/// h_mu(x) = prod_i h_{mu_i}(x).
fn evaluate_homogeneous(mu: &[usize], x: &[BigInt]) -> BigInt {
    mu.iter()
        .map(|&r| evaluate_homogeneous_single(r, x))
        .product()
}

fn compute_h_to_p_transition(n: usize) -> HashMap<Partition, HashMap<Partition, Rat>> {
    compute_basis_to_p_transition(n, evaluate_homogeneous)
}

/// Classical integer Littlewood-Richardson coefficients: s_lambda * s_mu =
/// sum_nu c^nu_{lambda,mu} s_nu, for all nu of degree |lambda|+|mu|.
/// Computed the same way as the basis transitions above (evaluate both
/// sides at several integer points, solve exactly for the integer
/// coefficients) rather than the combinatorial LR tableau rule, to avoid
/// re-deriving/verifying another formula from memory.
fn schur_lr_coefficients(lambda: &[usize], mu: &[usize]) -> HashMap<Partition, BigInt> {
    let n = partition_size(lambda) + partition_size(mu);
    if n == 0 {
        return HashMap::from([(Vec::new(), BigInt::one())]);
    }
    let partitions = integer_partitions(n);
    let p = partitions.len();

    let mut seed = 0u64;
    loop {
        let points: Vec<Vec<BigInt>> = (0..p).map(|k| random_point(n, seed, k)).collect();

        let mut a_matrix: Vec<Vec<Rat>> = points
            .iter()
            .map(|pt| {
                partitions
                    .iter()
                    .map(|nu| Rat::from_integer(evaluate_schur(nu, pt)))
                    .collect()
            })
            .collect();

        let rhs: Vec<Rat> = points
            .iter()
            .map(|pt| Rat::from_integer(evaluate_schur(lambda, pt) * evaluate_schur(mu, pt)))
            .collect();
        let mut b_matrix: Vec<Vec<Rat>> = rhs.into_iter().map(|r| vec![r]).collect();

        if gauss_jordan_solve(&mut a_matrix, &mut b_matrix) {
            let mut result = HashMap::new();
            for (i, nu) in partitions.iter().enumerate() {
                let c = &b_matrix[i][0];
                assert!(
                    c.denom().is_one(),
                    "LR coefficient came out non-integral: {nu:?} -> {c}"
                );
                if !c.is_zero() {
                    result.insert(nu.clone(), c.numer().clone());
                }
            }
            return result;
        }
        seed += 1;
        assert!(
            seed < 1000,
            "could not find non-singular evaluation points for degree {n} after 1000 attempts"
        );
    }
}

/// Inverts a square rational matrix via Gauss-Jordan elimination (on an
/// augmented identity), panicking on a singular input -- change-of-basis
/// matrices between two graded bases of the same degree are never
/// singular, so a failure here would indicate a real bug upstream.
fn invert_rat_matrix(matrix: &[Vec<Rat>]) -> Vec<Vec<Rat>> {
    let p = matrix.len();
    let mut a: Vec<Vec<Rat>> = matrix.to_vec();
    let mut identity: Vec<Vec<Rat>> = (0..p)
        .map(|i| {
            let mut row = vec![Rat::zero(); p];
            row[i] = Rat::one();
            row
        })
        .collect();
    let ok = gauss_jordan_solve(&mut a, &mut identity);
    assert!(ok, "change-of-basis matrix was unexpectedly singular");
    identity
}

/// Returns (partitions of n, M) where M is the inverse of the matrix
/// sending power-sum coordinates to Schur coordinates, i.e. for a
/// power-sum vector c (indexed the same way as `partitions`) representing
/// f = sum_kappa c_kappa p_kappa, the vector `M * c` gives f's coordinates
/// in the Schur basis (f = sum_mu d_mu s_mu).
fn schur_change_of_basis_inverse(n: usize) -> (Vec<Partition>, Vec<Vec<Rat>>) {
    let mut partitions = integer_partitions(n.max(1));
    if n == 0 {
        partitions = vec![Vec::new()];
    }
    let s_to_p = compute_s_to_p_transition(n);
    let p = partitions.len();

    // a[row=kappa][col=mu] = coefficient of p_kappa in s_mu, i.e. the matrix
    // whose columns are s_mu expressed in power-sum coordinates.
    let mut a = vec![vec![Rat::zero(); p]; p];
    for (col, mu) in partitions.iter().enumerate() {
        for (row, kappa) in partitions.iter().enumerate() {
            if let Some(c) = s_to_p[mu].get(kappa) {
                a[row][col] = c.clone();
            }
        }
    }
    (partitions, invert_rat_matrix(&a))
}

/// Applies the power-sum-to-Schur change of basis (see
/// `schur_change_of_basis_inverse`) to a power-sum-coordinate vector,
/// returning the same quantity in Schur coordinates.
fn schur_coordinates_of_power_sum_vector(
    partitions: &[Partition],
    inverse: &[Vec<Rat>],
    v: &PowerSumVector,
) -> HashMap<Partition, RationalFunction2> {
    let c: Vec<RationalFunction2> = partitions
        .iter()
        .map(|kappa| v.get(kappa).cloned().unwrap_or_else(RationalFunction2::zero))
        .collect();

    let mut result = HashMap::new();
    for (row, mu) in partitions.iter().enumerate() {
        let mut acc = RationalFunction2::zero();
        for (col, coeff) in inverse[row].iter().enumerate() {
            if coeff.is_zero() || c[col].is_zero() {
                continue;
            }
            acc = acc + rat_to_rf2(coeff) * c[col].clone();
        }
        if !acc.is_zero() {
            result.insert(mu.clone(), acc);
        }
    }
    result
}

fn permutation_sign(perm: &[usize]) -> i32 {
    let mut inversions = 0usize;
    for i in 0..perm.len() {
        for j in (i + 1)..perm.len() {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1
    } else {
        -1
    }
}

/// The k x k determinant from [LLM1998] Corollary 4.3, exactly as
/// implemented by Sage's `_creation_by_determinant_helper`: `part`
/// (zero-padded to length k) indexes an S-basis vector, and this computes
/// the action of the column-adding creation operator on it, expressed as
/// a Z[q,t]-linear combination of products of complete homogeneous
/// symmetric functions h_r ("h-basis coordinates": a term
/// h_{r_1}*h_{r_2}*...*h_{r_k} is keyed by the multiset {r_1,...,r_k}
/// sorted descending, dropping any r_i=0 factors since h_0=1).
fn creation_determinant_h_coords(k: usize, part: &[usize]) -> HashMap<Partition, Poly2> {
    assert!(part.len() <= k, "the column to add is too small");
    let mut padded = part.to_vec();
    padded.resize(k, 0);

    // matrix[i][j] = Some((coeff, h_index)), or None for a structural zero.
    let mut matrix: Vec<Vec<Option<(Poly2, usize)>>> = vec![vec![None; k]; k];
    for i in 0..k {
        let j0 = ((i as isize + 1) - 2 - padded[i] as isize).max(0) as usize;
        for j in j0..k {
            let value = padded[i] as isize + j as isize - i as isize + 1;
            assert!(
                value >= 0,
                "creation_determinant_h_coords: unexpected negative h-index"
            );
            let value = value as usize;
            let t_exponent = k - (j + 1);
            let coeff = Poly2::one() - Poly2::monomial(value, t_exponent, BigInt::one());
            matrix[i][j] = Some((coeff, value));
        }
    }

    let indices: Vec<usize> = (0..k).collect();
    let mut result: HashMap<Partition, Poly2> = HashMap::new();
    for perm in distinct_permutations(&indices) {
        let mut term = Poly2::one();
        let mut h_parts: Partition = Vec::new();
        let mut structural_zero = false;
        for (i, &pj) in perm.iter().enumerate() {
            match &matrix[i][pj] {
                None => {
                    structural_zero = true;
                    break;
                }
                Some((coeff, value)) => {
                    term = term * coeff.clone();
                    if term.is_zero() {
                        structural_zero = true;
                        break;
                    }
                    if *value > 0 {
                        h_parts.push(*value);
                    }
                }
            }
        }
        if structural_zero {
            continue;
        }
        h_parts.sort_unstable_by(|a, b| b.cmp(a));
        let signed = if permutation_sign(&perm) < 0 {
            -term
        } else {
            term
        };
        let entry = result.entry(h_parts).or_insert_with(Poly2::zero);
        *entry = entry.clone() + signed;
    }
    result
}

/// Converts an "h-basis coordinates" expression (as returned by
/// `creation_determinant_h_coords`) of total degree `m` into Schur
/// coordinates, via h -> power-sum (q,t-independent, plain rationals) ->
/// Schur (also q,t-independent).
fn h_expression_to_schur(
    expr: &HashMap<Partition, Poly2>,
    m: usize,
) -> HashMap<Partition, RationalFunction2> {
    let h_to_p = compute_h_to_p_transition(m);
    let mut ps_vector: PowerSumVector = HashMap::new();
    for (h_mu, poly_coeff) in expr {
        if poly_coeff.is_zero() {
            continue;
        }
        let rf2_coeff = RationalFunction2::new(poly_coeff.clone(), Poly2::one());
        if let Some(p_coords) = h_to_p.get(h_mu) {
            for (kappa, rat_c) in p_coords {
                let entry = ps_vector
                    .entry(kappa.clone())
                    .or_insert_with(RationalFunction2::zero);
                let updated = entry.clone() + rf2_coeff.clone() * rat_to_rf2(rat_c);
                *entry = updated;
            }
        }
    }
    let (partitions, p_to_s) = schur_change_of_basis_inverse(m);
    schur_coordinates_of_power_sum_vector(&partitions, &p_to_s, &ps_vector)
}

/// The full action of the LLM column-adding creation operator on the
/// S-basis vector `S[part]` (Sage's `_creation_by_determinant_helper`):
/// the k x k determinant, converted to Schur coordinates.
///
/// Note there is deliberately no "lift back into S-basis coordinates"
/// step here, despite Sage's own code calling `S._from_element(res)` at
/// this point: `_from_element` (defined generically in
/// `sfa.SymmetricFunctionAlgebra_generic`, not overridden for the `S`
/// basis) is NOT a change-of-basis -- its docstring is explicit that it
/// "returns the element of self with the same internal structure as x",
/// i.e. it just reinterprets x's existing coefficient dictionary as
/// coordinates in the new basis, unchanged. So Sage's "S basis" here is
/// purely a bookkeeping label: each `creation(k)` step's Schur-coordinate
/// output is fed directly back in as the next step's partition labels,
/// with no actual q,t-rescaling in between (confirmed by direct
/// comparison against `sage`'s own intermediate values for `k=1,
/// part=[]`: naively "inverting" the S<->Schur change of basis here, as
/// an earlier version of this function did, gives (1-q)^2/(1-t) instead
/// of Sage's actual (1-q)).
fn creation_operator_action(k: usize, part: &[usize]) -> HashMap<Partition, RationalFunction2> {
    let m = partition_size(part) + k;
    let det_h = creation_determinant_h_coords(k, part);
    h_expression_to_schur(&det_h, m)
}

/// The Macdonald `J` (integral form) polynomial `J_lambda(x;q,t)` in the
/// Schur basis, replicating Sage's `MacdonaldPolynomials_j._to_s`: fold
/// the creation operator over `reversed(lambda)` starting from the
/// constant `1`, then apply the `omega_qt` automorphism (conjugates every
/// partition key, coefficients unchanged -- see
/// `MacdonaldPolynomials_s::_omega_qt_in_schurs`) and finally swap q<->t
/// in every coefficient.
fn j_lambda_in_schur(lambda: &[usize]) -> HashMap<Partition, RationalFunction2> {
    let lambda = trim_partition(lambda);
    let mut res: HashMap<Partition, RationalFunction2> = HashMap::new();
    res.insert(Vec::new(), RationalFunction2::one());
    for &k in lambda.iter().rev() {
        let mut next: HashMap<Partition, RationalFunction2> = HashMap::new();
        for (mu, c_mu) in &res {
            if c_mu.is_zero() {
                continue;
            }
            for (mu2, c2) in creation_operator_action(k, mu) {
                if c2.is_zero() {
                    continue;
                }
                let entry = next.entry(mu2).or_insert_with(RationalFunction2::zero);
                let updated = entry.clone() + c_mu.clone() * c2;
                *entry = updated;
            }
        }
        res = next;
    }

    let mut result = HashMap::new();
    for (mu, c) in res {
        if c.is_zero() {
            continue;
        }
        result.insert(conjugate_partition(&mu), c.swap_qt());
    }
    result
}

/// The monic Macdonald `P_lambda(x;q,t)` in the Schur basis: `J_lambda`
/// (via `j_lambda_in_schur`) divided by the scalar `c_lambda(q,t)`
/// relating the two (Macdonald's book VI (6.19), see `hook_c_product_qt`).
fn p_lambda_in_schur(lambda: &[usize]) -> HashMap<Partition, RationalFunction2> {
    let lambda = trim_partition(lambda);
    let c_lambda = RationalFunction2::new(hook_c_product_qt(&lambda), Poly2::one());
    j_lambda_in_schur(&lambda)
        .into_iter()
        .map(|(mu, c)| (mu, c / c_lambda.clone()))
        .collect()
}

/// Gauss-Jordan elimination over `RationalFunction2` (mirrors
/// `gauss_jordan_solve`, but for the q,t-dependent Schur<->P
/// change-of-basis matrix built from `p_lambda_in_schur`, needed to
/// extract a single P_nu coefficient from a Schur-basis product).
fn gauss_jordan_solve_rf2(
    a: &mut [Vec<RationalFunction2>],
    b: &mut [Vec<RationalFunction2>],
) -> bool {
    let p = a.len();
    for col in 0..p {
        let mut pivot_row = col;
        while pivot_row < p && a[pivot_row][col].is_zero() {
            pivot_row += 1;
        }
        if pivot_row >= p {
            return false;
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot_val = a[col][col].clone();
        for j in 0..p {
            a[col][j] =
                std::mem::replace(&mut a[col][j], RationalFunction2::zero()) / pivot_val.clone();
        }
        for j in 0..b[col].len() {
            b[col][j] =
                std::mem::replace(&mut b[col][j], RationalFunction2::zero()) / pivot_val.clone();
        }

        for row in 0..p {
            if row == col {
                continue;
            }
            let factor = a[row][col].clone();
            if factor.is_zero() {
                continue;
            }
            for j in 0..p {
                let sub = factor.clone() * a[col][j].clone();
                a[row][j] = a[row][j].clone() - sub;
            }
            for j in 0..b[row].len() {
                let sub = factor.clone() * b[col][j].clone();
                b[row][j] = b[row][j].clone() - sub;
            }
        }
    }
    true
}

/// Inverts a square `RationalFunction2` matrix, panicking on a singular
/// input (a genuine change-of-basis matrix between two graded bases of
/// the same degree is never singular).
fn invert_rf2_matrix(matrix: &[Vec<RationalFunction2>]) -> Vec<Vec<RationalFunction2>> {
    let p = matrix.len();
    let mut a: Vec<Vec<RationalFunction2>> = matrix.to_vec();
    let mut identity: Vec<Vec<RationalFunction2>> = (0..p)
        .map(|i| {
            let mut row = vec![RationalFunction2::zero(); p];
            row[i] = RationalFunction2::one();
            row
        })
        .collect();
    let ok = gauss_jordan_solve_rf2(&mut a, &mut identity);
    assert!(ok, "Schur <-> P change-of-basis matrix was unexpectedly singular");
    identity
}

/// (partitions of n, Schur-to-P matrix [row=P, col=schur]), built from
/// `p_lambda_in_schur` (the LLM creation-operator construction) rather
/// than Gram-Schmidt, and inverted once per degree.
fn schur_to_p_matrix(n: usize) -> (Vec<Partition>, Vec<Vec<RationalFunction2>>) {
    let mut partitions = integer_partitions(n.max(1));
    if n == 0 {
        partitions = vec![Vec::new()];
    }
    let p = partitions.len();
    let p_basis: HashMap<Partition, HashMap<Partition, RationalFunction2>> = partitions
        .iter()
        .map(|lambda| (lambda.clone(), p_lambda_in_schur(lambda)))
        .collect();

    // forward[row=schur][col=P]: P[lambda] expressed in Schur coordinates.
    let mut forward = vec![vec![RationalFunction2::zero(); p]; p];
    for (col, lambda) in partitions.iter().enumerate() {
        for (row, mu) in partitions.iter().enumerate() {
            if let Some(c) = p_basis[lambda].get(mu) {
                forward[row][col] = c.clone();
            }
        }
    }
    (partitions, invert_rf2_matrix(&forward))
}

/// c^nu_{lambda,mu} in the monic Macdonald P basis, computed via the LLM
/// creation-operator construction throughout (no Gram-Schmidt anywhere):
/// P_lambda, P_mu in Schur coordinates (`p_lambda_in_schur`), multiplied
/// using the classical integer Littlewood-Richardson coefficients, then
/// converted to the P basis via `schur_to_p_matrix` (also creation-
/// operator-built) to read off the P_nu coefficient.
pub fn creation_operator_structure_constant(
    lambda: &[usize],
    mu: &[usize],
    nu: &[usize],
) -> RationalFunction2 {
    let lambda = trim_partition(lambda);
    let mu = trim_partition(mu);
    let nu = trim_partition(nu);

    if partition_size(&nu) != partition_size(&lambda) + partition_size(&mu) {
        return RationalFunction2::zero();
    }
    let n = partition_size(&nu);

    let p_lambda_schur = p_lambda_in_schur(&lambda);
    let p_mu_schur = p_lambda_in_schur(&mu);

    let mut lr_cache: HashMap<(Partition, Partition), HashMap<Partition, BigInt>> = HashMap::new();
    let mut product_schur: HashMap<Partition, RationalFunction2> = HashMap::new();
    for (a, ca) in &p_lambda_schur {
        if ca.is_zero() {
            continue;
        }
        for (b, cb) in &p_mu_schur {
            if cb.is_zero() {
                continue;
            }
            let lr = lr_cache
                .entry((a.clone(), b.clone()))
                .or_insert_with(|| schur_lr_coefficients(a, b));
            let coeff_ab = ca.clone() * cb.clone();
            for (s_nu, lr_coeff) in lr.iter() {
                if lr_coeff.is_zero() {
                    continue;
                }
                let entry = product_schur
                    .entry(s_nu.clone())
                    .or_insert_with(RationalFunction2::zero);
                let updated =
                    entry.clone() + coeff_ab.clone() * rat_to_rf2(&Rat::from_integer(lr_coeff.clone()));
                *entry = updated;
            }
        }
    }

    let (partitions, schur_to_p) = schur_to_p_matrix(n);
    let Some(row) = partitions.iter().position(|p| p == &nu) else {
        return RationalFunction2::zero();
    };
    let mut acc = RationalFunction2::zero();
    for (col, part) in partitions.iter().enumerate() {
        let coeff = &schur_to_p[row][col];
        if coeff.is_zero() {
            continue;
        }
        if let Some(c) = product_schur.get(part) {
            if !c.is_zero() {
                acc = acc + coeff.clone() * c.clone();
            }
        }
    }
    acc
}

/// c^nu_{lambda,mu} in the monic Macdonald P basis, computed by replicating
/// Sage's own internal technique: convert P_lambda, P_mu to the Schur
/// basis, multiply using the classical INTEGER Littlewood-Richardson
/// coefficients (no q,t arithmetic in this step at all), convert the
/// product back to power-sum coordinates, then read off the P_nu
/// coefficient via the diagonal inner product (same final step as
/// [`brute_force_structure_constant`]).
///
/// Note this still uses this module's own Gram-Schmidt (`compute_p_basis`)
/// to get P_lambda/P_mu/P_nu into power-sum coordinates in the first place
/// -- unlike Sage, which builds the Schur<->Macdonald-P transition directly
/// via the Lapointe-Lascoux-Morse creation-operator formula. So this
/// algorithm only isolates the benefit of doing the *multiplication* step
/// in pure integers; it does not avoid the Gram-Schmidt cost of building
/// P_lambda/P_mu/P_nu themselves.
pub fn schur_sandwich_structure_constant(
    lambda: &[usize],
    mu: &[usize],
    nu: &[usize],
) -> RationalFunction2 {
    let lambda = trim_partition(lambda);
    let mu = trim_partition(mu);
    let nu = trim_partition(nu);

    if partition_size(&nu) != partition_size(&lambda) + partition_size(&mu) {
        return RationalFunction2::zero();
    }

    let n_lambda = partition_size(&lambda);
    let n_mu = partition_size(&mu);
    let n = partition_size(&nu);

    let basis_lambda = compute_p_basis(n_lambda);
    let basis_mu = compute_p_basis(n_mu);
    let basis_nu = compute_p_basis(n);

    let p_lambda_ps = &basis_lambda[&lambda];
    let p_mu_ps = &basis_mu[&mu];
    let p_nu_ps = match basis_nu.get(&nu) {
        Some(p) => p,
        None => return RationalFunction2::zero(),
    };

    let (partitions_lambda, inv_lambda) = schur_change_of_basis_inverse(n_lambda);
    let (partitions_mu, inv_mu) = schur_change_of_basis_inverse(n_mu);

    let p_lambda_schur =
        schur_coordinates_of_power_sum_vector(&partitions_lambda, &inv_lambda, p_lambda_ps);
    let p_mu_schur = schur_coordinates_of_power_sum_vector(&partitions_mu, &inv_mu, p_mu_ps);

    let s_to_p_n = compute_s_to_p_transition(n);

    let mut lr_cache: HashMap<(Partition, Partition), HashMap<Partition, BigInt>> = HashMap::new();
    let mut product_ps: PowerSumVector = HashMap::new();
    for (a, ca) in &p_lambda_schur {
        if ca.is_zero() {
            continue;
        }
        for (b, cb) in &p_mu_schur {
            if cb.is_zero() {
                continue;
            }
            let lr = lr_cache
                .entry((a.clone(), b.clone()))
                .or_insert_with(|| schur_lr_coefficients(a, b));
            let coeff_ab = ca.clone() * cb.clone();
            for (s_nu, lr_coeff) in lr.iter() {
                if lr_coeff.is_zero() {
                    continue;
                }
                let Some(p_expansion) = s_to_p_n.get(s_nu) else {
                    continue;
                };
                let contribution = coeff_ab.clone() * rat_to_rf2(&Rat::from_integer(lr_coeff.clone()));
                for (kappa, p_coeff) in p_expansion {
                    let entry = product_ps
                        .entry(kappa.clone())
                        .or_insert_with(RationalFunction2::zero);
                    let updated = entry.clone() + contribution.clone() * rat_to_rf2(p_coeff);
                    *entry = updated;
                }
            }
        }
    }

    let ip_nu_nu = inner_product(p_nu_ps, p_nu_ps);
    let ip_product_nu = inner_product(&product_ps, p_nu_ps);
    ip_product_nu / ip_nu_nu
}

fn factorial(n: usize) -> BigInt {
    (1..=n).map(BigInt::from).product::<BigInt>().max(BigInt::one())
}

/// z_kappa(q,t) = z_kappa * prod_i (1-q^kappa_i)/(1-t^kappa_i), z_kappa the
/// usual prod_i i^{m_i} m_i! (m_i = multiplicity of part i in kappa).
fn z_qt(kappa: &[usize]) -> RationalFunction2 {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &k in kappa {
        *counts.entry(k).or_insert(0) += 1;
    }
    let mut z_int = BigInt::one();
    for (&part, &mult) in &counts {
        z_int *= num_traits::pow(BigInt::from(part), mult);
        z_int *= factorial(mult);
    }

    let mut result = RationalFunction2::new(
        Poly2::monomial(0, 0, z_int),
        Poly2::one(),
    );
    for &k in kappa {
        let numer = Poly2::one() - Poly2::monomial(k, 0, BigInt::one());
        let denom = Poly2::one() - Poly2::monomial(0, k, BigInt::one());
        result = result * RationalFunction2::new(numer, denom);
    }
    result
}

fn rat_to_rf2(r: &Rat) -> RationalFunction2 {
    let mut numer = r.numer().clone();
    let mut denom = r.denom().clone();
    if denom.is_negative() {
        numer = -numer;
        denom = -denom;
    }
    RationalFunction2::new(Poly2::monomial(0, 0, numer), Poly2::monomial(0, 0, denom))
}

type PowerSumVector = HashMap<Partition, RationalFunction2>;

fn inner_product(a: &PowerSumVector, b: &PowerSumVector) -> RationalFunction2 {
    let mut total = RationalFunction2::zero();
    for (k, av) in a {
        if av.is_zero() {
            continue;
        }
        if let Some(bv) = b.get(k) {
            if !bv.is_zero() {
                total = total + av.clone() * bv.clone() * z_qt(k);
            }
        }
    }
    total
}

/// Every P_lambda for |lambda| = n, in the power-sum basis, via Gram-Schmidt
/// (processing partitions in increasing lex order, a valid linear extension
/// of dominance order, starting from m_lambda's power-sum expansion).
fn compute_p_basis(n: usize) -> HashMap<Partition, PowerSumVector> {
    let m_to_p = compute_m_to_p_transition(n);
    let mut partitions = integer_partitions(n.max(1));
    if n == 0 {
        partitions = vec![Vec::new()];
    }
    // Increasing lex order: (1^n) first, (n) last.
    partitions.sort_by(|a, b| {
        let len = a.len().max(b.len());
        for i in 0..len {
            let av = a.get(i).copied().unwrap_or(0);
            let bv = b.get(i).copied().unwrap_or(0);
            if av != bv {
                return av.cmp(&bv);
            }
        }
        std::cmp::Ordering::Equal
    });

    let mut result: HashMap<Partition, PowerSumVector> = HashMap::new();
    let mut order: Vec<Partition> = Vec::new();

    for lambda in &partitions {
        let mut v: PowerSumVector = m_to_p[lambda]
            .iter()
            .map(|(k, r)| (k.clone(), rat_to_rf2(r)))
            .collect();

        for mu in &order {
            let p_mu = &result[mu];
            let ip_v_mu = inner_product(&v, p_mu);
            if ip_v_mu.is_zero() {
                continue;
            }
            let ip_mu_mu = inner_product(p_mu, p_mu);
            let coeff = ip_v_mu / ip_mu_mu;
            for (k, val) in p_mu {
                let entry = v.entry(k.clone()).or_insert_with(RationalFunction2::zero);
                let updated = entry.clone() - coeff.clone() * val.clone();
                *entry = updated;
            }
        }

        result.insert(lambda.clone(), v);
        order.push(lambda.clone());
    }

    result
}

fn merge_partition_parts(a: &[usize], b: &[usize]) -> Partition {
    let mut merged: Partition = a.iter().chain(b.iter()).copied().filter(|&x| x > 0).collect();
    merged.sort_unstable_by(|x, y| y.cmp(x));
    merged
}

/// c^nu_{lambda,mu} in the monic Macdonald P basis, computed entirely via
/// power-sum-basis Gram-Schmidt + the diagonal inner product -- no
/// dependence on [`crate::macdonald`]'s interpolation-theory machinery.
pub fn brute_force_structure_constant(
    lambda: &[usize],
    mu: &[usize],
    nu: &[usize],
) -> RationalFunction2 {
    let lambda = trim_partition(lambda);
    let mu = trim_partition(mu);
    let nu = trim_partition(nu);

    if partition_size(&nu) != partition_size(&lambda) + partition_size(&mu) {
        return RationalFunction2::zero();
    }

    let n_lambda = partition_size(&lambda);
    let n_mu = partition_size(&mu);
    let n = partition_size(&nu);

    let basis_lambda = compute_p_basis(n_lambda);
    let basis_mu = compute_p_basis(n_mu);
    let basis_nu = compute_p_basis(n);

    let p_lambda = &basis_lambda[&lambda];
    let p_mu = &basis_mu[&mu];
    let p_nu = match basis_nu.get(&nu) {
        Some(p) => p,
        None => return RationalFunction2::zero(),
    };

    let mut product: PowerSumVector = HashMap::new();
    for (k1, c1) in p_lambda {
        if c1.is_zero() {
            continue;
        }
        for (k2, c2) in p_mu {
            if c2.is_zero() {
                continue;
            }
            let merged = merge_partition_parts(k1, k2);
            let entry = product.entry(merged).or_insert_with(RationalFunction2::zero);
            let updated = entry.clone() + c1.clone() * c2.clone();
            *entry = updated;
        }
    }

    let ip_nu_nu = inner_product(p_nu, p_nu);
    let ip_product_nu = inner_product(&product, p_nu);
    ip_product_nu / ip_nu_nu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_to_p_transition_matches_hand_computation_for_n2() {
        // m_(2) = p_(2); m_(1,1) = (1/2)p_(1,1) - (1/2)p_(2).
        let transition = compute_m_to_p_transition(2);
        let m2 = &transition[&vec![2]];
        assert_eq!(m2.len(), 1);
        assert_eq!(m2[&vec![2usize]], Rat::one());

        let m11 = &transition[&vec![1, 1]];
        assert_eq!(m11[&vec![1, 1]], Rat::new(BigInt::one(), BigInt::from(2)));
        assert_eq!(m11[&vec![2]], -Rat::new(BigInt::one(), BigInt::from(2)));
    }

    #[test]
    fn brute_force_matches_sage_ground_truth_small_cases() {
        use crate::macdonald::RationalFunction2;

        // Sage ground truth (from macdonaldPStructureConstants3.jsonl):
        // {1},{1},{1,1} -> (q*t + q - t - 1)/(q*t - 1)
        let expected_11 = parse_test_rf("(q*t + q - t - 1)/(q*t - 1)");
        let actual_11 = brute_force_structure_constant(&[1], &[1], &[1, 1]);
        assert_eq!(actual_11, expected_11, "{{1}},{{1}} -> {{1,1}}");

        // {1},{1},{2} -> 1
        let actual_2 = brute_force_structure_constant(&[1], &[1], &[2]);
        assert_eq!(actual_2, RationalFunction2::one(), "{{1}},{{1}} -> {{2}}");

        // {2},{1},{2,1} -> (-q^3*t^2 + q*t^2 + q^2 - 1)/(-q^3*t^2 + q^2*t + q*t - 1)
        let expected_21 =
            parse_test_rf("(-q^3*t^2 + q*t^2 + q^2 - 1)/(-q^3*t^2 + q^2*t + q*t - 1)");
        let actual_21 = brute_force_structure_constant(&[2], &[1], &[2, 1]);
        assert_eq!(actual_21, expected_21, "{{2}},{{1}} -> {{2,1}}");
    }

    #[test]
    fn h_to_p_transition_matches_hand_computation_for_n2() {
        // Classical facts: h_(2) = (p_(2) + p_(1,1)) / 2 (same as s_(2));
        // h_(1,1) = h_1^2 = p_1^2 = p_(1,1) (trivial product of power sums).
        let transition = compute_h_to_p_transition(2);
        let h2 = &transition[&vec![2]];
        assert_eq!(h2[&vec![2usize]], Rat::new(BigInt::one(), BigInt::from(2)));
        assert_eq!(h2[&vec![1, 1]], Rat::new(BigInt::one(), BigInt::from(2)));

        let h11 = &transition[&vec![1, 1]];
        assert_eq!(h11.len(), 1);
        assert_eq!(h11[&vec![1, 1]], Rat::one());
    }

    #[test]
    fn creation_operator_action_matches_sage_examples() {
        // Sage ground truth (docstrings in macdonald.py):
        // a = S(1); a.creation(1) == -(q-1)*McdS[1]
        let a = creation_operator_action(1, &[]);
        assert_eq!(a.len(), 1);
        assert_eq!(a[&vec![1]], parse_test_rf("1 - q"));

        // a.creation(2) == (q^2*t-q*t-q+1)*McdS[1, 1] + (q^2-q*t-q+t)*McdS[2]
        let b = creation_operator_action(2, &[]);
        assert_eq!(b.len(), 2);
        assert_eq!(b[&vec![1, 1]], parse_test_rf("q^2*t - q*t - q + 1"));
        assert_eq!(b[&vec![2]], parse_test_rf("q^2 - q*t - q + t"));

        // a = S([2,1]); a._creation_by_determinant_helper(2,[1]) ==
        //   (q^3*t-q^2*t-q+1)*McdS[2, 1] + (q^3-q^2*t-q+t)*McdS[3]
        let c = creation_operator_action(2, &[1]);
        assert_eq!(c.len(), 2);
        assert_eq!(c[&vec![2, 1]], parse_test_rf("q^3*t - q^2*t - q + 1"));
        assert_eq!(c[&vec![3]], parse_test_rf("q^3 - q^2*t - q + t"));
    }

    #[test]
    fn j_lambda_in_schur_matches_sage_degree_2() {
        // Sage ground truth (J._s_cache(2) / J._self_to_s_cache[2]):
        // J[1,1] = (t^3-t^2-t+1)*s[1,1]
        // J[2]   = (-q*t+t^2+q-t)*s[1,1] + (q*t^2-q*t-t+1)*s[2]
        let j11 = j_lambda_in_schur(&[1, 1]);
        assert_eq!(j11.len(), 1);
        assert_eq!(
            j11[&vec![1, 1]],
            parse_test_rf("t^3 - t^2 - t + 1")
        );

        let j2 = j_lambda_in_schur(&[2]);
        assert_eq!(j2.len(), 2);
        assert_eq!(j2[&vec![1, 1]], parse_test_rf("-q*t + t^2 + q - t"));
        assert_eq!(j2[&vec![2]], parse_test_rf("q*t^2 - q*t - t + 1"));
    }

    #[test]
    fn j_lambda_in_schur_matches_sage_degree_3() {
        // Sage ground truth (direct sage run, s(J[lam]) for lam in Partitions(3)):
        // [3] -> -(q^3*t-q^2*t^2-q^3+q^2*t-q*t^2+t^3+q*t-t^2)*s[1,1,1]
        //        + (q^3*t^2-q^2*t^3-q^3*t+2*q^2*t^2-q*t^3-2*q^2*t+2*q*t^2+q^2-2*q*t+t^2+q-t)*s[2,1]
        //        - (q^3*t^3-q^3*t^2-q^2*t^2+q^2*t-q*t^2+q*t+t-1)*s[3]
        let j3 = j_lambda_in_schur(&[3]);
        assert_eq!(j3.len(), 3);
        assert_eq!(
            j3[&vec![1, 1, 1]],
            parse_test_rf(
                "-q^3*t + q^2*t^2 + q^3 - q^2*t + q*t^2 - t^3 - q*t + t^2"
            )
        );
        assert_eq!(
            j3[&vec![2, 1]],
            parse_test_rf(
                "q^3*t^2 - q^2*t^3 - q^3*t + 2*q^2*t^2 - q*t^3 - 2*q^2*t + 2*q*t^2 + q^2 - 2*q*t + t^2 + q - t"
            )
        );
        assert_eq!(
            j3[&vec![3]],
            parse_test_rf(
                "-q^3*t^3 + q^3*t^2 + q^2*t^2 - q^2*t + q*t^2 - q*t - t + 1"
            )
        );

        // [2, 1] -> (q*t^3-t^4-q*t^2+t^3-q*t+t^2+q-t)*s[1,1,1]
        //           - (q*t^4-2*q*t^3+q*t^2-t^2+2*t-1)*s[2,1]
        let j21 = j_lambda_in_schur(&[2, 1]);
        assert_eq!(j21.len(), 2);
        assert_eq!(
            j21[&vec![1, 1, 1]],
            parse_test_rf("q*t^3 - t^4 - q*t^2 + t^3 - q*t + t^2 + q - t")
        );
        assert_eq!(
            j21[&vec![2, 1]],
            parse_test_rf("-q*t^4 + 2*q*t^3 - q*t^2 + t^2 - 2*t + 1")
        );

        // [1, 1, 1] -> -(t^6-t^5-t^4+t^2+t-1)*s[1,1,1]
        let j111 = j_lambda_in_schur(&[1, 1, 1]);
        assert_eq!(j111.len(), 1);
        assert_eq!(
            j111[&vec![1, 1, 1]],
            parse_test_rf("-t^6 + t^5 + t^4 - t^2 - t + 1")
        );
    }

    #[test]
    fn creation_operator_structure_constant_matches_sage_ground_truth_small_cases() {
        let expected_11 = parse_test_rf("(q*t + q - t - 1)/(q*t - 1)");
        assert_eq!(
            creation_operator_structure_constant(&[1], &[1], &[1, 1]),
            expected_11
        );

        assert_eq!(
            creation_operator_structure_constant(&[1], &[1], &[2]),
            RationalFunction2::one()
        );

        let expected_21 =
            parse_test_rf("(-q^3*t^2 + q*t^2 + q^2 - 1)/(-q^3*t^2 + q^2*t + q*t - 1)");
        assert_eq!(
            creation_operator_structure_constant(&[2], &[1], &[2, 1]),
            expected_21
        );
    }

    #[test]
    fn creation_operator_structure_constant_matches_brute_force_on_a_harder_case() {
        let a = creation_operator_structure_constant(&[2, 1], &[2, 1], &[3, 2, 1]);
        let b = brute_force_structure_constant(&[2, 1], &[2, 1], &[3, 2, 1]);
        assert_eq!(a, b);
        assert!(!a.is_zero());
    }

    #[test]
    fn s_to_p_transition_matches_hand_computation_for_n2() {
        // Classical facts: s_(2) = (p_(2) + p_(1,1)) / 2; s_(1,1) = (p_(1,1) - p_(2)) / 2.
        let transition = compute_s_to_p_transition(2);
        let s2 = &transition[&vec![2]];
        assert_eq!(s2[&vec![2usize]], Rat::new(BigInt::one(), BigInt::from(2)));
        assert_eq!(s2[&vec![1, 1]], Rat::new(BigInt::one(), BigInt::from(2)));

        let s11 = &transition[&vec![1, 1]];
        assert_eq!(s11[&vec![1, 1]], Rat::new(BigInt::one(), BigInt::from(2)));
        assert_eq!(s11[&vec![2]], -Rat::new(BigInt::one(), BigInt::from(2)));
    }

    #[test]
    fn s_to_p_transition_matches_hand_computation_for_n3() {
        // Sage ground truth (`p(s(lam))` for lam in Partitions(3)):
        //   s_(3)       = 1/6 p_(1,1,1) + 1/2 p_(2,1) + 1/3 p_(3)
        //   s_(2,1)     = 1/3 p_(1,1,1)               - 1/3 p_(3)
        //   s_(1,1,1)   = 1/6 p_(1,1,1) - 1/2 p_(2,1) + 1/3 p_(3)
        let transition = compute_s_to_p_transition(3);
        let sixth = Rat::new(BigInt::one(), BigInt::from(6));
        let third = Rat::new(BigInt::one(), BigInt::from(3));

        let s3 = &transition[&vec![3]];
        assert_eq!(s3[&vec![3]], third);
        assert_eq!(s3[&vec![2, 1]], Rat::new(BigInt::from(3), BigInt::from(6)));
        assert_eq!(s3[&vec![1, 1, 1]], sixth);

        let s21 = &transition[&vec![2, 1]];
        assert_eq!(s21[&vec![1, 1, 1]], third);
        assert_eq!(s21[&vec![3]], -third);
        assert!(!s21.contains_key(&vec![2, 1]));

        let s111 = &transition[&vec![1, 1, 1]];
        assert_eq!(s111[&vec![1, 1, 1]], sixth);
        assert_eq!(
            s111[&vec![2, 1]],
            -Rat::new(BigInt::from(3), BigInt::from(6))
        );
        assert_eq!(s111[&vec![3]], Rat::new(BigInt::from(2), BigInt::from(6)));
    }

    #[test]
    fn schur_lr_coefficients_matches_known_pieri_and_general_cases() {
        // Pieri rule: s_(2) * s_(1) = s_(3) + s_(2,1).
        let lr = schur_lr_coefficients(&[2], &[1]);
        assert_eq!(lr.len(), 2);
        assert_eq!(lr[&vec![3]], BigInt::one());
        assert_eq!(lr[&vec![2, 1]], BigInt::one());

        // Sage ground truth (`s[2,1]*s[2,1]`): the only repeated LR
        // coefficient is the 2 at nu=(3,2,1) --
        //   s_(2,1)^2 = s_(4,2)+s_(4,1,1)+s_(3,3)+2s_(3,2,1)
        //             +s_(3,1,1,1)+s_(2,2,2)+s_(2,2,1,1).
        let lr2 = schur_lr_coefficients(&[2, 1], &[2, 1]);
        assert_eq!(lr2[&vec![3, 3]], BigInt::one());
        assert_eq!(lr2[&vec![3, 2, 1]], BigInt::from(2));
        assert_eq!(lr2[&vec![4, 2]], BigInt::one());
        assert_eq!(lr2[&vec![4, 1, 1]], BigInt::one());
        assert_eq!(lr2[&vec![3, 1, 1, 1]], BigInt::one());
        assert_eq!(lr2[&vec![2, 2, 2]], BigInt::one());
        assert_eq!(lr2[&vec![2, 2, 1, 1]], BigInt::one());
        assert_eq!(lr2.len(), 7);
    }

    #[test]
    fn schur_sandwich_matches_sage_ground_truth_small_cases() {
        let expected_11 = parse_test_rf("(q*t + q - t - 1)/(q*t - 1)");
        assert_eq!(
            schur_sandwich_structure_constant(&[1], &[1], &[1, 1]),
            expected_11
        );

        assert_eq!(
            schur_sandwich_structure_constant(&[1], &[1], &[2]),
            crate::macdonald::RationalFunction2::one()
        );

        let expected_21 =
            parse_test_rf("(-q^3*t^2 + q*t^2 + q^2 - 1)/(-q^3*t^2 + q^2*t + q*t - 1)");
        assert_eq!(
            schur_sandwich_structure_constant(&[2], &[1], &[2, 1]),
            expected_21
        );
    }

    #[test]
    fn schur_sandwich_matches_brute_force_on_a_harder_case() {
        let a = schur_sandwich_structure_constant(&[2, 1], &[2, 1], &[3, 2, 1]);
        let b = brute_force_structure_constant(&[2, 1], &[2, 1], &[3, 2, 1]);
        assert_eq!(a, b);
        assert!(!a.is_zero());
    }

    /// Minimal local parser for Sage's "(NUM)/(DEN)" or bare-polynomial
    /// output, for this module's own tests only.
    fn parse_test_rf(input: &str) -> crate::macdonald::RationalFunction2 {
        use crate::macdonald::{Poly2, RationalFunction2};
        fn parse_term(term: &str) -> Poly2 {
            let mut coeff = BigInt::from(1);
            let mut qd = 0usize;
            let mut td = 0usize;
            for factor in term.split('*') {
                let factor = factor.trim();
                if factor == "q" {
                    qd += 1;
                } else if factor == "t" {
                    td += 1;
                } else if let Some(r) = factor.strip_prefix("q^") {
                    qd += r.parse::<usize>().unwrap();
                } else if let Some(r) = factor.strip_prefix("t^") {
                    td += r.parse::<usize>().unwrap();
                } else {
                    coeff *= factor.parse::<BigInt>().unwrap();
                }
            }
            Poly2::monomial(qd, td, coeff)
        }
        fn parse_expr(input: &str) -> Poly2 {
            let bytes = input.as_bytes();
            let mut terms: Vec<(bool, &str)> = Vec::new();
            let mut negative = false;
            let mut start = 0usize;
            let mut i = 0usize;
            if bytes.first() == Some(&b'-') {
                negative = true;
                start = 1;
                i = 1;
            } else if bytes.first() == Some(&b'+') {
                start = 1;
                i = 1;
            }
            while i < bytes.len() {
                let boundary = bytes[i] == b' '
                    && matches!(bytes.get(i + 1), Some(b'+') | Some(b'-'))
                    && bytes.get(i + 2) == Some(&b' ');
                if boundary {
                    terms.push((negative, input[start..i].trim()));
                    negative = bytes[i + 1] == b'-';
                    i += 3;
                    start = i;
                } else {
                    i += 1;
                }
            }
            terms.push((negative, input[start..].trim()));
            terms.into_iter().fold(Poly2::zero(), |acc, (neg, term)| {
                let value = parse_term(term);
                if neg {
                    acc - value
                } else {
                    acc + value
                }
            })
        }
        let s = input.trim();
        if s.starts_with('(') {
            let split = s.find(")/(").unwrap();
            let num = &s[1..split];
            let den = &s[split + 3..s.len() - 1];
            RationalFunction2::new(parse_expr(num), parse_expr(den))
        } else {
            RationalFunction2::new(parse_expr(s), Poly2::one())
        }
    }

    #[test]
    fn m_to_p_transition_matches_hand_computation_for_n3() {
        // m_(3) = p_(3).
        // m_(2,1) = p_(2,1) - p_(3).
        // m_(1,1,1) = (1/6)p_(1,1,1) - (1/2)p_(2,1) + (1/3)p_(3).
        let transition = compute_m_to_p_transition(3);

        let m3 = &transition[&vec![3]];
        assert_eq!(m3.len(), 1);
        assert_eq!(m3[&vec![3usize]], Rat::one());

        let m21 = &transition[&vec![2, 1]];
        assert_eq!(m21[&vec![2, 1]], Rat::one());
        assert_eq!(m21[&vec![3]], -Rat::one());

        let m111 = &transition[&vec![1, 1, 1]];
        assert_eq!(m111[&vec![1, 1, 1]], Rat::new(BigInt::one(), BigInt::from(6)));
        assert_eq!(m111[&vec![2, 1]], -Rat::new(BigInt::one(), BigInt::from(2)));
        assert_eq!(m111[&vec![3]], Rat::new(BigInt::one(), BigInt::from(3)));
    }
}
