//! Macdonald Littlewood-Richardson coefficients (ordinary, non-shifted).
//!
//! This module will provide the two-parameter (q, t) analog of
//! [`crate::ShiftedJackCalculator`]'s top-degree structure constants: given
//! partitions lambda, mu, nu with |nu| = |lambda| + |mu|, the coefficient
//! c^nu_{lambda,mu}(q,t) in
//!
//!   P_lambda(x; q,t) * P_mu(x; q,t) = sum_nu c^nu_{lambda,mu}(q,t) P_nu(x; q,t)
//!
//! for the monic Macdonald P basis (coefficient of m_lambda in P_lambda is
//! 1), plus the analogous J-basis (integral form) structure constants.
//!
//! `Poly2` / `RationalFunction2` below give exact bivariate polynomial and
//! rational function arithmetic in q and t, with full GCD-based reduction to
//! lowest terms, mirroring [`crate::IntPoly`] / [`crate::RationalFunction`]
//! one level up: a `Poly2` is represented as a polynomial in q whose
//! coefficients are themselves polynomials in t (`IntPoly`), and bivariate
//! GCDs are computed by working over the fraction field of the t-coefficient
//! ring (i.e. `RationalFunction`, reused here purely as "rational functions
//! in t"), exactly as the existing univariate code works over the fraction
//! field of the integers.

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::{
    add_box_to_partition, conjugate_partition, gt_chain_from_tableau, int_poly_gcd,
    normalize_three, normalize_two, partition_size, remove_box_from_partition,
    semistandard_tableaux, skew_shape_contains, trim_partition, IntPoly, Partition,
    RationalFunction,
};

/// A polynomial in q and t with integer coefficients, stored as a
/// polynomial in q whose coefficients are polynomials in t.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly2 {
    /// `coeffs[i]` is the coefficient of `q^i`, itself a polynomial in t.
    coeffs: Vec<IntPoly>,
}

impl Poly2 {
    pub fn new(mut coeffs: Vec<IntPoly>) -> Self {
        while coeffs.last().is_some_and(IntPoly::is_zero) {
            coeffs.pop();
        }
        Self { coeffs }
    }

    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    pub fn one() -> Self {
        Self::from_i64(1)
    }

    pub fn from_i64(n: i64) -> Self {
        Self::monomial(0, 0, BigInt::from(n))
    }

    pub fn q() -> Self {
        Self::monomial(1, 0, BigInt::one())
    }

    pub fn t() -> Self {
        Self::monomial(0, 1, BigInt::one())
    }

    pub fn monomial(q_degree: usize, t_degree: usize, coefficient: BigInt) -> Self {
        if coefficient.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![IntPoly::zero(); q_degree + 1];
        coeffs[q_degree] = IntPoly::monomial(t_degree, coefficient);
        Self::new(coeffs)
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }

    fn coeff(&self, degree: usize) -> IntPoly {
        self.coeffs
            .get(degree)
            .cloned()
            .unwrap_or_else(IntPoly::zero)
    }

    /// The gcd (as a polynomial in t) of the q-coefficients: the "content"
    /// of this polynomial when viewed as a univariate polynomial in q over
    /// the ring Z[t].
    fn content_t(&self) -> IntPoly {
        // The gcd of the q-coefficients as elements of Z[t] has an integer
        // part and a polynomial part, and they must be tracked separately:
        // `int_poly_gcd` (borrowed from the univariate Jack code, where it
        // cancels polynomial factors between an already-separately-handled
        // numerator/denominator integer content) treats every nonzero
        // rational scalar as a unit -- exactly right for *that* use, but
        // wrong here, since it would report gcd(-2, 2t) = 1 instead of the
        // true 2. So: gcd the integer content of each coefficient via
        // BigInt gcd, and separately gcd their primitive parts via
        // `int_poly_gcd` (valid now, since primitive parts have no leftover
        // integer content to lose), then recombine.
        let mut int_content = BigInt::zero();
        let mut poly_content: Option<IntPoly> = None;
        for c in &self.coeffs {
            if c.is_zero() {
                continue;
            }
            int_content = int_content.gcd(&c.content_abs());
            let primitive = c.primitive_part_positive();
            poly_content = Some(match poly_content {
                None => primitive,
                Some(acc) => int_poly_gcd(&acc, &primitive),
            });
        }
        match poly_content {
            None => IntPoly::zero(),
            Some(poly_content) => IntPoly::monomial(0, int_content) * poly_content,
        }
    }

    fn div_by_intpoly_exact(&self, divisor: &IntPoly) -> Self {
        Self::new(self.coeffs.iter().map(|c| c.exact_div(divisor)).collect())
    }

    fn leading_intpoly(&self) -> &IntPoly {
        self.coeffs.last().expect("nonzero polynomial")
    }

    fn leading_sign_negative(&self) -> bool {
        let leading = self.leading_intpoly();
        let degree = leading.degree().expect("nonzero polynomial");
        leading.coeff(degree).is_negative()
    }

    fn primitive_part_positive(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let content = self.content_t();
        let mut result = if content.is_one() {
            self.clone()
        } else {
            self.div_by_intpoly_exact(&content)
        };
        if result.leading_sign_negative() {
            result = -result;
        }
        result
    }

    /// This polynomial's q-coefficients embedded as rational functions in
    /// t (i.e. as elements of the fraction field of Z[t]).
    fn to_rf_coeffs(&self) -> Vec<RationalFunction> {
        self.coeffs
            .iter()
            .map(|c| RationalFunction::new(c.clone(), IntPoly::one()))
            .collect()
    }

    /// Degree in q. Panics on the zero polynomial.
    fn deg_q(&self) -> usize {
        self.coeffs.len() - 1
    }

    #[cfg(test)]
    fn evaluate(
        &self,
        q: &num_rational::Ratio<BigInt>,
        t: &num_rational::Ratio<BigInt>,
    ) -> num_rational::Ratio<BigInt> {
        use num_rational::Ratio;
        let mut total = Ratio::from_integer(BigInt::from(0));
        let mut q_pow = Ratio::from_integer(BigInt::from(1));
        for tpoly in &self.coeffs {
            if !tpoly.is_zero() {
                let mut t_val = Ratio::from_integer(BigInt::from(0));
                let mut t_pow = Ratio::from_integer(BigInt::from(1));
                if let Some(max_dt) = tpoly.degree() {
                    for dt in 0..=max_dt {
                        let c = tpoly.coeff(dt);
                        if !c.is_zero() {
                            t_val += Ratio::from_integer(c) * &t_pow;
                        }
                        t_pow *= t;
                    }
                }
                total += t_val * &q_pow;
            }
            q_pow *= q;
        }
        total
    }

    /// Multiplies every q-coefficient by `factor` (an element of Z[t]).
    fn scale(&self, factor: &IntPoly) -> Self {
        if factor.is_zero() {
            return Self::zero();
        }
        Self::new(
            self.coeffs
                .iter()
                .map(|c| c.clone() * factor.clone())
                .collect(),
        )
    }

    /// Multiplies by `q^degree`.
    fn shift(&self, degree: usize) -> Self {
        if self.is_zero() || degree == 0 {
            return self.clone();
        }
        let mut coeffs = vec![IntPoly::zero(); degree];
        coeffs.extend(self.coeffs.iter().cloned());
        Self { coeffs }
    }

    /// Swaps q and t throughout (i.e. transposes the (q_degree, t_degree)
    /// exponent pair of every monomial). Used to replicate Sage's Macdonald
    /// `J` basis construction, which builds a q<->t-swapped intermediate
    /// result via the `S`-basis creation operators and corrects it with
    /// this swap at the end.
    pub(crate) fn swap_qt(&self) -> Self {
        let mut result = Self::zero();
        for (q_degree, t_poly) in self.coeffs.iter().enumerate() {
            if let Some(max_t) = t_poly.degree() {
                for t_degree in 0..=max_t {
                    let c = t_poly.coeff(t_degree);
                    if !c.is_zero() {
                        result = result + Self::monomial(t_degree, q_degree, c);
                    }
                }
            }
        }
        result
    }
}

impl Add for Poly2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        Self::new((0..n).map(|i| self.coeff(i) + rhs.coeff(i)).collect())
    }
}

impl Sub for Poly2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        Self::new((0..n).map(|i| self.coeff(i) - rhs.coeff(i)).collect())
    }
}

impl Neg for Poly2 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(self.coeffs.into_iter().map(|c| -c).collect())
    }
}

impl Mul for Poly2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![IntPoly::zero(); self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, a) in self.coeffs.into_iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + a.clone() * b.clone();
            }
        }
        Self::new(coeffs)
    }
}

impl fmt::Display for Poly2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for q_degree in (0..self.coeffs.len()).rev() {
            let t_poly = &self.coeffs[q_degree];
            if t_poly.is_zero() {
                continue;
            }
            let max_t_degree = t_poly.degree().expect("nonzero polynomial");
            for t_degree in (0..=max_t_degree).rev() {
                let coeff = t_poly.coeff(t_degree);
                if coeff.is_zero() {
                    continue;
                }

                let negative = coeff.is_negative();
                let abs_coeff = coeff.abs();
                if first {
                    if negative {
                        write!(f, "-")?;
                    }
                    first = false;
                } else if negative {
                    write!(f, " - ")?;
                } else {
                    write!(f, " + ")?;
                }

                let show_coeff = (q_degree == 0 && t_degree == 0) || !abs_coeff.is_one();
                if show_coeff {
                    write!(f, "{abs_coeff}")?;
                    if q_degree > 0 || t_degree > 0 {
                        write!(f, "*")?;
                    }
                }
                match q_degree {
                    0 => {}
                    1 => write!(f, "q")?,
                    _ => write!(f, "q^{q_degree}")?,
                }
                if q_degree > 0 && t_degree > 0 {
                    write!(f, "*")?;
                }
                match t_degree {
                    0 => {}
                    1 => write!(f, "t")?,
                    _ => write!(f, "t^{t_degree}")?,
                }
            }
        }
        Ok(())
    }
}

fn trim_rf(v: &mut Vec<RationalFunction>) {
    while v.last().is_some_and(RationalFunction::is_zero) {
        v.pop();
    }
}

/// Polynomial long division in q, with coefficients in the field of
/// rational functions in t.
fn rf_poly_div_rem(
    mut dividend: Vec<RationalFunction>,
    mut divisor: Vec<RationalFunction>,
) -> (Vec<RationalFunction>, Vec<RationalFunction>) {
    trim_rf(&mut dividend);
    trim_rf(&mut divisor);
    assert!(!divisor.is_empty(), "division by zero polynomial");

    let divisor_degree = divisor.len() - 1;
    let divisor_lead = divisor[divisor_degree].clone();

    if dividend.len() < divisor.len() {
        return (Vec::new(), dividend);
    }

    let mut quotient = vec![RationalFunction::zero(); dividend.len() - divisor.len() + 1];
    while !dividend.is_empty() && dividend.len() >= divisor.len() {
        let degree = dividend.len() - divisor.len();
        let coeff = dividend.last().unwrap().clone() / divisor_lead.clone();
        quotient[degree] = coeff.clone();
        for (i, divisor_coeff) in divisor.iter().enumerate().take(divisor_degree + 1) {
            dividend[degree + i] =
                dividend[degree + i].clone() - coeff.clone() * divisor_coeff.clone();
        }
        trim_rf(&mut dividend);
    }

    trim_rf(&mut quotient);
    (quotient, dividend)
}

/// Converts a polynomial-in-q with rational-function-in-t coefficients,
/// known to actually have integral (denominator-1) coefficients, back into a
/// `Poly2` exactly -- i.e. with no content-stripping or sign flip, since the
/// value is a specific, unique polynomial rather than a GCD representative
/// that's only defined up to a unit. The denominator-clearing here is a
/// no-op safety net: for a genuine exact quotient every coefficient's
/// denominator is already 1.
fn rf_coeffs_to_poly2_exact(coeffs: &[RationalFunction]) -> Poly2 {
    if coeffs.iter().all(RationalFunction::is_zero) {
        return Poly2::zero();
    }

    let mut lcm = IntPoly::one();
    for c in coeffs {
        if c.is_zero() {
            continue;
        }
        lcm = intpoly_lcm(&lcm, c.denominator());
    }

    let int_coeffs: Vec<IntPoly> = coeffs
        .iter()
        .map(|c| {
            if c.is_zero() {
                IntPoly::zero()
            } else {
                let factor = lcm.exact_div(c.denominator());
                c.numerator().clone() * factor
            }
        })
        .collect();

    Poly2::new(int_coeffs)
}

/// gcd of two IntPoly values (polynomials in t) as elements of Z[t],
/// i.e. correctly including their shared *integer* content. The raw
/// crate-level `int_poly_gcd` treats nonzero rational scalars as units --
/// exactly right for its original purpose (cancelling polynomial factors
/// between an already content-stripped univariate `RationalFunction`,
/// where integer content is tracked separately) but wrong whenever either
/// input can still carry nontrivial integer content of its own, e.g.
/// `int_poly_gcd(32+32t, 32)` comes out `1` instead of `32`, silently
/// losing a real shared factor. Every call site here that gcds two
/// possibly-content-bearing values (as opposed to two already-primitive
/// ones) must use this instead.
fn intpoly_gcd_z(a: &IntPoly, b: &IntPoly) -> IntPoly {
    if a.is_zero() {
        let degree = b.degree().expect("the nonzero gcd input has a degree");
        return if b.coeff(degree).is_negative() {
            -b.clone()
        } else {
            b.clone()
        };
    }
    if b.is_zero() {
        let degree = a.degree().expect("the nonzero gcd input has a degree");
        return if a.coeff(degree).is_negative() {
            -a.clone()
        } else {
            a.clone()
        };
    }
    let int_content = a.content_abs().gcd(&b.content_abs());
    let poly_content = int_poly_gcd(&a.primitive_part_positive(), &b.primitive_part_positive());
    IntPoly::monomial(0, int_content) * poly_content
}

fn intpoly_lcm(a: &IntPoly, b: &IntPoly) -> IntPoly {
    if a.is_zero() || b.is_zero() {
        return IntPoly::zero();
    }
    if a.is_one() {
        return b.clone();
    }
    if b.is_one() {
        return a.clone();
    }
    let gcd = intpoly_gcd_z(a, b);
    let product = a.clone() * b.clone();
    product.exact_div(&gcd)
}

fn pow_intpoly(base: &IntPoly, exponent: i64) -> IntPoly {
    if exponent == 0 {
        return IntPoly::one();
    }
    if exponent > 0 {
        let mut result = IntPoly::one();
        for _ in 0..exponent {
            result = result * base.clone();
        }
        return result;
    }
    // Negative exponents only ever arise (in the subresultant recurrence
    // below) applied to psi_1 = -1, a unit whose power is its own inverse
    // regardless of sign, so |exponent| gives the same answer.
    assert!(
        base.is_one() || (-base.clone()).is_one(),
        "negative exponent on a non-unit polynomial"
    );
    pow_intpoly(base, -exponent)
}

/// The pseudo-remainder prem(f, g): the unique r with deg(r) < deg(g) such
/// that lc(g)^(deg(f)-deg(g)+1) * f = q*g + r for some q, computed directly
/// in Z[t][q] (no field division), requiring deg(f) >= deg(g) and g != 0.
fn pseudo_remainder(f: &Poly2, g: &Poly2) -> Poly2 {
    let n = g.deg_q();
    let c = g.leading_intpoly().clone();
    let mut r = f.clone();
    let mut budget: i64 = f.deg_q() as i64 - n as i64 + 1;

    while !r.is_zero() && r.deg_q() >= n {
        let delta = r.deg_q() - n;
        let leading_r = r.leading_intpoly().clone();
        r = r.scale(&c) - g.scale(&leading_r).shift(delta);
        budget -= 1;
    }

    if budget > 0 {
        r = r.scale(&pow_intpoly(&c, budget));
    }
    r
}

/// Instrumentation for profiling where bivariate-gcd time actually goes,
/// bucketed by input size. Enabled by the `MACDONALD_GCD_STATS` env var
/// (checked once, so zero cost when unset); see `report`.
pub mod gcd_stats {
    use std::cell::Cell;

    thread_local! {
        pub(super) static CALLS: Cell<u64> = const { Cell::new(0) };
        pub(super) static NANOS: Cell<u128> = const { Cell::new(0) };
        /// Buckets by max(deg_q) of the two inputs: 0-1, 2-3, 4-7, 8-15, 16+.
        pub(super) static BUCKET_CALLS: [Cell<u64>; 5] = Default::default();
        pub(super) static BUCKET_NANOS: [Cell<u128>; 5] = Default::default();
        /// Largest single call seen: (deg_q, deg_t, nanos).
        pub(super) static WORST: Cell<(usize, usize, u128)> = const { Cell::new((0, 0, 0)) };
    }

    /// Zeroes the counters (thread-local, like all state in this module --
    /// call this on the same thread that will do the work being measured).
    pub fn reset() {
        CALLS.with(|c| c.set(0));
        NANOS.with(|c| c.set(0));
        BUCKET_CALLS.with(|b| b.iter().for_each(|c| c.set(0)));
        BUCKET_NANOS.with(|b| b.iter().for_each(|c| c.set(0)));
        WORST.with(|c| c.set((0, 0, 0)));
    }

    /// A human-readable summary of the counters accumulated on this thread
    /// since the last `reset`.
    pub fn report() -> String {
        let calls = CALLS.with(|c| c.get());
        let nanos = NANOS.with(|c| c.get());
        let worst = WORST.with(|c| c.get());
        let mut out = format!(
            "poly2_gcd: {calls} calls, {:.2}s total, worst single call \
             deg_q={} deg_t={} at {:.3}s\n",
            nanos as f64 / 1e9,
            worst.0,
            worst.1,
            worst.2 as f64 / 1e9,
        );
        let labels = [
            "deg_q 0-1",
            "deg_q 2-3",
            "deg_q 4-7",
            "deg_q 8-15",
            "deg_q 16+",
        ];
        for i in 0..5 {
            let c = BUCKET_CALLS.with(|b| b[i].get());
            let n = BUCKET_NANOS.with(|b| b[i].get());
            if c > 0 {
                out.push_str(&format!(
                    "  {:>10}: {c:>10} calls, {:>8.2}s ({:>5.1}%)\n",
                    labels[i],
                    n as f64 / 1e9,
                    100.0 * n as f64 / nanos.max(1) as f64,
                ));
            }
        }
        out
    }
}

/// Thin profiling wrapper around `poly2_gcd_inner` (the actual gcd
/// algorithm): records call counts/timings into `gcd_stats` when enabled,
/// otherwise adds nothing over calling it directly.
fn poly2_gcd(a: &Poly2, b: &Poly2) -> Poly2 {
    static STATS_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *STATS_ENABLED.get_or_init(|| std::env::var_os("MACDONALD_GCD_STATS").is_some()) {
        let start = std::time::Instant::now();
        let result = poly2_gcd_inner(a, b);
        let elapsed = start.elapsed().as_nanos();
        let deg_q = a.coeffs.len().max(b.coeffs.len()).saturating_sub(1);
        let deg_t = a
            .coeffs
            .iter()
            .chain(b.coeffs.iter())
            .filter_map(|c| c.degree())
            .max()
            .unwrap_or(0);
        let bucket = match deg_q {
            0..=1 => 0,
            2..=3 => 1,
            4..=7 => 2,
            8..=15 => 3,
            _ => 4,
        };
        gcd_stats::CALLS.with(|c| c.set(c.get() + 1));
        gcd_stats::NANOS.with(|c| c.set(c.get() + elapsed));
        gcd_stats::BUCKET_CALLS.with(|b| b[bucket].set(b[bucket].get() + 1));
        gcd_stats::BUCKET_NANOS.with(|b| b[bucket].set(b[bucket].get() + elapsed));
        gcd_stats::WORST.with(|c| {
            if elapsed > c.get().2 {
                c.set((deg_q, deg_t, elapsed));
            }
        });
        return result;
    }
    poly2_gcd_inner(a, b)
}

/// A prime just under 2^31, so products of two residues fit comfortably in
/// u128 and residues fit in u64.
const GCD_FILTER_PRIME: u64 = 2_147_483_647;

/// `x mod p`, normalized into `0..p` (BigInt's `%` can return a negative
/// remainder for a negative `x`).
fn bigint_mod_u64(x: &BigInt, p: u64) -> u64 {
    let modulus = BigInt::from(p);
    let mut r = x % &modulus;
    if r.is_negative() {
        r += &modulus;
    }
    r.to_u64().expect("a residue mod p always fits in u64")
}

/// Evaluates a polynomial in t at `t0`, in F_p (Horner).
fn eval_intpoly_mod(poly: &IntPoly, t0: u64, p: u64) -> u64 {
    let Some(degree) = poly.degree() else {
        return 0;
    };
    let mut acc = 0u64;
    for d in (0..=degree).rev() {
        acc = (acc as u128 * t0 as u128 % p as u128) as u64;
        acc = (acc + bigint_mod_u64(&poly.coeff(d), p)) % p;
    }
    acc
}

/// `base^exponent mod p`, via square-and-multiply.
fn mod_pow(mut base: u64, mut exponent: u64, p: u64) -> u64 {
    let mut result = 1u64;
    base %= p;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = (result as u128 * base as u128 % p as u128) as u64;
        }
        base = (base as u128 * base as u128 % p as u128) as u64;
        exponent >>= 1;
    }
    result
}

/// Drops trailing zero coefficients, the F_p analog of `Poly2::new`'s
/// leading-zero trim.
fn trim_mod_poly(v: &mut Vec<u64>) {
    while v.last() == Some(&0) {
        v.pop();
    }
}

/// gcd(a, b) for univariate polynomials over F_p, given as coefficient
/// vectors indexed by degree. An empty result means both inputs were zero.
fn univariate_gcd_mod(mut a: Vec<u64>, mut b: Vec<u64>, p: u64) -> Vec<u64> {
    trim_mod_poly(&mut a);
    trim_mod_poly(&mut b);
    while !b.is_empty() {
        let inverse = mod_pow(*b.last().expect("nonempty"), p - 2, p);
        while !a.is_empty() && a.len() >= b.len() {
            let factor =
                (*a.last().expect("nonempty") as u128 * inverse as u128 % p as u128) as u64;
            let shift = a.len() - b.len();
            for (i, &bc) in b.iter().enumerate() {
                let sub = (factor as u128 * bc as u128 % p as u128) as u64;
                a[i + shift] = (a[i + shift] + p - sub) % p;
            }
            trim_mod_poly(&mut a);
        }
        std::mem::swap(&mut a, &mut b);
    }
    a
}

/// True when `divisor` divides `target` in F_p[q] (both as coefficient
/// vectors indexed by degree). Used only as a cheap pre-filter ahead of
/// the exact bivariate division.
fn divides_mod(divisor: &[u64], target: &[u64], p: u64) -> bool {
    if divisor.is_empty() {
        return false;
    }
    let mut rem = target.to_vec();
    trim_mod_poly(&mut rem);
    let inverse = mod_pow(*divisor.last().expect("nonempty"), p - 2, p);
    while !rem.is_empty() && rem.len() >= divisor.len() {
        let factor = (*rem.last().expect("nonempty") as u128 * inverse as u128 % p as u128) as u64;
        let shift = rem.len() - divisor.len();
        for (i, &dc) in divisor.iter().enumerate() {
            let sub = (factor as u128 * dc as u128 % p as u128) as u64;
            rem[i + shift] = (rem[i + shift] + p - sub) % p;
        }
        trim_mod_poly(&mut rem);
    }
    rem.is_empty()
}

/// Reduces a `Poly2` to a univariate polynomial over F_p by evaluating t
/// at `t0`, as a coefficient vector indexed by q-degree.
fn poly2_reduce_mod(p2: &Poly2, t0: u64, p: u64) -> Vec<u64> {
    p2.coeffs
        .iter()
        .map(|c| eval_intpoly_mod(c, t0, p))
        .collect()
}

/// gcd(a_star, b_star) reduced at t = t0 over F_p, together with the t0
/// used. Only evaluation points keeping deg_q(a_star) intact are accepted,
/// which is what makes the degree of the result an upper bound on the
/// degree of the true gcd (see `provably_coprime_primitive_parts`).
fn modular_gcd_image(a_star: &Poly2, b_star: &Poly2) -> Option<(Vec<u64>, u64)> {
    let p = GCD_FILTER_PRIME;
    for t0 in [3u64, 5, 7, 11, 13, 17, 19] {
        if eval_intpoly_mod(a_star.leading_intpoly(), t0, p) == 0 {
            continue;
        }
        let a_bar = poly2_reduce_mod(a_star, t0, p);
        let b_bar = poly2_reduce_mod(b_star, t0, p);
        return Some((univariate_gcd_mod(a_bar, b_bar, p), t0));
    }
    None
}

/// Coefficients of the d-th cyclotomic polynomial, indexed by degree,
/// via Phi_d(x) = (x^d - 1) / prod_{e | d, e < d} Phi_e(x).
fn cyclotomic_coeffs(d: usize) -> Vec<BigInt> {
    let mut result = vec![BigInt::zero(); d + 1];
    result[0] = -BigInt::one();
    result[d] = BigInt::one();
    for e in 1..d {
        if d % e == 0 {
            result = exact_div_monic_int_poly(&result, &cyclotomic_coeffs(e));
        }
    }
    result
}

/// Exact division of integer coefficient vectors where the divisor is
/// monic (true of every cyclotomic polynomial).
fn exact_div_monic_int_poly(a: &[BigInt], b: &[BigInt]) -> Vec<BigInt> {
    let divisor_degree = b.len() - 1;
    let mut remainder = a.to_vec();
    let mut quotient = vec![BigInt::zero(); a.len() - divisor_degree];
    for i in (0..quotient.len()).rev() {
        let c = remainder[i + divisor_degree].clone();
        for (j, bj) in b.iter().enumerate() {
            remainder[i + j] -= &c * bj;
        }
        quotient[i] = c;
    }
    quotient
}

/// Phi_d(q^alpha t^beta) as a `Poly2`.
fn cyclotomic_factor(d: usize, alpha: usize, beta: usize) -> Poly2 {
    let mut result = Poly2::zero();
    for (k, c) in cyclotomic_coeffs(d).iter().enumerate() {
        if !c.is_zero() {
            result = result + Poly2::monomial(k * alpha, k * beta, c.clone());
        }
    }
    result
}

/// Exact division by a divisor whose constant coefficient in q is the unit
/// +/-1, done as an ascending recurrence in the q-degree (from
/// p_i = sum_j d_j q_{i-j} we get q_i = (p_i - sum_{j>=1} d_j q_{i-j})/d_0).
/// Returns `None` unless the division is exact, so a wrong guess at a
/// candidate factor can only waste time, never corrupt a result.
fn try_exact_divide_unit_constant(p: &Poly2, divisor: &Poly2) -> Option<Poly2> {
    let n = p.coeffs.len();
    let m = divisor.coeffs.len();
    if n == 0 || m == 0 || n < m {
        return None;
    }
    let d0 = &divisor.coeffs[0];
    let negate = if d0.is_one() {
        false
    } else if (-d0.clone()).is_one() {
        true
    } else {
        return None;
    };

    let mut quotient: Vec<IntPoly> = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = p.coeffs[i].clone();
        for j in 1..m.min(i + 1) {
            acc = acc - divisor.coeffs[j].clone() * quotient[i - j].clone();
        }
        if negate {
            acc = -acc;
        }
        quotient.push(acc);
    }
    if quotient[n - m + 1..].iter().any(|c| !c.is_zero()) {
        return None;
    }
    quotient.truncate(n - m + 1);
    Some(Poly2::new(quotient))
}

/// Strips the largest common power of q from both inputs, returning its
/// exponent. Monomial factors show up routinely as gcds here and are far
/// too cheap to leave to the PRS.
fn strip_common_q_power(a_star: &mut Poly2, b_star: &mut Poly2) -> usize {
    let leading_zeros = |p: &Poly2| p.coeffs.iter().position(|c| !c.is_zero()).unwrap_or(0);
    let k = leading_zeros(a_star).min(leading_zeros(b_star));
    if k > 0 {
        *a_star = Poly2::new(a_star.coeffs[k..].to_vec());
        *b_star = Poly2::new(b_star.coeffs[k..].to_vec());
    }
    k
}

/// Euler's totient function, via trial division over d's prime factors.
/// Used to bound a cyclotomic factor's q-degree during the search in
/// `strip_common_cyclotomic_factors` (`deg Phi_d(q^alpha t^beta) = alpha *
/// phi(d)`).
fn euler_phi(mut d: usize) -> usize {
    let mut result = d;
    let mut factor = 2;
    while factor * factor <= d {
        if d % factor == 0 {
            while d % factor == 0 {
                d /= factor;
            }
            result -= result / factor;
        }
        factor += 1;
    }
    if d > 1 {
        result -= result / d;
    }
    result
}

/// A cheap, *rigorous* filter answering "are these two primitive parts
/// coprime in Z[t][q]?" -- used to short-circuit the subresultant PRS,
/// whose cost is dominated by coefficient growth in Z[t] and which is at
/// its worst precisely on coprime inputs (it runs to completion).
///
/// Correctness (not probabilistic): let g = gcd(a_star, b_star). Since
/// g | a_star, the leading coefficient lc(g) divides lc(a_star) in Z[t],
/// so whenever lc(a_star)(t0) != 0 mod p we also have lc(g)(t0) != 0, and
/// the reduction q -> a_star(q, t0) mod p preserves deg(g). Reduction is a
/// ring homomorphism, so the reduced g still divides both reduced inputs
/// and hence divides their gcd. Therefore
///     deg gcd(a_bar, b_bar) >= deg g,
/// and a degree-0 modular gcd *proves* g is a constant. The converse can
/// fail (an unlucky t0 can inflate the modular gcd), but that only costs a
/// fall-through to the exact algorithm -- never a wrong answer.
fn provably_coprime_primitive_parts(a_star: &Poly2, b_star: &Poly2) -> bool {
    // Only lc(a_star) needs to survive the evaluation; see the argument above.
    matches!(modular_gcd_image(a_star, b_star), Some((image, _)) if image.len() <= 1)
}

fn max_t_degree(p: &Poly2) -> usize {
    p.coeffs
        .iter()
        .filter_map(|c| c.degree())
        .max()
        .unwrap_or(0)
}

/// Repeatedly strips irreducible factors of the form `Phi_d(q^alpha t^beta)`
/// common to both inputs, returning their product and dividing them out.
///
/// This is a structural shortcut for the hook-derived part of a denominator.
/// Many denominator factors in this module are Macdonald hook factors
/// `1 - q^a t^b`, coming from `c_poly`/`c_prime_poly`, `z_qt`'s
/// `(1-q^k)/(1-t^k)`, and the normalizing factors. Such a factor is generally
/// reducible:
/// `1 - u^g = prod_{d | g} Phi_d(u)` for `u = q^alpha t^beta`, so a shared
/// divisor can be any of those cyclotomic pieces -- `1 - q^2` contributing a
/// bare `1 + q`, for instance. Enumerating the `Phi_d` directly therefore
/// subsumes enumerating the hook factors (`Phi_1(u) = u - 1`) while also
/// catching the pieces, and each candidate costs one linear-time trial
/// division instead of a subresultant PRS whose Z[t] coefficients blow up
/// multiplicatively. Recursion divisors such as `norm(nu)-norm(mu)` can be
/// genuinely multi-term and non-cyclotomic when `nu/mu` spans several rows;
/// this shortcut deliberately leaves those residual factors to the exact PRS.
///
/// `gcd_image` (the true gcd's image mod p at t = t0) bounds the search:
/// only candidates whose own image divides it can possibly be factors, and
/// their q-degree `alpha * phi(d)` cannot exceed its degree. In practice
/// that collapses the candidate set to a handful.
fn strip_common_cyclotomic_factors(
    a_star: &mut Poly2,
    b_star: &mut Poly2,
    gcd_image: &[u64],
    t0: u64,
) -> Poly2 {
    let p = GCD_FILTER_PRIME;
    let target_degree = gcd_image.len().saturating_sub(1);
    let mut common = Poly2::one();
    if target_degree == 0 {
        return common;
    }

    loop {
        let max_alpha = a_star.deg_q().min(b_star.deg_q()).min(target_degree);
        let max_beta = max_t_degree(a_star).min(max_t_degree(b_star));
        let mut found = false;
        'search: for alpha in 1..=max_alpha {
            for beta in 0..=max_beta {
                if num_integer::gcd(alpha, beta) != 1 {
                    // Non-primitive directions are covered by the primitive
                    // one they are a multiple of.
                    continue;
                }
                for d in 1.. {
                    let degree = alpha * euler_phi(d);
                    if degree > target_degree {
                        // phi is not monotonic, so keep going a little past
                        // the first overshoot before giving up on this
                        // direction.
                        if d > 2 * target_degree + 4 {
                            break;
                        }
                        continue;
                    }
                    let candidate = cyclotomic_factor(d, alpha, beta);
                    if !divides_mod(&poly2_reduce_mod(&candidate, t0, p), gcd_image, p) {
                        continue;
                    }
                    let Some(quotient_a) = try_exact_divide_unit_constant(a_star, &candidate)
                    else {
                        continue;
                    };
                    let Some(quotient_b) = try_exact_divide_unit_constant(b_star, &candidate)
                    else {
                        continue;
                    };
                    *a_star = quotient_a;
                    *b_star = quotient_b;
                    common = common * candidate;
                    found = true;
                    break 'search;
                }
            }
        }
        if !found {
            return common;
        }
    }
}

/// Bivariate polynomial gcd. Tries a sequence of cheap structural shortcuts
/// first (common power of q, then a rigorous modular coprimality check,
/// then cyclotomic factor stripping bounded by that check's degree -- see
/// `strip_common_q_power`/`provably_coprime_primitive_parts`/
/// `strip_common_cyclotomic_factors`), falling back to the subresultant
/// pseudo-remainder-sequence algorithm (Collins/Brown-Traub) only for
/// whatever is left. The PRS stays entirely within Z[t][q] (pseudo-division
/// plus exact division by explicitly tracked scalars in Z[t], never field
/// division over Frac(Z[t])) and cancels pseudo-division's coefficient
/// blowup via cheap beta_i/psi_i scalar bookkeeping. See:
/// <https://en.wikipedia.org/wiki/Polynomial_greatest_common_divisor#Subresultants>
fn poly2_gcd_inner(a: &Poly2, b: &Poly2) -> Poly2 {
    if a.is_zero() {
        return b.primitive_part_positive();
    }
    if b.is_zero() {
        return a.primitive_part_positive();
    }

    // Standard Gauss's-lemma decomposition: gcd(a,b) = gcd(cont(a),cont(b))
    // * gcd(pp(a),pp(b)), where cont/pp are with respect to q. The content
    // factor is reincorporated (by direct multiplication, not re-stripped)
    // once the primitive-part gcd is found below.
    let content = intpoly_gcd_z(&a.content_t(), &b.content_t());
    let mut a_star = a.primitive_part_positive();
    let mut b_star = b.primitive_part_positive();
    if a_star.deg_q() < b_star.deg_q() {
        std::mem::swap(&mut a_star, &mut b_star);
    }

    // Bare monomials are common gcds here and far too cheap to hand to the
    // PRS, so take the common power of q out first.
    let q_power = strip_common_q_power(&mut a_star, &mut b_star);
    let mut easy_part = Poly2::monomial(q_power, 0, BigInt::one());

    // One cheap modular gcd both proves coprimality outright (the common
    // case, and the PRS's worst case since it then runs to completion) and,
    // when it does not, bounds the search for the structural factors below.
    if let Some((gcd_image, t0)) = modular_gcd_image(&a_star, &b_star) {
        if gcd_image.len() <= 1 {
            return easy_part.primitive_part_positive().scale(&content);
        }
        easy_part =
            easy_part * strip_common_cyclotomic_factors(&mut a_star, &mut b_star, &gcd_image, t0);
        // Stripping can expose a fresh common power of q.
        let extra = strip_common_q_power(&mut a_star, &mut b_star);
        easy_part = easy_part * Poly2::monomial(extra, 0, BigInt::one());
        if provably_coprime_primitive_parts(&a_star, &b_star) {
            return easy_part.primitive_part_positive().scale(&content);
        }
    }

    let mut r_prev = a_star; // r_0
    let mut r_curr = b_star; // r_1

    let mut d_prev: i64 = 0;
    let mut gamma_prev = IntPoly::zero();
    let mut psi_prev = IntPoly::zero();

    let mut i = 1u32;
    loop {
        let d_i = r_prev.deg_q() as i64 - r_curr.deg_q() as i64;
        assert!(d_i >= 0, "subresultant PRS: degrees must be non-increasing");
        let gamma_i = r_curr.leading_intpoly().clone();

        let (beta_i, psi_i) = if i == 1 {
            let psi_1 = IntPoly::from_i64(-1);
            let beta_1 = if (d_i + 1).rem_euclid(2) == 0 {
                IntPoly::one()
            } else {
                IntPoly::from_i64(-1)
            };
            (beta_1, psi_1)
        } else {
            // d_prev (= d_{i-1}) can only be 0 here when i == 2 (i.e. it's
            // d_1 = deg(a_star) - deg(b_star), which the caller's inputs
            // are free to make 0); for i >= 3, d_{i-1} >= 1 always, since a
            // pseudo-remainder always has strictly smaller degree than its
            // divisor. When d_prev == 0, psi_prev is exactly psi_1 = -1 (a
            // unit), so the negative exponent below is handled by
            // `pow_intpoly`'s unit special-case rather than needing d_prev
            // >= 1 unconditionally.
            let neg_gamma_prev = -gamma_prev.clone();
            let psi_i =
                pow_intpoly(&neg_gamma_prev, d_prev).exact_div(&pow_intpoly(&psi_prev, d_prev - 1));
            let beta_i = -(gamma_prev.clone() * pow_intpoly(&psi_i, d_i));
            (beta_i, psi_i)
        };

        let prem = pseudo_remainder(&r_prev, &r_curr);
        if prem.is_zero() {
            return (r_curr.primitive_part_positive() * easy_part.clone())
                .primitive_part_positive()
                .scale(&content);
        }
        let r_next = prem.div_by_intpoly_exact(&beta_i);

        r_prev = r_curr;
        r_curr = r_next;
        gamma_prev = gamma_i;
        psi_prev = psi_i;
        d_prev = d_i;
        i += 1;
    }
}

/// The leading (highest q-degree, then highest t-degree within that)
/// integer coefficient. Panics on the zero polynomial.
fn poly2_leading_bigint(p: &Poly2) -> BigInt {
    let top = p.leading_intpoly();
    let degree = top.degree().expect("nonzero polynomial");
    top.coeff(degree)
}

fn poly2_exact_div(a: &Poly2, divisor: &Poly2) -> Poly2 {
    let (quotient, remainder) = rf_poly_div_rem(a.to_rf_coeffs(), divisor.to_rf_coeffs());
    assert!(
        remainder.iter().all(RationalFunction::is_zero),
        "non-exact bivariate polynomial division"
    );
    let raw = rf_coeffs_to_poly2_exact(&quotient);
    if raw.is_zero() || a.is_zero() {
        return raw;
    }

    // Zero remainder in field-coefficient polynomial division does NOT
    // imply the quotient has integer-in-t coefficients (e.g. q divided by
    // 2q has zero remainder with quotient 1/2, a degree-0-in-q rational
    // function, not an integer). rf_coeffs_to_poly2_exact's per-coefficient
    // LCM-based denominator-clearing then returns an integer multiple of
    // the true quotient rather than the true quotient itself whenever this
    // happens. Detect and correct for that by verifying raw*divisor
    // reproduces `a` exactly; if not, the two differ by a pure integer
    // scalar (never a genuine polynomial factor, since both represent the
    // same q,t-shape up to that scalar), recoverable from their leading
    // coefficients.
    let check = raw.clone() * divisor.clone();
    if check == *a {
        return raw;
    }
    let (check_scale, remainder) = poly2_leading_bigint(&check).div_rem(&poly2_leading_bigint(a));
    assert!(
        remainder.is_zero(),
        "poly2_exact_div: non-integer correction scale"
    );
    let corrected = raw.div_by_intpoly_exact(&IntPoly::monomial(0, check_scale));
    debug_assert_eq!(corrected.clone() * divisor.clone(), *a);
    corrected
}

/// Cancels the full common factor of two integer bivariate polynomials.
/// Keeping this operation separate lets rational multiplication cancel
/// across numerator/denominator pairs before forming much larger products.
fn cancel_common_factor(a: Poly2, b: Poly2) -> (Poly2, Poly2) {
    let common = poly2_gcd(&a, &b);
    if common.is_one() {
        (a, b)
    } else {
        (poly2_exact_div(&a, &common), poly2_exact_div(&b, &common))
    }
}

/// An exact rational function in q and t, always kept in canonical lowest
/// terms (via bivariate gcd cancellation) with a sign-normalized
/// denominator, mirroring [`crate::RationalFunction`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RationalFunction2 {
    numerator: Poly2,
    denominator: Poly2,
}

impl RationalFunction2 {
    pub fn new(numerator: Poly2, denominator: Poly2) -> Self {
        assert!(!denominator.is_zero(), "zero denominator");
        if numerator.is_zero() {
            return Self {
                numerator: Poly2::zero(),
                denominator: Poly2::one(),
            };
        }

        let mut numerator = numerator;
        let mut denominator = denominator;

        let content_gcd = intpoly_gcd_z(&numerator.content_t(), &denominator.content_t());
        if !content_gcd.is_one() {
            numerator = numerator.div_by_intpoly_exact(&content_gcd);
            denominator = denominator.div_by_intpoly_exact(&content_gcd);
        }

        let common = poly2_gcd(&numerator, &denominator);
        if !common.is_one() {
            numerator = poly2_exact_div(&numerator, &common);
            denominator = poly2_exact_div(&denominator, &common);
        }

        if denominator.leading_sign_negative() {
            numerator = -numerator;
            denominator = -denominator;
        }

        Self {
            numerator,
            denominator,
        }
    }

    pub fn zero() -> Self {
        Self::from_i64(0)
    }

    pub fn one() -> Self {
        Self::from_i64(1)
    }

    pub fn from_i64(n: i64) -> Self {
        Self {
            numerator: Poly2::from_i64(n),
            denominator: Poly2::one(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    /// Builds from coprime numerator/denominator parts. This is used after
    /// cross-cancellation in multiplication and division: since both inputs
    /// are already canonical, cancelling gcd(a,d) and gcd(c,b) leaves the
    /// two resulting products coprime in the UFD Z[q,t].
    fn from_coprime_parts(numerator: Poly2, mut denominator: Poly2) -> Self {
        assert!(!denominator.is_zero(), "zero denominator");
        if numerator.is_zero() {
            return Self::zero();
        }
        let mut numerator = numerator;
        if denominator.leading_sign_negative() {
            numerator = -numerator;
            denominator = -denominator;
        }
        Self {
            numerator,
            denominator,
        }
    }

    /// Swaps q and t throughout (see [`Poly2::swap_qt`]).
    pub(crate) fn swap_qt(&self) -> Self {
        Self::new(self.numerator.swap_qt(), self.denominator.swap_qt())
    }

    #[cfg(test)]
    fn evaluate(
        &self,
        q: &num_rational::Ratio<BigInt>,
        t: &num_rational::Ratio<BigInt>,
    ) -> num_rational::Ratio<BigInt> {
        self.numerator.evaluate(q, t) / self.denominator.evaluate(q, t)
    }
}

impl Add for RationalFunction2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        // LCM-based addition (a/b + c/d = (a*(d/g) + c*(b/g)) / (b*(d/g)),
        // g = gcd(b,d)) rather than naive cross-multiplication (a/b + c/d =
        // (ad+bc)/(bd)): this recursion chains many additions of values
        // that tend to share common denominator factors (related hook-type
        // products), and using the raw product bd instead of the lcm let
        // coefficients balloon to hundreds of digits within a handful of
        // steps even though the final (fully reduced) values are small.
        if self.denominator == rhs.denominator {
            return Self::new(self.numerator + rhs.numerator, self.denominator);
        }
        let g = poly2_gcd(&self.denominator, &rhs.denominator);
        let self_denom_factor = poly2_exact_div(&self.denominator, &g);
        let rhs_denom_factor = poly2_exact_div(&rhs.denominator, &g);
        let common_denominator = self.denominator * rhs_denom_factor.clone();
        let numerator = self.numerator * rhs_denom_factor + rhs.numerator * self_denom_factor;
        Self::new(numerator, common_denominator)
    }
}

impl Sub for RationalFunction2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for RationalFunction2 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.numerator, self.denominator)
    }
}

impl Mul for RationalFunction2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let (a, d) = cancel_common_factor(self.numerator, rhs.denominator);
        let (c, b) = cancel_common_factor(rhs.numerator, self.denominator);
        Self::from_coprime_parts(a * c, b * d)
    }
}

impl std::ops::Div for RationalFunction2 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        assert!(!rhs.numerator.is_zero(), "division by zero");
        if self.is_zero() {
            return Self::zero();
        }
        let (a, c) = cancel_common_factor(self.numerator, rhs.numerator);
        let (d, b) = cancel_common_factor(rhs.denominator, self.denominator);
        Self::from_coprime_parts(a * d, b * c)
    }
}

impl fmt::Display for RationalFunction2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "({})/({})", self.numerator, self.denominator)
        }
    }
}

/// The (q,t) hook-ratio `b_lambda(s) = c_lambda(s)/c'_lambda(s)` for a cell
/// with the given arm and leg (Chen-Sahi Eq. 2.17-2.19).
fn c_poly(arm: usize, leg: usize) -> Poly2 {
    Poly2::one() - Poly2::monomial(arm, leg + 1, BigInt::one())
}

fn c_prime_poly(arm: usize, leg: usize) -> Poly2 {
    Poly2::one() - Poly2::monomial(arm + 1, leg, BigInt::one())
}

fn b_macdonald(arm: usize, leg: usize) -> RationalFunction2 {
    RationalFunction2::new(c_poly(arm, leg), c_prime_poly(arm, leg))
}

/// psi_{lambda/mu}(q,t) (Chen-Sahi Eq. 2.16), for lambda ⊇ mu with
/// lambda/mu a horizontal strip: the product of b_mu(s)/b_lambda(s) over
/// cells s of mu that are in a row of lambda meeting lambda/mu but not in a
/// column of lambda meeting lambda/mu. Structurally identical to
/// [`crate`]'s `macdonald_psi_with_parameter` cell-selection (same (R\C)
/// set), just with the two-parameter `b_macdonald` in place of the
/// single-parameter Jack hook ratio.
fn macdonald_psi_qt(lambda_in: &[usize], mu_in: &[usize]) -> RationalFunction2 {
    let [lambda, mu] = normalize_two(lambda_in, mu_in);
    let [lambdac, muc] =
        normalize_two(&conjugate_partition(lambda_in), &conjugate_partition(mu_in));
    let mut product = RationalFunction2::one();

    for r in 0..mu.len() {
        for c in 1..=mu[r] {
            if mu[r] < lambda[r] && muc[c - 1] == lambdac[c - 1] {
                let arm_l = lambda[r] - c;
                let leg_l = lambdac[c - 1] - r - 1;
                let arm_m = mu[r] - c;
                let leg_m = muc[c - 1] - r - 1;
                let b_m = b_macdonald(arm_m, leg_m);
                let b_l = b_macdonald(arm_l, leg_l);
                product = product * (b_m / b_l);
            }
        }
    }

    product
}

/// The "adjacent binomial coefficient" a_{lambda,mu} (Chen-Sahi Prop. 4.3,
/// Eq. 4.11) that drives the structure-constant recursion -- a different
/// object from `macdonald_psi_qt` above (that one weights tableaux in the
/// evaluation formula; this one is the recursion's own branching
/// coefficient). Requires `bigger` to cover `smaller` (differ by exactly
/// one box).
fn adjacent_binomial_coefficient(bigger_in: &[usize], smaller_in: &[usize]) -> RationalFunction2 {
    let [bigger, smaller] = normalize_two(bigger_in, smaller_in);
    let r0 = (0..bigger.len())
        .find(|&r| bigger[r] != smaller[r])
        .expect("bigger and smaller must differ by exactly one box");
    let c0 = smaller[r0] + 1;
    debug_assert_eq!(bigger[r0], smaller[r0] + 1);

    let [bigger_c, smaller_c] = normalize_two(
        &conjugate_partition(&bigger),
        &conjugate_partition(&smaller),
    );

    let mut product = RationalFunction2::one();

    // C = column c0 of `bigger`, excluding row r0: ratio c_bigger(s)/c_smaller(s).
    let col_height = bigger_c[c0 - 1];
    for r in 0..col_height {
        if r == r0 {
            continue;
        }
        let arm_b = bigger[r] - c0;
        let leg_b = bigger_c[c0 - 1] - r - 1;
        let arm_s = smaller[r] - c0;
        let leg_s = smaller_c[c0 - 1] - r - 1;
        product = product * RationalFunction2::new(c_poly(arm_b, leg_b), c_poly(arm_s, leg_s));
    }

    // R = row r0 of `bigger`, excluding column c0: ratio c'_bigger(s)/c'_smaller(s).
    for c in 1..=bigger[r0] {
        if c == c0 {
            continue;
        }
        let arm_b = bigger[r0] - c;
        let leg_b = bigger_c[c - 1] - r0 - 1;
        let arm_s = smaller[r0] - c;
        let leg_s = smaller_c[c - 1] - r0 - 1;
        product = product
            * RationalFunction2::new(c_prime_poly(arm_b, leg_b), c_prime_poly(arm_s, leg_s));
    }

    let prefactor = RationalFunction2::new(Poly2::one(), Poly2::monomial(0, r0, BigInt::one()));
    prefactor * product
}

/// `‖bigger_bar‖ - ‖smaller_bar‖` for the AM family, i.e.
/// `sum_r (q^bigger[r] - q^smaller[r]) * t^(n-1-r)`, using the ambient
/// variable count `n`. Used both as the recursion's main divisor and as the
/// per-term weight in its two sums (Chen-Sahi Eq. 3.17.ii).
fn norm_diff(bigger_in: &[usize], smaller_in: &[usize], n: usize) -> RationalFunction2 {
    let mut total = Poly2::zero();
    for r in 0..n {
        let b = bigger_in.get(r).copied().unwrap_or(0);
        let s = smaller_in.get(r).copied().unwrap_or(0);
        if b != s {
            let t_exp = n - 1 - r;
            total = total + Poly2::monomial(b, t_exp, BigInt::one())
                - Poly2::monomial(s, t_exp, BigInt::one());
        }
    }
    RationalFunction2::new(total, Poly2::one())
}

/// The (q,t) analog of `shifted_jack_evaluate`: evaluates the interpolation
/// Macdonald polynomial h_shape^monic at the point shape_bar (Chen-Sahi Eq.
/// 2.14, Okounkov's combinatorial formula), using ambient variable count
/// `n`. Sums over the same SSYT/GT-chain machinery as the Jack evaluator
/// (reverse tableaux there are represented via entries n+1-e relative to
/// the increasing-tableau convention already implemented, since the two
/// conventions produce the same GT-chain level-set sequence).
fn shifted_macdonald_evaluate(shape: &[usize], point: &[usize], n: usize) -> RationalFunction2 {
    let shape = trim_partition(shape);
    let mut total = RationalFunction2::zero();

    for tableau in semistandard_tableaux(&shape, n) {
        let mut term = RationalFunction2::one();
        let chain = gt_chain_from_tableau(&tableau, &shape, n);
        for adjacent in chain.windows(2) {
            term = term * macdonald_psi_qt(&adjacent[0], &adjacent[1]);
            if term.is_zero() {
                break;
            }
        }
        if term.is_zero() {
            continue;
        }

        for (r, row) in tableau.iter().enumerate() {
            for (c, &entry) in row.iter().enumerate() {
                let x_index = n - entry;
                let point_value = point.get(x_index).copied().unwrap_or(0);
                let x_term = Poly2::monomial(point_value, entry - 1, BigInt::one());
                let subtrahend = Poly2::monomial(c, entry - 1 - r, BigInt::one());
                term = term * RationalFunction2::new(x_term - subtrahend, Poly2::one());
                if term.is_zero() {
                    break;
                }
            }
            if term.is_zero() {
                break;
            }
        }
        total = total + term;
    }

    total
}

/// c'_mu(q,t) = prod_{s in mu} (1 - q^(arm(s)+1) t^leg(s)), the (q,t) analog
/// of `hook_prime_factor`.
fn hook_prime_product_qt(mu: &[usize]) -> Poly2 {
    let mu = trim_partition(mu);
    let muc = conjugate_partition(&mu);
    let mut product = Poly2::one();
    for (r, &row_len) in mu.iter().enumerate() {
        for c in 1..=row_len {
            let arm = row_len - c;
            let leg = muc[c - 1] - r - 1;
            product = product * c_prime_poly(arm, leg);
        }
    }
    product
}

/// c_mu(q,t) = prod_{s in mu} (1 - q^arm(s) t^(leg(s)+1)): the scalar
/// relating the Macdonald `J` (integral) and monic `P` bases,
/// `J_mu(x;q,t) = c_mu(q,t) * P_mu(x;q,t)` (Macdonald's book VI (6.19)).
/// Verified numerically against Sage's `J`/`P` ratio for all mu with
/// |mu| <= 3 (see `hook_c_product_qt_matches_sage_j_to_p_ratio`).
pub(crate) fn hook_c_product_qt(mu: &[usize]) -> Poly2 {
    let mu = trim_partition(mu);
    let muc = conjugate_partition(&mu);
    let mut product = Poly2::one();
    for (r, &row_len) in mu.iter().enumerate() {
        for c in 1..=row_len {
            let arm = row_len - c;
            let leg = muc[c - 1] - r - 1;
            product = product * c_poly(arm, leg);
        }
    }
    product
}

/// n(mu) := sum_r r*mu[r] (0-indexed rows), the standard partition
/// statistic -- unrelated to the ambient variable count `n` despite the
/// unfortunate notational clash in Chen-Sahi Eq. 4.4.
fn n_of_partition(mu: &[usize]) -> usize {
    mu.iter().enumerate().map(|(r, &part)| r * part).sum()
}

/// H(mu;q,t) (Chen-Sahi Eq. 4.4), converting between the "unital"
/// interpolation-polynomial normalization (h_mu(mu_bar) = 1) and the monic
/// one (coefficient of m_mu is 1): h_mu = h_mu^monic / H(mu).
fn normalizing_factor(mu: &[usize], n: usize) -> RationalFunction2 {
    let mu_t = trim_partition(mu);
    let mu_c = conjugate_partition(&mu_t);
    let size = partition_size(&mu_t) as i64;
    let n_mu = n_of_partition(&mu_t) as i64;
    let n_mu_conj = n_of_partition(&mu_c);

    let sign: i64 = if size % 2 == 0 { 1 } else { -1 };
    let t_exponent = (n as i64 - 1) * size - 2 * n_mu;
    assert!(
        t_exponent >= 0,
        "normalizing_factor: unexpected negative t exponent"
    );

    let monomial = Poly2::monomial(n_mu_conj, t_exponent as usize, BigInt::from(sign));
    RationalFunction2::new(monomial * hook_prime_product_qt(&mu_t), Poly2::one())
}

/// `sum_r q^part[r] * t^(n-1-r)`, the shifted-norm used in Eq. 3.19-3.20's
/// weight formula and in `norm_diff`.
fn norm(part: &[usize], n: usize) -> RationalFunction2 {
    let mut total = Poly2::zero();
    for r in 0..n {
        let p = part.get(r).copied().unwrap_or(0);
        total = total + Poly2::monomial(p, n - 1 - r, BigInt::one());
    }
    RationalFunction2::new(total, Poly2::one())
}

/// All maximal chains top = z0 :> z1 :> ... :> zk = bottom in Young's
/// lattice (each step a single-box covering relation), for top ⊇ bottom.
/// The count equals the number of standard Young tableaux of shape
/// top/bottom.
fn enumerate_maximal_chains(top: &[usize], bottom: &[usize]) -> Vec<Vec<Partition>> {
    let top = trim_partition(top);
    let bottom = trim_partition(bottom);
    if top == bottom {
        return vec![vec![top]];
    }
    let mut result = Vec::new();
    for candidate in remove_box_from_partition(&top) {
        if skew_shape_contains(&candidate, &bottom) {
            for mut sub_chain in enumerate_maximal_chains(&candidate, &bottom) {
                let mut chain = vec![top.clone()];
                chain.append(&mut sub_chain);
                result.push(chain);
            }
        }
    }
    result
}

/// c^nu_{mu,lambda} (unital, our (lambda,mu,nu) convention -- see
/// `unital_h_structure_constant`'s doc comment for the mapping to
/// Chen-Sahi's (lambda,mu,nu)) via Theorem 3.7 ("Theorem E", the weighted
/// sum / chain formula, Eq. 3.19-3.20): a sum over maximal chains from nu
/// down to mu, each contributing a product of closed-form `a` values and a
/// weighted sum of closed-form `b` evaluations. Unlike
/// `unital_h_structure_constant`'s recursion, this never computes an
/// intermediate structure constant recursively -- every term is closed-form
/// -- at the cost of the chain count growing with the number of standard
/// Young tableaux of the skew shape nu/mu.
fn weighted_chain_sum_structure_constant(
    lambda: &[usize],
    mu: &[usize],
    nu: &[usize],
    n: usize,
) -> RationalFunction2 {
    if !skew_shape_contains(nu, mu) || !skew_shape_contains(nu, lambda) {
        return RationalFunction2::zero();
    }

    let chains = enumerate_maximal_chains(nu, mu);
    let mut total = RationalFunction2::zero();
    for chain in &chains {
        let k = chain.len() - 1;
        let norms: Vec<RationalFunction2> = chain.iter().map(|z| norm(z, n)).collect();

        let a_product = (0..k)
            .map(|i| adjacent_binomial_coefficient(&chain[i], &chain[i + 1]))
            .fold(RationalFunction2::one(), |acc, x| acc * x);

        let numerator: RationalFunction2 = (0..k)
            .map(|i| norms[i].clone() - norms[i + 1].clone())
            .fold(RationalFunction2::one(), |acc, x| acc * x);

        let mut wt = RationalFunction2::zero();
        for j in 0..=k {
            let mut denom = RationalFunction2::one();
            for i in 0..=k {
                if i != j {
                    denom = denom * (norms[j].clone() - norms[i].clone());
                }
            }
            let b_j =
                shifted_macdonald_evaluate(lambda, &chain[j], n) / normalizing_factor(lambda, n);
            wt = wt + (numerator.clone() / denom) * b_j;
        }

        total = total + a_product * wt;
    }
    total
}

type StructureConstantKey = (Partition, Partition, Partition, usize);

/// The general (shifted/binomial-type) structure constant c^nu_{lambda,mu}
/// in Chen-Sahi's "unital" h-basis normalization (Theorem 3.6/3.7, Eq.
/// 3.17), computed via the same box-by-box recursion as
/// [`crate::ShiftedJackCalculator::jack_p_structure_constant`], with
/// `macdonald_psi_qt`/`adjacent_binomial_coefficient`/`norm_diff` standing
/// in for the Jack case's psi/psi'/plain-size-difference. This is nonzero
/// for a range of |nu| (not just top degree) -- [`MacdonaldCalculator`]
/// only surfaces the top-degree slice publicly, per the "regular, not
/// shifted" scope, but needs this general recursion as an internal step to
/// get there (there's no simpler top-degree-only closed form beyond the
/// single-box Pieri case).
fn unital_h_structure_constant(
    cache: &mut HashMap<StructureConstantKey, RationalFunction2>,
    lambda: &[usize],
    mu: &[usize],
    nu: &[usize],
    n: usize,
) -> RationalFunction2 {
    if !skew_shape_contains(nu, mu) || !skew_shape_contains(nu, lambda) {
        return RationalFunction2::zero();
    }
    if partition_size(lambda) > partition_size(mu) {
        return unital_h_structure_constant(cache, mu, lambda, nu, n);
    }

    let [lambda, mu, nu] = normalize_three(lambda, mu, nu);
    let key = (lambda.clone(), mu.clone(), nu.clone(), n);
    if let Some(value) = cache.get(&key) {
        return value.clone();
    }

    let value = if mu == nu {
        // shifted_macdonald_evaluate computes h_lambda^monic(nu_bar) (Eq.
        // 2.14 is stated for the monic normalization); the recursion needs
        // the unital h_lambda(nu_bar) = h_lambda^monic(nu_bar) / H(lambda).
        shifted_macdonald_evaluate(&lambda, &nu, n) / normalizing_factor(&lambda, n)
    } else if lambda == nu {
        shifted_macdonald_evaluate(&mu, &nu, n) / normalizing_factor(&mu, n)
    } else {
        let mut total = RationalFunction2::zero();
        for mup in add_box_to_partition(&mu) {
            let a_coef = adjacent_binomial_coefficient(&mup, &mu);
            let weight = norm_diff(&mup, &mu, n);
            let sub = unital_h_structure_constant(cache, &lambda, &mup, &nu, n);
            total = total + a_coef * weight * sub;
        }
        for nup in remove_box_from_partition(&nu) {
            let a_coef = adjacent_binomial_coefficient(&nu, &nup);
            let weight = norm_diff(&nu, &nup, n);
            let sub = unital_h_structure_constant(cache, &lambda, &mu, &nup, n);
            total = total - a_coef * weight * sub;
        }
        let divisor = norm_diff(&nu, &mu, n);
        total / divisor
    };

    cache.insert(key, value.clone());
    value
}

/// Computes ordinary (non-shifted) Macdonald Littlewood-Richardson
/// coefficients: for the monic P basis, c^nu_{lambda,mu}(q,t) such that
///
///   P_lambda(x; q,t) * P_mu(x; q,t) = sum_nu c^nu_{lambda,mu}(q,t) P_nu(x; q,t),
///
/// which is nonzero only when |nu| = |lambda| + |mu|.
#[derive(Default)]
pub struct MacdonaldCalculator {
    cache: HashMap<StructureConstantKey, RationalFunction2>,
}

/// Which method to use for the internal "unital" structure constant
/// computation that [`MacdonaldCalculator::p_structure_constant_with`]
/// converts into the ordinary monic P-basis answer. Both are exact and
/// always agree; they differ only in how the work is organized, and can
/// have very different performance depending on the shapes involved (see
/// each variant's doc comment).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// The box-by-box recursion (Chen-Sahi Eq. 3.17.ii), memoized across
    /// calls on the same [`MacdonaldCalculator`]. Generally the better
    /// default: benefits from the cache when computing many related
    /// coefficients, and empirically outperforms `ChainSum` on every case
    /// tested so far.
    #[default]
    Recursive,
    /// The weighted sum over maximal chains from nu down to mu (Chen-Sahi
    /// Theorem 3.7 / Eq. 3.19-3.20). Never recurses into sub-structure-
    /// constants, but the number of chains equals the number of standard
    /// Young tableaux of the skew shape nu/mu, and each chain's weight
    /// formula does O(k^2) rational-function arithmetic on already-complex
    /// values -- empirically flat-to-worse vs. `Recursive`, not memoized
    /// (no shared state to cache), kept mainly for cross-checking and in
    /// case some shape profile favors it.
    ChainSum,
    /// Completely independent computation ([`crate::brute_force`]): builds
    /// P_lambda, P_mu, P_nu via Gram-Schmidt in the power-sum basis (where
    /// multiplication and the Macdonald inner product are both simple),
    /// then c^nu_{lambda,mu} = <P_lambda*P_mu, P_nu> / <P_nu,P_nu>. Shares
    /// no code with the other two variants -- useful as a correctness
    /// reference -- but its Gram-Schmidt setup (a monomial-to-power-sum
    /// transition via evaluating at several integer points and solving a
    /// p(n) x p(n) linear system, p(n) = number of partitions of the
    /// relevant degree) has its own scaling profile, distinct from either
    /// recursion or chain enumeration.
    BruteForce,
    /// Replicates Sage's own internal technique: convert P_lambda, P_mu to
    /// the Schur basis, multiply using the classical INTEGER
    /// Littlewood-Richardson coefficients (no q,t arithmetic at all in
    /// this step), convert back to power-sum coordinates, then extract the
    /// P_nu coefficient via the same diagonal inner product as
    /// `BruteForce`. Note this still uses this crate's own Gram-Schmidt
    /// (not Sage's Lapointe-Lascoux-Morse creation-operator formula) to
    /// build P_lambda/P_mu/P_nu in the first place, so it only isolates
    /// the benefit of doing the multiplication step in pure integers --
    /// see [`crate::brute_force::schur_sandwich_structure_constant`].
    SchurSandwich,
    /// A genuine (not Gram-Schmidt-based) replication of Sage's technique:
    /// P_lambda, P_mu are built directly in the Schur basis via the
    /// Lapointe-Lascoux-Morse column-adding creation operators ([LLM1998]
    /// Cor. 4.3, the same formula Sage's own Macdonald `J`/`P` bases use
    /// internally), multiplied via the classical integer
    /// Littlewood-Richardson coefficients, then converted to the P basis
    /// via a Schur<->P change-of-basis matrix that is itself built from
    /// the same creation-operator construction (inverted once per
    /// degree) -- no Gram-Schmidt anywhere in this variant. See
    /// [`crate::brute_force::creation_operator_structure_constant`].
    CreationOperator,
}

impl MacdonaldCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn p_structure_constant(
        &mut self,
        lambda: &[usize],
        mu: &[usize],
        nu: &[usize],
    ) -> RationalFunction2 {
        self.p_structure_constant_with(lambda, mu, nu, Algorithm::default())
    }

    pub fn p_structure_constant_with(
        &mut self,
        lambda: &[usize],
        mu: &[usize],
        nu: &[usize],
        algorithm: Algorithm,
    ) -> RationalFunction2 {
        if !skew_shape_contains(nu, mu) || !skew_shape_contains(nu, lambda) {
            return RationalFunction2::zero();
        }
        if partition_size(nu) != partition_size(lambda) + partition_size(mu) {
            // Regular (ordinary, non-shifted) Macdonald LR coefficients are
            // only defined/nonzero at the top degree.
            return RationalFunction2::zero();
        }

        // BruteForce computes the monic P-basis structure constant
        // directly (via the power-sum-basis inner product), unlike the
        // other two algorithms, which compute Chen-Sahi's "unital" h-basis
        // value first and need the H-conversion below -- so it returns
        // straight away.
        if algorithm == Algorithm::BruteForce {
            return crate::brute_force::brute_force_structure_constant(lambda, mu, nu);
        }
        if algorithm == Algorithm::SchurSandwich {
            return crate::brute_force::schur_sandwich_structure_constant(lambda, mu, nu);
        }
        if algorithm == Algorithm::CreationOperator {
            return crate::brute_force::creation_operator_structure_constant(lambda, mu, nu);
        }

        let n = [lambda.len(), mu.len(), nu.len()]
            .into_iter()
            .max()
            .unwrap_or(0)
            .max(1);

        let unital = match algorithm {
            Algorithm::Recursive => unital_h_structure_constant(&mut self.cache, lambda, mu, nu, n),
            Algorithm::ChainSum => weighted_chain_sum_structure_constant(lambda, mu, nu, n),
            Algorithm::BruteForce | Algorithm::SchurSandwich | Algorithm::CreationOperator => {
                unreachable!("handled above")
            }
        };
        if unital.is_zero() {
            return RationalFunction2::zero();
        }

        // Convert from Chen-Sahi's "unital" h-basis normalization to the
        // ordinary monic Macdonald P-basis (Eq. 4.16, extrapolated from the
        // Pieri case to general nu via the same top-degree/uniqueness
        // argument as Prop. 2.2(5)):
        //   c~^nu_{lambda,mu} = c^nu_{lambda,mu} * H(lambda) * H(mu) / H(nu)
        let h_lambda = normalizing_factor(lambda, n);
        let h_mu = normalizing_factor(mu, n);
        let h_nu = normalizing_factor(nu, n);
        unital * h_lambda * h_mu / h_nu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer_partitions;
    use num_rational::Ratio;

    fn q() -> Poly2 {
        Poly2::q()
    }

    fn t() -> Poly2 {
        Poly2::t()
    }

    /// Classical Littlewood--Richardson tableaux, implemented only for an
    /// independent q=t specialization check. Cells are filled in reverse
    /// row-reading order, so the row/column and lattice-word conditions can
    /// all be checked on each prefix.
    fn classical_lr_coefficient(lambda: &[usize], mu: &[usize], nu: &[usize]) -> usize {
        let lambda = trim_partition(lambda);
        let mu = trim_partition(mu);
        let nu = trim_partition(nu);
        if !skew_shape_contains(&nu, &lambda)
            || partition_size(&nu) != partition_size(&lambda) + partition_size(&mu)
        {
            return 0;
        }

        let mut cells = Vec::new();
        for (r, &row_len) in nu.iter().enumerate() {
            let inner_len = lambda.get(r).copied().unwrap_or(0);
            for c in (inner_len..row_len).rev() {
                cells.push((r, c));
            }
        }

        fn fill(
            index: usize,
            cells: &[(usize, usize)],
            lambda: &[usize],
            mu: &[usize],
            values: &mut HashMap<(usize, usize), usize>,
            used: &mut [usize],
        ) -> usize {
            if index == cells.len() {
                return usize::from(used == mu);
            }
            let (r, c) = cells[index];
            let mut total = 0;
            for value in 1..=mu.len() {
                if used[value - 1] == mu[value - 1] {
                    continue;
                }
                if let Some(&right) = values.get(&(r, c + 1)) {
                    if value > right {
                        continue;
                    }
                }
                if r > 0
                    && c >= lambda.get(r - 1).copied().unwrap_or(0)
                    && values.get(&(r - 1, c)).is_some_and(|&above| value <= above)
                {
                    continue;
                }

                used[value - 1] += 1;
                let lattice = (0..used.len().saturating_sub(1)).all(|i| used[i] >= used[i + 1]);
                if lattice {
                    values.insert((r, c), value);
                    total += fill(index + 1, cells, lambda, mu, values, used);
                    values.remove(&(r, c));
                }
                used[value - 1] -= 1;
            }
            total
        }

        fill(
            0,
            &cells,
            &lambda,
            &mu,
            &mut HashMap::new(),
            &mut vec![0; mu.len()],
        )
    }

    #[test]
    fn p_basis_specializes_to_classical_lr_when_q_equals_t() {
        let value = Ratio::from_integer(BigInt::from(2));
        let factors: Vec<Partition> = (1..=3).flat_map(integer_partitions).collect();
        let mut calculator = MacdonaldCalculator::new();

        for (i, lambda) in factors.iter().enumerate() {
            for mu in &factors[i..] {
                for nu in integer_partitions(partition_size(lambda) + partition_size(mu)) {
                    let expected = classical_lr_coefficient(lambda, mu, &nu);
                    let actual = calculator
                        .p_structure_constant(lambda, mu, &nu)
                        .evaluate(&value, &value);
                    assert_eq!(
                        actual,
                        Ratio::from_integer(BigInt::from(expected)),
                        "q=t specialization for {lambda:?} x {mu:?} -> {nu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn recursive_matches_gram_schmidt_for_all_outputs_through_factor_size_two() {
        let factors: Vec<Partition> = (1..=2).flat_map(integer_partitions).collect();
        let mut calculator = MacdonaldCalculator::new();

        for (i, lambda) in factors.iter().enumerate() {
            for mu in &factors[i..] {
                for nu in integer_partitions(partition_size(lambda) + partition_size(mu)) {
                    let recursive = calculator.p_structure_constant(lambda, mu, &nu);
                    let gram_schmidt = calculator.p_structure_constant_with(
                        lambda,
                        mu,
                        &nu,
                        Algorithm::BruteForce,
                    );
                    assert_eq!(
                        recursive, gram_schmidt,
                        "algorithm cross-check for {lambda:?} x {mu:?} -> {nu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn hook_c_product_qt_matches_sage_j_to_p_ratio() {
        // Sage: J[lam] / P[lam], expressed as a scalar ratio, for every
        // lam with |lam| <= 3 (verified via a direct sage run comparing
        // the monomial-basis leading coefficients of J and P).
        assert_eq!(hook_c_product_qt(&[1]), Poly2::one() - t());
        assert_eq!(
            hook_c_product_qt(&[2]),
            (Poly2::one() - t()) * (Poly2::one() - q() * t())
        );
        assert_eq!(
            hook_c_product_qt(&[1, 1]),
            (Poly2::one() - t()) * (Poly2::one() - t() * t())
        );
        assert_eq!(
            hook_c_product_qt(&[3]),
            (Poly2::one() - t()) * (Poly2::one() - q() * t()) * (Poly2::one() - q() * q() * t())
        );
        assert_eq!(
            hook_c_product_qt(&[2, 1]),
            (Poly2::one() - t()) * (Poly2::one() - t()) * (Poly2::one() - q() * t() * t())
        );
        assert_eq!(
            hook_c_product_qt(&[1, 1, 1]),
            (Poly2::one() - t()) * (Poly2::one() - t() * t()) * (Poly2::one() - t() * t() * t())
        );
    }

    #[test]
    fn rational_function2_arithmetic_is_evaluation_homomorphism() {
        use num_rational::Ratio;
        let q_val = Ratio::from_integer(BigInt::from(2));
        let t_val = Ratio::from_integer(BigInt::from(3));

        // Build a handful of moderately-nested values the way the
        // recursion does: several additions and multiplications of small
        // rational functions, then check evaluate() commutes with +,-,*,/
        // at every step -- this isolates whether the bug is in the core
        // Poly2/RationalFunction2 arithmetic (this test) versus the
        // Macdonald-specific formulas (tested elsewhere).
        let a = RationalFunction2::new(q() + Poly2::one(), q() - Poly2::one());
        let b = RationalFunction2::new(t() - Poly2::one(), q() * t() - Poly2::one());
        let c = RationalFunction2::new(q() * q() - t(), q() + t());

        let av = a.evaluate(&q_val, &t_val);
        let bv = b.evaluate(&q_val, &t_val);
        let cv = c.evaluate(&q_val, &t_val);

        assert_eq!(
            (a.clone() + b.clone()).evaluate(&q_val, &t_val),
            &av + &bv,
            "add"
        );
        assert_eq!(
            (a.clone() - b.clone()).evaluate(&q_val, &t_val),
            &av - &bv,
            "sub"
        );
        assert_eq!(
            (a.clone() * b.clone()).evaluate(&q_val, &t_val),
            &av * &bv,
            "mul"
        );
        assert_eq!(
            (a.clone() / b.clone()).evaluate(&q_val, &t_val),
            &av / &bv,
            "div"
        );

        // 2 rounds, not more: repeated multiplication by `a` with no
        // opportunity for cancellation is a genuinely adversarial
        // degree-growth stress case (true degree grows each round
        // regardless of how good the gcd/reduction is, unlike the real
        // Macdonald recursion's pattern where hook-type factors naturally
        // cancel a lot) -- this is a regression check on the arithmetic
        // laws, not a performance benchmark; correctness on the actual
        // workload is separately proven by the ground-truth tests.
        let mut acc = RationalFunction2::zero();
        let mut acc_val = Ratio::from_integer(BigInt::from(0));
        for _ in 0..2 {
            acc = acc * a.clone() + b.clone() * c.clone() - a.clone() / c.clone();
            acc_val = acc_val * &av + &bv * &cv - &av / &cv;
            assert_eq!(
                acc.evaluate(&q_val, &t_val),
                acc_val,
                "chained accumulation"
            );
        }
    }

    #[test]
    fn p_structure_constant_2_2_31_matches_sage_numerically() {
        // Sage ground truth: (t-1)(q-1)(q+1)^2(qt+1) / [(qt-1)(q^3*t-1)]
        use num_rational::Ratio;
        let q_val = Ratio::from_integer(BigInt::from(2));
        let t_val = Ratio::from_integer(BigInt::from(3));
        let expected = (&t_val - Ratio::from_integer(BigInt::from(1)))
            * (&q_val - Ratio::from_integer(BigInt::from(1)))
            * (&q_val + Ratio::from_integer(BigInt::from(1)))
            * (&q_val + Ratio::from_integer(BigInt::from(1)))
            * (&q_val * &t_val + Ratio::from_integer(BigInt::from(1)))
            / ((&q_val * &t_val - Ratio::from_integer(BigInt::from(1)))
                * (&q_val * &q_val * &q_val * &t_val - Ratio::from_integer(BigInt::from(1))));

        let mut calculator = MacdonaldCalculator::new();
        let actual = calculator.p_structure_constant(&[2], &[2], &[3, 1]);
        let actual_val = actual.evaluate(&q_val, &t_val);
        assert_eq!(actual_val, expected, "numeric mismatch at q=2,t=3");
    }

    #[test]
    fn adjacent_binomial_coefficient_cross_check_via_corollary_3_8() {
        // Sage: {2,1},{1},{3,1} -> 1, i.e. c~^{[3,1]}_{[2,1],[1]} = 1.
        // Corollary 3.8: c^{[3,1]}_{[2,1],[1]}(unital) = a_{[3,1],[2,1]} *
        // (b_{[3,1],[1]} - b_{[2,1],[1]}), and c~ = c(unital) * H([2,1]) *
        // H([1]) / H([3,1]), so we can solve for a_{[3,1],[2,1]}
        // independently of the main recursion and compare it to what
        // adjacent_binomial_coefficient actually computes.
        let n = 2;
        let h_31 = normalizing_factor(&[3, 1], n);
        let h_21 = normalizing_factor(&[2, 1], n);
        let h_1 = normalizing_factor(&[1], n);

        let unital_target = h_31.clone() / (h_21.clone() * h_1.clone()); // = c(unital), since c~=1

        let b_31_1 = shifted_macdonald_evaluate(&[1], &[3, 1], n) / normalizing_factor(&[1], n);
        let b_21_1 = shifted_macdonald_evaluate(&[1], &[2, 1], n) / normalizing_factor(&[1], n);
        let diff = b_31_1 - b_21_1;

        let implied_a = unital_target / diff;
        let actual_a = adjacent_binomial_coefficient(&[3, 1], &[2, 1]);

        eprintln!("implied a_[3,1],[2,1] = {implied_a}");
        eprintln!("actual  a_[3,1],[2,1] = {actual_a}");
        assert_eq!(implied_a, actual_a);
    }

    #[test]
    fn unital_h_structure_constant_matches_corollary_3_8_for_call4() {
        // nu=[3,1] covers mu=[2,1] directly, so Corollary 3.8 applies to
        // unital_h_structure_constant(lambda=[2],mu=[2,1],nu=[3,1]) itself:
        // c^{[3,1]}_{[2,1],[2]}(unital) = a_{[3,1],[2,1]} * (b_{[3,1],[2]} - b_{[2,1],[2]}).
        let n = 2;
        let a = adjacent_binomial_coefficient(&[3, 1], &[2, 1]);
        let b_31_2 = shifted_macdonald_evaluate(&[2], &[3, 1], n) / normalizing_factor(&[2], n);
        let b_21_2 = shifted_macdonald_evaluate(&[2], &[2, 1], n) / normalizing_factor(&[2], n);
        let expected = a * (b_31_2 - b_21_2);

        let mut cache = HashMap::new();
        let actual = unital_h_structure_constant(&mut cache, &[2], &[2, 1], &[3, 1], n);

        eprintln!("expected (Corollary 3.8) = {expected}");
        eprintln!("actual (recursion)       = {actual}");
        assert_eq!(expected, actual);
    }

    #[test]
    fn unital_h_structure_constant_matches_corollary_3_8_for_call1() {
        // nu=[3,1] covers mu=[3,0] directly too.
        let n = 2;
        let a = adjacent_binomial_coefficient(&[3, 1], &[3, 0]);
        let b_31_2 = shifted_macdonald_evaluate(&[2], &[3, 1], n) / normalizing_factor(&[2], n);
        let b_30_2 = shifted_macdonald_evaluate(&[2], &[3, 0], n) / normalizing_factor(&[2], n);
        let expected = a * (b_31_2 - b_30_2);

        let mut cache = HashMap::new();
        let actual = unital_h_structure_constant(&mut cache, &[2], &[3, 0], &[3, 1], n);

        eprintln!("expected (Corollary 3.8) = {expected}");
        eprintln!("actual (recursion)       = {actual}");
        assert_eq!(expected, actual);
    }

    #[test]
    fn unital_h_structure_constant_matches_corollary_3_8_for_call7_and_call9() {
        let n = 2;

        // call7: lambda=[2],mu=[2],nu=[2,1]; [2,1] covers [2].
        let a7 = adjacent_binomial_coefficient(&[2, 1], &[2]);
        let b_21_2 = shifted_macdonald_evaluate(&[2], &[2, 1], n) / normalizing_factor(&[2], n);
        let b_2_2 = shifted_macdonald_evaluate(&[2], &[2], n) / normalizing_factor(&[2], n);
        let expected7 = a7 * (b_21_2 - b_2_2.clone());
        let mut cache = HashMap::new();
        let actual7 = unital_h_structure_constant(&mut cache, &[2], &[2], &[2, 1], n);
        eprintln!("call7 expected = {expected7}");
        eprintln!("call7 actual   = {actual7}");
        assert_eq!(expected7, actual7);

        // call9: lambda=[2],mu=[2],nu=[3,0]; [3] covers [2].
        let a9 = adjacent_binomial_coefficient(&[3], &[2]);
        let b_3_2 = shifted_macdonald_evaluate(&[2], &[3], n) / normalizing_factor(&[2], n);
        let expected9 = a9 * (b_3_2 - b_2_2);
        let mut cache9 = HashMap::new();
        let actual9 = unital_h_structure_constant(&mut cache9, &[2], &[2], &[3, 0], n);
        eprintln!("call9 expected = {expected9}");
        eprintln!("call9 actual   = {actual9}");
        assert_eq!(expected9, actual9);
    }

    #[test]
    fn unital_h_structure_constant_call0_combination() {
        let n = 2;
        // All four direct children verified correct individually; check
        // the combination formula itself now: call0 = lambda=[2],mu=[2],nu=[3,1].
        let mut cache = HashMap::new();
        let call1 = unital_h_structure_constant(&mut cache, &[2], &[3, 0], &[3, 1], n);
        let call4 = unital_h_structure_constant(&mut cache, &[2], &[2, 1], &[3, 1], n);
        let call7 = unital_h_structure_constant(&mut cache, &[2], &[2], &[2, 1], n);
        let call9 = unital_h_structure_constant(&mut cache, &[2], &[2], &[3, 0], n);

        let a_30 = adjacent_binomial_coefficient(&[3, 0], &[2, 0]);
        let w_30 = norm_diff(&[3, 0], &[2, 0], n);
        let a_21 = adjacent_binomial_coefficient(&[2, 1], &[2, 0]);
        let w_21 = norm_diff(&[2, 1], &[2, 0], n);
        let a_nup21 = adjacent_binomial_coefficient(&[3, 1], &[2, 1]);
        let w_nup21 = norm_diff(&[3, 1], &[2, 1], n);
        let a_nup30 = adjacent_binomial_coefficient(&[3, 1], &[3, 0]);
        let w_nup30 = norm_diff(&[3, 1], &[3, 0], n);
        let divisor = norm_diff(&[3, 1], &[2, 0], n);

        let manual = (a_30 * w_30 * call1 + a_21 * w_21 * call4
            - a_nup21 * w_nup21 * call7
            - a_nup30 * w_nup30 * call9)
            / divisor;

        let mut cache2 = HashMap::new();
        let actual = unital_h_structure_constant(&mut cache2, &[2], &[2], &[3, 1], n);

        eprintln!("manual combination = {manual}");
        eprintln!("actual call0       = {actual}");
        assert_eq!(manual, actual);
    }

    #[test]
    fn call0_numeric_check_against_sage() {
        use num_rational::Ratio;
        let n = 2;
        let q_val = Ratio::from_integer(BigInt::from(2));
        let t_val = Ratio::from_integer(BigInt::from(3));

        let mut cache = HashMap::new();
        let unital = unital_h_structure_constant(&mut cache, &[2], &[2], &[3, 1], n);
        let unital_val = unital.evaluate(&q_val, &t_val);
        eprintln!("unital numeric value = {unital_val}");

        let h_lambda = normalizing_factor(&[2], n).evaluate(&q_val, &t_val);
        let h_mu = normalizing_factor(&[2], n).evaluate(&q_val, &t_val);
        let h_nu = normalizing_factor(&[3, 1], n).evaluate(&q_val, &t_val);
        eprintln!("H(lambda)={h_lambda} H(mu)={h_mu} H(nu)={h_nu}");

        let combined = unital_val * h_lambda * h_mu / h_nu;
        eprintln!("combined = {combined}, expected 126/115");
        assert_eq!(combined, Ratio::new(BigInt::from(126), BigInt::from(115)));
    }

    #[test]
    fn adjacent_binomial_coefficient_3_2_cross_check() {
        // Sage: {2},{1},{3} -> 1, i.e. c~^{[3]}_{[2],[1]} = 1.
        let n = 2;
        let h_3 = normalizing_factor(&[3], n);
        let h_2 = normalizing_factor(&[2], n);
        let h_1 = normalizing_factor(&[1], n);
        let unital_target = h_3 / (h_2 * h_1);

        let b_3_1 = shifted_macdonald_evaluate(&[1], &[3], n) / normalizing_factor(&[1], n);
        let b_2_1 = shifted_macdonald_evaluate(&[1], &[2], n) / normalizing_factor(&[1], n);
        let diff = b_3_1 - b_2_1;

        let implied_a = unital_target / diff;
        let actual_a = adjacent_binomial_coefficient(&[3], &[2]);
        let actual_a_padded = adjacent_binomial_coefficient(&[3, 0], &[2, 0]);

        eprintln!("implied a_[3],[2]      = {implied_a}");
        eprintln!("actual  a_[3],[2]      = {actual_a}");
        eprintln!("actual  a_[3,0],[2,0]  = {actual_a_padded}");
        assert_eq!(implied_a, actual_a);
        assert_eq!(implied_a, actual_a_padded);
    }

    #[test]
    fn weighted_chain_sum_cross_check_for_call0() {
        // Theorem 3.7 (Eq 3.19-3.20): c^lambda_{mu,nu} = sum over maximal
        // chains lambda=z0 :> z1 :> ... :> zk=mu of wt^LR_nu(chain) * prod
        // a_{z_i,z_{i+1}}, an independent (non-recursive) formula for the
        // SAME quantity as unital_h_structure_constant. For
        // lambda_p=[3,1],mu_p=[2],nu_p=[2] (k=2) there are two maximal
        // chains: [3,1]>[3]>[2] and [3,1]>[2,1]>[2].
        let n = 2;
        let lambda_p = vec![3usize, 1];
        let nu_p = vec![2usize];

        // norm(partition) = sum_r q^part[r] * t^(n-1-r), as a RationalFunction2.
        let norm = |part: &[usize]| -> RationalFunction2 {
            let mut total = Poly2::zero();
            for r in 0..n {
                let p = part.get(r).copied().unwrap_or(0);
                total = total + Poly2::monomial(p, n - 1 - r, BigInt::one());
            }
            RationalFunction2::new(total, Poly2::one())
        };
        let b = |shape: &[usize], point: &[usize]| -> RationalFunction2 {
            shifted_macdonald_evaluate(shape, point, n) / normalizing_factor(shape, n)
        };

        let chains: Vec<Vec<Vec<usize>>> = vec![
            vec![lambda_p.clone(), vec![3], vec![2]],
            vec![lambda_p.clone(), vec![2, 1], vec![2]],
        ];

        let mut total = RationalFunction2::zero();
        for chain in &chains {
            let k = chain.len() - 1;
            let norms: Vec<RationalFunction2> = chain.iter().map(|z| norm(z)).collect();

            let a_product = (0..k)
                .map(|i| adjacent_binomial_coefficient(&chain[i], &chain[i + 1]))
                .fold(RationalFunction2::one(), |acc, x| acc * x);

            let numerator: RationalFunction2 = (0..k)
                .map(|i| norms[i].clone() - norms[i + 1].clone())
                .fold(RationalFunction2::one(), |acc, x| acc * x);

            let mut wt = RationalFunction2::zero();
            for j in 0..=k {
                let mut denom = RationalFunction2::one();
                for i in 0..=k {
                    if i != j {
                        denom = denom * (norms[j].clone() - norms[i].clone());
                    }
                }
                let b_j = b(&nu_p, &chain[j]);
                wt = wt + (numerator.clone() / denom) * b_j;
            }

            total = total + a_product * wt;
        }

        let mut cache = HashMap::new();
        let actual = unital_h_structure_constant(&mut cache, &nu_p, &[2], &lambda_p, n);

        eprintln!("weighted chain sum = {total}");
        eprintln!("recursion (actual) = {actual}");
        assert_eq!(total, actual);
    }

    #[test]
    fn call0_manual_combination_numeric_trace() {
        use num_rational::Ratio;
        let n = 2;
        let q_val = Ratio::from_integer(BigInt::from(2));
        let t_val = Ratio::from_integer(BigInt::from(3));

        let mut cache = HashMap::new();
        let call1 = unital_h_structure_constant(&mut cache, &[2], &[3, 0], &[3, 1], n);
        let call4 = unital_h_structure_constant(&mut cache, &[2], &[2, 1], &[3, 1], n);
        let call7 = unital_h_structure_constant(&mut cache, &[2], &[2], &[2, 1], n);
        let call9 = unital_h_structure_constant(&mut cache, &[2], &[2], &[3, 0], n);

        let a_30 = adjacent_binomial_coefficient(&[3, 0], &[2, 0]);
        let w_30 = norm_diff(&[3, 0], &[2, 0], n);
        let a_21 = adjacent_binomial_coefficient(&[2, 1], &[2, 0]);
        let w_21 = norm_diff(&[2, 1], &[2, 0], n);
        let a_nup21 = adjacent_binomial_coefficient(&[3, 1], &[2, 1]);
        let w_nup21 = norm_diff(&[3, 1], &[2, 1], n);
        let a_nup30 = adjacent_binomial_coefficient(&[3, 1], &[3, 0]);
        let w_nup30 = norm_diff(&[3, 1], &[3, 0], n);
        let divisor = norm_diff(&[3, 1], &[2, 0], n);

        let e = |x: &RationalFunction2| x.evaluate(&q_val, &t_val);

        let numeric_manual = (e(&a_30) * e(&w_30) * e(&call1) + e(&a_21) * e(&w_21) * e(&call4)
            - e(&a_nup21) * e(&w_nup21) * e(&call7)
            - e(&a_nup30) * e(&w_nup30) * e(&call9))
            / e(&divisor);

        eprintln!("numeric manual combination = {numeric_manual}");

        // Compare against the symbolic manual combination's numeric value.
        let symbolic_manual = (a_30 * w_30 * call1 + a_21 * w_21 * call4
            - a_nup21 * w_nup21 * call7
            - a_nup30 * w_nup30 * call9)
            / divisor;
        let numeric_of_symbolic = e(&symbolic_manual);
        eprintln!("numeric-of-symbolic manual  = {numeric_of_symbolic}");

        assert_eq!(numeric_manual, numeric_of_symbolic);
    }

    #[test]
    fn call0_narrow_down_bad_operation() {
        use num_rational::Ratio;
        let n = 2;
        let q_val = Ratio::from_integer(BigInt::from(2));
        let t_val = Ratio::from_integer(BigInt::from(3));
        let e = |x: &RationalFunction2| x.evaluate(&q_val, &t_val);

        let mut cache = HashMap::new();
        let call1 = unital_h_structure_constant(&mut cache, &[2], &[3, 0], &[3, 1], n);
        let call4 = unital_h_structure_constant(&mut cache, &[2], &[2, 1], &[3, 1], n);
        let call7 = unital_h_structure_constant(&mut cache, &[2], &[2], &[2, 1], n);
        let call9 = unital_h_structure_constant(&mut cache, &[2], &[2], &[3, 0], n);

        let a_30 = adjacent_binomial_coefficient(&[3, 0], &[2, 0]);
        let w_30 = norm_diff(&[3, 0], &[2, 0], n);
        let a_21 = adjacent_binomial_coefficient(&[2, 1], &[2, 0]);
        let w_21 = norm_diff(&[2, 1], &[2, 0], n);
        let a_nup21 = adjacent_binomial_coefficient(&[3, 1], &[2, 1]);
        let w_nup21 = norm_diff(&[3, 1], &[2, 1], n);
        let a_nup30 = adjacent_binomial_coefficient(&[3, 1], &[3, 0]);
        let w_nup30 = norm_diff(&[3, 1], &[3, 0], n);
        let divisor = norm_diff(&[3, 1], &[2, 0], n);

        macro_rules! check {
            ($name:expr, $symbolic:expr, $numeric:expr) => {
                let sv = e(&$symbolic);
                if sv != $numeric {
                    eprintln!(
                        "MISMATCH at {}: symbolic={} numeric={}",
                        $name, sv, $numeric
                    );
                } else {
                    eprintln!("ok at {}: {}", $name, sv);
                }
            };
        }

        let term1 = a_30.clone() * w_30.clone() * call1.clone();
        check!("term1", term1, e(&a_30) * e(&w_30) * e(&call1));

        let term2 = a_21.clone() * w_21.clone() * call4.clone();
        check!("term2", term2, e(&a_21) * e(&w_21) * e(&call4));

        let sum12 = term1.clone() + term2.clone();
        check!("sum12", sum12, e(&term1) + e(&term2));

        let term3 = a_nup21.clone() * w_nup21.clone() * call7.clone();
        check!("term3", term3, e(&a_nup21) * e(&w_nup21) * e(&call7));

        let term4 = a_nup30.clone() * w_nup30.clone() * call9.clone();
        check!("term4", term4, e(&a_nup30) * e(&w_nup30) * e(&call9));

        let sum34 = term3.clone() + term4.clone();
        check!("sum34", sum34, e(&term3) + e(&term4));

        let sum1234 = sum12.clone() - sum34.clone();
        check!("sum1234", sum1234, e(&sum12) - e(&sum34));

        let final_val = sum1234.clone() / divisor.clone();
        check!("final", final_val, e(&sum1234) / e(&divisor));
    }

    #[test]
    fn sum12_minus_sum34_repro() {
        use num_rational::Ratio;
        let n = 2;
        let q_val = Ratio::from_integer(BigInt::from(2));
        let t_val = Ratio::from_integer(BigInt::from(3));
        let e = |x: &RationalFunction2| x.evaluate(&q_val, &t_val);

        let mut cache = HashMap::new();
        let call1 = unital_h_structure_constant(&mut cache, &[2], &[3, 0], &[3, 1], n);
        let call4 = unital_h_structure_constant(&mut cache, &[2], &[2, 1], &[3, 1], n);
        let call7 = unital_h_structure_constant(&mut cache, &[2], &[2], &[2, 1], n);
        let call9 = unital_h_structure_constant(&mut cache, &[2], &[2], &[3, 0], n);

        let a_30 = adjacent_binomial_coefficient(&[3, 0], &[2, 0]);
        let w_30 = norm_diff(&[3, 0], &[2, 0], n);
        let a_21 = adjacent_binomial_coefficient(&[2, 1], &[2, 0]);
        let w_21 = norm_diff(&[2, 1], &[2, 0], n);
        let a_nup21 = adjacent_binomial_coefficient(&[3, 1], &[2, 1]);
        let w_nup21 = norm_diff(&[3, 1], &[2, 1], n);
        let a_nup30 = adjacent_binomial_coefficient(&[3, 1], &[3, 0]);
        let w_nup30 = norm_diff(&[3, 1], &[3, 0], n);

        let sum12 = a_30 * w_30 * call1 + a_21 * w_21 * call4;
        let sum34 = a_nup21 * w_nup21 * call7 + a_nup30 * w_nup30 * call9;

        eprintln!("sum12 = {sum12}");
        eprintln!("sum34 = {sum34}");
        eprintln!(
            "sum12 denom degq={} sum34 denom degq={}",
            sum12.denominator.coeffs.len(),
            sum34.denominator.coeffs.len()
        );

        let g = poly2_gcd(&sum12.denominator, &sum34.denominator);
        eprintln!("gcd(denom12,denom34) = {g}");

        // Verify g actually divides both denominators exactly by checking
        // the remainder directly, rather than trusting poly2_exact_div's
        // internal assert (which is what we're trying to falsify).
        let (_, rem1) = rf_poly_div_rem(sum12.denominator.to_rf_coeffs(), g.to_rf_coeffs());
        let (_, rem2) = rf_poly_div_rem(sum34.denominator.to_rf_coeffs(), g.to_rf_coeffs());
        eprintln!(
            "rem1 all zero: {}",
            rem1.iter().all(RationalFunction::is_zero)
        );
        eprintln!(
            "rem2 all zero: {}",
            rem2.iter().all(RationalFunction::is_zero)
        );

        let diff = sum12.clone() - sum34.clone();
        eprintln!("diff (sum12-sum34) evaluated = {}", e(&diff));
        eprintln!("expected (e(sum12)-e(sum34)) = {}", e(&sum12) - e(&sum34));
    }

    #[test]
    fn poly2_exact_div_round_trip_repro() {
        let n = 2;

        let mut cache = HashMap::new();
        let call1 = unital_h_structure_constant(&mut cache, &[2], &[3, 0], &[3, 1], n);
        let call4 = unital_h_structure_constant(&mut cache, &[2], &[2, 1], &[3, 1], n);
        let call7 = unital_h_structure_constant(&mut cache, &[2], &[2], &[2, 1], n);
        let call9 = unital_h_structure_constant(&mut cache, &[2], &[2], &[3, 0], n);

        let a_30 = adjacent_binomial_coefficient(&[3, 0], &[2, 0]);
        let w_30 = norm_diff(&[3, 0], &[2, 0], n);
        let a_21 = adjacent_binomial_coefficient(&[2, 1], &[2, 0]);
        let w_21 = norm_diff(&[2, 1], &[2, 0], n);
        let a_nup21 = adjacent_binomial_coefficient(&[3, 1], &[2, 1]);
        let w_nup21 = norm_diff(&[3, 1], &[2, 1], n);
        let a_nup30 = adjacent_binomial_coefficient(&[3, 1], &[3, 0]);
        let w_nup30 = norm_diff(&[3, 1], &[3, 0], n);

        let sum12 = a_30 * w_30 * call1 + a_21 * w_21 * call4;
        let sum34 = a_nup21 * w_nup21 * call7 + a_nup30 * w_nup30 * call9;

        // Reconstruct exactly what Add computes before calling Self::new,
        // for sum12 + (-sum34).
        let neg_sum34_num = -sum34.numerator.clone();
        let rhs_denom = sum34.denominator.clone();
        let g = poly2_gcd(&sum12.denominator, &rhs_denom);
        let self_denom_factor = poly2_exact_div(&sum12.denominator, &g);
        let rhs_denom_factor = poly2_exact_div(&rhs_denom, &g);
        let common_denominator = sum12.denominator.clone() * rhs_denom_factor.clone();
        let raw_numerator =
            sum12.numerator.clone() * rhs_denom_factor + neg_sum34_num * self_denom_factor;

        eprintln!("raw_numerator = {raw_numerator}");
        eprintln!("common_denominator = {common_denominator}");

        let content_gcd = int_poly_gcd(&raw_numerator.content_t(), &common_denominator.content_t());
        eprintln!("content_gcd = {content_gcd}");
        let num_after_content = raw_numerator.div_by_intpoly_exact(&content_gcd);
        let den_after_content = common_denominator.div_by_intpoly_exact(&content_gcd);
        eprintln!("num_after_content = {num_after_content}");
        eprintln!("den_after_content = {den_after_content}");

        let common = poly2_gcd(&num_after_content, &den_after_content);
        eprintln!("common (poly2_gcd of num,den) = {common}");

        let quotient_num = poly2_exact_div(&num_after_content, &common);
        let quotient_den = poly2_exact_div(&den_after_content, &common);
        eprintln!("quotient_num = {quotient_num}");
        eprintln!("quotient_den = {quotient_den}");

        // Round-trip check: quotient * common should reproduce the input, exactly (as Poly2).
        let recovered_num = quotient_num.clone() * common.clone();
        let recovered_den = quotient_den.clone() * common.clone();
        eprintln!(
            "recovered_num == num_after_content: {}",
            recovered_num == num_after_content
        );
        eprintln!(
            "recovered_den == den_after_content: {}",
            recovered_den == den_after_content
        );
        if recovered_num != num_after_content {
            eprintln!("recovered_num = {recovered_num}");
        }
        if recovered_den != den_after_content {
            eprintln!("recovered_den = {recovered_den}");
        }
    }

    #[test]
    fn primitive_part_positive_of_2qt_minus_2() {
        let p = q() * t() * Poly2::from_i64(2) - Poly2::from_i64(2);
        eprintln!("p = {p}");
        eprintln!("p.content_t() = {}", p.content_t());
        let pp = p.primitive_part_positive();
        eprintln!("p.primitive_part_positive() = {pp}");
        assert_eq!(format!("{pp}"), "q*t - 1");
    }

    #[test]
    fn intpoly_gcd_z_preserves_content_when_one_input_is_zero() {
        let p = IntPoly::new(vec![BigInt::from(32), BigInt::from(32)]);
        assert_eq!(intpoly_gcd_z(&IntPoly::zero(), &p), p);

        let negative = IntPoly::new(vec![BigInt::from(-32), BigInt::from(-32)]);
        assert_eq!(intpoly_gcd_z(&negative, &IntPoly::zero()), p);
    }

    #[test]
    fn algorithm_brute_force_wired_correctly() {
        let mut calc = MacdonaldCalculator::new();
        let via_recursive =
            calc.p_structure_constant_with(&[2], &[2], &[3, 1], Algorithm::Recursive);
        let via_brute_force =
            calc.p_structure_constant_with(&[2], &[2], &[3, 1], Algorithm::BruteForce);
        assert_eq!(via_recursive, via_brute_force);
    }

    #[test]
    fn weighted_chain_sum_matches_recursion_on_several_cases() {
        let cases: Vec<(Vec<usize>, Vec<usize>, Vec<usize>, usize)> = vec![
            (vec![2], vec![2], vec![3, 1], 2),
            (vec![3], vec![3], vec![4, 2], 3),
            (vec![2, 1], vec![2, 1], vec![3, 1, 1], 3),
            (vec![1, 1], vec![1, 1], vec![2, 2], 2),
        ];
        for (lambda, mu, nu, n) in cases {
            let mut cache = HashMap::new();
            let recursive = unital_h_structure_constant(&mut cache, &lambda, &mu, &nu, n);
            let chain_sum = weighted_chain_sum_structure_constant(&lambda, &mu, &nu, n);
            assert_eq!(
                recursive, chain_sum,
                "mismatch for lambda={lambda:?} mu={mu:?} nu={nu:?}"
            );
        }
    }

    #[test]
    fn weighted_chain_sum_timing_on_hard_cases() {
        let cases: Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> =
            vec![(vec![2, 1], vec![2, 1], vec![3, 2, 1])];
        for (lambda, mu, nu) in &cases {
            let n = lambda.len().max(mu.len()).max(nu.len()).max(1);
            let chain_count = enumerate_maximal_chains(nu, mu).len();
            let start = std::time::Instant::now();
            let v = weighted_chain_sum_structure_constant(lambda, mu, nu, n);
            eprintln!(
                "{lambda:?} x {mu:?} -> {nu:?}: {chain_count} chains, {:?}",
                start.elapsed()
            );
            eprintln!("  = {v}");
        }
    }

    #[test]
    fn shifted_macdonald_evaluate_matches_sage_at_shape_2() {
        use num_rational::Ratio;
        let q_val = Ratio::from_integer(BigInt::from(2));
        let t_val = Ratio::from_integer(BigInt::from(3));

        let h2_monic = shifted_macdonald_evaluate(&[2], &[3, 1], 2);
        let h2_val = h2_monic.evaluate(&q_val, &t_val);
        eprintln!("h_2^monic((3,1)bar) = {h2_val}");
        assert_eq!(h2_val, Ratio::new(BigInt::from(1998), BigInt::from(5)));

        let big_h2 = normalizing_factor(&[2], 2);
        let big_h2_val = big_h2.evaluate(&q_val, &t_val);
        eprintln!("H(2) = {big_h2_val}");
        assert_eq!(big_h2_val, Ratio::from_integer(BigInt::from(54)));
    }

    #[test]
    fn poly2_arithmetic_and_display() {
        let p = q() * q() + t() * Poly2::from_i64(2) - Poly2::one();
        assert_eq!(format!("{p}"), "q^2 + 2*t - 1");
    }

    #[test]
    fn rational_function2_reduces_bivariate_common_factors() {
        // (q^2*t - t) / (q - 1) = t*(q-1)*(q+1) / (q-1) = t*(q+1)
        let numerator = q() * q() * t() - t();
        let denominator = q() - Poly2::one();
        let reduced = RationalFunction2::new(numerator, denominator);
        assert_eq!(format!("{reduced}"), "q*t + t");
    }

    #[test]
    fn rational_function2_reduces_pure_t_common_factor() {
        // (q*t^2 - q*t) / (t - 1) = q*t
        let numerator = q() * t() * t() - q() * t();
        let denominator = t() - Poly2::one();
        let reduced = RationalFunction2::new(numerator, denominator);
        assert_eq!(format!("{reduced}"), "q*t");
    }

    #[test]
    fn rational_function2_equality_is_representation_independent() {
        // (q^2 - t^2) / (q - t) and (q + t) should reduce to the same value.
        let numerator = q() * q() - t() * t();
        let denominator = q() - t();
        let reduced = RationalFunction2::new(numerator, denominator);
        let expected = RationalFunction2::new(q() + t(), Poly2::one());
        assert_eq!(reduced, expected);
    }

    #[test]
    fn rational_function2_arithmetic() {
        let a = RationalFunction2::new(Poly2::one(), q() - Poly2::one());
        let b = RationalFunction2::new(Poly2::one(), q() + Poly2::one());
        // 1/(q-1) + 1/(q+1) = 2q / (q^2 - 1)
        let sum = a + b;
        let expected = RationalFunction2::new(Poly2::from_i64(2) * q(), q() * q() - Poly2::one());
        assert_eq!(sum, expected);
    }

    #[test]
    fn rational_function2_cross_cancellation_matches_naive_reduction() {
        let p1 = q() + t() + Poly2::one();
        let p2 = q() * t() + q() + Poly2::from_i64(2);
        let p3 = q() * q() + t() + Poly2::from_i64(3);
        let p4 = q() + t() * t() + Poly2::from_i64(5);

        let a = RationalFunction2::new(p1.clone() * p2.clone(), p3.clone());
        let b = RationalFunction2::new(p3.clone(), p2.clone() * p4.clone());

        let naive_product = RationalFunction2::new(
            a.numerator.clone() * b.numerator.clone(),
            a.denominator.clone() * b.denominator.clone(),
        );
        assert_eq!(a.clone() * b.clone(), naive_product);
        assert_eq!(
            a.clone() * b.clone(),
            RationalFunction2::new(p1.clone(), p4.clone())
        );

        let naive_quotient = RationalFunction2::new(
            a.numerator.clone() * b.denominator.clone(),
            a.denominator.clone() * b.numerator.clone(),
        );
        assert_eq!(a / b, naive_quotient);
    }
}
