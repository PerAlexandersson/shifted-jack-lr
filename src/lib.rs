use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

pub mod brute_force;
pub mod macdonald;

type Rat = Ratio<BigInt>;

/// A partition, stored in weakly decreasing order.
pub type Partition = Vec<usize>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntPoly {
    coeffs: Vec<BigInt>,
}

impl IntPoly {
    pub fn new(mut coeffs: Vec<BigInt>) -> Self {
        while coeffs.last().is_some_and(BigInt::is_zero) {
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
        if n == 0 {
            Self::zero()
        } else {
            Self::new(vec![BigInt::from(n)])
        }
    }

    pub fn variable() -> Self {
        Self::new(vec![BigInt::zero(), BigInt::one()])
    }

    pub fn monomial(degree: usize, coefficient: BigInt) -> Self {
        if coefficient.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![BigInt::zero(); degree + 1];
        coeffs[degree] = coefficient;
        Self::new(coeffs)
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.coeffs.len() == 1 && self.coeffs[0].is_one()
    }

    pub fn degree(&self) -> Option<usize> {
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    pub(crate) fn coeff(&self, degree: usize) -> BigInt {
        self.coeffs
            .get(degree)
            .cloned()
            .unwrap_or_else(BigInt::zero)
    }

    fn leading_coefficient(&self) -> &BigInt {
        self.coeffs.last().expect("nonzero polynomial")
    }

    pub(crate) fn content_abs(&self) -> BigInt {
        self.coeffs
            .iter()
            .fold(BigInt::zero(), |acc, c| acc.gcd(&c.abs()))
    }

    fn div_by_bigint_exact(&self, divisor: &BigInt) -> Self {
        assert!(!divisor.is_zero(), "division by zero");
        Self::new(
            self.coeffs
                .iter()
                .map(|c| {
                    let (q, r) = c.div_rem(divisor);
                    assert!(r.is_zero(), "non-exact integer polynomial division");
                    q
                })
                .collect(),
        )
    }

    pub(crate) fn primitive_part_positive(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let content = self.content_abs();
        let mut result = self.div_by_bigint_exact(&content);
        if result.leading_coefficient().is_negative() {
            result = -result;
        }
        result
    }

    pub(crate) fn exact_div(&self, divisor: &Self) -> Self {
        assert!(!divisor.is_zero(), "division by zero polynomial");
        let (quotient, remainder) = rat_poly_div_rem(
            int_poly_to_rat_coeffs(self),
            int_poly_to_rat_coeffs(divisor),
        );
        assert!(remainder.is_empty(), "non-exact polynomial division");
        rat_poly_to_integer_exact(&quotient)
    }
}

impl Add for IntPoly {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        Self::new((0..n).map(|i| self.coeff(i) + rhs.coeff(i)).collect())
    }
}

impl Sub for IntPoly {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        Self::new((0..n).map(|i| self.coeff(i) - rhs.coeff(i)).collect())
    }
}

impl Neg for IntPoly {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(self.coeffs.into_iter().map(|c| -c).collect())
    }
}

impl Mul for IntPoly {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![BigInt::zero(); self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, a) in self.coeffs.into_iter().enumerate() {
            for (j, b) in rhs.coeffs.iter().enumerate() {
                coeffs[i + j] += &a * b;
            }
        }
        Self::new(coeffs)
    }
}

impl fmt::Display for IntPoly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for degree in (0..self.coeffs.len()).rev() {
            let coeff = &self.coeffs[degree];
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

            let show_coeff = degree == 0 || !abs_coeff.is_one();
            if show_coeff {
                write!(f, "{abs_coeff}")?;
                if degree > 0 {
                    write!(f, "*")?;
                }
            }
            match degree {
                0 => {}
                1 => write!(f, "a")?,
                _ => write!(f, "a^{degree}")?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RationalFunction {
    numerator: IntPoly,
    denominator: IntPoly,
}

impl RationalFunction {
    pub fn new(numerator: IntPoly, denominator: IntPoly) -> Self {
        assert!(!denominator.is_zero(), "zero denominator");
        if numerator.is_zero() {
            return Self {
                numerator: IntPoly::zero(),
                denominator: IntPoly::one(),
            };
        }

        let mut numerator = numerator;
        let mut denominator = denominator;
        let common_content = numerator.content_abs().gcd(&denominator.content_abs());
        if common_content > BigInt::one() {
            numerator = numerator.div_by_bigint_exact(&common_content);
            denominator = denominator.div_by_bigint_exact(&common_content);
        }

        let common_poly = int_poly_gcd(&numerator, &denominator);
        if !common_poly.is_one() {
            numerator = numerator.exact_div(&common_poly);
            denominator = denominator.exact_div(&common_poly);
        }

        if denominator.leading_coefficient().is_negative() {
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
            numerator: IntPoly::from_i64(n),
            denominator: IntPoly::one(),
        }
    }

    pub fn variable() -> Self {
        Self {
            numerator: IntPoly::variable(),
            denominator: IntPoly::one(),
        }
    }

    pub fn a_power(degree: usize) -> Self {
        Self {
            numerator: IntPoly::monomial(degree, BigInt::one()),
            denominator: IntPoly::one(),
        }
    }

    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    pub(crate) fn numerator(&self) -> &IntPoly {
        &self.numerator
    }

    pub(crate) fn denominator(&self) -> &IntPoly {
        &self.denominator
    }

    pub fn inverse(self) -> Self {
        assert!(!self.numerator.is_zero(), "division by zero");
        Self::new(self.denominator, self.numerator)
    }
}

impl Add for RationalFunction {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.numerator.clone() * rhs.denominator.clone()
                + rhs.numerator.clone() * self.denominator.clone(),
            self.denominator * rhs.denominator,
        )
    }
}

impl Sub for RationalFunction {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for RationalFunction {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.numerator, self.denominator)
    }
}

impl Mul for RationalFunction {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.numerator * rhs.numerator,
            self.denominator * rhs.denominator,
        )
    }
}

impl Div for RationalFunction {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        assert!(!rhs.numerator.is_zero(), "division by zero");
        Self::new(
            self.numerator * rhs.denominator,
            self.denominator * rhs.numerator,
        )
    }
}

impl fmt::Display for RationalFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator.is_one() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "({})/({})", self.numerator, self.denominator)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Normalization {
    P,
    J,
}

#[derive(Default)]
pub struct ShiftedJackCalculator {
    c_cache: HashMap<(Partition, Partition, Partition), RationalFunction>,
    j_cache: HashMap<(Partition, Partition, Partition), RationalFunction>,
    eval_cache: HashMap<(Partition, Partition), RationalFunction>,
    hook_cache: HashMap<Partition, RationalFunction>,
    hook_prime_cache: HashMap<Partition, RationalFunction>,
    psi_cache: HashMap<(Partition, Partition), RationalFunction>,
    psi_prime_cache: HashMap<(Partition, Partition), RationalFunction>,
}

impl ShiftedJackCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn coefficient(
        &mut self,
        lambda: &[usize],
        mu: &[usize],
        nu: &[usize],
        normalization: Normalization,
    ) -> RationalFunction {
        match normalization {
            Normalization::P => self.jack_p_structure_constant(lambda, mu, nu),
            Normalization::J => self.jack_j_structure_constant(lambda, mu, nu),
        }
    }

    pub fn jack_p_structure_constant(
        &mut self,
        lambda: &[usize],
        mu: &[usize],
        nu: &[usize],
    ) -> RationalFunction {
        if !skew_shape_contains(nu, mu) || !skew_shape_contains(nu, lambda) {
            return RationalFunction::zero();
        }

        if partition_size(lambda) > partition_size(mu) {
            return self.jack_p_structure_constant(mu, lambda, nu);
        }

        let [lambda, mu, nu] = normalize_three(lambda, mu, nu);
        let key = (lambda.clone(), mu.clone(), nu.clone());
        if let Some(value) = self.c_cache.get(&key) {
            return value.clone();
        }

        let value = if mu == nu {
            self.shifted_jack_evaluate(&lambda, &nu)
        } else if lambda == nu {
            self.shifted_jack_evaluate(&mu, &nu)
        } else {
            let mut total = RationalFunction::zero();
            for mup in add_box_to_partition(&mu) {
                let psi = self.macdonald_psi_prime(&mup, &mu);
                let sub = self.jack_p_structure_constant(&lambda, &mup, &nu);
                total = total + psi * sub;
            }
            for num in remove_box_from_partition(&nu) {
                let psi = self.macdonald_psi_prime(&nu, &num);
                let sub = self.jack_p_structure_constant(&lambda, &mu, &num);
                total = total - psi * sub;
            }
            total / RationalFunction::from_i64((partition_size(&nu) - partition_size(&mu)) as i64)
        };

        self.c_cache.insert(key, value.clone());
        value
    }

    pub fn jack_j_structure_constant(
        &mut self,
        lambda: &[usize],
        mu: &[usize],
        nu: &[usize],
    ) -> RationalFunction {
        if !skew_shape_contains(nu, mu) || !skew_shape_contains(nu, lambda) {
            return RationalFunction::zero();
        }

        if partition_size(lambda) > partition_size(mu) {
            return self.jack_j_structure_constant(mu, lambda, nu);
        }

        let [lambda, mu, nu] = normalize_three(lambda, mu, nu);
        let key = (lambda.clone(), mu.clone(), nu.clone());
        if let Some(value) = self.j_cache.get(&key) {
            return value.clone();
        }

        let value = if mu == nu {
            self.jack_j_boundary_value(&lambda, &lambda, &mu, &nu)
        } else if lambda == nu {
            self.jack_j_boundary_value(&mu, &lambda, &mu, &nu)
        } else {
            let hp_mu = self.hook_prime_factor(&mu);
            let h_nu = self.hook_factor(&nu);
            let mut total = RationalFunction::zero();

            for mup in add_box_to_partition(&mu) {
                let psi = self.macdonald_psi_prime(&mup, &mu);
                let hp_mup = self.hook_prime_factor(&mup);
                let sub = self.jack_j_structure_constant(&lambda, &mup, &nu);
                total = total + psi * hp_mu.clone() / hp_mup * sub;
            }

            for num in remove_box_from_partition(&nu) {
                let psi = self.macdonald_psi_prime(&nu, &num);
                let h_num = self.hook_factor(&num);
                let sub = self.jack_j_structure_constant(&lambda, &mu, &num);
                total = total - psi * RationalFunction::variable() * h_nu.clone() / h_num * sub;
            }

            total / RationalFunction::from_i64((partition_size(&nu) - partition_size(&mu)) as i64)
        };

        self.j_cache.insert(key, value.clone());
        value
    }

    pub fn product_coefficients(
        &mut self,
        lambda: &[usize],
        mu: &[usize],
        max_size: Option<usize>,
        normalization: Normalization,
    ) -> Vec<(Partition, RationalFunction)> {
        let lower = partition_size(lambda).max(partition_size(mu));
        let upper = max_size.unwrap_or_else(|| partition_size(lambda) + partition_size(mu));
        let mut terms = Vec::new();
        for size in lower..=upper {
            for nu in integer_partitions(size) {
                if !skew_shape_contains(&nu, lambda) || !skew_shape_contains(&nu, mu) {
                    continue;
                }
                let coefficient = self.coefficient(lambda, mu, &nu, normalization);
                if !coefficient.is_zero() {
                    terms.push((nu, coefficient));
                }
            }
        }
        terms
    }

    pub fn shifted_jack_evaluate(&mut self, mu: &[usize], nu: &[usize]) -> RationalFunction {
        let shape = trim_partition(mu);
        let nu = nu.to_vec();
        let key = (shape.clone(), nu.clone());
        if let Some(value) = self.eval_cache.get(&key) {
            return value.clone();
        }

        let k = nu.len();
        let mut total = RationalFunction::zero();
        for tableau in semistandard_tableaux(&shape, k) {
            let mut term = RationalFunction::one();
            let chain = gt_chain_from_tableau(&tableau, &shape, k);
            for adjacent in chain.windows(2) {
                term = term * self.macdonald_psi(&adjacent[0], &adjacent[1]);
            }

            for (r, row) in tableau.iter().enumerate() {
                for (c, entry) in row.iter().enumerate() {
                    let x_index = k - entry;
                    let x_value = nu.get(x_index).copied().unwrap_or(0) as i64;
                    let mut factor = RationalFunction::from_i64(x_value - c as i64);
                    if r > 0 {
                        factor = factor
                            + RationalFunction::from_i64(r as i64) / RationalFunction::variable();
                    }
                    term = term * factor;
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

        self.eval_cache.insert(key, total.clone());
        total
    }

    fn macdonald_psi(&mut self, lambda: &[usize], mu: &[usize]) -> RationalFunction {
        let [lambda, mu] = normalize_two(lambda, mu);
        let key = (lambda, mu);
        if let Some(value) = self.psi_cache.get(&key) {
            return value.clone();
        }
        let value = macdonald_psi_with_parameter(&key.0, &key.1, RationalFunction::variable());
        self.psi_cache.insert(key, value.clone());
        value
    }

    fn macdonald_psi_prime(&mut self, lambda: &[usize], mu: &[usize]) -> RationalFunction {
        let [lambda, mu] = normalize_two(lambda, mu);
        let key = (lambda, mu);
        if let Some(value) = self.psi_prime_cache.get(&key) {
            return value.clone();
        }
        let lambda_c = conjugate_partition(&key.0);
        let mu_c = conjugate_partition(&key.1);
        let value =
            macdonald_psi_with_parameter(&lambda_c, &mu_c, RationalFunction::variable().inverse());
        self.psi_prime_cache.insert(key, value.clone());
        value
    }

    fn jack_j_boundary_value(
        &mut self,
        evaluation_shape: &[usize],
        lambda: &[usize],
        mu: &[usize],
        nu: &[usize],
    ) -> RationalFunction {
        RationalFunction::a_power(partition_size(nu))
            * self.hook_prime_factor(lambda)
            * self.hook_prime_factor(mu)
            * self.hook_factor(nu)
            * self.shifted_jack_evaluate(evaluation_shape, nu)
    }

    fn hook_factor(&mut self, mu: &[usize]) -> RationalFunction {
        let mu = trim_partition(mu);
        if let Some(value) = self.hook_cache.get(&mu) {
            return value.clone();
        }
        let value = hook_factor(&mu);
        self.hook_cache.insert(mu, value.clone());
        value
    }

    fn hook_prime_factor(&mut self, mu: &[usize]) -> RationalFunction {
        let mu = trim_partition(mu);
        if let Some(value) = self.hook_prime_cache.get(&mu) {
            return value.clone();
        }
        let value = hook_prime_factor(&mu);
        self.hook_prime_cache.insert(mu, value.clone());
        value
    }
}

pub fn parse_partition(input: &str) -> Result<Partition, String> {
    let trimmed = input
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim_start_matches('[')
        .trim_end_matches(']');
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut partition = Vec::new();
    for part in trimmed.split(|c: char| c == ',' || c.is_ascii_whitespace()) {
        if part.is_empty() {
            continue;
        }
        let value = part
            .parse::<usize>()
            .map_err(|_| format!("invalid partition part `{part}`"))?;
        if value > 0 {
            partition.push(value);
        }
    }
    if !is_partition(&partition) {
        return Err(format!(
            "partition parts must be weakly decreasing: {}",
            format_partition(&partition)
        ));
    }
    Ok(partition)
}

pub fn format_partition(partition: &[usize]) -> String {
    let trimmed = trim_partition(partition);
    if trimmed.is_empty() {
        "{}".to_string()
    } else {
        format!(
            "{{{}}}",
            trimmed
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

pub fn partition_size(partition: &[usize]) -> usize {
    partition.iter().sum()
}

pub fn integer_partitions(n: usize) -> Vec<Partition> {
    fn rec(remaining: usize, max_part: usize, current: &mut Partition, out: &mut Vec<Partition>) {
        if remaining == 0 {
            out.push(current.clone());
            return;
        }
        let upper = remaining.min(max_part);
        for part in (1..=upper).rev() {
            current.push(part);
            rec(remaining - part, part, current, out);
            current.pop();
        }
    }

    let mut out = Vec::new();
    rec(n, n, &mut Vec::new(), &mut out);
    out
}

fn is_partition(partition: &[usize]) -> bool {
    partition.windows(2).all(|w| w[0] >= w[1])
}

pub(crate) fn trim_partition(partition: &[usize]) -> Partition {
    let mut result = partition.to_vec();
    while result.last().is_some_and(|part| *part == 0) {
        result.pop();
    }
    result
}

pub(crate) fn normalize_two(a: &[usize], b: &[usize]) -> [Partition; 2] {
    let len = a.len().max(b.len());
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    aa.resize(len, 0);
    bb.resize(len, 0);
    [aa, bb]
}

pub(crate) fn normalize_three(a: &[usize], b: &[usize], c: &[usize]) -> [Partition; 3] {
    let len = a.len().max(b.len()).max(c.len());
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    let mut cc = c.to_vec();
    aa.resize(len, 0);
    bb.resize(len, 0);
    cc.resize(len, 0);
    [aa, bb, cc]
}

pub(crate) fn skew_shape_contains(outer: &[usize], inner: &[usize]) -> bool {
    let [outer, inner] = normalize_two(outer, inner);
    outer.iter().zip(inner.iter()).all(|(o, i)| o >= i)
}

pub(crate) fn conjugate_partition(partition: &[usize]) -> Partition {
    let partition = trim_partition(partition);
    let Some(&width) = partition.first() else {
        return Vec::new();
    };
    (1..=width)
        .map(|column| partition.iter().filter(|&&row| row >= column).count())
        .collect()
}

pub(crate) fn add_box_to_partition(mu_in: &[usize]) -> Vec<Partition> {
    let mut mu = if mu_in.is_empty() {
        vec![0]
    } else {
        mu_in.to_vec()
    };
    let mut top: Partition = mu.iter().map(|part| part + 1).collect();
    if mu.last().copied().unwrap_or(0) > 0 {
        mu.push(0);
    }
    let len = mu.len().max(top.len());
    mu.resize(len, 0);
    top.resize(len, 0);

    let mut out = Vec::new();
    for j in 0..len {
        if (j == 0 || mu[j] < mu[j - 1]) && mu[j] < top[j] {
            let mut next = mu.clone();
            next[j] += 1;
            out.push(next);
        }
    }
    out
}

pub(crate) fn remove_box_from_partition(mu_in: &[usize]) -> Vec<Partition> {
    let mut mu = mu_in.to_vec();
    let mut bot: Partition = mu.iter().map(|part| part.saturating_sub(1)).collect();
    let len = mu.len().max(bot.len());
    mu.resize(len, 0);
    bot.resize(len, 0);

    let mut out = Vec::new();
    for j in 0..len {
        if (j + 1 == len || mu[j + 1] < mu[j]) && mu[j] > bot[j] {
            let mut next = mu.clone();
            next[j] -= 1;
            out.push(next);
        }
    }
    out
}

fn hook_factor(mu: &[usize]) -> RationalFunction {
    let mu = trim_partition(mu);
    let muc = conjugate_partition(&mu);
    let a = RationalFunction::variable();
    let mut product = RationalFunction::one();
    for (r, row_len) in mu.iter().enumerate() {
        for c in 1..=*row_len {
            let arm = (*row_len - c) as i64;
            let leg = (muc[c - 1] - r - 1) as i64;
            let factor =
                RationalFunction::from_i64(arm + 1) + RationalFunction::from_i64(leg) / a.clone();
            product = product * factor;
        }
    }
    product
}

fn hook_prime_factor(mu: &[usize]) -> RationalFunction {
    let mu = trim_partition(mu);
    let muc = conjugate_partition(&mu);
    let a = RationalFunction::variable();
    let mut product = RationalFunction::one();
    for (r, row_len) in mu.iter().enumerate() {
        for c in 1..=*row_len {
            let arm = (*row_len - c) as i64;
            let leg = (muc[c - 1] - r - 1) as i64;
            let factor =
                a.clone() * RationalFunction::from_i64(arm) + RationalFunction::from_i64(leg + 1);
            product = product * factor;
        }
    }
    product
}

fn macdonald_psi_with_parameter(
    lambda_in: &[usize],
    mu_in: &[usize],
    parameter: RationalFunction,
) -> RationalFunction {
    let [lambda, mu] = normalize_two(lambda_in, mu_in);
    let [lambdac, muc] =
        normalize_two(&conjugate_partition(lambda_in), &conjugate_partition(mu_in));
    let mut product = RationalFunction::one();

    for r in 0..mu.len() {
        for c in 1..=mu[r] {
            if mu[r] < lambda[r] && muc[c - 1] == lambdac[c - 1] {
                let arm_l = (lambda[r] - c) as i64;
                let leg_l = (lambdac[c - 1] - r - 1) as i64;
                let arm_m = (mu[r] - c) as i64;
                let leg_m = (muc[c - 1] - r - 1) as i64;
                let b_m = b_factor(&parameter, arm_m, leg_m);
                let b_l = b_factor(&parameter, arm_l, leg_l);
                product = product * (b_m / b_l);
            }
        }
    }

    product
}

fn b_factor(parameter: &RationalFunction, arm: i64, leg: i64) -> RationalFunction {
    let p_arm = parameter.clone() * RationalFunction::from_i64(arm);
    let numerator = p_arm.clone() + RationalFunction::from_i64(leg + 1);
    let denominator = p_arm + RationalFunction::from_i64(leg) + parameter.clone();
    numerator / denominator
}

pub(crate) fn semistandard_tableaux(shape: &[usize], max_entry: usize) -> Vec<Vec<Vec<usize>>> {
    if shape.is_empty() {
        return vec![Vec::new()];
    }
    let cells: Vec<(usize, usize)> = shape
        .iter()
        .enumerate()
        .flat_map(|(r, row_len)| (0..*row_len).map(move |c| (r, c)))
        .collect();
    let mut tableau: Vec<Vec<usize>> = shape.iter().map(|row_len| vec![0; *row_len]).collect();
    let mut out = Vec::new();
    fill_tableaux(shape, max_entry, &cells, 0, &mut tableau, &mut out);
    out
}

fn fill_tableaux(
    shape: &[usize],
    max_entry: usize,
    cells: &[(usize, usize)],
    index: usize,
    tableau: &mut Vec<Vec<usize>>,
    out: &mut Vec<Vec<Vec<usize>>>,
) {
    if index == cells.len() {
        out.push(tableau.clone());
        return;
    }

    let (r, c) = cells[index];
    let mut lower = 1;
    if c > 0 {
        lower = lower.max(tableau[r][c - 1]);
    }
    if r > 0 && c < shape[r - 1] {
        lower = lower.max(tableau[r - 1][c] + 1);
    }

    for entry in lower..=max_entry {
        tableau[r][c] = entry;
        fill_tableaux(shape, max_entry, cells, index + 1, tableau, out);
    }
}

pub(crate) fn gt_chain_from_tableau(
    tableau: &[Vec<usize>],
    shape: &[usize],
    max_entry: usize,
) -> Vec<Partition> {
    (0..=max_entry)
        .rev()
        .map(|threshold| {
            (0..shape.len())
                .map(|r| {
                    tableau[r]
                        .iter()
                        .filter(|&&entry| entry <= threshold)
                        .count()
                })
                .collect()
        })
        .collect()
}

fn int_poly_to_rat_coeffs(poly: &IntPoly) -> Vec<Rat> {
    poly.coeffs.iter().cloned().map(Rat::from_integer).collect()
}

fn rat_poly_trim(poly: &mut Vec<Rat>) {
    while poly.last().is_some_and(Rat::is_zero) {
        poly.pop();
    }
}

fn rat_poly_div_rem(mut dividend: Vec<Rat>, divisor: Vec<Rat>) -> (Vec<Rat>, Vec<Rat>) {
    assert!(!divisor.is_empty(), "division by zero polynomial");
    rat_poly_trim(&mut dividend);
    let divisor_degree = divisor.len() - 1;
    let divisor_lead = divisor[divisor_degree].clone();

    if dividend.len() < divisor.len() {
        return (Vec::new(), dividend);
    }

    let mut quotient = vec![Rat::zero(); dividend.len() - divisor.len() + 1];
    while !dividend.is_empty() && dividend.len() >= divisor.len() {
        let degree = dividend.len() - divisor.len();
        let coeff = dividend.last().unwrap().clone() / divisor_lead.clone();
        quotient[degree] = coeff.clone();
        for (i, divisor_coeff) in divisor.iter().enumerate().take(divisor_degree + 1) {
            dividend[degree + i] -= coeff.clone() * divisor_coeff.clone();
        }
        rat_poly_trim(&mut dividend);
    }

    rat_poly_trim(&mut quotient);
    (quotient, dividend)
}

pub(crate) fn int_poly_gcd(a: &IntPoly, b: &IntPoly) -> IntPoly {
    if a.is_zero() {
        return b.primitive_part_positive();
    }
    if b.is_zero() {
        return a.primitive_part_positive();
    }

    let mut aa = int_poly_to_rat_coeffs(a);
    let mut bb = int_poly_to_rat_coeffs(b);
    while !bb.is_empty() {
        let (_, remainder) = rat_poly_div_rem(aa, bb.clone());
        aa = bb;
        bb = remainder;
    }

    if aa.is_empty() {
        return IntPoly::one();
    }
    let lead = aa.last().cloned().expect("nonzero gcd");
    for coeff in &mut aa {
        *coeff /= lead.clone();
    }
    rat_poly_to_primitive_int(&aa)
}

fn rat_poly_to_primitive_int(poly: &[Rat]) -> IntPoly {
    if poly.is_empty() {
        return IntPoly::zero();
    }
    let mut lcm = BigInt::one();
    for coeff in poly {
        lcm = lcm.lcm(coeff.denom());
    }
    let mut coeffs: Vec<BigInt> = poly
        .iter()
        .map(|coeff| coeff.numer() * (&lcm / coeff.denom()))
        .collect();
    while coeffs.last().is_some_and(BigInt::is_zero) {
        coeffs.pop();
    }
    let mut result = IntPoly::new(coeffs);
    if result.is_zero() {
        return result;
    }
    let content = result.content_abs();
    result = result.div_by_bigint_exact(&content);
    if result.leading_coefficient().is_negative() {
        result = -result;
    }
    result
}

fn rat_poly_to_integer_exact(poly: &[Rat]) -> IntPoly {
    IntPoly::new(
        poly.iter()
            .map(|coeff| {
                assert!(coeff.denom().is_one(), "non-integral polynomial quotient");
                coeff.numer().clone()
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_functions_reduce_polynomial_factors() {
        let a = IntPoly::variable();
        let numerator = a.clone() * a.clone() - IntPoly::one();
        let denominator = a - IntPoly::one();
        let reduced = RationalFunction::new(numerator, denominator);
        assert_eq!(format!("{reduced}"), "a + 1");
    }

    #[test]
    fn partition_parsing_and_formatting() {
        assert_eq!(parse_partition("{3,1,1}").unwrap(), vec![3, 1, 1]);
        assert_eq!(parse_partition("3 1 0").unwrap(), vec![3, 1]);
        assert!(parse_partition("1,3").is_err());
        assert_eq!(format_partition(&[3, 1, 0, 0]), "{3,1}");
    }

    #[test]
    fn shifted_jack_eval_for_one_box() {
        let mut calc = ShiftedJackCalculator::new();
        assert_eq!(format!("{}", calc.shifted_jack_evaluate(&[1], &[2])), "2");
        assert_eq!(
            format!("{}", calc.shifted_jack_evaluate(&[1], &[1, 0])),
            "1"
        );
    }

    #[test]
    fn smallest_structure_constants() {
        let mut calc = ShiftedJackCalculator::new();
        assert_eq!(
            format!("{}", calc.coefficient(&[1], &[1], &[1], Normalization::P)),
            "1"
        );
        assert_eq!(
            format!("{}", calc.coefficient(&[1], &[1], &[2], Normalization::P)),
            "1"
        );
        assert_eq!(
            format!("{}", calc.coefficient(&[1], &[1], &[2], Normalization::J)),
            "2*a^2"
        );
        assert_eq!(
            format!("{}", calc.coefficient(&[1], &[1], &[1], Normalization::J)),
            "a"
        );
        assert_eq!(
            format!("{}", calc.coefficient(&[2], &[2], &[2], Normalization::J)),
            "4*a^4 + 8*a^3 + 4*a^2"
        );
    }
}
