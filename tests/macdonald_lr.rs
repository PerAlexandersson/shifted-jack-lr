//! Cross-checks `MacdonaldCalculator::p_structure_constant` against
//! Sage-generated ground truth: ordinary (non-shifted, top-degree) Macdonald
//! P-basis structure constants for all unordered factor pairs lambda, mu
//! with |lambda|, |mu| <= 3 (see scripts/generate_macdonald_test_data.sage).
//!
//! The P basis (monic: coefficient of m_lambda in P_lambda is 1) is used
//! because it's canonical and convention-independent, unlike Sage's own "J"
//! basis. This methodology was validated against the already-trusted Jack
//! implementation: Sage's Jack P-basis products match
//! `shifted-jack-lr coefficient --normalization p` exactly, and combining
//! those P-basis values with this repo's own hook-factor formulas
//! reproduces the trusted jackStructureConstants6.m J-basis data exactly.

use num_bigint::BigInt;
use shifted_jack_lr::macdonald::{Algorithm, MacdonaldCalculator, Poly2, RationalFunction2};

/// Parses Sage's default `str()` rendering of an element of
/// `Frac(QQ[q,t])`, e.g. `"(q*t + q - t - 1)/(q*t - 1)"` or `"1"`.
fn parse_rational_function(input: &str) -> RationalFunction2 {
    let s = input.trim();
    if s.starts_with('(') {
        let split = s
            .find(")/(")
            .unwrap_or_else(|| panic!("expected \"(NUM)/(DEN)\" form in `{s}`"));
        let numerator_str = &s[1..split];
        let denominator_str = &s[split + 3..s.len() - 1];
        RationalFunction2::new(parse_expr(numerator_str), parse_expr(denominator_str))
    } else {
        RationalFunction2::new(parse_expr(s), Poly2::one())
    }
}

/// Parses a sum/difference of terms, e.g. `"q*t + q - t - 1"` or
/// `"-q^3*t^2 + q*t^2 + q^2 - 1"`, into a `Poly2`.
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
        let is_boundary = bytes[i] == b' '
            && matches!(bytes.get(i + 1), Some(b'+') | Some(b'-'))
            && bytes.get(i + 2) == Some(&b' ');
        if is_boundary {
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

/// Parses a single term, e.g. `"q^3*t^2"` or `"4"` or `"q"`, into a `Poly2`.
fn parse_term(term: &str) -> Poly2 {
    let mut coefficient = BigInt::from(1);
    let mut q_degree = 0usize;
    let mut t_degree = 0usize;
    for factor in term.split('*') {
        let factor = factor.trim();
        if factor == "q" {
            q_degree += 1;
        } else if factor == "t" {
            t_degree += 1;
        } else if let Some(rest) = factor.strip_prefix("q^") {
            q_degree += rest.parse::<usize>().expect("integer exponent");
        } else if let Some(rest) = factor.strip_prefix("t^") {
            t_degree += rest.parse::<usize>().expect("integer exponent");
        } else {
            coefficient *= factor.parse::<BigInt>().expect("integer coefficient");
        }
    }
    Poly2::monomial(q_degree, t_degree, coefficient)
}

struct Record {
    lambda: Vec<usize>,
    mu: Vec<usize>,
    nu: Vec<usize>,
    value: RationalFunction2,
}

fn parse_partition(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .expect("partition array")
        .iter()
        .map(|x| x.as_u64().expect("partition part") as usize)
        .collect()
}

fn load_ground_truth() -> Vec<Record> {
    let data = include_str!("../macdonaldPStructureConstants3.jsonl");
    data.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).expect("valid JSON line");
            Record {
                lambda: parse_partition(&v["lambda"]),
                mu: parse_partition(&v["mu"]),
                nu: parse_partition(&v["nu"]),
                value: parse_rational_function(v["value"].as_str().expect("value string")),
            }
        })
        .collect()
}

#[test]
fn parser_round_trips_sample_sage_output() {
    assert_eq!(format!("{}", parse_rational_function("1")), "1");
    let simple = parse_rational_function("(q*t + q - t - 1)/(q*t - 1)");
    assert_eq!(format!("{simple}"), "(q*t + q - t - 1)/(q*t - 1)");
}

#[test]
fn ground_truth_data_loads_and_covers_expected_pair_count() {
    let records = load_ground_truth();
    // Unordered pairs (lambda, mu) with |lambda|, |mu| <= 3, i.e. all pairs
    // among the 6 factor partitions of size 1..=3: C(6,2) + 6 = 21 pairs.
    // Each pair can contribute multiple nu terms, so just sanity check the
    // count matches what the generator script reported (64 records).
    assert_eq!(records.len(), 64);
}

#[test]
fn macdonald_p_structure_constants_match_sage_ground_truth() {
    let records = load_ground_truth();
    let mut calculator = MacdonaldCalculator::new();
    for record in &records {
        let start = std::time::Instant::now();
        let actual = calculator.p_structure_constant(&record.lambda, &record.mu, &record.nu);
        eprintln!(
            "{:?} x {:?} -> {:?} in {:?}",
            record.lambda,
            record.mu,
            record.nu,
            start.elapsed()
        );
        assert_eq!(
            actual, record.value,
            "c^{:?}_{{{:?},{:?}}}(q,t): expected {}, got {}",
            record.nu, record.lambda, record.mu, record.value, actual
        );
    }
}

#[test]
fn macdonald_p_structure_constant_is_zero_off_top_degree() {
    let mut calculator = MacdonaldCalculator::new();
    // The ordinary (non-shifted) product P_lambda * P_mu is homogeneous of
    // degree |lambda| + |mu|, so any nu of a different size must be zero --
    // true regardless of q, t, so this holds no matter how the coefficient
    // is computed.
    assert!(calculator.p_structure_constant(&[1], &[1], &[1]).is_zero());
    assert!(calculator
        .p_structure_constant(&[2], &[1], &[1, 1, 1, 1])
        .is_zero());
}

#[test]
#[ignore = "CreationOperator is ~650s per case; run explicitly with --ignored"]
fn creation_operator_matches_sage_ground_truth() {
    let records = load_ground_truth();
    let mut calculator = MacdonaldCalculator::new();
    for record in &records {
        let start = std::time::Instant::now();
        let actual = calculator.p_structure_constant_with(
            &record.lambda,
            &record.mu,
            &record.nu,
            Algorithm::CreationOperator,
        );
        eprintln!(
            "{:?} x {:?} -> {:?} in {:?}",
            record.lambda,
            record.mu,
            record.nu,
            start.elapsed()
        );
        assert_eq!(
            actual, record.value,
            "c^{:?}_{{{:?},{:?}}}(q,t): expected {}, got {}",
            record.nu, record.lambda, record.mu, record.value, actual
        );
    }
}

#[test]
fn size_ten_case_matches_sage_exactly() {
    // Sage ground truth for the size-10 case, from a direct
    // `P[3,2,1] * P[2,1,1]` product (which Sage computes in ~112s).
    let expected = parse_rational_function(
        "(3*q^12*t^12 + 2*q^12*t^11 - 2*q^11*t^12 - q^12*t^10 + 4*q^11*t^11 - 3*q^10*t^12 - q^12*t^9 + 5*q^11*t^10 - 9*q^10*t^11 + 2*q^9*t^12 - 5*q^11*t^9 + 5*q^10*t^10 - 3*q^9*t^11 - 5*q^11*t^8 + 10*q^10*t^9 - 17*q^9*t^10 + 6*q^8*t^11 - 6*q^10*t^8 + 7*q^9*t^9 - 2*q^8*t^10 + q^7*t^11 - 7*q^10*t^7 + 20*q^9*t^8 - 24*q^8*t^9 + 12*q^7*t^10 - q^6*t^11 + 2*q^10*t^6 - 7*q^9*t^7 + 8*q^8*t^8 + 3*q^7*t^9 + 2*q^10*t^5 - 9*q^9*t^6 + 30*q^8*t^7 - 31*q^7*t^8 + 16*q^6*t^9 - 2*q^5*t^10 + 5*q^9*t^5 - 7*q^8*t^6 + 5*q^7*t^7 + 5*q^6*t^8 - 2*q^5*t^9 + 4*q^9*t^4 - 14*q^8*t^5 + 35*q^7*t^6 - 35*q^6*t^7 + 16*q^5*t^8 - 3*q^4*t^9 + q^9*t^3 + 3*q^8*t^4 - 7*q^7*t^5 + 7*q^5*t^7 - 3*q^4*t^8 - q^3*t^9 + 3*q^8*t^3 - 16*q^7*t^4 + 35*q^6*t^5 - 35*q^5*t^6 + 14*q^4*t^7 - 4*q^3*t^8 + 2*q^7*t^3 - 5*q^6*t^4 - 5*q^5*t^5 + 7*q^4*t^6 - 5*q^3*t^7 + 2*q^7*t^2 - 16*q^6*t^3 + 31*q^5*t^4 - 30*q^4*t^5 + 9*q^3*t^6 - 2*q^2*t^7 - 3*q^5*t^3 - 8*q^4*t^4 + 7*q^3*t^5 - 2*q^2*t^6 + q^6*t - 12*q^5*t^2 + 24*q^4*t^3 - 20*q^3*t^4 + 7*q^2*t^5 - q^5*t + 2*q^4*t^2 - 7*q^3*t^3 + 6*q^2*t^4 - 6*q^4*t + 17*q^3*t^2 - 10*q^2*t^3 + 5*q*t^4 + 3*q^3*t - 5*q^2*t^2 + 5*q*t^3 - 2*q^3 + 9*q^2*t - 5*q*t^2 + t^3 + 3*q^2 - 4*q*t + t^2 + 2*q - 2*t - 3)/(q^12*t^12 - q^10*t^11 - q^11*t^9 - q^10*t^9 - q^9*t^10 + q^9*t^8 - q^8*t^9 + q^8*t^8 + q^7*t^9 + q^9*t^6 + q^8*t^7 + q^7*t^7 + q^6*t^8 + q^7*t^6 - q^7*t^5 + q^5*t^7 - q^5*t^6 - q^6*t^4 - q^5*t^5 - q^4*t^5 - q^3*t^6 - q^5*t^3 - q^4*t^4 + q^4*t^3 - q^3*t^4 + q^3*t^2 + q^2*t^3 + q*t^3 + q^2*t - 1)",
    );
    let mut calculator = MacdonaldCalculator::new();
    let start = std::time::Instant::now();
    let actual = calculator.p_structure_constant(&[3, 2, 1], &[2, 1, 1], &[4, 3, 2, 1]);
    eprintln!("[3,2,1] x [2,1,1] -> [4,3,2,1] in {:?}", start.elapsed());
    assert_eq!(actual, expected);
}
