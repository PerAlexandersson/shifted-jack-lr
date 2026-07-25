use std::fs;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use shifted_jack_lr::macdonald::{Algorithm, MacdonaldCalculator};
use shifted_jack_lr::{
    format_partition, integer_partitions, parse_partition, partition_size, Normalization,
    Partition, RationalFunction, ShiftedJackCalculator,
};

#[derive(Parser)]
#[command(
    author,
    version,
    about,
    long_about = "Compute shifted Jack Littlewood-Richardson structure constants exactly as rational functions in a."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute one shifted Jack structure constant.
    Coefficient {
        #[arg(long)]
        lambda: String,
        #[arg(long)]
        mu: String,
        #[arg(long)]
        nu: String,
        #[arg(long, value_enum, default_value_t = CliNormalization::J)]
        normalization: CliNormalization,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Compute all nonzero coefficients for a product.
    Product {
        #[arg(long)]
        lambda: String,
        #[arg(long)]
        mu: String,
        /// Highest top partition size to include. Defaults to |lambda|+|mu|.
        #[arg(long)]
        max_size: Option<usize>,
        #[arg(long, value_enum, default_value_t = CliNormalization::J)]
        normalization: CliNormalization,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Generate a Mathematica or JSON Lines table of coefficients.
    GenerateData {
        /// Include partitions lambda, mu with sizes at most this value.
        #[arg(long)]
        max_factor_size: usize,
        /// Only emit pairs whose larger factor has at least this size.
        #[arg(long, default_value_t = 1)]
        from_factor_size: usize,
        /// Highest top partition size to include. Defaults to |lambda|+|mu| for each pair.
        #[arg(long)]
        max_nu_size: Option<usize>,
        /// Omit coefficients that are exactly zero.
        #[arg(long)]
        skip_zero: bool,
        /// Print progress on stderr.
        #[arg(long)]
        progress: bool,
        /// Write to this file instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = CliNormalization::J)]
        normalization: CliNormalization,
        #[arg(long, value_enum, default_value_t = DataFormat::Mathematica)]
        format: DataFormat,
    },
    /// Evaluate a shifted Jack polynomial at a partition.
    Evaluate {
        #[arg(long)]
        shape: String,
        #[arg(long)]
        at: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Check a Mathematica JackJStructureConstant data file.
    VerifyData {
        #[arg(long)]
        path: PathBuf,
        #[arg(long, default_value_t = 10)]
        max_mismatches: usize,
        /// Stop after this many coefficients. Useful for quick smoke checks.
        #[arg(long)]
        max_checks: Option<usize>,
    },
    /// Compute one ordinary Macdonald (q,t) P-basis structure constant:
    /// the coefficient of P_nu in P_lambda * P_mu.
    #[command(name = "mac-lr")]
    MacLr {
        #[arg(long)]
        lambda: String,
        #[arg(long)]
        mu: String,
        #[arg(long)]
        nu: String,
        #[arg(long, value_enum, default_value_t = CliAlgorithm::Recursive)]
        algorithm: CliAlgorithm,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
    /// Compute every nonzero coefficient in P_lambda * P_mu (Macdonald
    /// products are homogeneous, so nu always has size |lambda|+|mu|).
    #[command(name = "mac-lr-product")]
    MacLrProduct {
        #[arg(long)]
        lambda: String,
        #[arg(long)]
        mu: String,
        #[arg(long, value_enum, default_value_t = CliAlgorithm::Recursive)]
        algorithm: CliAlgorithm,
        #[arg(long, value_enum, default_value_t = OutputFormat::Text)]
        format: OutputFormat,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliNormalization {
    P,
    J,
}

impl From<CliNormalization> for Normalization {
    fn from(value: CliNormalization) -> Self {
        match value {
            CliNormalization::P => Normalization::P,
            CliNormalization::J => Normalization::J,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliAlgorithm {
    /// Box-by-box recursion, memoized. The default: fastest in general and
    /// benefits from the cache across repeated calls.
    Recursive,
    /// Weighted sum over maximal chains. Kept mainly for cross-checking.
    ChainSum,
    /// Independent Gram-Schmidt computation in the power-sum basis.
    BruteForce,
    /// Gram-Schmidt P_lambda/P_mu/P_nu, multiplied via Schur-basis integer
    /// Littlewood-Richardson coefficients.
    SchurSandwich,
    /// Replicates Sage's own technique (Lapointe-Lascoux-Morse creation
    /// operators); much slower here than `Recursive` -- see
    /// `crate::macdonald::Algorithm::CreationOperator`.
    CreationOperator,
}

impl From<CliAlgorithm> for Algorithm {
    fn from(value: CliAlgorithm) -> Self {
        match value {
            CliAlgorithm::Recursive => Algorithm::Recursive,
            CliAlgorithm::ChainSum => Algorithm::ChainSum,
            CliAlgorithm::BruteForce => Algorithm::BruteForce,
            CliAlgorithm::SchurSandwich => Algorithm::SchurSandwich,
            CliAlgorithm::CreationOperator => Algorithm::CreationOperator,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DataFormat {
    Mathematica,
    Jsonl,
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(cli) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let mut calculator = ShiftedJackCalculator::new();
    let mut macdonald_calculator = MacdonaldCalculator::new();
    match cli.command {
        Command::Coefficient {
            lambda,
            mu,
            nu,
            normalization,
            format,
        } => {
            let lambda = parse_partition(&lambda)?;
            let mu = parse_partition(&mu)?;
            let nu = parse_partition(&nu)?;
            let value = calculator.coefficient(&lambda, &mu, &nu, normalization.into());
            match format {
                OutputFormat::Text => {
                    println!(
                        "c_{{{}, {}}}^{} ({:?}) = {}",
                        format_partition(&lambda),
                        format_partition(&mu),
                        format_partition(&nu),
                        normalization,
                        value
                    );
                }
                OutputFormat::Json => {
                    println!(
                        "{{\"lambda\":{},\"mu\":{},\"nu\":{},\"normalization\":\"{:?}\",\"value\":\"{}\"}}",
                        partition_json(&lambda),
                        partition_json(&mu),
                        partition_json(&nu),
                        normalization,
                        json_escape(&value.to_string())
                    );
                }
            }
        }
        Command::Product {
            lambda,
            mu,
            max_size,
            normalization,
            format,
        } => {
            let lambda = parse_partition(&lambda)?;
            let mu = parse_partition(&mu)?;
            let terms =
                calculator.product_coefficients(&lambda, &mu, max_size, normalization.into());
            match format {
                OutputFormat::Text => {
                    println!(
                        "{} * {} ({:?})",
                        format_partition(&lambda),
                        format_partition(&mu),
                        normalization
                    );
                    for (nu, value) in terms {
                        println!("{} : {}", format_partition(&nu), value);
                    }
                }
                OutputFormat::Json => {
                    let terms_json = terms
                        .iter()
                        .map(|(nu, value)| {
                            format!(
                                "{{\"nu\":{},\"value\":\"{}\"}}",
                                partition_json(nu),
                                json_escape(&value.to_string())
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    println!(
                        "{{\"lambda\":{},\"mu\":{},\"normalization\":\"{:?}\",\"terms\":[{}]}}",
                        partition_json(&lambda),
                        partition_json(&mu),
                        normalization,
                        terms_json
                    );
                }
            }
        }
        Command::GenerateData {
            max_factor_size,
            from_factor_size,
            max_nu_size,
            skip_zero,
            progress,
            output,
            normalization,
            format,
        } => generate_data(
            &mut calculator,
            GenerateDataOptions {
                max_factor_size,
                from_factor_size,
                max_nu_size,
                skip_zero,
                progress,
                output,
                normalization,
                format,
            },
        )?,
        Command::Evaluate { shape, at, format } => {
            let shape = parse_partition(&shape)?;
            let at = parse_partition(&at)?;
            let value = calculator.shifted_jack_evaluate(&shape, &at);
            match format {
                OutputFormat::Text => {
                    println!(
                        "P^*_{}({}) = {}",
                        format_partition(&shape),
                        format_partition(&at),
                        value
                    );
                }
                OutputFormat::Json => {
                    println!(
                        "{{\"shape\":{},\"at\":{},\"value\":\"{}\"}}",
                        partition_json(&shape),
                        partition_json(&at),
                        json_escape(&value.to_string())
                    );
                }
            }
        }
        Command::VerifyData {
            path,
            max_mismatches,
            max_checks,
        } => verify_data_file(&mut calculator, &path, max_mismatches, max_checks)?,
        Command::MacLr {
            lambda,
            mu,
            nu,
            algorithm,
            format,
        } => {
            let lambda = parse_partition(&lambda)?;
            let mu = parse_partition(&mu)?;
            let nu = parse_partition(&nu)?;
            let value = macdonald_calculator.p_structure_constant_with(
                &lambda,
                &mu,
                &nu,
                algorithm.into(),
            );
            match format {
                OutputFormat::Text => {
                    println!(
                        "c^{}_{{{}, {}}}(q,t) = {}",
                        format_partition(&nu),
                        format_partition(&lambda),
                        format_partition(&mu),
                        value
                    );
                }
                OutputFormat::Json => {
                    println!(
                        "{{\"lambda\":{},\"mu\":{},\"nu\":{},\"value\":\"{}\"}}",
                        partition_json(&lambda),
                        partition_json(&mu),
                        partition_json(&nu),
                        json_escape(&value.to_string())
                    );
                }
            }
        }
        Command::MacLrProduct {
            lambda,
            mu,
            algorithm,
            format,
        } => {
            let lambda = parse_partition(&lambda)?;
            let mu = parse_partition(&mu)?;
            let size = partition_size(&lambda) + partition_size(&mu);
            let mut terms = Vec::new();
            for nu in integer_partitions(size) {
                if !partition_contains(&nu, &lambda) || !partition_contains(&nu, &mu) {
                    continue;
                }
                let value = macdonald_calculator.p_structure_constant_with(
                    &lambda,
                    &mu,
                    &nu,
                    algorithm.into(),
                );
                if !value.is_zero() {
                    terms.push((nu, value));
                }
            }
            match format {
                OutputFormat::Text => {
                    println!("{} * {} (q,t)", format_partition(&lambda), format_partition(&mu));
                    for (nu, value) in &terms {
                        println!("{} : {}", format_partition(nu), value);
                    }
                }
                OutputFormat::Json => {
                    let terms_json = terms
                        .iter()
                        .map(|(nu, value)| {
                            format!(
                                "{{\"nu\":{},\"value\":\"{}\"}}",
                                partition_json(nu),
                                json_escape(&value.to_string())
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    println!(
                        "{{\"lambda\":{},\"mu\":{},\"terms\":[{}]}}",
                        partition_json(&lambda),
                        partition_json(&mu),
                        terms_json
                    );
                }
            }
        }
    }
    Ok(())
}

fn partition_json(partition: &[usize]) -> String {
    format!(
        "[{}]",
        partition
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_escape(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

struct GenerateDataOptions {
    max_factor_size: usize,
    from_factor_size: usize,
    max_nu_size: Option<usize>,
    skip_zero: bool,
    progress: bool,
    output: Option<PathBuf>,
    normalization: CliNormalization,
    format: DataFormat,
}

fn generate_data(
    calculator: &mut ShiftedJackCalculator,
    options: GenerateDataOptions,
) -> Result<(), String> {
    if options.max_factor_size == 0 {
        return Err("--max-factor-size must be positive".to_string());
    }
    if options.from_factor_size == 0 {
        return Err("--from-factor-size must be positive".to_string());
    }
    if options.from_factor_size > options.max_factor_size {
        return Err("--from-factor-size cannot exceed --max-factor-size".to_string());
    }

    let mut writer: Box<dyn Write> = match &options.output {
        Some(path) => {
            Box::new(BufWriter::new(File::create(path).map_err(|error| {
                format!("could not create output file: {error}")
            })?))
        }
        None => Box::new(BufWriter::new(io::stdout())),
    };

    let factors = factor_partitions(options.max_factor_size);
    let pairs = unordered_factor_pairs(&factors, options.from_factor_size);
    let total_pairs = pairs.len();
    let mut emitted = 0usize;

    for (pair_index, (lambda, mu)) in pairs.into_iter().enumerate() {
        if options.progress {
            eprintln!(
                "pair {}/{}: {} * {}",
                pair_index + 1,
                total_pairs,
                format_partition(lambda),
                format_partition(mu)
            );
        }

        let lower_size = partition_size(lambda).max(partition_size(mu));
        let upper_size = options
            .max_nu_size
            .unwrap_or_else(|| partition_size(lambda) + partition_size(mu));
        if upper_size < lower_size {
            continue;
        }

        for size in lower_size..=upper_size {
            for nu in integer_partitions(size) {
                if !partition_contains(&nu, lambda) || !partition_contains(&nu, mu) {
                    continue;
                }
                let value = calculator.coefficient(lambda, mu, &nu, options.normalization.into());
                if options.skip_zero && value.is_zero() {
                    continue;
                }
                write_data_record(
                    &mut writer,
                    lambda,
                    mu,
                    &nu,
                    options.normalization,
                    options.format,
                    &value,
                )?;
                emitted += 1;
            }
        }
    }

    writer
        .flush()
        .map_err(|error| format!("could not flush output: {error}"))?;
    if options.progress {
        eprintln!("emitted {emitted} coefficients");
    }
    Ok(())
}

fn factor_partitions(max_factor_size: usize) -> Vec<Partition> {
    (1..=max_factor_size).flat_map(integer_partitions).collect()
}

fn unordered_factor_pairs(
    factors: &[Partition],
    from_factor_size: usize,
) -> Vec<(&Partition, &Partition)> {
    let mut pairs = Vec::new();
    for (i, lambda) in factors.iter().enumerate() {
        for mu in factors.iter().take(i + 1) {
            let pair_size = partition_size(lambda).max(partition_size(mu));
            if pair_size >= from_factor_size {
                pairs.push((lambda, mu));
            }
        }
    }
    pairs
}

fn partition_contains(outer: &[usize], inner: &[usize]) -> bool {
    let len = outer.len().max(inner.len());
    (0..len).all(|idx| outer.get(idx).copied().unwrap_or(0) >= inner.get(idx).copied().unwrap_or(0))
}

fn write_data_record(
    writer: &mut dyn Write,
    lambda: &[usize],
    mu: &[usize],
    nu: &[usize],
    normalization: CliNormalization,
    format: DataFormat,
    value: &RationalFunction,
) -> Result<(), String> {
    match format {
        DataFormat::Mathematica => writeln!(
            writer,
            "{}[{{{}, {}, {}}}]:={};",
            mathematica_function_name(normalization),
            format_partition(lambda),
            format_partition(mu),
            format_partition(nu),
            value
        ),
        DataFormat::Jsonl => writeln!(
            writer,
            "{{\"lambda\":{},\"mu\":{},\"nu\":{},\"normalization\":\"{:?}\",\"value\":\"{}\"}}",
            partition_json(lambda),
            partition_json(mu),
            partition_json(nu),
            normalization,
            json_escape(&value.to_string())
        ),
    }
    .map_err(|error| format!("could not write output: {error}"))
}

fn mathematica_function_name(normalization: CliNormalization) -> &'static str {
    match normalization {
        CliNormalization::P => "JackPStructureConstant",
        CliNormalization::J => "JackJStructureConstant",
    }
}

fn verify_data_file(
    calculator: &mut ShiftedJackCalculator,
    path: &PathBuf,
    max_mismatches: usize,
    max_checks: Option<usize>,
) -> Result<(), String> {
    let data =
        fs::read_to_string(path).map_err(|error| format!("could not read data file: {error}"))?;
    let mut checked = 0usize;
    let mut mismatches = Vec::new();

    for (line_number, line) in data.lines().enumerate() {
        let line_number = line_number + 1;
        let Some(rest) = line.strip_prefix("JackJStructureConstant[") else {
            continue;
        };
        let Some((lhs, rhs)) = rest.split_once("]:=") else {
            return Err(format!("line {line_number}: expected `]:=`"));
        };
        let [lambda, mu, nu] = parse_mathematica_triple(lhs)
            .map_err(|error| format!("line {line_number}: {error}"))?;
        let expected = parse_mathematica_expression(rhs)
            .map_err(|error| format!("line {line_number}: {error}"))?;
        let actual = calculator.coefficient(&lambda, &mu, &nu, Normalization::J);
        checked += 1;

        if actual != expected {
            mismatches.push(format!(
                "line {line_number}: c_{{{}, {}}}^{} expected {}, got {}",
                format_partition(&lambda),
                format_partition(&mu),
                format_partition(&nu),
                expected,
                actual
            ));
            if mismatches.len() >= max_mismatches {
                break;
            }
        }

        if checked.is_multiple_of(1000) {
            eprintln!("checked {checked} coefficients");
        }
        if max_checks.is_some_and(|limit| checked >= limit) {
            break;
        }
    }

    if mismatches.is_empty() {
        println!("checked {checked} coefficients; all matched");
        Ok(())
    } else {
        for mismatch in &mismatches {
            eprintln!("{mismatch}");
        }
        Err(format!(
            "{} mismatches found after checking {checked} coefficients",
            mismatches.len()
        ))
    }
}

fn parse_mathematica_triple(input: &str) -> Result<[Partition; 3], String> {
    let inner = input
        .trim()
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .ok_or_else(|| "expected outer braces around partition triple".to_string())?;
    let mut parts = Vec::new();
    let mut start = None;
    let mut depth = 0usize;
    for (idx, ch) in inner.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    start = Some(idx);
                }
                depth += 1;
            }
            '}' => {
                if depth == 0 {
                    return Err("unbalanced closing brace".to_string());
                }
                depth -= 1;
                if depth == 0 {
                    let part_start = start.ok_or_else(|| "missing partition start".to_string())?;
                    parts.push(parse_partition(&inner[part_start..=idx])?);
                    start = None;
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("unbalanced opening brace".to_string());
    }
    parts
        .try_into()
        .map_err(|_| "expected exactly three partitions".to_string())
}

fn parse_mathematica_expression(input: &str) -> Result<RationalFunction, String> {
    let mut parser = ExpressionParser::new(input.trim().trim_end_matches(';'));
    let value = parser.parse_sum()?;
    parser.skip_spaces();
    if parser.is_done() {
        Ok(value)
    } else {
        Err(format!("unexpected input near `{}`", parser.remaining()))
    }
}

struct ExpressionParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> ExpressionParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse_sum(&mut self) -> Result<RationalFunction, String> {
        let mut value = self.parse_product()?;
        loop {
            self.skip_spaces();
            if !self.consume('+') {
                return Ok(value);
            }
            value = value + self.parse_product()?;
        }
    }

    fn parse_product(&mut self) -> Result<RationalFunction, String> {
        let mut value = self.parse_power()?;
        loop {
            self.skip_spaces();
            if self.consume('*') {
                value = value * self.parse_power()?;
            } else if self.consume('/') {
                value = value / self.parse_power()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_power(&mut self) -> Result<RationalFunction, String> {
        let base = self.parse_atom()?;
        self.skip_spaces();
        if !self.consume('^') {
            return Ok(base);
        }
        let exponent = self.parse_usize()?;
        let mut value = RationalFunction::one();
        for _ in 0..exponent {
            value = value * base.clone();
        }
        Ok(value)
    }

    fn parse_atom(&mut self) -> Result<RationalFunction, String> {
        self.skip_spaces();
        match self.peek() {
            Some('a') => {
                self.pos += 1;
                Ok(RationalFunction::variable())
            }
            Some('(') => {
                self.pos += 1;
                let value = self.parse_sum()?;
                self.skip_spaces();
                if self.consume(')') {
                    Ok(value)
                } else {
                    Err("expected `)`".to_string())
                }
            }
            Some(ch) if ch.is_ascii_digit() => {
                let value = self.parse_i64()?;
                Ok(RationalFunction::from_i64(value))
            }
            Some(ch) => Err(format!("unexpected character `{ch}`")),
            None => Err("unexpected end of expression".to_string()),
        }
    }

    fn parse_i64(&mut self) -> Result<i64, String> {
        let start = self.pos;
        while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
            self.pos += 1;
        }
        self.input[start..self.pos]
            .parse()
            .map_err(|error| format!("could not parse integer: {error}"))
    }

    fn parse_usize(&mut self) -> Result<usize, String> {
        let start = self.pos;
        while self.peek().is_some_and(|ch| ch.is_ascii_digit()) {
            self.pos += 1;
        }
        self.input[start..self.pos]
            .parse()
            .map_err(|error| format!("could not parse exponent: {error}"))
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn skip_spaces(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.pos += 1;
        }
    }

    fn is_done(&self) -> bool {
        self.pos == self.input.len()
    }

    fn remaining(&self) -> &str {
        &self.input[self.pos..]
    }
}
