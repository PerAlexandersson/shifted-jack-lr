# shifted-jack-lr

`shifted-jack-lr` is a command-line tool for exact shifted Jack
Littlewood-Richardson structure constants. It computes rational functions in
the Jack parameter `a`.

The default `j` normalization matches data files of the form

```mathematica
JackJStructureConstant[{{lambda}, {mu}, {nu}}] := value;
```

such as `jackStructureConstants6.m`.

## Install

Install Rust from <https://rustup.rs/>, then build the release binary:

```bash
cargo build --release
```

The binary is written to:

```text
target/release/shifted-jack-lr
```

On Windows the binary is:

```text
target\release\shifted-jack-lr.exe
```

To install from a GitHub checkout into Cargo's binary directory:

```bash
cargo install --path .
```

After publishing the repository, collaborators can also use:

```bash
cargo install --git https://github.com/PerAlexandersson/shifted-jack-lr
```

## Prebuilt Binaries

This repository includes a GitHub Actions release workflow. When a tag such as
`v0.1.0` is pushed, GitHub builds release binaries for:

- Linux x86_64
- Windows x86_64

The archives are attached to the GitHub Release page.

## Basic Usage

Partitions can be written with commas, spaces, or braces:

```text
3,2,1
"3 2 1"
{3,2,1}
```

Compute one coefficient:

```bash
shifted-jack-lr coefficient --lambda 3 --mu 2 --nu 4,1
```

Output:

```text
c_{{3}, {2}}^{4,1} (J) = 108*a^7 + 180*a^6 + 72*a^5
```

Compute all nonzero coefficients in one product:

```bash
shifted-jack-lr product --lambda 2,1 --mu 1 --normalization j
```

Get machine-readable JSON:

```bash
shifted-jack-lr coefficient --lambda 2,1 --mu 1 --nu 3,1 --format json
```

Evaluate a shifted Jack polynomial:

```bash
shifted-jack-lr evaluate --shape 2,1 --at 3,1
```

## Generate Data

Generate a Mathematica `.m` table for all unordered pairs of factor partitions
with sizes at most 7:

```bash
shifted-jack-lr generate-data --max-factor-size 7 --output jackStructureConstants7.m --progress
```

If existing data already covers factor sizes up to 6, generate only the new
size-7 layer:

```bash
shifted-jack-lr generate-data \
  --from-factor-size 7 \
  --max-factor-size 7 \
  --output jackStructureConstants7-layer.m \
  --progress
```

The default output format is Mathematica syntax:

```mathematica
JackJStructureConstant[{{1}, {1}, {1}}]:=a;
JackJStructureConstant[{{1}, {1}, {2}}]:=2*a^2;
JackJStructureConstant[{{1}, {1}, {1,1}}]:=2*a^2;
```

Use JSON Lines for scripting:

```bash
shifted-jack-lr generate-data --max-factor-size 3 --format jsonl --output data.jsonl
```

Skip zero coefficients:

```bash
shifted-jack-lr generate-data --max-factor-size 7 --skip-zero --output nonzero.m
```

Batch generation is much faster than calling `coefficient` in a shell loop,
because one process reuses the recurrence cache.

## Normalizations

The `p` normalization is the recursive shifted Jack `P` coefficient
`c^nu_{lambda,mu}`.

The default `j` normalization is the data-file normalization

```text
a^|nu| HookPrimeFactor(lambda) HookPrimeFactor(mu) HookFactor(nu) c^nu_{lambda,mu}.
```

This is the normalization used by `JackJStructureConstant` in
`jackStructureConstants6.m`.

Select a normalization with:

```bash
shifted-jack-lr coefficient --lambda 1 --mu 1 --nu 2 --normalization p
shifted-jack-lr coefficient --lambda 1 --mu 1 --nu 2 --normalization j
```

## Verify Existing Data

Check a Mathematica data file against the Rust implementation:

```bash
shifted-jack-lr verify-data --path jackStructureConstants6.m
```

Run a quick smoke check:

```bash
shifted-jack-lr verify-data --path jackStructureConstants6.m --max-checks 1000
```

For large files this can take a while, since it performs exact symbolic
comparisons.

## Development

Run tests:

```bash
cargo test
```

Build an optimized binary:

```bash
cargo build --release
```

In the original Dropbox/Docker workspace, use the shared target directory to
avoid writing build artifacts into synced folders:

```bash
CARGO_TARGET_DIR=/tmp/shifted-jack-target cargo test
```
