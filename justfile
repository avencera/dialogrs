# dialogrs builds against ../scriptrs and ../speakrs, whose master moves, so no
# recipe passes --locked. speakrs has no backend without cuda, so every build here
# turns it on

fmt:
    cargo fmt

# advisories, licenses and sources
deny:
    cargo deny --all-features check

# the docs as docs.rs builds them
doc:
    DOCS_RS=1 RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features cuda

clippy:
    cargo clippy --all-targets --features cuda -- -D warnings

# unit tests only; the pipeline itself needs a GPU and models
test *args:
    cargo test --features cuda --lib {{args}}

# the crate as crates.io receives it: cargo package builds it against the
# scriptrs and speakrs releases on crates.io, not the sibling checkouts, so this
# fails while dialogrs needs something they haven't released
package *args:
    #!/usr/bin/env bash
    set -euo pipefail
    # cargo package writes under the effective target directory, which
    # CARGO_TARGET_DIR or build.target-dir can move away from ./target
    target=$(cargo metadata --format-version 1 --no-deps | jq -r .target_directory)
    rm -rf "$target/package"
    cargo package --features cuda {{args}}
    crate=$(find "$target/package" -maxdepth 1 -name 'dialogrs-*.crate' -print -quit)
    test -n "$crate"
    # crates.io rejects uploads over 10 MiB unless it raises the crate's limit
    size=$(wc -c < "$crate")
    echo "$(basename "$crate"): $size bytes"
    test "$size" -le $((10 * 1024 * 1024))

    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
    tar -xzf "$crate" -C "$work"
    package=$(find "$work" -mindepth 1 -maxdepth 1 -type d -print -quit)
    # a consumer resolves the newest compatible dependencies rather than the
    # packaged lockfile; its build files stay in this target directory too
    export CARGO_TARGET_DIR="$target/package-consumers"
    export CARGO_BUILD_BUILD_DIR="$CARGO_TARGET_DIR/build"
    mkdir -p "$work/consumer/src"
    printf '[package]\nname = "consumer"\nversion = "0.0.0"\nedition = "2024"\npublish = false\n\n[workspace]\n\n[dependencies]\ndialogrs = { path = "%s", features = ["cuda"] }\n' \
        "$package" > "$work/consumer/Cargo.toml"
    printf '%s\n' 'pub fn words() -> usize {
        dialogrs::words_from_tokens(&[]).len()
    }' > "$work/consumer/src/lib.rs"
    echo "building the packaged crate as a dependency"
    cargo build --manifest-path "$work/consumer/Cargo.toml"

check:
    just fmt
    just deny
    just clippy
    just doc
    just test
