# dialogrs builds against ../scriptrs and ../speakrs, whose master moves, so no
# recipe passes --locked. speakrs has no backend without cuda, so every build here
# turns it on

fmt:
    cargo fmt
    cargo fmt --manifest-path crates/dialogrs-python/Cargo.toml
    uvx ruff@0.17.4 format crates/dialogrs-python

# ruff's lints for the Python package; `just fmt` runs its formatter. ruff is pinned
# because a release that restyles code would fail CI's formatting check
python-lint:
    uvx ruff@0.17.4 check crates/dialogrs-python

# advisories, licenses and sources for the crate and the Python bindings, which are
# separate workspaces
deny:
    cargo deny --all-features check
    cargo deny --manifest-path crates/dialogrs-python/Cargo.toml --all-features check

# the docs as docs.rs builds them
doc:
    DOCS_RS=1 RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features cuda

clippy:
    cargo clippy --all-targets --features cuda -- -D warnings
    cargo clippy --manifest-path crates/dialogrs-python/Cargo.toml --all-targets -- -D warnings

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
    just python-lint
    just deny
    just clippy
    just doc
    just test

# build the Python wheel for this platform into crates/dialogrs-python/target/wheels,
# without the manylinux check that CI runs in its container
python-build:
    uvx --from 'maturin>=1.9,<2' maturin build --release --locked --auditwheel skip -m crates/dialogrs-python/Cargo.toml

# install the wheel into a fresh environment and run its tests; on Linux the model
# tests need the GPU, so submit `just python-test -m model` through the homebased
# GPU queue there, and run `just python-test -m "not model"` directly
python-test *args: python-build
    #!/usr/bin/env bash
    set -euo pipefail
    cd crates/dialogrs-python
    venv=target/python-test-venv
    rm -rf "$venv"
    uv venv -q --python 3.12 "$venv"
    version=$(cargo pkgid | sed 's/.*[#@]//')
    uv pip install -q --python "$venv" "target/wheels/dialogrs-${version}"-*.whl pytest numpy
    # Windows venvs keep the interpreter in Scripts, not bin
    python="$venv/bin/python"
    if [ -x "$venv/Scripts/python.exe" ]; then python="$venv/Scripts/python.exe"; fi
    "$python" -m pytest -rA {{args}}
