# Contributing to Telvax

Thanks for trying Telvax. At 0.0.x the most useful contribution is a bug report or a program that
did not behave as you expected.

## Reporting problems

Open an issue and include:

* the output of `telvax --version` and your operating system;
* the smallest `.tvx` program that shows the problem;
* what you ran (`telvax run ...`, `telvax serve ...`) and the full output.

Language ideas are welcome too, ideally with a short example of the code you wish you could write.

## Building

Nightly Rust is required; `rust-toolchain.toml` selects it.

```sh
cargo build --release -p telvax_cli      # target/release/telvax
cargo test --workspace                  # default build, no LLVM needed
```

Optional pieces:

| Feature | What it adds | Needs |
|---|---|---|
| `--features telvax_cli/llvm` | native code, `--emit ptx` | LLVM 22 with MLIR (`. scripts/llvm-env.sh`) |
| `--features telvax_cli/gpu` | `--device gpu` | a GPU with Metal, Vulkan or DX12 |

The Python bindings (`python/`), the VS Code extension (`editors/vscode`) and the Wasm playground
(`playground/`) have their own README and tests.

## How changes get into `main`

`main` is protected: nothing is pushed to it directly, not even by the maintainer. Work happens on a
branch and reaches `main` through a pull request, and the pull request has to pass every check
(formatting, clippy, tests on Linux and Windows, the LLVM and GPU builds, the lite build, the
Python bindings, the playground in a real browser, the extension in a real VS Code, the Docker
image, `cargo deny` and the secret scan) before it can be merged. Merges are squashed, so the pull
request title becomes the commit message: keep it short and in the imperative (`feat: add ...`).

```sh
git switch -c fix/short-name
# ...edit, commit...
git push -u origin fix/short-name
gh pr create --fill
gh pr merge --auto --squash      # merges by itself once every check is green
```

You can still run the checks locally while you work (see below), but a green pull request is what
counts.

## Before sending a pull request

```sh
cargo fmt --all
cargo clippy --workspace --all-targets
cargo test --workspace
cargo deny check
```

* Add a test with the change. Compiler tests live next to the crate (`compiler/*/tests`), runtime
  tests in `runtime/tests`.
* Keep commits short and in the imperative, with the existing prefixes: `feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `style:`, `ci:`, `build:`, `chore:`.
* If a change alters the language, update `docs/guide/tour.md` and `CHANGELOG.md`.
* `docs/design/status.md` says what has and has not been run on real hardware. Keep it honest: do
  not mark something as working because it compiles.

## Where things are

See the table in the README. A quick map for the language itself: `compiler/lexer` (tokens),
`compiler/parser` (syntax), `compiler/semantic` (types and checks), `runtime` (interpreter),
`compiler/ir` and `compiler/codegen` (native), `compiler/accel` (PTX and StableHLO).

## Releases

Raising `version` in the root `Cargo.toml` (and the matching `CHANGELOG.md` section) on `main`
makes CI tag the commit and publish a pre-release with binaries.

## Security

See [SECURITY.md](SECURITY.md).
