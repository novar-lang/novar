# Working on Telvax

Rules for anyone (person or tool) changing this repository. `CONTRIBUTING.md` has the details.

## Git and pull requests

* Never work on `main`. Start every change on a new branch cut from an up-to-date `main`:
  `git fetch && git switch -c type/short-name origin/main`.
* One branch, one small pull request, one idea. Do not stack pull requests on each other: wait for the
  one below to merge, then cut the next branch from `main`.
* Branch names are `type/short-name` (`feat/std-json`, `fix/openai-errors`, `ci/faster-docker`).
* Titles are Conventional Commits, short and in the imperative: `feat: std.json`, `fix: ...`, `ci: ...`,
  `docs: ...`, `test: ...`. The title becomes the squashed commit and picks the next version (see the
  table in `CONTRIBUTING.md`; `feat!:` is a breaking change).
* Commit messages and pull request descriptions are short and direct: what changed and why, as a few
  bullets. State what was checked and what was not.
* Do not mention Claude, AI or any tool anywhere: no `Co-Authored-By` trailer, no "Generated with" line,
  not in commits, pull requests, code, comments or docs.
* Merge with `gh pr merge --squash --delete-branch --auto` once the checks pass. `main` is protected
  for administrators too: wait for the checks instead of forcing the merge.
* Do not commit `docs/design/telvax-rumo-ao-1.0.md`: it names private repositories.

## Before every pull request

```sh
cargo fmt --all
cargo clippy --workspace --all-targets      # CI builds with RUSTFLAGS=-D warnings
cargo test --workspace
. scripts/llvm-env.sh && cargo test --workspace --features llvm   # when the native backend is touched
```

* A change that touches the lexer, parser or syntax also needs `cd editors/vscode && npm test`, and the
  examples must still pass `telvax check` and `telvax fmt --check`.
* Add a test with every behavior change. Keep tests next to the code they cover.

## The language

* Source files end in `.tvx`. The syntax is documented in `docs/design/sintaxe-pythonica.md` and
  `docs/guide/tour.md`; update both when it changes, and `CHANGELOG.md` for anything a user sees.
* Blocks are made by indentation. Words: `fn`, `let` (fixed), `var` (can change), `type`, `enum`, `trait`,
  `extend`, `x => ...`. Types: `Int`, `Float`, `Text`, `Bool`, `List[T]`, `Map[K, V]`, `T?`,
  `Tensor[f32, 2, 3]`, `Table[Row]`. The last expression of a function is its value.
* Error messages say what to write, in the same words the language uses.
* New standard library functions go in `compiler/semantic/src/stdlib.rs` (the signature) and
  `runtime/src/std_impl.rs` (the behavior), under a `std.<module>.<name>` path.

## Style

* Rust: nightly, edition 2024, `rustfmt` defaults. Match the surrounding code and comment density.
* Documentation is plain: short sentences, no marketing words, say what exists today.
* Keep the status page (`docs/design/status.md`) true: if a claim stops being true, fix it in the same
  pull request.
