# Security Policy

## Supported versions

Telvax is an early alpha. Only the latest release receives security fixes; older versions are not
patched, so please update before reporting.

## Reporting a vulnerability

Report vulnerabilities **privately** through GitHub: open the repository's Security tab and choose
"Report a vulnerability". Please do not open a public issue, pull request or discussion for a
security problem.

Include what you can of: the affected version (`telvax --version`), your operating system, a minimal
`.tvx` program or request that reproduces the problem, and what you expected to happen.

## What to expect

* We aim to acknowledge a report within **7 days** and to tell you whether we consider it a
  vulnerability within **14 days**. This is a small project, so these are goals, not guarantees.
* A confirmed vulnerability is fixed in a new release, and the advisory is published after the
  release is out. We credit reporters who want to be credited.
* Please give us a reasonable time to fix the problem before you disclose it publicly.

## Scope

In scope: the `telvax` and `tvpm` binaries, the language server, the HTTP server that `telvax serve`
and `telvax build --serve` produce (request handling, validation), the package manager and its
registry client, the install scripts, and the VS Code extension.

Out of scope: the content of programs you choose to run (the interpreter runs the code it is given),
denial of service by deliberately huge inputs to a development server, and issues in third-party
services that Telvax can call (Ollama, OpenAI, PostgreSQL).
