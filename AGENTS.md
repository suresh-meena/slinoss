# AGENTS.md

## Commit Conventions

Use lightweight Conventional Commits for all new commits:

- `feat:` new capabilities or APIs
- `fix:` correctness fixes
- `refactor:` structure-preserving code changes
- `perf:` measurable performance work
- `test:` test-only changes
- `docs:` documentation-only changes
- `chore:` repo maintenance that does not affect behavior

Keep commit subjects short and specific. If a layout decision, kernel contract, or
benchmark change is important, make that visible in the subject line.

## Changelog Policy

Do not maintain a hand-written `CHANGELOG.md` during active development.

Rely on:

- disciplined commit messages
- milestone tags for important checkpoints

Add a real `CHANGELOG.md` only when one of these happens:

- the repo is prepared for a public release
- the paper artifact is being frozen
- the kernel/benchmark stack reaches a release candidate state
