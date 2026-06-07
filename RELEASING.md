# Releasing

This project ships to [PyPI](https://pypi.org/project/trainscope/) via **PyPI
Trusted Publishing** (OIDC) from GitHub Actions — no API tokens or long-lived
secrets live in the repository. Releases are cut from `main` by tagging.

## Versioning policy

trainscope follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** — incompatible public API changes (the importable API, the
  `Finding` schema, the `steps.jsonl` / `run.json` on-disk format, or CLI
  behavior scripts depend on).
- **MINOR** — backward-compatible additions (new analyzers, new diagnosis
  rules, new CLI subcommands/flags, new integrations).
- **PATCH** — backward-compatible bug fixes and tuning that doesn't change
  documented behavior.

While the project is pre-1.0 (`0.y.z`), the public surface may still move; we'll
bump MINOR for anything a user could notice and call it out in the changelog.

The version lives in **one place**: `trainscope/__init__.py` (`__version__`).
`pyproject.toml` reads it dynamically via Hatch, so there is nothing to keep in
sync. New diagnosis codes are additive and don't themselves require a major bump.

## One-time setup (maintainer)

Configure Trusted Publishing once per index so the workflow can authenticate:

1. On **PyPI** → project `trainscope` → *Settings → Publishing → Add a pending
   publisher*:
   - Owner: `Sumu004`, Repository: `trainscope`
   - Workflow: `publish.yml`
   - Environment: `pypi`
2. Repeat on **TestPyPI** with environment `testpypi` (for dry runs).
3. In GitHub → *Settings → Environments*, create `pypi` and `testpypi`
   (optionally add required reviewers to `pypi` so a human approves each push).

No secrets are added; authentication is via short-lived OIDC tokens minted per
run.

## Cutting a release

1. **Land all changes** on `main` via PR; CI (lint + tests on the support
   matrix + `twine check`) must be green.
2. **Bump the version** in `trainscope/__init__.py`.
3. **Update `CHANGELOG.md`**: move items from `[Unreleased]` into a new
   `[X.Y.Z] - YYYY-MM-DD` section and refresh the compare links at the bottom.
4. **Open and merge** the release PR ("Release vX.Y.Z").
5. **(Optional) Dry-run to TestPyPI:** Actions → *Publish to PyPI* → *Run
   workflow* → target `testpypi`. Then verify in a clean venv:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ trainscope
   ```
6. **Tag and push** from the merged commit:
   ```bash
   git tag -a vX.Y.Z -m "trainscope vX.Y.Z"
   git push origin vX.Y.Z
   ```
   The tag push triggers `publish.yml`, which re-runs tests, builds the sdist +
   wheel, verifies the tag matches `__version__`, `twine check`s the artifacts,
   and publishes to PyPI via Trusted Publishing.
7. **Create the GitHub Release** from the tag, pasting the changelog section.

## After releasing

- Confirm the new version installs cleanly: `pip install trainscope==X.Y.Z`.
- Add a fresh empty `[Unreleased]` section back to `CHANGELOG.md`.

## Yanking

If a release is broken, **yank** it on PyPI (don't delete — deletion breaks
pinned installs) and ship a fixed PATCH release. Note the reason in the
changelog.
