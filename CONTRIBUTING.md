# Contributing

Thank you for improving this executable survey of CNN architectures.

## Development setup

1. Fork and clone the repository.
2. Create a branch from `main`.
3. Install the project and development tools:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   pre-commit install
   ```

4. Verify your environment:

   ```bash
   make check
   python -m famous_cnns list
   ```

## Contribution guidelines

- Keep architecture implementations readable and close to their papers.
- Put shared behavior in `famous_cnns`; keep architecture scripts as thin wrappers.
- Add or update tests for behavior changes.
- Do not commit datasets, checkpoints, credentials, generated plots, or build artifacts.
- Preserve backwards compatibility when practical and document intentional breaking changes.
- Use clear commit messages and keep pull requests focused.

## Adding an architecture

1. Add the implementation in its own folder.
2. Register it in `famous_cnns/factory.py` with task and input metadata.
3. Add `scripts/train.py` and `scripts/infer.py` wrappers.
4. Add a smoke test and a short architecture README.
5. Update the root architecture table and changelog.

## Pull requests

Complete the pull request template, link related issues, and confirm tests, lint, package build, and documentation. CI must pass before review. Security issues must follow [SECURITY.md](SECURITY.md), not the normal pull request process.
