# Recommended GitHub repository settings

Some protections cannot be committed as files. Apply this checklist in GitHub after merging the repository infrastructure.

## General

- Set `main` as the default branch.
- Add topics: `pytorch`, `cnn`, `deep-learning`, `computer-vision`, `from-scratch`, `research`.
- Enable Issues, Discussions, Releases, and the social preview image.
- Disable wiki if documentation remains in the repository.
- Enable automatic deletion of merged branches.

## Ruleset for `main`

- Require pull requests and at least one approving review.
- Require CODEOWNERS review for owned paths.
- Require conversation resolution and block force pushes/deletions.
- Require linear history and up-to-date branches.
- Require these checks: `quality`, both `tests` matrix jobs, `package`, `dependency-review`, and Docker `build` when applicable.
- Allow Dependabot pull requests to run CI; do not grant workflows unnecessary write permissions.

## Security and analysis

- Enable the dependency graph, Dependabot alerts, and Dependabot security updates.
- Enable private vulnerability reporting.
- Enable secret scanning and push protection when available.
- Use the committed CodeQL advanced workflow; do not enable a duplicate default CodeQL setup.
- Review and export the dependency graph/SBOM before releases.

## Actions

- Set the default `GITHUB_TOKEN` permission to read-only.
- Allow GitHub-authored, Docker-authored, and verified actions required by the workflows.
- Require approval for first-time external contributors.
- Keep workflow retention appropriate for the project and protect environment secrets if publishing packages later.

## Releases

- Tag releases as `vMAJOR.MINOR.PATCH`.
- Generate notes using `.github/release.yml` and copy user-visible changes into `CHANGELOG.md`.
- Build artifacts from a clean tagged commit; add PyPI or container publishing only after trusted-publisher/OIDC configuration is ready.
