# Changelog

All notable changes to this project are documented in this file.

## [0.2.0] - 2026-02-15
### Added
- GitHub publishing essentials: `LICENSE`, `CITATION.cff`, `requirements.txt`, `environment.yml`.
- Reproducibility entrypoint: `reproduce.sh` and `make reproduce`.
- Documentation assets in `docs/figures/`.
- CI workflow for linting, formatting, and tests.

### Changed
- Renamed key modules to clearer names (`run_experiment.py`, `trust_region_solver.py`, `mip_subproblem.py`, etc.).
- Added compatibility wrappers under legacy filenames.
- Refined README with thesis context, results summary, and limitations.

## [0.1.0] - 2025-05-29
### Added
- Initial research implementation for TV-regularized integer optimal control.
