# AdaAttn - Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (December 28, 2025)
- Initial project structure and scaffolding
- Memory-bank documentation system
- Core package structure under `src/adaattn/`
- Configuration files (pyproject.toml, .editorconfig, .gitignore)
- Docker development environment
- VS Code workspace settings
- GitHub templates and workflows placeholder
- **Virtual environment setup with core dependencies**
- **Comprehensive unit test suite (96 tests passing)**
  - `test_precision.py`: 13 tests for precision control
  - `test_entropy.py`: 5 tests for entropy estimation
  - `test_low_rank.py`: 9 tests for low-rank approximations (fixed tolerance)
  - `test_utils.py`: 20 tests for utility functions
  - `test_attention_base.py`: 21 tests for base classes & config
  - `test_adaptive_precision.py`: 12 tests for adaptive precision attention
  - `test_adaptive_rank.py`: 16 tests for adaptive rank attention ⭐ NEW
- **Integration test suite (10 tests, 1 skipped - no CUDA)**
  - `test_attention_pipeline.py`: End-to-end attention pipeline tests
- **GitHub Actions CI/CD workflow**
  - Python 3.9-3.12 matrix testing
  - Unit and integration tests
  - Coverage reporting with Codecov
  - Linting with black, ruff, mypy
- **Fixed import errors in linalg module**
- **Package installation in editable mode**
- **Coverage reporting: 35.81% (target: 80%+)**

### Changed (December 28, 2025)
- Fixed mermaid diagram syntax in README.md (sequence diagram)
- Updated pyproject.toml license configuration for modern setuptools
- Corrected function imports in `src/adaattn/linalg/__init__.py`
  - Changed `entropy_based_rank` to `entropy_based_rank_hint`
  - Removed non-existent `softmax_entropy` import
  - Added missing `estimate_entropy` import
- Updated project plan with accurate completion tracking (Phase 1: 90% complete)
- Improved test reliability with adjusted tolerances

### Deprecated
- N/A

### Removed
- N/A

### Fixed (December 28, 2025)
- License configuration error in pyproject.toml (removed classifier, used license.text)
- Import errors preventing test execution
- Mermaid sequence diagram rendering issues in README.md

### Security
- N/A

---

## Change Log Entry Template

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- Feature/component added
- Testing notes: [describe tests run]
- Contributors: [names]

### Changed
- Feature/component modified
- Brief description of changes
- Testing notes: [describe tests run]

### Fixed
- Bug fixed
- Root cause analysis
- Testing notes: [verification steps]
```

---

## Version History

| Version | Date | Status         | Notes                 |
| ------- | ---- | -------------- | --------------------- |
| 0.1.0   | TBD  | In Development | Initial alpha release |
