# Changelog

All notable changes to this project will be documented in this file.

## [0.1.3] - 2025-08-04
### Added
- **Performance Optimizations**:
  - Precompute and broadcast `beta` values in `menchero` to replace row-wise adjustment loops.
  - Vectorized aggregation and safe division in `attributions` to replace `apply()` calls.
  - Vectorized selection-to-value mapping and effect swapping to remove `apply()` in stock-level adjustments.

### Changed
- **API Changes**:
  - Renamed internal columns (`A_`, `S_`, `I_` → `allocation_`, `selection_`, `interaction_`) for consistency.

### Removed
- Legacy row-wise loops and `apply()`-based code blocks.

## [0.1.2] - 2025-02-04
- Initial PyPI release with core Menchero multiperiod smoothing methods.
- Provided `sectorAttributions` and `stockAttributions` user-facing functions.

