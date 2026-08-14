"""Global GPR acquisition geometry and reproducibility settings.

These values match the research notebook so that the pre-trained U-Net
(128 x 128 input) remains compatible with the generated B-scans.
"""

# ------------------------------------------------------------------
# Acquisition geometry
# ------------------------------------------------------------------
GPR_NT = 128    # time samples per trace
GPR_NX = 128    # traces per B-scan
GPR_DT = 0.6    # ns per time sample
GPR_DX = 0.08   # m per trace -> 10.24 m line length

# ------------------------------------------------------------------
# Reproducible seed ranges (disjoint ranges prevent data leakage)
# ------------------------------------------------------------------
GLOBAL_SEED = 42
TRAIN_SEEDS = range(1000, 1200)
VAL_SEEDS = range(2000, 2050)
TEST_SEEDS = range(3000, 3050)

# ------------------------------------------------------------------
# Hyperbola detection / inversion settings
# ------------------------------------------------------------------
V_GRID = [round(v, 3) for v in [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]]
