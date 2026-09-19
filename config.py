"""
Central configuration for the Poker Decision Fragility project.

This file contains simulation settings and heuristic modeling
assumptions.

Keeping these values in one place makes the model easier to:

- understand
- reproduce
- experiment with
- eventually replace with values learned from poker data
"""


# ============================================================
# RANDOMNESS
# ============================================================

# Fixed during development so identical inputs produce
# reproducible Monte Carlo results.
RANDOM_SEED = 42


# ============================================================
# BASELINE HAND-STRENGTH SIMULATION
# ============================================================

# Number of Monte Carlo trials used to estimate the baseline
# strength of each of the 169 starting-hand classes.
BASELINE_TRIALS_PER_CLASS = 1000


# Cache filename automatically changes if the baseline trial
# count changes.
HAND_STRENGTH_CACHE = (
    "hand_strengths_"
    + str(BASELINE_TRIALS_PER_CLASS)
    + ".json"
)


# ============================================================
# DECISION FRAGILITY SIMULATION
# ============================================================

# Number of plausible opponent models generated for each
# Decision Fragility calculation.
NUMBER_OF_MODELS = 50


# Number of Monte Carlo deals run for each plausible model.
TRIALS_PER_MODEL = 6000


# Hero + 1 modeled opponent + 4 random opponents = 6 players.
RANDOM_OPPONENTS = 4


# ============================================================
# BETA UNCERTAINTY MODEL
# ============================================================

# The current Beta model acts approximately like a uniform
# Beta(1, 1) prior before observed player statistics are added.
BETA_PRIOR_ALPHA = 1.0
BETA_PRIOR_BETA = 1.0


# ============================================================
# POSITION ADJUSTMENT FACTORS
# ============================================================

# These are heuristic assumptions.
#
# Values below 1 tighten a player's modeled frequency.
# Values above 1 widen it.
#
# These should eventually be candidates for replacement with
# estimates learned from real poker hand histories.

VPIP_POSITION_FACTOR = {
    "UTG": 0.70,
    "HJ": 0.85,
    "CO": 1.05,
    "BTN": 1.25,
    "SB": 1.20,
    "BB": 1.35,
}


PFR_POSITION_FACTOR = {
    "UTG": 0.65,
    "HJ": 0.82,
    "CO": 1.05,
    "BTN": 1.30,
    "SB": 1.15,
    "BB": 0.80,
}


THREEBET_POSITION_FACTOR = {
    "UTG": 0.80,
    "HJ": 0.90,
    "CO": 1.00,
    "BTN": 1.10,
    "SB": 1.20,
    "BB": 1.25,
}


# ============================================================
# POSITION-ADJUSTED RATE LIMITS
# ============================================================

VPIP_MIN = 0.001
VPIP_MAX = 0.95

PFR_MIN = 0.001
PFR_MAX = 0.90

THREEBET_MIN = 0.001
THREEBET_MAX = 0.50


# ============================================================
# LOGISTIC ACTION MODEL
# ============================================================

# Larger slopes make the transition from low action probability
# to high action probability more abrupt.

RAISE_SLOPE = 18
CALL_SLOPE = 16
THREEBET_SLOPE = 24


# Every plausible opponent model receives a slightly different
# slope to represent uncertainty in the exact shape of the range.
SLOPE_SCALE_MIN = 0.85
SLOPE_SCALE_MAX = 1.15


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

# Binary-search bounds used when finding the hand-strength
# threshold that matches a player's observed action frequency.
CALIBRATION_LOW = -1.0
CALIBRATION_HIGH = 2.0


# Number of binary-search iterations.
CALIBRATION_ITERATIONS = 70


# ============================================================
# NUMERICAL SAFETY
# ============================================================

# Prevent CALL likelihoods from becoming exactly zero, which
# would completely eliminate that hand from the posterior range.
MIN_ACTION_LIKELIHOOD = 1e-12