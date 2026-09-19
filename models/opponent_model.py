"""
Opponent behavior model.

This module converts observed player statistics such as VPIP, PFR,
and 3-bet frequency into position-adjusted action probabilities
for each starting-hand class.

The current model is heuristic.

The position multipliers, logistic slopes, and calibration settings
are defined in config.py so that the modeling assumptions are kept
separate from the implementation logic.
"""


import math

from config import (
    CALIBRATION_HIGH,
    CALIBRATION_ITERATIONS,
    CALIBRATION_LOW,
    CALL_SLOPE,
    MIN_ACTION_LIKELIHOOD,
    PFR_MAX,
    PFR_MIN,
    PFR_POSITION_FACTOR,
    RAISE_SLOPE,
    THREEBET_MAX,
    THREEBET_MIN,
    THREEBET_POSITION_FACTOR,
    THREEBET_SLOPE,
    VPIP_MAX,
    VPIP_MIN,
    VPIP_POSITION_FACTOR,
)


# ============================================================
# CLAMP
# ============================================================

def clamp(
    value,
    minimum,
    maximum,
):
    """
    Restrict a value to a specified range.

    Examples:

        clamp(0.5, 0, 1) -> 0.5
        clamp(-1, 0, 1)  -> 0
        clamp(2, 0, 1)   -> 1
    """

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


# ============================================================
# POSITION-ADJUSTED PLAYER STATISTICS
# ============================================================

def adjust_rates_for_position(
    vpip_rate,
    pfr_rate,
    three_bet_rate,
    position,
):
    """
    Adjust a player's overall preflop statistics based on position.

    Parameters
    ----------
    vpip_rate:
        Overall VPIP as a decimal.

        Example:
            25% VPIP is represented as 0.25.

    pfr_rate:
        Overall preflop raise frequency as a decimal.

    three_bet_rate:
        Overall 3-bet frequency as a decimal.

    position:
        One of:

        UTG
        HJ
        CO
        BTN
        SB
        BB

    Returns
    -------
    A tuple containing:

        (
            adjusted_vpip,
            adjusted_pfr,
            adjusted_three_bet
        )

    The position multipliers themselves are defined in config.py.
    """

    # Adjust VPIP based on position.
    #
    # A factor below 1 tightens the modeled frequency.
    # A factor above 1 widens it.

    adjusted_vpip = clamp(
        vpip_rate
        * VPIP_POSITION_FACTOR[
            position
        ],
        VPIP_MIN,
        VPIP_MAX,
    )


    # Adjust preflop raise frequency based on position.

    adjusted_pfr = clamp(
        pfr_rate
        * PFR_POSITION_FACTOR[
            position
        ],
        PFR_MIN,
        PFR_MAX,
    )


    # A player's preflop raise frequency cannot logically
    # exceed their VPIP.
    #
    # Therefore:
    #
    # PFR <= VPIP

    adjusted_pfr = min(
        adjusted_pfr,
        adjusted_vpip,
    )


    # Adjust 3-bet frequency based on position.

    adjusted_three_bet = clamp(
        three_bet_rate
        * THREEBET_POSITION_FACTOR[
            position
        ],
        THREEBET_MIN,
        THREEBET_MAX,
    )


    return (
        adjusted_vpip,
        adjusted_pfr,
        adjusted_three_bet,
    )


# ============================================================
# SIGMOID FUNCTION
# ============================================================

def sigmoid(x):
    """
    Convert a raw value into a smooth probability between 0 and 1.

    The opponent model uses a sigmoid so that action probabilities
    change gradually as hand strength increases.

    Rather than using a hard cutoff such as:

        raise every hand above 60% equity

    the model can instead represent:

        weak hand   -> low raise probability
        medium hand -> medium raise probability
        strong hand -> high raise probability
    """

    # Standard sigmoid expression for non-negative values.

    if x >= 0:

        return (
            1
            /
            (
                1
                + math.exp(-x)
            )
        )


    # This mathematically equivalent form is used for negative x.
    #
    # It avoids numerical overflow that could occur from
    # calculating math.exp(-x) when x is very negative.

    exp_x = math.exp(x)


    return (
        exp_x
        /
        (
            1
            + exp_x
        )
    )


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def calibrate_threshold(
    target_frequency,
    slope,
    hand_classes,
    hand_strengths,
    combo_counts,
):
    """
    Find the hand-strength threshold that makes the modeled action
    frequency approximately equal the player's observed frequency.

    Example:

        If adjusted PFR is 20%, this function finds a threshold
        such that the weighted average:

            P(raise | hand)

        across all available physical hand combinations is
        approximately 20%.

    Binary search is used because changing the threshold moves
    the modeled frequency in a predictable direction.
    """

    # These search limits are stored in config.py.

    low = CALIBRATION_LOW
    high = CALIBRATION_HIGH


    # Repeatedly narrow the search interval.

    for iteration in range(
        CALIBRATION_ITERATIONS
    ):

        midpoint = (
            low + high
        ) / 2


        weighted_total = 0
        combination_total = 0


        # Calculate the implied action frequency across every
        # available starting-hand class.

        for hand_class in hand_classes:

            # Number of remaining physical combinations.
            #
            # Examples before blockers:
            #
            # AA  -> 6 combinations
            # AKs -> 4 combinations
            # AKo -> 12 combinations

            count = (
                combo_counts[
                    hand_class
                ]
            )


            # Baseline Monte Carlo strength estimate for this
            # starting-hand class.

            strength = (
                hand_strengths[
                    hand_class
                ]
            )


            # Convert hand strength relative to the current
            # threshold into an action probability.

            probability = sigmoid(
                slope
                * (
                    strength
                    - midpoint
                )
            )


            # Weight by physical combo count so a hand class with
            # more possible combinations contributes more probability
            # mass than one with fewer combinations.

            weighted_total += (
                probability
                * count
            )


            combination_total += (
                count
            )


        current_frequency = (
            weighted_total
            / combination_total
        )


        # If the model predicts the action too frequently,
        # increase the threshold.
        #
        # A higher threshold requires stronger hands to receive
        # high action probabilities.

        if (
            current_frequency
            > target_frequency
        ):

            low = midpoint


        # If the modeled action is too rare, lower the threshold.

        else:

            high = midpoint


    return (
        low + high
    ) / 2


# ============================================================
# ACTION LIKELIHOOD MODEL
# ============================================================

def build_action_likelihoods(
    action,
    vpip_rate,
    pfr_rate,
    three_bet_rate,
    hand_classes,
    hand_strengths,
    combo_counts,
    slope_scale=1.0,
):
    """
    Estimate P(action | hand class) for every starting-hand class.

    Supported actions:

        RAISE
        CALL
        3BET

    These probabilities are later used in the Bayesian-style
    opponent range update:

        P(hand | action)
            proportional to
        P(action | hand) * P(hand)

    Important:
        This is currently a heuristic behavioral model.

        The logistic slopes, position multipliers, and calibration
        assumptions are not learned from real poker hand data.
    """

    likelihoods = {}


    # ========================================================
    # RAISE MODEL
    # ========================================================

    if action == "RAISE":

        # The configured raise slope controls how sharply
        # raise probability increases with hand strength.

        slope = (
            RAISE_SLOPE
            * slope_scale
        )


        # Calibrate the threshold so the weighted average
        # raise probability matches the player's adjusted PFR.

        threshold = (
            calibrate_threshold(
                pfr_rate,
                slope,
                hand_classes,
                hand_strengths,
                combo_counts,
            )
        )


        for hand_class in hand_classes:

            # Estimate:
            #
            # P(raise | this hand)

            likelihoods[
                hand_class
            ] = sigmoid(
                slope
                * (
                    hand_strengths[
                        hand_class
                    ]
                    - threshold
                )
            )


    # ========================================================
    # CALL MODEL
    # ========================================================

    elif action == "CALL":

        slope = (
            CALL_SLOPE
            * slope_scale
        )


        # VPIP approximates the frequency with which a player
        # voluntarily enters the pot.

        vpip_threshold = (
            calibrate_threshold(
                vpip_rate,
                slope,
                hand_classes,
                hand_strengths,
                combo_counts,
            )
        )


        # PFR approximates the subset of those hands that the
        # player raises.

        raise_threshold = (
            calibrate_threshold(
                pfr_rate,
                slope,
                hand_classes,
                hand_strengths,
                combo_counts,
            )
        )


        for hand_class in hand_classes:

            strength = (
                hand_strengths[
                    hand_class
                ]
            )


            # Approximate:
            #
            # P(voluntarily play | hand)

            play_probability = sigmoid(
                slope
                * (
                    strength
                    - vpip_threshold
                )
            )


            # Approximate:
            #
            # P(raise | hand)

            raise_probability = sigmoid(
                slope
                * (
                    strength
                    - raise_threshold
                )
            )


            # Simplifying assumption:
            #
            # VPIP roughly consists of calls + raises.
            #
            # Therefore:
            #
            # P(call | hand)
            #     ~= P(play | hand)
            #        - P(raise | hand)

            call_probability = (
                play_probability
                - raise_probability
            )


            # Keep every hand's likelihood slightly above zero.
            #
            # An exact zero would completely eliminate that hand
            # from the posterior range.

            likelihoods[
                hand_class
            ] = max(
                call_probability,
                MIN_ACTION_LIKELIHOOD,
            )


    # ========================================================
    # 3-BET MODEL
    # ========================================================

    elif action == "3BET":

        # The 3-bet slope is intentionally steeper in the current
        # heuristic configuration.

        slope = (
            THREEBET_SLOPE
            * slope_scale
        )


        # Calibrate the threshold so the average modeled 3-bet
        # probability matches the player's adjusted 3-bet rate.

        threshold = (
            calibrate_threshold(
                three_bet_rate,
                slope,
                hand_classes,
                hand_strengths,
                combo_counts,
            )
        )


        for hand_class in hand_classes:

            # Estimate:
            #
            # P(3-bet | this hand)

            likelihoods[
                hand_class
            ] = sigmoid(
                slope
                * (
                    hand_strengths[
                        hand_class
                    ]
                    - threshold
                )
            )


    return (
        likelihoods
    )