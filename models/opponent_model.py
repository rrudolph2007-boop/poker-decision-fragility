import math


# ============================================================
# POSITION ADJUSTMENT FACTORS
# ============================================================

# These multipliers are heuristic assumptions.
#
# They are not learned from real poker data and they are not
# solver-derived GTO frequencies.
#
# The purpose is to approximate the general idea that players
# tend to play tighter ranges from early positions and wider
# ranges from later positions.
#
# These values can eventually be replaced by machine-learning
# estimates trained on real hand histories.

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
# CLAMP
# ============================================================

def clamp(
    value,
    minimum,
    maximum,
):
    """
    Restrict a number to a specified range.

    Example:
        clamp(1.2, 0, 1) returns 1
        clamp(-0.2, 0, 1) returns 0
        clamp(0.4, 0, 1) returns 0.4
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

    Parameters:
        vpip_rate:
            Overall VPIP as a decimal.
            Example: 25% VPIP is represented as 0.25.

        pfr_rate:
            Overall preflop raise frequency as a decimal.

        three_bet_rate:
            Overall 3-bet frequency as a decimal.

        position:
            One of:
            UTG, HJ, CO, BTN, SB, BB

    Returns:
        A tuple containing:

        (
            adjusted_vpip,
            adjusted_pfr,
            adjusted_three_bet
        )

    These adjustments are currently heuristic assumptions.
    """

    # Multiply the player's overall VPIP by the positional factor.
    #
    # For example, a BTN factor above 1 widens the modeled range,
    # while a UTG factor below 1 tightens it.

    adjusted_vpip = clamp(
        vpip_rate
        * VPIP_POSITION_FACTOR[
            position
        ],
        0.001,
        0.95,
    )


    # Apply the same idea to the player's preflop raise frequency.

    adjusted_pfr = clamp(
        pfr_rate
        * PFR_POSITION_FACTOR[
            position
        ],
        0.001,
        0.90,
    )


    # A player cannot raise preflop more often than they voluntarily
    # enter the pot.
    #
    # Therefore:
    #
    # PFR <= VPIP

    adjusted_pfr = min(
        adjusted_pfr,
        adjusted_vpip,
    )


    # Adjust 3-bet frequency for position.

    adjusted_three_bet = clamp(
        three_bet_rate
        * THREEBET_POSITION_FACTOR[
            position
        ],
        0.001,
        0.50,
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
    Convert a raw number into a smooth probability from 0 to 1.

    The opponent model uses the sigmoid function so that stronger
    hands gradually become more likely to take an action rather
    than using a hard cutoff.

    Instead of saying:

        raise every hand above 60% equity

    the model can say:

        weak hand   -> low raise probability
        medium hand -> medium raise probability
        strong hand -> high raise probability
    """

    # Standard sigmoid form.
    #
    # This version is safe when x is positive.

    if x >= 0:

        return (
            1
            /
            (
                1
                + math.exp(-x)
            )
        )


    # For very negative numbers, math.exp(-x) can become extremely
    # large and cause numerical overflow.
    #
    # This mathematically equivalent version avoids that problem.

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
    Find the hand-strength threshold that makes the model's average
    action frequency approximately match the player's observed rate.

    Example:

        Suppose the player's adjusted PFR is 20%.

        The model needs to find a threshold such that, across all
        possible starting-hand combinations:

            average P(raise | hand) ~= 20%

    The function uses binary search to find that threshold.

    The resulting threshold separates weaker hands from stronger hands,
    but the sigmoid keeps the transition smooth rather than creating
    a hard range cutoff.
    """

    # Search across a deliberately wide range of possible thresholds.

    low = -1.0
    high = 2.0


    # 70 binary-search iterations gives much more precision than
    # this project actually needs, while still being extremely fast.

    for iteration in range(
        70
    ):

        midpoint = (
            low + high
        ) / 2


        weighted_total = 0
        combination_total = 0


        # Calculate the average modeled action probability across
        # every available physical starting-hand combination.

        for hand_class in hand_classes:

            # Number of physical card combinations remaining for
            # this hand class after blockers.
            #
            # Example:
            #
            # AA  = normally 6 combinations
            # AKs = normally 4 combinations
            # AKo = normally 12 combinations

            count = (
                combo_counts[
                    hand_class
                ]
            )


            # Baseline Monte Carlo equity of this starting-hand class.

            strength = (
                hand_strengths[
                    hand_class
                ]
            )


            # Convert the hand's strength relative to the threshold
            # into an action probability.

            probability = sigmoid(
                slope
                * (
                    strength
                    - midpoint
                )
            )


            # Weight the probability by the number of physical
            # combinations represented by this hand class.

            weighted_total += (
                probability
                * count
            )


            combination_total += (
                count
            )


        # This is the action frequency implied by the current threshold.

        current_frequency = (
            weighted_total
            / combination_total
        )


        # If the model currently predicts the action too frequently,
        # increase the threshold.
        #
        # A higher threshold makes the action require stronger hands,
        # which lowers the overall action frequency.

        if (
            current_frequency
            > target_frequency
        ):

            low = midpoint


        # If the modeled action is too rare, lower the threshold.
        #
        # This allows weaker hands to receive higher probabilities.

        else:

            high = midpoint


    # Return the midpoint of the final search interval.

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

    In other words:

        If the opponent holds a particular hand,
        how likely are they to take the observed action?

    The supported actions are:

        RAISE
        CALL
        3BET

    These probabilities are later used in the Bayesian-style range
    update:

        P(hand | action)
            proportional to
        P(action | hand) * P(hand)

    Important:
        This is currently a heuristic behavioral model.

        The logistic curve shapes and slope values are assumptions,
        not probabilities learned from real poker decisions.
    """

    likelihoods = {}


    # ========================================================
    # RAISE MODEL
    # ========================================================

    if action == "RAISE":

        # The slope controls how sharply action probability rises
        # as hand strength increases.
        #
        # A larger slope produces a more abrupt transition between
        # hands that rarely raise and hands that frequently raise.

        slope = (
            18
            * slope_scale
        )


        # Find the threshold that makes the overall modeled raise
        # frequency match the opponent's adjusted PFR.

        threshold = calibrate_threshold(
            pfr_rate,
            slope,
            hand_classes,
            hand_strengths,
            combo_counts,
        )


        for hand_class in hand_classes:

            # Estimate:
            #
            # P(raise | this hand class)

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
            16
            * slope_scale
        )


        # First estimate the threshold for voluntarily entering
        # the pot at all.

        vpip_threshold = (
            calibrate_threshold(
                vpip_rate,
                slope,
                hand_classes,
                hand_strengths,
                combo_counts,
            )
        )


        # Then estimate the threshold for raising.

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
            # VPIP consists approximately of calls plus raises.
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


            # Prevent the likelihood from becoming exactly zero.
            #
            # A zero likelihood would completely eliminate a hand
            # from the inferred posterior range.

            likelihoods[
                hand_class
            ] = max(
                call_probability,
                1e-12,
            )


    # ========================================================
    # 3-BET MODEL
    # ========================================================

    elif action == "3BET":

        # The 3-bet model uses a steeper curve because 3-betting
        # generally represents a narrower and stronger subset of
        # the player's preflop strategy in this heuristic model.

        slope = (
            24
            * slope_scale
        )


        # Calibrate the model so the average probability matches
        # the opponent's adjusted 3-bet frequency.

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
            # P(3-bet | this hand class)

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