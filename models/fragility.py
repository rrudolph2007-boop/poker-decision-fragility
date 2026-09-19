"""
Decision Fragility simulation engine.

This module measures how sensitive a poker decision is to uncertainty
about one modeled opponent's range.

For each plausible opponent model, the simulation:

1. Samples plausible VPIP, PFR, and 3-bet statistics.
2. Builds an uncertain range for the modeled opponent.
3. Samples the modeled opponent from that range.
4. Deals additional opponents uniformly random hands.
5. Deals a five-card board.
6. Calculates Hero's multiway equity.
7. Calculates the expected value of calling.
8. Records whether the call is profitable.

Decision Fragility measures disagreement across plausible models.

A fragility score near 0 means the models mostly agree.
A fragility score near 1 means the models are divided.
"""


import bisect
import random
import time

from itertools import accumulate

from config import (
    BETA_PRIOR_ALPHA,
    BETA_PRIOR_BETA,
    SLOPE_SCALE_MAX,
    SLOPE_SCALE_MIN,
)

from models.opponent_model import (
    adjust_rates_for_position,
    build_action_likelihoods,
)

from poker.cards import full_deck
from poker.evaluator import evaluate_hand


# ============================================================
# BETA UNCERTAINTY
# ============================================================

def sample_rate_from_beta(
    estimated_rate,
    sample_size,
):
    """
    Sample a plausible version of an observed player statistic.

    The Beta distribution represents uncertainty around statistics
    such as VPIP, PFR, and 3-bet frequency.

    Smaller observed samples produce more variation.

    Larger observed samples produce values more tightly clustered
    around the observed rate.

    The prior parameters are defined in config.py.
    """

    alpha = (
        BETA_PRIOR_ALPHA
        + estimated_rate
        * sample_size
    )

    beta = (
        BETA_PRIOR_BETA
        + (
            1
            - estimated_rate
        )
        * sample_size
    )

    return random.betavariate(
        alpha,
        beta,
    )


# ============================================================
# WEIGHTED OPPONENT RANGE
# ============================================================

def build_weighted_combo_population(
    hand_classes,
    available_combos,
    action_likelihoods,
):
    """
    Convert hand-class action likelihoods into a weighted population
    of actual physical two-card combinations.

    Example:

        AKs has 4 physical combinations.
        AKo has 12 physical combinations.

    Every physical combo receives the action likelihood assigned
    to its hand class.
    """

    combo_population = []
    combo_weights = []

    for hand_class in hand_classes:

        weight = (
            action_likelihoods[
                hand_class
            ]
        )

        for combo in available_combos[
            hand_class
        ]:

            combo_population.append(
                combo
            )

            combo_weights.append(
                weight
            )

    cumulative_weights = list(
        accumulate(
            combo_weights
        )
    )

    total_weight = (
        cumulative_weights[-1]
    )

    return (
        combo_population,
        cumulative_weights,
        total_weight,
    )


# ============================================================
# ACTION RATE FOR OUTPUT
# ============================================================

def get_model_action_rate(
    observed_action,
    model_vpip,
    model_pfr,
    model_three_bet,
):
    """
    Return the statistic associated with the modeled action.

    RAISE:
        Uses PFR.

    CALL:
        Approximated as VPIP minus PFR.

    3BET:
        Uses 3-bet frequency.
    """

    if observed_action == "RAISE":

        return (
            model_pfr
        )

    elif observed_action == "CALL":

        return max(
            model_vpip
            - model_pfr,
            0,
        )

    else:

        return (
            model_three_bet
        )


# ============================================================
# DECISION FRAGILITY ENGINE
# ============================================================

def run_decision_fragility_simulation(
    hero_hand,
    vpip,
    pfr,
    three_bet,
    observed_hands,
    position,
    observed_action,
    current_pot,
    call_amount,
    hand_classes,
    hand_strengths,
    available_combos,
    combo_counts,
    number_of_models,
    trials_per_model,
    random_opponents,
    show_progress=True,
):
    """
    Run the full multi-model Decision Fragility simulation.

    One opponent is modeled using uncertain statistics and an inferred
    range.

    The remaining opponents receive uniformly random legal hands.

    Returns a dictionary containing the model-level results and
    summary statistics.
    """

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if number_of_models <= 0:

        raise ValueError(
            "number_of_models must be positive."
        )

    if trials_per_model <= 0:

        raise ValueError(
            "trials_per_model must be positive."
        )

    if random_opponents < 0:

        raise ValueError(
            "random_opponents cannot be negative."
        )


    # --------------------------------------------------------
    # BREAK-EVEN EQUITY
    # --------------------------------------------------------

    # Minimum equity Hero needs for calling to have
    # non-negative expected value.

    break_even_equity = (
        call_amount
        /
        (
            current_pot
            + call_amount
        )
    )


    # --------------------------------------------------------
    # RESULT STORAGE
    # --------------------------------------------------------

    model_equities = []
    model_evs = []
    model_action_rates = []

    profitable_models = 0


    # Hero's known cards cannot appear anywhere else.

    base_deck = [
        card
        for card in full_deck
        if card not in hero_hand
    ]


    simulation_start = (
        time.perf_counter()
    )


    # ========================================================
    # PLAUSIBLE OPPONENT MODELS
    # ========================================================

    for model_number in range(
        1,
        number_of_models + 1,
    ):

        # ----------------------------------------------------
        # SAMPLE PLAUSIBLE PLAYER STATISTICS
        # ----------------------------------------------------

        sampled_vpip = (
            sample_rate_from_beta(
                vpip,
                observed_hands,
            )
        )

        sampled_pfr = (
            sample_rate_from_beta(
                pfr,
                observed_hands,
            )
        )

        sampled_three_bet = (
            sample_rate_from_beta(
                three_bet,
                observed_hands,
            )
        )


        # PFR cannot logically exceed VPIP.

        sampled_pfr = min(
            sampled_pfr,
            sampled_vpip,
        )


        # Apply the heuristic position adjustments from
        # models/opponent_model.py.

        (
            model_vpip,
            model_pfr,
            model_three_bet,
        ) = adjust_rates_for_position(
            sampled_vpip,
            sampled_pfr,
            sampled_three_bet,
            position,
        )


        # ----------------------------------------------------
        # RANGE-SHAPE UNCERTAINTY
        # ----------------------------------------------------

        # Different plausible models receive slightly different
        # logistic curve steepness.
        #
        # The bounds are stored in config.py.

        slope_scale = (
            random.uniform(
                SLOPE_SCALE_MIN,
                SLOPE_SCALE_MAX,
            )
        )


        model_likelihoods = (
            build_action_likelihoods(
                observed_action,
                model_vpip,
                model_pfr,
                model_three_bet,
                hand_classes,
                hand_strengths,
                combo_counts,
                slope_scale,
            )
        )


        # ----------------------------------------------------
        # BUILD PHYSICAL MODELED-OPPONENT RANGE
        # ----------------------------------------------------

        (
            combo_population,
            cumulative_weights,
            total_weight,
        ) = build_weighted_combo_population(
            hand_classes,
            available_combos,
            model_likelihoods,
        )


        equity_total = 0


        # ====================================================
        # MONTE CARLO DEALS FOR THIS MODEL
        # ====================================================

        for simulation in range(
            trials_per_model
        ):

            # ------------------------------------------------
            # SAMPLE MODELED OPPONENT
            # ------------------------------------------------

            random_weight = (
                random.random()
                * total_weight
            )

            combo_index = (
                bisect.bisect_left(
                    cumulative_weights,
                    random_weight,
                )
            )

            modeled_opponent_hand = (
                combo_population[
                    combo_index
                ]
            )


            simulation_deck = (
                base_deck.copy()
            )


            for card in modeled_opponent_hand:

                simulation_deck.remove(
                    card
                )


            # ------------------------------------------------
            # DEAL ADDITIONAL RANDOM OPPONENTS
            # ------------------------------------------------

            random_opponent_hands = []


            for opponent_number in range(
                random_opponents
            ):

                random_hand = (
                    random.sample(
                        simulation_deck,
                        2,
                    )
                )

                random_opponent_hands.append(
                    random_hand
                )

                # Remove dealt cards so every physical card
                # can only exist in one place.

                for card in random_hand:

                    simulation_deck.remove(
                        card
                    )


            # ------------------------------------------------
            # DEAL COMMUNITY CARDS
            # ------------------------------------------------

            board = random.sample(
                simulation_deck,
                5,
            )


            # ------------------------------------------------
            # HERO SCORE
            # ------------------------------------------------

            hero_score = (
                evaluate_hand(
                    hero_hand
                    + board
                )
            )


            # ------------------------------------------------
            # MODELED OPPONENT SCORE
            # ------------------------------------------------

            modeled_opponent_score = (
                evaluate_hand(
                    modeled_opponent_hand
                    + board
                )
            )


            opponent_scores = [
                modeled_opponent_score
            ]


            # ------------------------------------------------
            # RANDOM OPPONENT SCORES
            # ------------------------------------------------

            for random_hand in random_opponent_hands:

                random_opponent_score = (
                    evaluate_hand(
                        random_hand
                        + board
                    )
                )

                opponent_scores.append(
                    random_opponent_score
                )


            # ------------------------------------------------
            # MULTIWAY POT EQUITY
            # ------------------------------------------------

            all_scores = (
                [hero_score]
                + opponent_scores
            )

            best_score = max(
                all_scores
            )


            # If another player has a stronger hand,
            # Hero receives zero equity for this deal.

            if hero_score < best_score:

                hero_equity_share = 0


            else:

                # Hero has the best score, but may be tied with
                # one or more opponents.

                number_of_winners = 0

                for score in all_scores:

                    if score == best_score:

                        number_of_winners += 1


                # Examples:
                #
                # Hero wins outright      -> 1
                # Hero ties one player    -> 1/2
                # Hero ties two players   -> 1/3

                hero_equity_share = (
                    1
                    / number_of_winners
                )


            equity_total += (
                hero_equity_share
            )


        # ====================================================
        # MODEL EQUITY
        # ====================================================

        model_equity = (
            equity_total
            / trials_per_model
        )


        # ====================================================
        # EXPECTED VALUE
        # ====================================================

        # Net expected value of calling:
        #
        # EV = equity * final pot - call amount

        model_ev = (
            model_equity
            * (
                current_pot
                + call_amount
            )
            - call_amount
        )


        model_equities.append(
            model_equity
        )

        model_evs.append(
            model_ev
        )


        model_action_rate = (
            get_model_action_rate(
                observed_action,
                model_vpip,
                model_pfr,
                model_three_bet,
            )
        )


        model_action_rates.append(
            model_action_rate
        )


        # ----------------------------------------------------
        # PROFITABILITY CLASSIFICATION
        # ----------------------------------------------------

        if model_ev > 0:

            profitable_models += 1


        if show_progress:

            print(
                "Model",
                model_number,
                "/",
                number_of_models,
                "| Six Player Equity:",
                round(
                    model_equity * 100,
                    2,
                ),
                "%",
                "| EV:",
                round(
                    model_ev,
                    2,
                ),
            )


    simulation_end = (
        time.perf_counter()
    )


    # ========================================================
    # DECISION FRAGILITY
    # ========================================================

    profitable_fraction = (
        profitable_models
        / number_of_models
    )


    # q = fraction of models where calling has positive EV.
    #
    # F = 1 - |2q - 1|
    #
    # F = 0 when all models agree.
    # F = 1 when the models are split 50/50.

    fragility_score = (
        1
        - abs(
            2
            * profitable_fraction
            - 1
        )
    )


    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    average_equity = (
        sum(model_equities)
        / len(model_equities)
    )

    minimum_equity = min(
        model_equities
    )

    maximum_equity = max(
        model_equities
    )


    average_ev = (
        sum(model_evs)
        / len(model_evs)
    )

    minimum_ev = min(
        model_evs
    )

    maximum_ev = max(
        model_evs
    )


    sorted_equities = sorted(
        model_equities
    )


    # Approximate empirical 5th percentile.

    lower_index = int(
        0.05
        * len(sorted_equities)
    )


    # Approximate empirical 95th percentile.

    upper_index = (
        int(
            0.95
            * len(sorted_equities)
        )
        - 1
    )


    # Keep the indices valid even if the number of models
    # becomes very small during testing.

    lower_index = max(
        0,
        min(
            lower_index,
            len(sorted_equities) - 1,
        ),
    )

    upper_index = max(
        0,
        min(
            upper_index,
            len(sorted_equities) - 1,
        ),
    )


    lower_equity = (
        sorted_equities[
            lower_index
        ]
    )

    upper_equity = (
        sorted_equities[
            upper_index
        ]
    )


    simulation_runtime = (
        simulation_end
        - simulation_start
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "break_even_equity":
            break_even_equity,

        "model_equities":
            model_equities,

        "model_evs":
            model_evs,

        "model_action_rates":
            model_action_rates,

        "profitable_models":
            profitable_models,

        "profitable_fraction":
            profitable_fraction,

        "fragility_score":
            fragility_score,

        "average_equity":
            average_equity,

        "minimum_equity":
            minimum_equity,

        "maximum_equity":
            maximum_equity,

        "average_ev":
            average_ev,

        "minimum_ev":
            minimum_ev,

        "maximum_ev":
            maximum_ev,

        "lower_equity":
            lower_equity,

        "upper_equity":
            upper_equity,

        "simulation_runtime":
            simulation_runtime,
    }