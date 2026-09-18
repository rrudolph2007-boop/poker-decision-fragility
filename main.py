import bisect
import csv
import random
import time

from itertools import accumulate

import matplotlib.pyplot as plt

from models.opponent_model import (
    adjust_rates_for_position,
    build_action_likelihoods,
)

from poker.cards import full_deck
from poker.equity import load_or_build_hand_strengths
from poker.evaluator import evaluate_hand
from poker.hands import (
    generate_class_combos,
    hand_classes,
)


# ============================================================
# PROJECT SETTINGS
# ============================================================

RANDOM_SEED = 42

BASELINE_TRIALS_PER_CLASS = 1000

NUMBER_OF_MODELS = 50
TRIALS_PER_MODEL = 6000

# Hero + 1 modeled opponent + 4 random opponents = 6 players
RANDOM_OPPONENTS = 4

HAND_STRENGTH_CACHE = (
    "hand_strengths_"
    + str(BASELINE_TRIALS_PER_CLASS)
    + ".json"
)


# ============================================================
# USER INPUT HELPERS
# ============================================================

def get_percentage(
    prompt,
):
    """
    Request a percentage from the user and convert it to a decimal.

    Example:
        25 becomes 0.25
    """

    while True:

        try:

            value = float(
                input(prompt)
            )

            if (
                0
                <= value
                <= 100
            ):

                return (
                    value / 100
                )

            print(
                "Enter a value from 0 to 100."
            )

        except ValueError:

            print(
                "Enter a number."
            )


# ============================================================
# BETA UNCERTAINTY MODEL
# ============================================================

def sample_rate_from_beta(
    estimated_rate,
    sample_size,
):
    """
    Sample a plausible version of an observed player statistic.

    A smaller sample of observed hands creates more uncertainty.
    A larger sample creates a tighter distribution around the
    observed statistic.
    """

    alpha = (
        1
        + estimated_rate
        * sample_size
    )

    beta = (
        1
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
# MAIN PROGRAM
# ============================================================

def main():

    # Use a fixed seed so identical inputs produce reproducible
    # Monte Carlo results during development and testing.

    random.seed(
        RANDOM_SEED
    )


    print(
        "Starting hand classes:",
        len(hand_classes),
    )


    # ========================================================
    # LOAD OR BUILD BASELINE HAND STRENGTH MODEL
    # ========================================================

    (
        hand_strengths,
        baseline_simulations_run,
        baseline_runtime,
    ) = load_or_build_hand_strengths(
        hand_classes,
        BASELINE_TRIALS_PER_CLASS,
        HAND_STRENGTH_CACHE,
    )


    if baseline_simulations_run > 0:

        print()

        print(
            "Baseline model runtime:",
            round(
                baseline_runtime,
                2,
            ),
            "seconds",
        )


    # ========================================================
    # HERO HAND INPUT
    # ========================================================

    while True:

        hero_hand = input(
            "\nEnter hero hand. Example: AS QH: "
        ).upper().split()


        if len(hero_hand) != 2:

            print(
                "Enter exactly two cards."
            )

            continue


        if (
            hero_hand[0] not in full_deck
            or
            hero_hand[1] not in full_deck
        ):

            print(
                "Enter valid cards."
            )

            continue


        if (
            hero_hand[0]
            == hero_hand[1]
        ):

            print(
                "Cards cannot be identical."
            )

            continue


        break


    # ========================================================
    # MODELED OPPONENT STATISTICS
    # ========================================================

    while True:

        vpip = get_percentage(
            "Modeled opponent VPIP percentage: "
        )

        pfr = get_percentage(
            "Modeled opponent PFR percentage: "
        )

        three_bet = get_percentage(
            "Modeled opponent 3 bet percentage: "
        )


        if pfr <= vpip:

            break


        print()

        print(
            "PFR should not exceed VPIP."
        )


    while True:

        try:

            observed_hands = int(
                input(
                    "Number of observed hands: "
                )
            )


            if observed_hands > 0:

                break


        except ValueError:

            pass


        print(
            "Enter a positive whole number."
        )


    # ========================================================
    # MODELED OPPONENT POSITION
    # ========================================================

    valid_positions = [
        "UTG",
        "HJ",
        "CO",
        "BTN",
        "SB",
        "BB",
    ]


    while True:

        position = input(
            "Modeled opponent position "
            "(UTG, HJ, CO, BTN, SB, BB): "
        ).upper()


        if position in valid_positions:

            break


        print(
            "Enter a valid position."
        )


    # ========================================================
    # OBSERVED ACTION
    # ========================================================

    valid_actions = [
        "RAISE",
        "CALL",
        "3BET",
    ]


    while True:

        observed_action = input(
            "Modeled opponent action "
            "(RAISE, CALL, 3BET): "
        ).upper().replace(
            " ",
            "",
        )


        if observed_action in valid_actions:

            break


        print(
            "Enter RAISE, CALL, or 3BET."
        )


    # ========================================================
    # AVAILABLE HAND COMBINATIONS AFTER HERO BLOCKERS
    # ========================================================

    available_combos = {}

    combo_counts = {}


    for hand_class in hand_classes:

        combos = generate_class_combos(
            hand_class,
            hero_hand,
        )


        available_combos[
            hand_class
        ] = combos


        combo_counts[
            hand_class
        ] = len(
            combos
        )


    # ========================================================
    # CENTRAL MODELED OPPONENT
    # ========================================================

    # The implementation of position adjustment now lives in:
    #
    # models/opponent_model.py

    (
        adjusted_vpip,
        adjusted_pfr,
        adjusted_three_bet,
    ) = adjust_rates_for_position(
        vpip,
        pfr,
        three_bet,
        position,
    )


    # The logistic action model and threshold calibration also
    # live in models/opponent_model.py.

    central_likelihoods = (
        build_action_likelihoods(
            observed_action,
            adjusted_vpip,
            adjusted_pfr,
            adjusted_three_bet,
            hand_classes,
            hand_strengths,
            combo_counts,
        )
    )


    # ========================================================
    # BAYESIAN-STYLE RANGE UPDATE
    # ========================================================

    central_range_mass = {}

    normalizing_constant = 0


    for hand_class in hand_classes:

        # Prior range probability is proportional to the number
        # of available physical combinations for the hand class.
        #
        # The action likelihood then increases or decreases that
        # probability depending on how likely the opponent would
        # be to take the observed action with that hand.

        mass = (
            combo_counts[
                hand_class
            ]
            * central_likelihoods[
                hand_class
            ]
        )


        central_range_mass[
            hand_class
        ] = mass


        normalizing_constant += (
            mass
        )


    # Convert the unnormalized masses into probabilities that
    # sum to 1.

    for hand_class in hand_classes:

        central_range_mass[
            hand_class
        ] /= (
            normalizing_constant
        )


    # ========================================================
    # DISPLAY INFERRED RANGE
    # ========================================================

    sorted_range = sorted(
        hand_classes,
        key=lambda hand:
            central_range_mass[
                hand
            ],
        reverse=True,
    )


    print()

    print(
        "========================================"
    )

    print(
        "INFERRED MODELED OPPONENT RANGE"
    )

    print(
        "========================================"
    )

    print()


    print(
        "Position:",
        position,
    )


    print(
        "Observed Action:",
        observed_action,
    )


    print()


    print(
        "Adjusted VPIP:",
        round(
            adjusted_vpip * 100,
            2,
        ),
        "%",
    )


    print(
        "Adjusted PFR:",
        round(
            adjusted_pfr * 100,
            2,
        ),
        "%",
    )


    print(
        "Adjusted 3 Bet:",
        round(
            adjusted_three_bet * 100,
            2,
        ),
        "%",
    )


    print()

    print(
        "Top 25 hands by posterior range mass:"
    )

    print()


    for index, hand_class in enumerate(
        sorted_range[:25],
        start=1,
    ):

        print(
            index,
            hand_class,
            "| Action Likelihood:",
            round(
                central_likelihoods[
                    hand_class
                ] * 100,
                2,
            ),
            "%",
            "| Range Mass:",
            round(
                central_range_mass[
                    hand_class
                ] * 100,
                3,
            ),
            "%",
        )


    # ========================================================
    # POT AND CALL INPUT
    # ========================================================

    while True:

        try:

            current_pot = float(
                input(
                    "\nCurrent pot before your call: "
                )
            )


            if current_pot > 0:

                break


        except ValueError:

            pass


        print(
            "Enter a positive number."
        )


    while True:

        try:

            call_amount = float(
                input(
                    "Amount required to call: "
                )
            )


            if call_amount > 0:

                break


        except ValueError:

            pass


        print(
            "Enter a positive number."
        )


    # Minimum equity Hero needs for the call to break even.

    break_even_equity = (
        call_amount
        /
        (
            current_pot
            + call_amount
        )
    )


    # ========================================================
    # SIX-PLAYER DECISION FRAGILITY SIMULATION
    # ========================================================

    model_equities = []

    model_evs = []

    model_action_rates = []

    profitable_models = 0


    # Hero's two known cards can never appear in an opponent
    # hand or on the board.

    base_deck = [
        card
        for card in full_deck
        if card not in hero_hand
    ]


    decision_start = (
        time.perf_counter()
    )


    for model_number in range(
        1,
        NUMBER_OF_MODELS + 1,
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


        # Introduce uncertainty about the exact shape of the
        # opponent's action-probability curve.

        slope_scale = (
            random.uniform(
                0.85,
                1.15,
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
        # BUILD WEIGHTED MODELED-OPPONENT RANGE
        # ----------------------------------------------------

        combo_population = []

        combo_weights = []


        for hand_class in hand_classes:

            weight = (
                model_likelihoods[
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


        # Cumulative weights allow efficient weighted sampling
        # using binary search.

        cumulative_weights = list(
            accumulate(
                combo_weights
            )
        )


        total_weight = (
            cumulative_weights[-1]
        )


        equity_total = 0


        # ----------------------------------------------------
        # MONTE CARLO TRIALS FOR THIS PLAUSIBLE MODEL
        # ----------------------------------------------------

        for simulation in range(
            TRIALS_PER_MODEL
        ):

            # ------------------------------------------------
            # SAMPLE THE MODELED OPPONENT
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
            # DEAL FOUR ADDITIONAL RANDOM OPPONENTS
            # ------------------------------------------------

            random_opponent_hands = []


            for opponent_number in range(
                RANDOM_OPPONENTS
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


                # Remove the dealt cards so no physical card can
                # be dealt to more than one player.

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
            # EVALUATE HERO
            # ------------------------------------------------

            hero_score = (
                evaluate_hand(
                    hero_hand
                    + board
                )
            )


            # ------------------------------------------------
            # EVALUATE MODELED OPPONENT
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
            # EVALUATE FOUR RANDOM OPPONENTS
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
            # MULTIWAY EQUITY
            # ------------------------------------------------

            all_scores = (
                [hero_score]
                + opponent_scores
            )


            best_score = max(
                all_scores
            )


            # If another player has a stronger hand,
            # Hero receives no share of the pot.

            if hero_score < best_score:

                hero_equity_share = 0


            else:

                # Hero has the best hand, but multiple players
                # may have tied for that same best hand.

                number_of_winners = 0


                for score in all_scores:

                    if score == best_score:

                        number_of_winners += 1


                # Examples:
                #
                # Hero wins alone       -> 1
                # Hero ties one player  -> 1/2
                # Hero ties two players -> 1/3

                hero_equity_share = (
                    1
                    / number_of_winners
                )


            equity_total += (
                hero_equity_share
            )


        # ----------------------------------------------------
        # EQUITY FOR THIS PLAUSIBLE OPPONENT MODEL
        # ----------------------------------------------------

        model_equity = (
            equity_total
            / TRIALS_PER_MODEL
        )


        # ----------------------------------------------------
        # EXPECTED VALUE OF CALL
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # STORE ACTION RATE FOR OUTPUT
        # ----------------------------------------------------

        if observed_action == "RAISE":

            model_action_rate = (
                model_pfr
            )


        elif observed_action == "CALL":

            model_action_rate = max(
                model_vpip
                - model_pfr,
                0,
            )


        else:

            model_action_rate = (
                model_three_bet
            )


        model_action_rates.append(
            model_action_rate
        )


        # Decision Fragility ultimately depends on whether each
        # plausible model considers calling profitable.

        if model_ev > 0:

            profitable_models += 1


        print(
            "Model",
            model_number,
            "/",
            NUMBER_OF_MODELS,
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


    decision_end = (
        time.perf_counter()
    )


    # ========================================================
    # DECISION FRAGILITY
    # ========================================================

    profitable_fraction = (
        profitable_models
        / NUMBER_OF_MODELS
    )


    # q = fraction of models where calling has positive EV.
    #
    # F = 1 - |2q - 1|
    #
    # F = 0 when all models agree.
    # F = 1 when the models split 50/50.

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


    lower_index = int(
        0.05
        * len(sorted_equities)
    )


    upper_index = (
        int(
            0.95
            * len(sorted_equities)
        )
        - 1
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


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "SIX PLAYER DECISION FRAGILITY RESULTS"
    )

    print(
        "========================================"
    )

    print()


    print(
        "Hero Hand:",
        hero_hand,
    )


    print()


    print(
        "Table Model:"
    )


    print(
        "Hero + 1 inferred opponent + 4 random opponents"
    )


    print()


    print(
        "Modeled Opponent Position:",
        position,
    )


    print(
        "Modeled Opponent Action:",
        observed_action,
    )


    print()


    print(
        "Break Even Equity:",
        round(
            break_even_equity * 100,
            2,
        ),
        "%",
    )


    print()


    print(
        "Average Hero Equity:",
        round(
            average_equity * 100,
            2,
        ),
        "%",
    )


    print(
        "Minimum Hero Equity:",
        round(
            minimum_equity * 100,
            2,
        ),
        "%",
    )


    print(
        "Maximum Hero Equity:",
        round(
            maximum_equity * 100,
            2,
        ),
        "%",
    )


    print()


    print(
        "Approximate 90% Model Equity Interval:",
        round(
            lower_equity * 100,
            2,
        ),
        "%",
        "to",
        round(
            upper_equity * 100,
            2,
        ),
        "%",
    )


    print()


    print(
        "Average Call EV:",
        round(
            average_ev,
            2,
        ),
    )


    print(
        "Minimum Call EV:",
        round(
            minimum_ev,
            2,
        ),
    )


    print(
        "Maximum Call EV:",
        round(
            maximum_ev,
            2,
        ),
    )


    print()


    print(
        "Profitable Models:",
        profitable_models,
        "/",
        NUMBER_OF_MODELS,
    )


    print(
        "Call Profitable Across Models:",
        round(
            profitable_fraction * 100,
            2,
        ),
        "%",
    )


    print()


    print(
        "Decision Fragility Score:",
        round(
            fragility_score * 100,
            2,
        ),
        "%",
    )


    print()


    print(
        "Decision Simulation Runtime:",
        round(
            decision_end
            - decision_start,
            2,
        ),
        "seconds",
    )


    # ========================================================
    # SAVE INFERRED RANGE
    # ========================================================

    with open(
        "inferred_range.csv",
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "Hand Class",
                "Baseline Equity",
                "Action Likelihood",
                "Posterior Range Mass",
            ]
        )


        for hand_class in sorted_range:

            writer.writerow(
                [
                    hand_class,

                    hand_strengths[
                        hand_class
                    ],

                    central_likelihoods[
                        hand_class
                    ],

                    central_range_mass[
                        hand_class
                    ],
                ]
            )


    # ========================================================
    # SAVE DECISION MODELS
    # ========================================================

    with open(
        "decision_fragility_models.csv",
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "Model",
                "Six Player Equity",
                "Call EV",
                "Profitable",
                "Estimated Action Rate",
            ]
        )


        for index in range(
            NUMBER_OF_MODELS
        ):

            writer.writerow(
                [
                    index + 1,

                    model_equities[
                        index
                    ],

                    model_evs[
                        index
                    ],

                    (
                        model_evs[
                            index
                        ] > 0
                    ),

                    model_action_rates[
                        index
                    ],
                ]
            )


    # ========================================================
    # GRAPH 1: DECISION FRAGILITY
    # ========================================================

    model_numbers = list(
        range(
            1,
            NUMBER_OF_MODELS + 1,
        )
    )


    equity_percentages = []


    for equity in model_equities:

        equity_percentages.append(
            equity * 100
        )


    plt.figure(
        figsize=(11, 6)
    )


    plt.scatter(
        model_numbers,
        equity_percentages,
    )


    plt.axhline(
        y=break_even_equity * 100,
        linestyle="--",
        label="Break Even Equity",
    )


    plt.xlabel(
        "Plausible Modeled Opponent"
    )


    plt.ylabel(
        "Hero Six Player Equity (%)"
    )


    plt.title(
        "Decision Fragility in a Six Player Pot"
    )


    plt.legend()

    plt.grid()

    plt.show()


    # ========================================================
    # GRAPH 2: INFERRED OPPONENT RANGE
    # ========================================================

    top_20 = (
        sorted_range[:20]
    )


    top_20_mass = []


    for hand_class in top_20:

        top_20_mass.append(
            central_range_mass[
                hand_class
            ] * 100
        )


    plt.figure(
        figsize=(11, 6)
    )


    plt.bar(
        top_20,
        top_20_mass,
    )


    plt.xlabel(
        "Modeled Opponent Starting Hand"
    )


    plt.ylabel(
        "Posterior Range Mass (%)"
    )


    plt.title(
        "Most Likely Holdings for the Modeled Opponent"
    )


    plt.xticks(
        rotation=60
    )


    plt.grid(
        axis="y"
    )


    plt.show()


    # ========================================================
    # FINAL NOTES
    # ========================================================

    print()

    print(
        "Saved inferred_range.csv"
    )


    print(
        "Saved decision_fragility_models.csv"
    )


    print()

    print(
        "Simulation structure:"
    )


    print(
        "1 modeled opponent uses inferred range."
    )


    print(
        "4 additional opponents receive uniformly random hands."
    )


    print()

    print(
        "Important: this is a probabilistic research prototype."
    )


    print(
        "Position multipliers and logistic action probabilities "
        "are heuristic model assumptions, not solver-derived GTO ranges."
    )


    print(
        "The four additional opponents are currently modeled as "
        "uniform random hands rather than action-conditioned ranges."
    )


if __name__ == "__main__":

    main()