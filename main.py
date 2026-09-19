import csv
import random

import matplotlib.pyplot as plt

from models.fragility import (
    run_decision_fragility_simulation,
)

from models.opponent_model import (
    adjust_rates_for_position,
    build_action_likelihoods,
)

from poker.cards import full_deck
from poker.equity import load_or_build_hand_strengths
from poker.hands import (
    generate_class_combos,
    hand_classes,
)

from config import (
    BASELINE_TRIALS_PER_CLASS,
    HAND_STRENGTH_CACHE,
    NUMBER_OF_MODELS,
    RANDOM_OPPONENTS,
    RANDOM_SEED,
    TRIALS_PER_MODEL,
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

        User enters 25
        Function returns 0.25
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
# MAIN PROGRAM
# ============================================================

def main():

    # Use a fixed random seed during development so identical
    # inputs produce reproducible simulations.

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
    # AVAILABLE COMBINATIONS AFTER HERO BLOCKERS
    # ========================================================

    available_combos = {}

    combo_counts = {}


    for hand_class in hand_classes:

        combos = (
            generate_class_combos(
                hand_class,
                hero_hand,
            )
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
    # CENTRAL OPPONENT MODEL
    # ========================================================

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

        # Prior mass is proportional to the number of remaining
        # physical combinations of the hand class.
        #
        # Multiplying by P(action | hand) produces an
        # unnormalized posterior mass.

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


    # Normalize so the posterior masses sum to 1.

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


    # ========================================================
    # RUN DECISION FRAGILITY ENGINE
    # ========================================================

    # The Monte Carlo simulation, uncertainty sampling,
    # multiway equity calculation, EV calculation, and
    # Decision Fragility calculation now live entirely in:
    #
    # models/fragility.py

    results = (
        run_decision_fragility_simulation(
            hero_hand=hero_hand,
            vpip=vpip,
            pfr=pfr,
            three_bet=three_bet,
            observed_hands=observed_hands,
            position=position,
            observed_action=observed_action,
            current_pot=current_pot,
            call_amount=call_amount,
            hand_classes=hand_classes,
            hand_strengths=hand_strengths,
            available_combos=available_combos,
            combo_counts=combo_counts,
            number_of_models=NUMBER_OF_MODELS,
            trials_per_model=TRIALS_PER_MODEL,
            random_opponents=RANDOM_OPPONENTS,
        )
    )


    # ========================================================
    # EXTRACT RESULTS
    # ========================================================

    break_even_equity = (
        results[
            "break_even_equity"
        ]
    )


    model_equities = (
        results[
            "model_equities"
        ]
    )


    model_evs = (
        results[
            "model_evs"
        ]
    )


    model_action_rates = (
        results[
            "model_action_rates"
        ]
    )


    profitable_models = (
        results[
            "profitable_models"
        ]
    )


    profitable_fraction = (
        results[
            "profitable_fraction"
        ]
    )


    fragility_score = (
        results[
            "fragility_score"
        ]
    )


    average_equity = (
        results[
            "average_equity"
        ]
    )


    minimum_equity = (
        results[
            "minimum_equity"
        ]
    )


    maximum_equity = (
        results[
            "maximum_equity"
        ]
    )


    average_ev = (
        results[
            "average_ev"
        ]
    )


    minimum_ev = (
        results[
            "minimum_ev"
        ]
    )


    maximum_ev = (
        results[
            "maximum_ev"
        ]
    )


    lower_equity = (
        results[
            "lower_equity"
        ]
    )


    upper_equity = (
        results[
            "upper_equity"
        ]
    )


    decision_runtime = (
        results[
            "simulation_runtime"
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
            decision_runtime,
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