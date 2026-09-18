import random
import math
import time
import os
import json
import csv
import bisect

from itertools import accumulate

import matplotlib.pyplot as plt

from poker.cards import (
    ranks,
    rank_order,
    suits,
    rank_values,
    full_deck
)

from poker.evaluator import evaluate_hand

from poker.hands import (
    hand_classes,
    generate_class_combos
)

# ============================================================
# PROJECT SETTINGS
# ============================================================

RANDOM_SEED = 42

BASELINE_TRIALS_PER_CLASS = 1000

NUMBER_OF_MODELS = 50
TRIALS_PER_MODEL = 6000

# One modeled opponent + four random opponents + Hero = 6 players
RANDOM_OPPONENTS = 4

random.seed(RANDOM_SEED)

HAND_STRENGTH_CACHE = (
    "hand_strengths_"
    + str(BASELINE_TRIALS_PER_CLASS)
    + ".json"
)

# ============================================================
# BASELINE MONTE CARLO HAND STRENGTH
# ============================================================

def estimate_class_equity(
    hand_class,
    trials
):

    hero_combos = (
        generate_class_combos(
            hand_class
        )
    )


    equity_total = 0


    for simulation in range(
        trials
    ):

        hero_hand = random.choice(
            hero_combos
        )


        simulation_deck = (
            full_deck.copy()
        )


        for card in hero_hand:

            simulation_deck.remove(
                card
            )


        opponent_hand = random.sample(
            simulation_deck,
            2
        )


        for card in opponent_hand:

            simulation_deck.remove(
                card
            )


        board = random.sample(
            simulation_deck,
            5
        )


        hero_score = evaluate_hand(
            hero_hand
            + board
        )


        opponent_score = evaluate_hand(
            opponent_hand
            + board
        )


        if hero_score > opponent_score:

            equity_total += 1


        elif hero_score == opponent_score:

            equity_total += 0.5


    return (
        equity_total
        / trials
    )


# ============================================================
# LOAD OR BUILD BASELINE HAND STRENGTH MODEL
# ============================================================

def main():

    baseline_simulations_run = 0


    if os.path.exists(
        HAND_STRENGTH_CACHE
    ):

        print()

        print(
            "Loading cached hand strength model..."
        )


        with open(
            HAND_STRENGTH_CACHE,
            "r"
        ) as file:

            hand_strengths = (
                json.load(file)
            )


    else:

        print()

        print(
            "Building baseline strength model..."
        )


        hand_strengths = {}


        baseline_start = (
            time.perf_counter()
        )


        for index, hand_class in enumerate(
            hand_classes,
            start=1
        ):

            equity = (
                estimate_class_equity(
                    hand_class,
                    BASELINE_TRIALS_PER_CLASS
                )
            )


            hand_strengths[
                hand_class
            ] = equity


            print(
                index,
                "/",
                len(hand_classes),
                hand_class,
                round(
                    equity * 100,
                    2
                ),
                "%"
            )


        baseline_end = (
            time.perf_counter()
        )


        baseline_simulations_run = (
            len(hand_classes)
            * BASELINE_TRIALS_PER_CLASS
        )


        with open(
            HAND_STRENGTH_CACHE,
            "w"
        ) as file:

            json.dump(
                hand_strengths,
                file
            )


        print()

        print(
            "Baseline model runtime:",
            round(
                baseline_end
                - baseline_start,
                2
            ),
            "seconds"
        )


# ============================================================
# HERO HAND INPUT
# ============================================================

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


# ============================================================
# PLAYER STAT INPUT
# ============================================================

    def get_percentage(
        prompt
    ):

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


# ============================================================
# POSITION
# ============================================================

    valid_positions = [
        "UTG",
        "HJ",
        "CO",
        "BTN",
        "SB",
        "BB"
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


# ============================================================
# OBSERVED ACTION
# ============================================================

    valid_actions = [
        "RAISE",
        "CALL",
        "3BET"
    ]


    while True:

        observed_action = input(
            "Modeled opponent action "
            "(RAISE, CALL, 3BET): "
        ).upper().replace(
            " ",
            ""
        )


        if observed_action in valid_actions:

            break


        print(
            "Enter RAISE, CALL, or 3BET."
        )


# ============================================================
# POSITION ADJUSTMENTS
# ============================================================

    VPIP_POSITION_FACTOR = {
        "UTG": 0.70,
        "HJ": 0.85,
        "CO": 1.05,
        "BTN": 1.25,
        "SB": 1.20,
        "BB": 1.35
    }

    PFR_POSITION_FACTOR = {
        "UTG": 0.65,
        "HJ": 0.82,
        "CO": 1.05,
        "BTN": 1.30,
        "SB": 1.15,
        "BB": 0.80
    }

    THREEBET_POSITION_FACTOR = {
        "UTG": 0.80,
        "HJ": 0.90,
        "CO": 1.00,
        "BTN": 1.10,
        "SB": 1.20,
        "BB": 1.25
    }


    def clamp(
        value,
        minimum,
        maximum
    ):

        return max(
            minimum,
            min(
                value,
                maximum
            )
        )


    def adjust_rates_for_position(
        vpip_rate,
        pfr_rate,
        three_bet_rate,
        position
    ):

        adjusted_vpip = clamp(
            vpip_rate
            * VPIP_POSITION_FACTOR[
                position
            ],
            0.001,
            0.95
        )


        adjusted_pfr = clamp(
            pfr_rate
            * PFR_POSITION_FACTOR[
                position
            ],
            0.001,
            0.90
        )


        adjusted_pfr = min(
            adjusted_pfr,
            adjusted_vpip
        )


        adjusted_three_bet = clamp(
            three_bet_rate
            * THREEBET_POSITION_FACTOR[
                position
            ],
            0.001,
            0.50
        )


        return (
            adjusted_vpip,
            adjusted_pfr,
            adjusted_three_bet
        )


# ============================================================
# AVAILABLE COMBOS AFTER HERO BLOCKERS
# ============================================================

    available_combos = {}

    combo_counts = {}


    for hand_class in hand_classes:

        combos = (
            generate_class_combos(
                hand_class,
                hero_hand
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


# ============================================================
# LOGISTIC ACTION MODEL
# ============================================================

    def sigmoid(x):

        if x >= 0:

            return (
                1
                /
                (
                    1
                    + math.exp(-x)
                )
            )


        exp_x = (
            math.exp(x)
        )


        return (
            exp_x
            /
            (
                1
                + exp_x
            )
        )


    def calibrate_threshold(
        target_frequency,
        slope
    ):

        low = -1.0
        high = 2.0


        for iteration in range(
            70
        ):

            midpoint = (
                low + high
            ) / 2


            weighted_total = 0
            combination_total = 0


            for hand_class in hand_classes:

                count = (
                    combo_counts[
                        hand_class
                    ]
                )


                strength = (
                    hand_strengths[
                        hand_class
                    ]
                )


                probability = sigmoid(
                    slope
                    * (
                        strength
                        - midpoint
                    )
                )


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


            if (
                current_frequency
                > target_frequency
            ):

                low = midpoint

            else:

                high = midpoint


        return (
            low + high
        ) / 2


    def build_action_likelihoods(
        action,
        vpip_rate,
        pfr_rate,
        three_bet_rate,
        slope_scale=1.0
    ):

        likelihoods = {}


    # --------------------------------------------------------
    # RAISE
    # --------------------------------------------------------

        if action == "RAISE":

            slope = (
                18
                * slope_scale
            )


            threshold = (
                calibrate_threshold(
                    pfr_rate,
                    slope
                )
            )


            for hand_class in hand_classes:

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


    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

        elif action == "CALL":

            slope = (
                16
                * slope_scale
            )


            vpip_threshold = (
                calibrate_threshold(
                    vpip_rate,
                    slope
                )
            )


            raise_threshold = (
                calibrate_threshold(
                    pfr_rate,
                    slope
                )
            )


            for hand_class in hand_classes:

                strength = (
                    hand_strengths[
                        hand_class
                    ]
                )


                play_probability = sigmoid(
                    slope
                    * (
                        strength
                        - vpip_threshold
                    )
                )


                raise_probability = sigmoid(
                    slope
                    * (
                        strength
                        - raise_threshold
                    )
                )


                call_probability = (
                    play_probability
                    - raise_probability
                )


                likelihoods[
                    hand_class
                ] = max(
                    call_probability,
                    1e-12
                )


    # --------------------------------------------------------
    # 3 BET
    # --------------------------------------------------------

        elif action == "3BET":

            slope = (
                24
                * slope_scale
            )


            threshold = (
                calibrate_threshold(
                    three_bet_rate,
                    slope
                )
            )


            for hand_class in hand_classes:

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


# ============================================================
# CENTRAL MODELED OPPONENT
# ============================================================

    (
        adjusted_vpip,
        adjusted_pfr,
        adjusted_three_bet
    ) = adjust_rates_for_position(
        vpip,
        pfr,
        three_bet,
        position
    )


    central_likelihoods = (
        build_action_likelihoods(
            observed_action,
            adjusted_vpip,
            adjusted_pfr,
            adjusted_three_bet
        )
    )


# ============================================================
# BAYESIAN RANGE UPDATE
# ============================================================

    central_range_mass = {}

    normalizing_constant = 0


    for hand_class in hand_classes:

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


    for hand_class in hand_classes:

        central_range_mass[
            hand_class
        ] /= (
            normalizing_constant
        )


# ============================================================
# DISPLAY INFERRED RANGE
# ============================================================

    sorted_range = sorted(
        hand_classes,
        key=lambda hand:
            central_range_mass[
                hand
            ],
        reverse=True
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
        position
    )

    print(
        "Observed Action:",
        observed_action
    )

    print()

    print(
        "Adjusted VPIP:",
        round(
            adjusted_vpip * 100,
            2
        ),
        "%"
    )

    print(
        "Adjusted PFR:",
        round(
            adjusted_pfr * 100,
            2
        ),
        "%"
    )

    print(
        "Adjusted 3 Bet:",
        round(
            adjusted_three_bet * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Top 25 hands by posterior range mass:"
    )

    print()


    for index, hand_class in enumerate(
        sorted_range[:25],
        start=1
    ):

        print(
            index,
            hand_class,
            "| Action Likelihood:",
            round(
                central_likelihoods[
                    hand_class
                ] * 100,
                2
            ),
            "%",
            "| Range Mass:",
            round(
                central_range_mass[
                    hand_class
                ] * 100,
                3
            ),
            "%"
        )


# ============================================================
# POT AND CALL INPUT
# ============================================================

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


    break_even_equity = (
        call_amount
        /
        (
            current_pot
            + call_amount
        )
    )


# ============================================================
# BETA UNCERTAINTY MODEL
# ============================================================

    def sample_rate_from_beta(
        estimated_rate,
        sample_size
    ):

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
            beta
        )


# ============================================================
# SIX PLAYER DECISION FRAGILITY SIMULATION
#
# PLAYER 1: HERO
# PLAYER 2: MODELED OPPONENT
# PLAYERS 3-6: RANDOM OPPONENTS
# ============================================================

    model_equities = []

    model_evs = []

    model_action_rates = []

    profitable_models = 0


    # Hero cards cannot appear anywhere else

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
        NUMBER_OF_MODELS + 1
    ):

    # --------------------------------------------------------
    # SAMPLE PLAUSIBLE OPPONENT STATISTICS
    # --------------------------------------------------------

        sampled_vpip = (
            sample_rate_from_beta(
                vpip,
                observed_hands
            )
        )


        sampled_pfr = (
            sample_rate_from_beta(
                pfr,
                observed_hands
            )
        )


        sampled_three_bet = (
            sample_rate_from_beta(
                three_bet,
                observed_hands
            )
        )


        sampled_pfr = min(
            sampled_pfr,
            sampled_vpip
        )


        (
            model_vpip,
            model_pfr,
            model_three_bet
        ) = adjust_rates_for_position(
            sampled_vpip,
            sampled_pfr,
            sampled_three_bet,
            position
        )


    # --------------------------------------------------------
    # UNCERTAINTY ABOUT RANGE SHAPE
    # --------------------------------------------------------

        slope_scale = (
            random.uniform(
                0.85,
                1.15
            )
        )


        model_likelihoods = (
            build_action_likelihoods(
                observed_action,
                model_vpip,
                model_pfr,
                model_three_bet,
                slope_scale
            )
        )


    # --------------------------------------------------------
    # BUILD WEIGHTED PHYSICAL COMBO POPULATION
    # --------------------------------------------------------

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


        cumulative_weights = list(
            accumulate(
                combo_weights
            )
        )


        total_weight = (
            cumulative_weights[-1]
        )


    # --------------------------------------------------------
    # MONTE CARLO EQUITY
    # --------------------------------------------------------

        equity_total = 0


        for simulation in range(
            TRIALS_PER_MODEL
        ):

        # ====================================================
        # MODELED OPPONENT
        # ====================================================

            random_weight = (
                random.random()
                * total_weight
            )


            combo_index = (
                bisect.bisect_left(
                    cumulative_weights,
                    random_weight
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


        # ====================================================
        # FOUR RANDOM OPPONENTS
        # ====================================================

            random_opponent_hands = []


            for opponent_number in range(
                RANDOM_OPPONENTS
            ):

                random_hand = (
                    random.sample(
                        simulation_deck,
                        2
                    )
                )


                random_opponent_hands.append(
                    random_hand
                )


                for card in random_hand:

                    simulation_deck.remove(
                        card
                    )


        # ====================================================
        # COMMUNITY CARDS
        # ====================================================

            board = random.sample(
                simulation_deck,
                5
            )


        # ====================================================
        # HERO SCORE
        # ====================================================

            hero_score = (
                evaluate_hand(
                    hero_hand
                    + board
                )
            )


        # ====================================================
        # MODELED OPPONENT SCORE
        # ====================================================

            modeled_opponent_score = (
                evaluate_hand(
                    modeled_opponent_hand
                    + board
                )
            )


            opponent_scores = [
                modeled_opponent_score
            ]


        # ====================================================
        # RANDOM OPPONENT SCORES
        # ====================================================

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


        # ====================================================
        # MULTIWAY POT EQUITY
        # ====================================================

            all_scores = (
                [hero_score]
                + opponent_scores
            )


            best_score = max(
                all_scores
            )


            # Somebody beats Hero

            if hero_score < best_score:

                hero_equity_share = 0


            else:

                # Hero tied for best hand.
                # Count number of players sharing the pot.

                number_of_winners = 0


                for score in all_scores:

                    if score == best_score:

                        number_of_winners += 1


                hero_equity_share = (
                    1
                    / number_of_winners
                )


            equity_total += (
                hero_equity_share
            )


    # --------------------------------------------------------
    # EQUITY FOR THIS PLAUSIBLE MODEL
    # --------------------------------------------------------

        model_equity = (
            equity_total
            / TRIALS_PER_MODEL
        )


    # --------------------------------------------------------
    # EXPECTED VALUE OF CALL
    #
    # Assumes current_pot is the total pot Hero can win
    # before Hero contributes the call.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MODELED OPPONENT ACTION RATE
    # --------------------------------------------------------

        if observed_action == "RAISE":

            model_action_rate = (
                model_pfr
            )


        elif observed_action == "CALL":

            model_action_rate = max(
                model_vpip
                - model_pfr,
                0
            )


        else:

            model_action_rate = (
                model_three_bet
            )


        model_action_rates.append(
            model_action_rate
        )


    # --------------------------------------------------------
    # IS CALL PROFITABLE UNDER THIS MODEL?
    # --------------------------------------------------------

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
                2
            ),
            "%",
            "| EV:",
            round(
                model_ev,
                2
            )
        )


    decision_end = (
        time.perf_counter()
    )


# ============================================================
# DECISION FRAGILITY
# ============================================================

    profitable_fraction = (
        profitable_models
        / NUMBER_OF_MODELS
    )


    fragility_score = (
        1
        - abs(
            2
            * profitable_fraction
            - 1
        )
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

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


# ============================================================
# FINAL RESULTS
# ============================================================

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
        hero_hand
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
        position
    )

    print(
        "Modeled Opponent Action:",
        observed_action
    )

    print()

    print(
        "Break Even Equity:",
        round(
            break_even_equity * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Average Hero Equity:",
        round(
            average_equity * 100,
            2
        ),
        "%"
    )

    print(
        "Minimum Hero Equity:",
        round(
            minimum_equity * 100,
            2
        ),
        "%"
    )

    print(
        "Maximum Hero Equity:",
        round(
            maximum_equity * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Approximate 90% Model Equity Interval:",
        round(
            lower_equity * 100,
            2
        ),
        "%",
        "to",
        round(
            upper_equity * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Average Call EV:",
        round(
            average_ev,
            2
        )
    )

    print(
        "Minimum Call EV:",
        round(
            minimum_ev,
            2
        )
    )

    print(
        "Maximum Call EV:",
        round(
            maximum_ev,
            2
        )
    )

    print()

    print(
        "Profitable Models:",
        profitable_models,
        "/",
        NUMBER_OF_MODELS
    )

    print(
        "Call Profitable Across Models:",
        round(
            profitable_fraction * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Decision Fragility Score:",
        round(
            fragility_score * 100,
            2
        ),
        "%"
    )

    print()

    print(
        "Decision Simulation Runtime:",
        round(
            decision_end
            - decision_start,
            2
        ),
        "seconds"
    )


# ============================================================
# SAVE INFERRED RANGE
# ============================================================

    with open(
        "inferred_range.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "Hand Class",
                "Baseline Equity",
                "Action Likelihood",
                "Posterior Range Mass"
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
                    ]
                ]
            )


# ============================================================
# SAVE DECISION MODELS
# ============================================================

    with open(
        "decision_fragility_models.csv",
        "w",
        newline=""
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
                "Estimated Action Rate"
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
                    ]
                ]
            )


# ============================================================
# GRAPH 1
# DECISION FRAGILITY
# ============================================================

    model_numbers = list(
        range(
            1,
            NUMBER_OF_MODELS + 1
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
        equity_percentages
    )


    plt.axhline(
        y=break_even_equity * 100,
        linestyle="--",
        label="Break Even Equity"
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


# ============================================================
# GRAPH 2
# MODELED OPPONENT POSTERIOR RANGE
# ============================================================

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
        top_20_mass
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


# ============================================================
# FINAL NOTES
# ============================================================

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