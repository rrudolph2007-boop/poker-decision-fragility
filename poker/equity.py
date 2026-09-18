import json
import os
import random
import time

from poker.cards import full_deck
from poker.evaluator import evaluate_hand
from poker.hands import generate_class_combos


def estimate_class_equity(
    hand_class,
    trials,
):
    hero_combos = generate_class_combos(
        hand_class
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
            2,
        )

        for card in opponent_hand:
            simulation_deck.remove(
                card
            )

        board = random.sample(
            simulation_deck,
            5,
        )

        hero_score = evaluate_hand(
            hero_hand + board
        )

        opponent_score = evaluate_hand(
            opponent_hand + board
        )

        if hero_score > opponent_score:
            equity_total += 1

        elif hero_score == opponent_score:
            equity_total += 0.5

    return (
        equity_total
        / trials
    )


def load_or_build_hand_strengths(
    hand_classes,
    trials_per_class,
    cache_path,
):
    baseline_simulations_run = 0
    baseline_runtime = 0

    if os.path.exists(
        cache_path
    ):
        print()

        print(
            "Loading cached hand strength model..."
        )

        with open(
            cache_path,
            "r",
        ) as file:
            hand_strengths = json.load(
                file
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
            start=1,
        ):
            equity = estimate_class_equity(
                hand_class,
                trials_per_class,
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
                    2,
                ),
                "%",
            )

        baseline_end = (
            time.perf_counter()
        )

        baseline_runtime = (
            baseline_end
            - baseline_start
        )

        baseline_simulations_run = (
            len(hand_classes)
            * trials_per_class
        )

        with open(
            cache_path,
            "w",
        ) as file:
            json.dump(
                hand_strengths,
                file,
            )

    return (
        hand_strengths,
        baseline_simulations_run,
        baseline_runtime,
    )