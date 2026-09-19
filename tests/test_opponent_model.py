import pytest

from models.opponent_model import (
    adjust_rates_for_position,
    build_action_likelihoods,
    clamp,
    sigmoid,
)

from poker.hands import (
    generate_class_combos,
    hand_classes,
)


def test_clamp_inside_range():

    assert clamp(
        0.5,
        0,
        1,
    ) == 0.5


def test_clamp_below_range():

    assert clamp(
        -1,
        0,
        1,
    ) == 0


def test_clamp_above_range():

    assert clamp(
        2,
        0,
        1,
    ) == 1


def test_sigmoid_zero():

    assert sigmoid(
        0
    ) == pytest.approx(
        0.5
    )


def test_sigmoid_increases():

    assert (
        sigmoid(2)
        > sigmoid(0)
        > sigmoid(-2)
    )


def test_position_adjustment_keeps_pfr_below_vpip():

    (
        adjusted_vpip,
        adjusted_pfr,
        adjusted_three_bet,
    ) = adjust_rates_for_position(
        0.20,
        0.19,
        0.08,
        "BB",
    )

    assert (
        adjusted_pfr
        <= adjusted_vpip
    )


def test_position_adjustment_stays_in_valid_range():

    (
        adjusted_vpip,
        adjusted_pfr,
        adjusted_three_bet,
    ) = adjust_rates_for_position(
        0.95,
        0.90,
        0.50,
        "BTN",
    )

    assert (
        0
        < adjusted_vpip
        <= 0.95
    )

    assert (
        0
        < adjusted_pfr
        <= 0.90
    )

    assert (
        0
        < adjusted_three_bet
        <= 0.50
    )


def test_raise_model_matches_target_frequency():

    # Give the 169 hand classes deterministic artificial
    # strength values from 0 to 1.
    #
    # This isolates the calibration logic from Monte Carlo noise.

    hand_strengths = {}

    for index, hand_class in enumerate(
        hand_classes
    ):

        hand_strengths[
            hand_class
        ] = (
            index
            /
            (
                len(hand_classes)
                - 1
            )
        )


    combo_counts = {}

    for hand_class in hand_classes:

        combo_counts[
            hand_class
        ] = len(
            generate_class_combos(
                hand_class
            )
        )


    target_pfr = 0.20


    likelihoods = (
        build_action_likelihoods(
            action="RAISE",
            vpip_rate=0.30,
            pfr_rate=target_pfr,
            three_bet_rate=0.08,
            hand_classes=hand_classes,
            hand_strengths=hand_strengths,
            combo_counts=combo_counts,
        )
    )


    weighted_probability = 0

    total_combos = 0


    for hand_class in hand_classes:

        combo_count = (
            combo_counts[
                hand_class
            ]
        )

        weighted_probability += (
            likelihoods[
                hand_class
            ]
            * combo_count
        )

        total_combos += (
            combo_count
        )


    modeled_frequency = (
        weighted_probability
        / total_combos
    )


    assert modeled_frequency == (
        pytest.approx(
            target_pfr,
            abs=0.001,
        )
    )


def test_action_likelihoods_are_probabilities():

    hand_strengths = {}

    combo_counts = {}


    for index, hand_class in enumerate(
        hand_classes
    ):

        hand_strengths[
            hand_class
        ] = (
            index
            /
            (
                len(hand_classes)
                - 1
            )
        )

        combo_counts[
            hand_class
        ] = len(
            generate_class_combos(
                hand_class
            )
        )


    likelihoods = (
        build_action_likelihoods(
            action="3BET",
            vpip_rate=0.30,
            pfr_rate=0.20,
            three_bet_rate=0.08,
            hand_classes=hand_classes,
            hand_strengths=hand_strengths,
            combo_counts=combo_counts,
        )
    )


    assert len(
        likelihoods
    ) == 169


    for probability in likelihoods.values():

        assert (
            0
            <= probability
            <= 1
        )