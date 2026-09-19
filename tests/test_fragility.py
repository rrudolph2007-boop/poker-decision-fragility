import random

import pytest

from models.fragility import (
    build_weighted_combo_population,
    get_model_action_rate,
    sample_rate_from_beta,
)


def test_beta_sample_is_probability():

    random.seed(
        42
    )


    sample = (
        sample_rate_from_beta(
            0.30,
            100,
        )
    )


    assert (
        0
        <= sample
        <= 1
    )


def test_beta_sampling_is_reproducible_with_seed():

    random.seed(
        42
    )

    first_sample = (
        sample_rate_from_beta(
            0.30,
            100,
        )
    )


    random.seed(
        42
    )

    second_sample = (
        sample_rate_from_beta(
            0.30,
            100,
        )
    )


    assert (
        first_sample
        == second_sample
    )


def test_weighted_combo_population():

    hand_classes = [
        "AA",
        "AKs",
    ]


    available_combos = {
        "AA": [
            ["AS", "AH"],
            ["AS", "AD"],
        ],

        "AKs": [
            ["AS", "KS"],
        ],
    }


    action_likelihoods = {
        "AA": 0.90,
        "AKs": 0.50,
    }


    (
        combo_population,
        cumulative_weights,
        total_weight,
    ) = build_weighted_combo_population(
        hand_classes,
        available_combos,
        action_likelihoods,
    )


    assert len(
        combo_population
    ) == 3


    assert cumulative_weights == (
        pytest.approx(
            [
                0.90,
                1.80,
                2.30,
            ]
        )
    )


    assert total_weight == (
        pytest.approx(
            2.30
        )
    )


def test_raise_action_rate_uses_pfr():

    rate = get_model_action_rate(
        observed_action="RAISE",
        model_vpip=0.30,
        model_pfr=0.20,
        model_three_bet=0.08,
    )


    assert rate == (
        pytest.approx(
            0.20
        )
    )


def test_call_action_rate_uses_vpip_minus_pfr():

    rate = get_model_action_rate(
        observed_action="CALL",
        model_vpip=0.30,
        model_pfr=0.20,
        model_three_bet=0.08,
    )


    assert rate == (
        pytest.approx(
            0.10
        )
    )


def test_three_bet_action_rate():

    rate = get_model_action_rate(
        observed_action="3BET",
        model_vpip=0.30,
        model_pfr=0.20,
        model_three_bet=0.08,
    )


    assert rate == (
        pytest.approx(
            0.08
        )
    )


def test_call_action_rate_cannot_be_negative():

    rate = get_model_action_rate(
        observed_action="CALL",
        model_vpip=0.20,
        model_pfr=0.25,
        model_three_bet=0.08,
    )


    assert rate == 0