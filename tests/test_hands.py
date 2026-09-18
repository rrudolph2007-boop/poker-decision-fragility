from poker.hands import (
    hand_classes,
    generate_class_combos
)


def test_number_of_starting_hand_classes():

    assert len(hand_classes) == 169


def test_starting_hand_classes_are_unique():

    assert len(
        set(hand_classes)
    ) == 169


def test_pocket_pair_has_six_combinations():

    combos = generate_class_combos(
        "AA"
    )

    assert len(combos) == 6


def test_suited_hand_has_four_combinations():

    combos = generate_class_combos(
        "AKs"
    )

    assert len(combos) == 4


def test_offsuit_hand_has_twelve_combinations():

    combos = generate_class_combos(
        "AKo"
    )

    assert len(combos) == 12


def test_pair_blocker_removes_combinations():

    combos = generate_class_combos(
        "AA",
        blocked_cards=["AS"]
    )

    assert len(combos) == 3


def test_suited_blocker_removes_combination():

    combos = generate_class_combos(
        "AKs",
        blocked_cards=["AS"]
    )

    assert len(combos) == 3


def test_offsuit_blocker_removes_combinations():

    combos = generate_class_combos(
        "AKo",
        blocked_cards=["AS"]
    )

    assert len(combos) == 9