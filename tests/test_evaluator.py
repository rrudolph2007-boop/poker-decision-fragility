from poker.evaluator import evaluate_hand

def test_straight_flush():

    hand = [
        "9S",
        "TS",
        "JS",
        "QS",
        "KS",
        "2D",
        "3C"
    ]

    assert evaluate_hand(hand) == [
        8,
        13
    ]


def test_four_of_a_kind():

    hand = [
        "AH",
        "AD",
        "AS",
        "AC",
        "KD",
        "2S",
        "3H"
    ]

    assert evaluate_hand(hand) == [
        7,
        14,
        13
    ]


def test_full_house():

    hand = [
        "KH",
        "KD",
        "KS",
        "9C",
        "9D",
        "2S",
        "3H"
    ]

    assert evaluate_hand(hand) == [
        6,
        13,
        9
    ]


def test_flush():

    hand = [
        "AS",
        "JS",
        "8S",
        "5S",
        "2S",
        "KD",
        "QC"
    ]

    assert evaluate_hand(hand) == [
        5,
        14,
        11,
        8,
        5,
        2
    ]


def test_straight():

    hand = [
        "9S",
        "TH",
        "JD",
        "QC",
        "KS",
        "2D",
        "3C"
    ]

    assert evaluate_hand(hand) == [
        4,
        13
    ]


def test_wheel_straight():

    hand = [
        "AS",
        "2H",
        "3D",
        "4C",
        "5S",
        "KD",
        "QC"
    ]

    assert evaluate_hand(hand) == [
        4,
        5
    ]


def test_three_of_a_kind():

    hand = [
        "QH",
        "QD",
        "QS",
        "AC",
        "9D",
        "2S",
        "3H"
    ]

    assert evaluate_hand(hand) == [
        3,
        12,
        14,
        9
    ]


def test_two_pair():

    hand = [
        "AH",
        "AD",
        "KC",
        "KD",
        "QS",
        "2D",
        "3C"
    ]

    assert evaluate_hand(hand) == [
        2,
        14,
        13,
        12
    ]


def test_one_pair():

    hand = [
        "JH",
        "JD",
        "AS",
        "KC",
        "9D",
        "2S",
        "3H"
    ]

    assert evaluate_hand(hand) == [
        1,
        11,
        14,
        13,
        9
    ]


def test_high_card():

    hand = [
        "AS",
        "KD",
        "QC",
        "9H",
        "7S",
        "4D",
        "2C"
    ]

    assert evaluate_hand(hand) == [
        0,
        14,
        13,
        12,
        9,
        7
    ]


def test_stronger_hand_beats_weaker_hand():

    full_house = evaluate_hand(
        [
            "KH",
            "KD",
            "KS",
            "9C",
            "9D",
            "2S",
            "3H"
        ]
    )

    flush = evaluate_hand(
        [
            "AS",
            "JS",
            "8S",
            "5S",
            "2S",
            "KD",
            "QC"
        ]
    )

    assert full_house > flush


def test_board_play_tie():

    board = [
        "AS",
        "KD",
        "QH",
        "JC",
        "TS"
    ]

    player_one = [
        "2C",
        "3D"
    ]

    player_two = [
        "4C",
        "5D"
    ]

    player_one_score = evaluate_hand(
        player_one + board
    )

    player_two_score = evaluate_hand(
        player_two + board
    )

    assert (
        player_one_score
        == player_two_score
    )