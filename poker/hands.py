from poker.cards import rank_order, suits

# ============================================================
# GENERATE ALL 169 STARTING HAND CLASSES
# ============================================================

hand_classes = []


# Pocket pairs

for rank in rank_order:

    hand_classes.append(
        rank + rank
    )


# Suited hands

for i in range(
    len(rank_order)
):

    for j in range(
        i + 1,
        len(rank_order)
    ):

        hand_classes.append(
            rank_order[i]
            + rank_order[j]
            + "s"
        )


# Offsuit hands

for i in range(
    len(rank_order)
):

    for j in range(
        i + 1,
        len(rank_order)
    ):

        hand_classes.append(
            rank_order[i]
            + rank_order[j]
            + "o"
        )


# ============================================================
# PHYSICAL CARD COMBINATIONS
# ============================================================

def generate_class_combos(
    hand_class,
    blocked_cards=None
):

    if blocked_cards is None:

        blocked_cards = []


    combos = []


    # --------------------------------------------------------
    # POCKET PAIR
    # --------------------------------------------------------

    if len(hand_class) == 2:

        rank = hand_class[0]


        for i in range(
            len(suits)
        ):

            for j in range(
                i + 1,
                len(suits)
            ):

                card1 = (
                    rank
                    + suits[i]
                )

                card2 = (
                    rank
                    + suits[j]
                )


                if (
                    card1 not in blocked_cards
                    and
                    card2 not in blocked_cards
                ):

                    combos.append(
                        [
                            card1,
                            card2
                        ]
                    )


    # --------------------------------------------------------
    # SUITED HAND
    # --------------------------------------------------------

    elif (
        hand_class[2].lower()
        == "s"
    ):

        rank1 = hand_class[0]
        rank2 = hand_class[1]


        for suit in suits:

            card1 = (
                rank1
                + suit
            )

            card2 = (
                rank2
                + suit
            )


            if (
                card1 not in blocked_cards
                and
                card2 not in blocked_cards
            ):

                combos.append(
                    [
                        card1,
                        card2
                    ]
                )


    # --------------------------------------------------------
    # OFFSUIT HAND
    # --------------------------------------------------------

    elif (
        hand_class[2].lower()
        == "o"
    ):

        rank1 = hand_class[0]
        rank2 = hand_class[1]


        for suit1 in suits:

            for suit2 in suits:

                if suit1 != suit2:

                    card1 = (
                        rank1
                        + suit1
                    )

                    card2 = (
                        rank2
                        + suit2
                    )


                    if (
                        card1 not in blocked_cards
                        and
                        card2 not in blocked_cards
                    ):

                        combos.append(
                            [
                                card1,
                                card2
                            ]
                        )


    return combos