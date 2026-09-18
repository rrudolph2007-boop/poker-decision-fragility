from poker.cards import rank_values

# ============================================================
# HAND EVALUATOR
# ============================================================

def evaluate_hand(cards):

    values = []

    for card in cards:

        values.append(
            rank_values[card[0]]
        )


    # Count how many times each rank appears

    counts = {}

    for value in values:

        if value in counts:

            counts[value] += 1

        else:

            counts[value] = 1


    pair_count = 0
    three_count = 0

    has_pair = False
    has_three_of_a_kind = False
    has_four_of_a_kind = False


    for count in counts.values():

        if count == 2:

            pair_count += 1
            has_pair = True

        elif count == 3:

            three_count += 1
            has_three_of_a_kind = True

        elif count == 4:

            has_four_of_a_kind = True


    has_two_pair = (
        pair_count >= 2
    )


    has_full_house = (
        (
            pair_count >= 1
            and three_count >= 1
        )
        or three_count >= 2
    )


    # --------------------------------------------------------
    # PAIR RANKS
    # --------------------------------------------------------

    pair_ranks = []

    for rank in counts:

        if counts[rank] == 2:

            pair_ranks.append(
                rank
            )


    pair_ranks.sort(
        reverse=True
    )


    # --------------------------------------------------------
    # TRIP RANKS
    # --------------------------------------------------------

    trip_ranks = []

    for rank in counts:

        if counts[rank] == 3:

            trip_ranks.append(
                rank
            )


    trip_ranks.sort(
        reverse=True
    )


    # --------------------------------------------------------
    # FLUSH
    # --------------------------------------------------------

    suit_counts = {}


    for card in cards:

        suit = card[1]

        if suit in suit_counts:

            suit_counts[suit] += 1

        else:

            suit_counts[suit] = 1


    has_flush = False
    flush_suit = None


    for suit in suit_counts:

        if suit_counts[suit] >= 5:

            has_flush = True
            flush_suit = suit


    # --------------------------------------------------------
    # STRAIGHT
    # --------------------------------------------------------

    straight_values = list(
        set(values)
    )


    # Allow Ace to function as low card in A-2-3-4-5

    if 14 in straight_values:

        straight_values.append(
            1
        )


    straight_values = list(
        set(straight_values)
    )


    straight_values.sort()


    consecutive = 1

    has_straight = False
    straight_high = 0


    for i in range(
        1,
        len(straight_values)
    ):

        if (
            straight_values[i]
            == straight_values[i - 1] + 1
        ):

            consecutive += 1

        else:

            consecutive = 1


        if consecutive >= 5:

            has_straight = True

            straight_high = (
                straight_values[i]
            )


    # --------------------------------------------------------
    # FLUSH VALUES
    # --------------------------------------------------------

    flush_values = []


    if has_flush:

        for card in cards:

            if card[1] == flush_suit:

                flush_values.append(
                    rank_values[
                        card[0]
                    ]
                )


    # --------------------------------------------------------
    # STRAIGHT FLUSH
    # --------------------------------------------------------

    straight_flush_values = (
        flush_values.copy()
    )


    if 14 in straight_flush_values:

        straight_flush_values.append(
            1
        )


    straight_flush_values = list(
        set(straight_flush_values)
    )


    straight_flush_values.sort()


    consecutive = 1

    has_straight_flush = False
    straight_flush_high = 0


    for i in range(
        1,
        len(straight_flush_values)
    ):

        if (
            straight_flush_values[i]
            == straight_flush_values[i - 1] + 1
        ):

            consecutive += 1

        else:

            consecutive = 1


        if consecutive >= 5:

            has_straight_flush = True

            straight_flush_high = (
                straight_flush_values[i]
            )


    # --------------------------------------------------------
    # HIGH CARD
    # --------------------------------------------------------

    sorted_values = (
        values.copy()
    )


    sorted_values.sort(
        reverse=True
    )


    high_card_score = (
        [0]
        + sorted_values[:5]
    )


    # --------------------------------------------------------
    # PAIR
    # --------------------------------------------------------

    if has_pair:

        pair_rank = (
            pair_ranks[0]
        )


        kickers = []


        for value in values:

            if value != pair_rank:

                kickers.append(
                    value
                )


        kickers.sort(
            reverse=True
        )


        pair_score = (
            [1, pair_rank]
            + kickers[:3]
        )


    # --------------------------------------------------------
    # TWO PAIR
    # --------------------------------------------------------

    if has_two_pair:

        best_pair_ranks = (
            pair_ranks[:2]
        )


        kickers = []


        for value in values:

            if value not in best_pair_ranks:

                kickers.append(
                    value
                )


        kickers.sort(
            reverse=True
        )


        two_pair_score = (
            [2]
            + best_pair_ranks
            + [kickers[0]]
        )


    # --------------------------------------------------------
    # THREE OF A KIND
    # --------------------------------------------------------

    if has_three_of_a_kind:

        trip_rank = (
            trip_ranks[0]
        )


        kickers = []


        for value in values:

            if value != trip_rank:

                kickers.append(
                    value
                )


        kickers.sort(
            reverse=True
        )


        three_of_a_kind_score = (
            [3, trip_rank]
            + kickers[:2]
        )


    # --------------------------------------------------------
    # STRAIGHT
    # --------------------------------------------------------

    if has_straight:

        straight_score = [
            4,
            straight_high
        ]


    # --------------------------------------------------------
    # FLUSH
    # --------------------------------------------------------

    if has_flush:

        flush_values.sort(
            reverse=True
        )


        flush_score = (
            [5]
            + flush_values[:5]
        )


    # --------------------------------------------------------
    # FULL HOUSE
    # --------------------------------------------------------

    if has_full_house:

        full_house_trip = (
            trip_ranks[0]
        )


        if len(trip_ranks) >= 2:

            full_house_pair = (
                trip_ranks[1]
            )

        else:

            full_house_pair = (
                pair_ranks[0]
            )


        full_house_score = [
            6,
            full_house_trip,
            full_house_pair
        ]


    # --------------------------------------------------------
    # FOUR OF A KIND
    # --------------------------------------------------------

    if has_four_of_a_kind:

        quad_rank = 0


        for rank in counts:

            if counts[rank] == 4:

                quad_rank = rank


        kickers = []


        for value in values:

            if value != quad_rank:

                kickers.append(
                    value
                )


        kickers.sort(
            reverse=True
        )


        four_of_a_kind_score = [
            7,
            quad_rank,
            kickers[0]
        ]


    # --------------------------------------------------------
    # STRAIGHT FLUSH
    # --------------------------------------------------------

    if has_straight_flush:

        straight_flush_score = [
            8,
            straight_flush_high
        ]


    # --------------------------------------------------------
    # RETURN STRONGEST HAND
    # --------------------------------------------------------

    if has_straight_flush:

        return straight_flush_score

    elif has_four_of_a_kind:

        return four_of_a_kind_score

    elif has_full_house:

        return full_house_score

    elif has_flush:

        return flush_score

    elif has_straight:

        return straight_score

    elif has_three_of_a_kind:

        return three_of_a_kind_score

    elif has_two_pair:

        return two_pair_score

    elif has_pair:

        return pair_score

    else:

        return high_card_score