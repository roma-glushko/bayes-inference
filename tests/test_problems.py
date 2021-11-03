import pytest as pytest

from enigma.hypotheses import Hypotheses


def test__problems__dungeons_and_dragons():
    """
    Problem:
        There is a box of dice that contains:
            - a 4-sided die,
            - a 6-sided die,
            - an 8-sided die,
            - a 12-sided die,
            - and a 20-sided die.
        Suppose you select a die from the box at random, roll it, and get a 6.
        What is the probability that I rolled each die?
    """

    def likelihood(hypothesis: float, data: float) -> float:
        if hypothesis < data:
            return 0

        return 1 / hypothesis

    hypotheses = Hypotheses([4, 6, 8, 12, 20], likelihood=likelihood)
    hypotheses.evaluate([6, 8, 7, 7, 5, 4])

    assert dict(hypotheses) == {
        4: 0,
        6: 0,
        8: 0.91584527196901,
        12: 0.08040342579700496,
        20: 0.0037513022339850668,
    }


def test__problems__two_bowls_with_chocolate_and_vanilla_cookies():
    """
    Problem:
        There are two bowls of cookies.
        Bowl 1 contains 30 vanilla cookies and 10 chocolate cookies.
        Bowl 2 contains 20 of each.
        You choose one of the bowls at random and, without looking, then select a cookie at random.
        The cookie is vanilla.
        What is the probability that it came from Bowl 1?
    """

    cookie_likelihood = {
        "bowl1": {"chocolate": 1 / 4, "vanilla": 3 / 4},
        "bowl2": {
            "chocolate": 1 / 2,
            "vanilla": 1 / 2,
        },
    }

    hypotheses = Hypotheses(["bowl1", "bowl2"], likelihood=cookie_likelihood)
    hypotheses.evaluate("vanilla")

    assert hypotheses["bowl1"] == pytest.approx(0.6)
    assert hypotheses["bowl2"] == pytest.approx(0.4)


def test__problems__monty_hall():
    """
    Problem:
        - Monty shows you three closed doors and tells you that there is a prize behind each door: one prize is a car,
            the other two are less valuable prizes like peanut butter and fake finger nails.
            The prizes are arranged at random.
            The object of the game is to guess which door has the car. If you guess right, you get to keep the car.
        - You pick a door, which we will call Door A. We’ll call the other doors B and C.
        - Before opening the door you chose, Monty increases the suspense by opening either Door B or C,
            whichever does not have the car.
            (If the car is actually behind Door A, Monty can safely open B or C, so he chooses one at random.)
        - Then Monty offers you the option to stick with your original choice or switch to
            the one remaining unopened door.
    """
    pass


def test__problems__m_and_ms():
    """
    Problem:
        In 1995, M&Ms introduced blue M&M’s.
        Before then, the color mix in a bag of plain M&M’s was
            - 30% Brown,
            - 20% Yellow,
            - 20% Red,
            - 10% Green,
            - 10% Orange,
            - 10% Tan.
        Afterward it was:
            - 24% Blue,
            - 20% Green,
            - 16% Orange
            - 14% Yellow
            - 13% Red
            - 13% Brown.
        You have two bags of M&M’s, and you don't know which one is from 1994 and which one is from 1996.
        You take one M&M from each bag. One is yellow and one is green.
        What is the probability that the yellow one came from the 1994 bag?
    """

    mm94 = {
        "brown": 0.3,
        "yellow": 0.2,
        "red": 0.2,
        "green": 0.1,
        "orange": 0.1,
        "tan": 0.1,
    }

    mm96 = {
        "brown": 0.13,
        "yellow": 0.14,
        "red": 0.13,
        "green": 0.2,
        "orange": 0.16,
        "blue": 0.24,
    }

    hypotheses_likelihood = {
        "94 96": {
            "bag1": mm94,
            "bag2": mm96,
        },
        "96 94": {
            "bag1": mm96,
            "bag2": mm94,
        },
    }

    bayes = Hypotheses(["94 96", "96 94"], likelihood=hypotheses_likelihood)
    bayes.evaluate(
        [
            "bag1.yellow",
            "bag2.green",
        ]
    )

    assert dict(bayes) == {
        "94 96": 0.7407407407407408,
        "96 94": 0.25925925925925924,
    }


def test_problems__locomotives():
    """
    Problem:
        A railroad numbers its locomotives in order 1..N.
        One day you see a locomotive with the number 60.
        Estimate how many locomotives the railroad has.
    """

    def num_likelihood(hypothesis, new_outcome) -> float:
        if hypothesis < new_outcome:
            return 0

        return 1 / hypothesis

    hypotheses = Hypotheses(
        range(1, 1001),
        probability_func=lambda number, alpha=1.0: number ** (-alpha),
        likelihood=num_likelihood,
    )
    hypotheses.evaluate([30, 60, 90])

    assert pytest.approx(164, hypotheses.mean)
