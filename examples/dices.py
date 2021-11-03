from enigma.hypotheses import Hypotheses


def likelihood(hypothesis: float, data: float) -> float:
    if hypothesis < data:
        return 0

    return 1 / hypothesis


hypotheses = Hypotheses([4, 6, 8, 12, 20], likelihood=likelihood)
hypotheses.evaluate([6, 8, 7, 7, 5, 4])

print(str(hypotheses))
