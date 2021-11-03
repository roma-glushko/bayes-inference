# Posterior

A lightweight framework for Bayes's inference.

## Usage

```python

"""
Problem:
    There are two bowls of cookies.
    Bowl 1 contains 30 vanilla cookies and 10 chocolate cookies.
    Bowl 2 contains 20 of each.
    You choose one of the bowls at random and, without looking, then select a cookie at random.
    The cookie is vanilla.
    What is the probability that it came from Bowl 1?
"""

from posterior.hypotheses import Hypotheses

cookie_likelihood = {
    "bowl1": {
        "chocolate": 1 / 4,
        "vanilla": 3 / 4
    },
    "bowl2": {
        "chocolate": 1 / 2,
        "vanilla": 1 / 2,
    },
}

hypotheses = Hypotheses(["bowl1", "bowl2"], likelihood=cookie_likelihood)
hypotheses.evaluate("vanilla")

print(hypotheses)
```

Output:

```bash
╒════════════╤═════════════════╕
│ Outcomes   │   Probabilities │
╞════════════╪═════════════════╡
│ bowl1      │             0.6 │
├────────────┼─────────────────┤
│ bowl2      │             0.4 │
╘════════════╧═════════════════╛
```