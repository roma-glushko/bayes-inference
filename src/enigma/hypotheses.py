from collections import Callable
from numbers import Number
from typing import Any, Iterable, Union

from enigma.random_variables import RandomVariable

Likelihood = Callable[[Any, Number], float]


class Hypotheses(RandomVariable):
    def __init__(self, *args, likelihood: Likelihood, **kwargs):
        super().__init__(*args, **kwargs)

        self.likelihood = likelihood

    def evaluate(self, new_data: Union[Number, Iterable[Number]]):
        """
        Considers evidence of new data to update believes about hypothesis
        """
        if isinstance(new_data, Number):
            new_data = [new_data]

        for data in new_data:
            for hypothesis, probability in self.items():
                self[hypothesis] = probability * self.get_likelihood(hypothesis, data)

        self.normalize()

    def get_likelihood(self, hypothesis, new_data: Number):
        """
        Computes likelihood of a hypothesis given evidence of the data
        """
        return self.likelihood(hypothesis, new_data)
