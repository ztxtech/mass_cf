import math

from cf.distribution import distribution, vec2mass
from cf.moment import calculate_high_order_moments_from_mgf, calculate_high_order_moments_from_cf


class FMassDistribution:
    def __init__(self, generation_function, ignore_zero=True):
        self.generation_function = generation_function
        self.ignore_zero = ignore_zero
        self.cache = {}

    def set_cardinality(self, cardinality):
        if cardinality in self.cache:
            return
        else:
            self.cache[cardinality] = {}
            self.cache[cardinality]['vector'] = distribution(cardinality, self.generation_function, self.ignore_zero)
            self.cache[cardinality]['mass'] = vec2mass(self.cache[cardinality]['vector'])

    def sampling(self, cardinality):
        self.set_cardinality(cardinality)
        return self.cache[cardinality]['vector'], self.cache[cardinality]['mass']

    def high_order_moments_from_mgf(self, order, x, f=None):
        cardinality = int(math.log2(len(x)))
        self.set_cardinality(cardinality)
        if order == 0:
            return 1
        else:
            p = self.cache[cardinality]['vector']
            if f is not None:
                x = f(x)
            return calculate_high_order_moments_from_mgf(p, x, order)

    def high_order_moments_from_cf(self, order, x, f=None):
        cardinality = int(math.log2(len(x)))
        self.set_cardinality(cardinality)
        if order == 0:
            return 1
        else:
            p = self.cache[cardinality]['vector']
            if f is not None:
                x = f(x)
            return calculate_high_order_moments_from_cf(p, x, order)
