import numpy as np
from ga import continuous_genetic_algorithm


def test_x_2():
    real_fitness = 0.0
    real_solution = np.zeros(4)
    fitness, solution = continuous_genetic_algorithm([1, 1, 1, 2],
                                                     (-10, 10),
                                                     100,
                                                     98,
                                                     800, 0.95, 1.0)
    np.testing.assert_almost_equal(fitness, real_fitness, 5)
    np.testing.assert_almost_equal(solution, real_solution, 3)
    print('fitness',fitness)
    print('solution', solution)


if __name__ == '__main__':
    test_x_2()
