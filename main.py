from benchmark_functions import *


def generate_population(count, dimension):
    population = []
    for i in range(count):
        population.append(list(np.random.randint(
            low=-100, high=101, size=dimension)))
    return population


# def de_rand_1_bin(dimensions, population_size, repetitions):

def de_rand_1_bin(func, dim, pop_size, F=0.8, CR=0.9):

    population = generate_population(pop_size, dim)
    max_gen = dim*2000

    # Hlavní smyčka algoritmu
    for gen in range(max_gen):
        for i in range(len(population)):
            # Výběr tří různých indexů (r1, r2, r3) náhodně z populace
            candidates = list(range(0, i)) + list(range(i + 1, pop_size))
            r1, r2, r3 = population[np.random.choice(
                candidates, 3, replace=False)]

            # Mutace
            mutant = r1 + F * (r2 - r3)
            mutant = np.clip(mutant, 0, 1)  # Omezení mutantů v rozsahu [0, 1]

            # Binární křížení
            trial = np.copy(population[i])
            for j in range(len(bounds)):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]
                # Kontrola hranic a upravení metodou odrazu
                if trial[j] < bounds[j][0] or trial[j] > bounds[j][1]:
                    trial[j] = bounds[j][0] + \
                        abs(trial[j] - bounds[j][0]
                            ) % (bounds[j][1] - bounds[j][0])

            # Selekcí
            if func(trial) < func(population[i]):
                population[i] = trial

    # Nalezení a vrácení nejlepšího řešení
    best_idx = np.argmin([func(ind) for ind in population])
    return population[best_idx]
