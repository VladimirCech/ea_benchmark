from benchmark_functions import *

lower_bound = -100
upper_bound = 101

test_functions = [
    sphere,
    rastrigin,
    ackley,
    rosenbrock,
    griewank,
    levy,
    schwefel,
    zakharov,
    expanded_griewank_rosenbrock,
    booth_general,
    three_hump_camel_general,
    sum2,
    powell,
    trid,
    drop_wave_general,
    perm,
    michalewicz,
    saddle,
    ellipse,
    salomon,
    sum_of_different_powers,
    alpine_function,
    modified_bent_cigar,
]


def generate_population(count, dimension):
    population = []
    for i in range(count):
        population.append(list(np.random.randint(low=lower_bound, high=upper_bound, size=dimension)))
    return population


def de_rand_1_bin(func, dim, pop_size, F=0.8, CR=0.9):
    population = generate_population(pop_size, dim)
    population = np.array(population)

    max_gen = dim * 2000

    # Hlavní smyčka algoritmu
    for gen in range(max_gen):
        for i in range(len(population)):
            # Výběr tří různých indexů (r1, r2, r3) náhodně z populace
            candidates = list(range(0, i)) + list(range(i + 1, pop_size))
            indexes = np.random.choice(candidates, 3, replace=False)
            r1, r2, r3 = population[indexes[0]], population[indexes[1]], population[indexes[2]]

            # Mutace
            mutant = r1 + F * (r2 - r3)

            # Binární křížení
            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]

                # Kontrola hranic a upravení metodou odrazu
                if trial[j] < lower_bound:
                    trial[j] = lower_bound + abs(trial[j] - lower_bound)
                elif trial[j] > upper_bound:
                    trial[j] = upper_bound - abs(trial[j] - upper_bound)

            # Selekcí
            if func(trial) < func(population[i]):
                population[i] = trial

    # Nalezení a vrácení nejlepšího řešení
    best_idx = np.argmin([func(ind) for ind in population])
    return population[best_idx]


def de_best_1_bin(func, dim, pop_size, F=0.8, CR=0.9):
    population = generate_population(pop_size, dim)
    population = np.array(population)

    max_gen = dim * 2000

    # Hlavní smyčka algoritmu
    for gen in range(max_gen):
        # Nalezení nejlepšího jedince
        best_idx = np.argmin([func(ind) for ind in population])
        best = population[best_idx]

        for i in range(len(population)):
            # Výběr dvou různých indexů (r1, r2) náhodně z populace
            candidates = list(range(0, i)) + list(range(i + 1, pop_size))
            indexes = np.random.choice(candidates, 2, replace=False)
            r1, r2 = population[indexes[0]], population[indexes[1]]

            # Mutace
            mutant = best + F * (r1 - r2)

            # Binární křížení
            trial = np.copy(population[i])
            for j in range(dim):
                if np.random.rand() < CR:
                    trial[j] = mutant[j]

                # Kontrola hranic a upravení metodou odrazu
                if trial[j] < lower_bound:
                    trial[j] = lower_bound + abs(trial[j] - lower_bound)
                elif trial[j] > upper_bound:
                    trial[j] = upper_bound - abs(trial[j] - upper_bound)

            # Selekcí
            if func(trial) < func(population[i]):
                population[i] = trial

    # Nalezení a vrácení nejlepšího řešení
    best_idx = np.argmin([func(ind) for ind in population])
    return population[best_idx]


dim_count = 5
pop_count = 20

print("----------------- DE RAND/1/BIN -----------------")

for x, function in enumerate(test_functions):
    print(f"{x}.", de_rand_1_bin(function, dim_count, pop_count), function.__name__)

print("----------------- DE BEST/1/BIN -----------------")

for x, function in enumerate(test_functions):
    print(f"{x}.",de_best_1_bin(function, dim_count, pop_count), function.__name__)
