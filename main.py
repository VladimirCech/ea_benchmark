from benchmark_functions import *
from concurrent.futures import ProcessPoolExecutor

lower_bound = -100
upper_bound = 101

funcs = [
    sphere,
    rastrigin,
    ackley,
    rosenbrock,
    griewank,
    levy,
    schwefel,
    zakharov,
    styblinski_tang,
    expanded_griewank_rosenbrock,
    booth_general,
    three_hump_camel_general,
    sum2,
    powell,
    trid,
    drop_wave_general,
    perm,
    michalewicz,
    dixonprice,
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


def de_best_1_bin(func, dim, pop_size, F=0.5, CR=0.9):
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


# def pso(func, dim, pop_size, c1=1.49618, c2=1.49618, w=0.7298):
#     # Inicializace populace (hejna)
#     population = generate_population(pop_size, dim)
#     population = np.array(population, dtype=np.float64)
#     velocity = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), (pop_size, dim))

#     # Inicializace osobních a globálních nejlepších hodnot
#     personal_best = population.copy()
#     personal_best_scores = np.array([func(ind) for ind in population])
#     global_best_idx = np.argmin(personal_best_scores)
#     global_best = population[global_best_idx]

#     max_gen = dim * 2000

#     # Hlavní smyčka algoritmu PSO
#     for gen in range(max_gen):
#         for i in range(pop_size):
#             # Aktualizace rychlosti
#             r1, r2 = np.random.rand(dim), np.random.rand(dim)
#             velocity[i] = w * velocity[i] + c1 * r1 * (personal_best[i] - population[i]) + c2 * r2 * (global_best - population[i])

#             # Aktualizace pozice
#             population[i] += velocity[i]

#             # Kontrola hranic a upravení metodou odrazu
#             for j in range(dim):
#                 if population[i, j] < lower_bound:
#                     population[i, j] = lower_bound + abs(population[i, j] - lower_bound)
#                     velocity[i, j] *= -1
#                 elif population[i, j] > upper_bound:
#                     population[i, j] = upper_bound - abs(population[i, j] - upper_bound)
#                     velocity[i, j] *= -1

#             # Hodnocení a aktualizace osobních a globálních nejlepších hodnot
#             score = func(population[i])
#             if score < personal_best_scores[i]:
#                 personal_best[i] = population[i]
#                 personal_best_scores[i] = score

#             # Aktualizace globálního nejlepšího
#             if score < func(global_best):
#                 global_best = population[i]

#     # Nalezení a vrácení nejlepšího řešení
#     best_idx = np.argmin(personal_best_scores)
#     return population[best_idx]


def pso(func, dim, pop_size, c1=1.49618, c2=1.49618, w=0.7298):
    population = generate_population(pop_size, dim)
    population = np.array(population, dtype=np.float64)
    # velocity = np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), (pop_size, dim))

    max_gen = dim * 2000

    # Initialize velocity for each individual
    velocity = np.zeros((len(population), dim))

    # Initialize personal best positions and their fitnesses
    pbest_positions = np.copy(population)
    pbest_fitness = np.array([func(ind) for ind in population])

    # Initialize global best position and its fitness
    gbest_position = population[np.argmin(pbest_fitness)]
    gbest_fitness = np.min(pbest_fitness)

    for t in range(max_gen):
        for i, individual in enumerate(population):
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            velocity[i] = w * velocity[i] + \
                c1 * r1 * (pbest_positions[i] - individual) + \
                c2 * r2 * (gbest_position - individual)

            # Update position
            population[i] = individual + velocity[i]

            # Update personal best
            fitness = func(population[i])
            if fitness < pbest_fitness[i]:
                pbest_fitness[i] = fitness
                pbest_positions[i] = np.copy(population[i])

                # Update global best
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = np.copy(population[i])

    return gbest_position


dim_count = 2
pop_count = 10


def evaluate_de_rand_1_bin(function, dim_count, pop_count):
    return de_rand_1_bin(function, dim_count, pop_count), function.__name__


def evaluate_de_best_1_bin(function, dim_count, pop_count):
    return de_best_1_bin(function, dim_count, pop_count), function.__name__


if __name__ == "__main__":
    print("----------------- DE RAND/1/BIN -----------------")

    for x, function in enumerate(funcs):
        print(f"{x}.", pso(function, dim_count, pop_count), function.__name__)

    # for x, function in enumerate(funcs):
    # print(f"{x}.", de_rand_1_bin(function, dim_count, pop_count), function.__name__)

    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(evaluate_de_rand_1_bin, function, dim_count, pop_count) for function in funcs]
    #     for x, future in enumerate(futures):
    #         result, name = future.result()
    #         print(f"{x}. {result} {name}")

    # print("----------------- DE BEST/1/BIN -----------------")

    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(evaluate_de_best_1_bin, function, dim_count, pop_count) for function in funcs]
    #     for x, future in enumerate(futures):
    #         result, name = future.result()
    #         print(f"{x}. {result} {name}")

    # for x, function in enumerate(funcs):
    #     print(f"{x}.",de_best_1_bin(function, dim_count, pop_count), function.__name__)
