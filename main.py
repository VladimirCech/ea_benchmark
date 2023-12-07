from benchmark_functions import *

lower_bound = -100
upper_bound = 101


def generate_population(count, dimension):
    population = []
    for i in range(count):
        population.append(
            list(np.random.randint(low=lower_bound, high=upper_bound, size=dimension))
        )
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


dim_count = 5
pop_count = 20

print(de_rand_1_bin(sphere, dim_count, pop_count), sphere.__name__)
print(de_rand_1_bin(rastrigin, dim_count, pop_count), rastrigin.__name__)
print(de_rand_1_bin(ackley, dim_count, pop_count), ackley.__name__)
print(de_rand_1_bin(rosenbrock, dim_count, pop_count), rosenbrock.__name__)
print(de_rand_1_bin(griewank, dim_count, pop_count), griewank.__name__)
print(de_rand_1_bin(levy, dim_count, pop_count), levy.__name__)
print(de_rand_1_bin(schwefel, dim_count, pop_count), schwefel.__name__)
print(de_rand_1_bin(zakharov, dim_count, pop_count), zakharov.__name__)
print(de_rand_1_bin(styblinski_tang, dim_count, pop_count), styblinski_tang.__name__)
print(de_rand_1_bin(matyas_general, dim_count, pop_count), matyas_general.__name__)
print(de_rand_1_bin(booth_general, dim_count, pop_count), booth_general.__name__)
print(de_rand_1_bin(three_hump_camel_general, dim_count, pop_count), three_hump_camel_general.__name__)
print(de_rand_1_bin(easom_general, dim_count, pop_count), easom_general.__name__)
print(de_rand_1_bin(cross_in_tray_general, dim_count, pop_count), cross_in_tray_general.__name__)
print(de_rand_1_bin(six_hump_camel_back_general, dim_count, pop_count), six_hump_camel_back_general.__name__)
print(de_rand_1_bin(drop_wave_general, dim_count, pop_count), drop_wave_general.__name__)
print(de_rand_1_bin(bukin_n6_general, dim_count, pop_count), bukin_n6_general.__name__)
print(de_rand_1_bin(goldstein_price_general, dim_count, pop_count), goldstein_price_general.__name__)
print(de_rand_1_bin(de_jong_f4, dim_count, pop_count), de_jong_f4.__name__)
print(de_rand_1_bin(schaffer_n4, dim_count, pop_count), schaffer_n4.__name__)
print(de_rand_1_bin(salomon, dim_count, pop_count), salomon.__name__)
print(de_rand_1_bin(schaffer_n2, dim_count, pop_count), schaffer_n2.__name__)
print(de_rand_1_bin(sine_envelope_sine_wave, dim_count, pop_count), sine_envelope_sine_wave.__name__)
print(de_rand_1_bin(modified_exponential, dim_count, pop_count), modified_exponential.__name__)
print(de_rand_1_bin(modified_bent_cigar, dim_count, pop_count), modified_bent_cigar.__name__)
