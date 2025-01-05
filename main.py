""" Main script for running all algorithms on all benchmark functions. """

# optimization_algorithms.py

from typing import Callable, List, Tuple, Dict
import numpy as np
from benchmark_functions import functions
import multiprocessing
import pandas as pd
from scipy.stats import friedmanchisquare

# Parametry algoritmů
DIMENSIONS = [2, 10, 20]
POP_SIZES = {2: 10, 10: 20, 20: 40}
FEs = {2: 2000 * 2, 10: 2000 * 10, 20: 2000 * 20}
REPEATS = 20

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

# DE parameters
DE_RAND_1_BIN_F = 0.8
DE_RAND_1_BIN_CR = 0.9
DE_BEST_1_BIN_F = 0.5
DE_BEST_1_BIN_CR = 0.9

# PSO parameters
PSO_C1 = 1.49618
PSO_C2 = 1.49618
PSO_W = 0.7298

def generate_population(pop_size: int, dimension: int) -> np.ndarray:
    """Generuje počáteční populaci s plovoucí desetinnou čárkou."""
    return np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(pop_size, dimension))

def de_rand_1_bin(
    func: Callable[[np.ndarray], float],
    dim: int,
    pop_size: int,
    F: float = DE_RAND_1_BIN_F,
    CR: float = DE_RAND_1_BIN_CR,
    max_gen: int = 1000
) -> Tuple[np.ndarray, float]:
    """Differential Evolution DE/rand/1/bin"""
    population = generate_population(pop_size, dim)
    fitness = np.array([func(ind) for ind in population])

    for gen in range(max_gen):
        for i in range(pop_size):
            # Výběr tří různých jedinců
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2, r3 = population[np.random.choice(indices, 3, replace=False)]

            # Mutace
            mutant = r1 + F * (r2 - r3)
            mutant = np.clip(mutant, LOWER_BOUND, UPPER_BOUND)

            # Binární křížení
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            # Vyhodnocení trial jedince
            trial_fitness = func(trial)

            # Selekce
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

    # Nalezení nejlepšího jedince
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

def de_best_1_bin(
    func: Callable[[np.ndarray], float],
    dim: int,
    pop_size: int,
    F: float = DE_BEST_1_BIN_F,
    CR: float = DE_BEST_1_BIN_CR,
    max_gen: int = 1000
) -> Tuple[np.ndarray, float]:
    """Differential Evolution DE/best/1/bin"""
    population = generate_population(pop_size, dim)
    fitness = np.array([func(ind) for ind in population])

    for gen in range(max_gen):
        # Najdi nejlepší jedince v populaci
        best_idx = np.argmin(fitness)
        best = population[best_idx]

        for i in range(pop_size):
            # Výběr dvou různých jedinců
            indices = list(range(pop_size))
            indices.remove(i)
            r1, r2 = population[np.random.choice(indices, 2, replace=False)]

            # Mutace
            mutant = best + F * (r1 - r2)
            mutant = np.clip(mutant, LOWER_BOUND, UPPER_BOUND)

            # Binární křížení
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            # Vyhodnocení trial jedince
            trial_fitness = func(trial)

            # Selekce
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

    # Nalezení nejlepšího jedince
    best_idx = np.argmin(fitness)
    return population[best_idx], fitness[best_idx]

def pso(
    func: Callable[[np.ndarray], float],
    dim: int,
    pop_size: int,
    c1: float = PSO_C1,
    c2: float = PSO_C2,
    w: float = PSO_W,
    max_gen: int = 1000
) -> Tuple[np.ndarray, float]:
    """Particle Swarm Optimization (PSO)"""
    # Inicializace populace a rychlosti
    population = generate_population(pop_size, dim)
    velocity = np.zeros((pop_size, dim))

    # Inicializace osobních a globálních nejlepších hodnot
    fitness = np.array([func(ind) for ind in population])
    personal_best_positions = population.copy()
    personal_best_fitness = fitness.copy()
    global_best_idx = np.argmin(fitness)
    global_best_position = population[global_best_idx].copy()
    global_best_fitness = fitness[global_best_idx]

    for gen in range(max_gen):
        r1 = np.random.rand(pop_size, dim)
        r2 = np.random.rand(pop_size, dim)

        # Aktualizace rychlosti
        velocity = w * velocity + c1 * r1 * (personal_best_positions - population) + c2 * r2 * (global_best_position - population)

        # Aktualizace pozice
        population += velocity
        population = np.clip(population, LOWER_BOUND, UPPER_BOUND)

        # Vyhodnocení nové populace
        fitness = np.array([func(ind) for ind in population])

        # Aktualizace osobních nejlepších
        better_fitness_mask = fitness < personal_best_fitness
        personal_best_positions[better_fitness_mask] = population[better_fitness_mask]
        personal_best_fitness[better_fitness_mask] = fitness[better_fitness_mask]

        # Aktualizace globálního nejlepšího
        current_global_best_idx = np.argmin(personal_best_fitness)
        current_global_best_fitness = personal_best_fitness[current_global_best_idx]
        if current_global_best_fitness < global_best_fitness:
            global_best_fitness = current_global_best_fitness
            global_best_position = personal_best_positions[current_global_best_idx].copy()

    return global_best_position, global_best_fitness

def run_single_test(
    func: Callable[[np.ndarray], float],
    dim: int,
    pop_size: int,
    max_gen: int,
    algorithms: List[Callable]
) -> Dict[str, float]:
    """Runs all algorithms for a single function, dimension, and population size."""
    results = {}
    solutions = {}
    for algorithm in algorithms:
        solution, fitness = algorithm(func, dim, pop_size, max_gen=max_gen)
        results[algorithm.__name__] = fitness
        # Převést řešení na řetězec pro uložení do CSV
        solutions[algorithm.__name__ + '_solution'] = ','.join(map(str, solution))
    # Kombinace výsledků a řešení do jednoho slovníku
    combined_results = {}
    for alg in algorithms:
        combined_results[alg.__name__] = results[alg.__name__]
        combined_results[alg.__name__ + '_solution'] = solutions[alg.__name__ + '_solution']
    return combined_results

def process_task(task: Tuple[Callable, int, int, int, List[Callable], int]) -> Dict[str, float]:
    """
    Procesní funkce pro běh jednotlivých testů.
    Musí být na úrovni modulu pro multiprocessing.
    """
    func, dim, pop_size, max_gen, algorithms, repeat = task
    print(f"Running Repeat {repeat}/{REPEATS}: Function: {func.__name__}, Dimension: {dim}, Population Size: {pop_size}")
    fitness_results = run_single_test(func, dim, pop_size, max_gen, algorithms)
    record = {
        'Function': func.__name__,
        'Dimension': dim,
        'Population Size': pop_size,
        'Repeat': repeat
    }
    # Přidání fitness a solution pro každý algoritmus
    for key, value in fitness_results.items():
        record[key] = value
    return record

def run_all_tests(
    functions: List[Callable[[np.ndarray], float]],
    dimensions: List[int],
    pop_sizes: Dict[int, int],
    FEs: Dict[int, int],
    repeats: int,
    algorithms: List[Callable]
) -> pd.DataFrame:
    """Runs all tests across functions, dimensions, population sizes, algorithms and repeats."""
    records = []

    # Vytvoření seznamu úkolů
    tasks = []
    for dim in dimensions:
        pop_size = pop_sizes[dim]
        max_gen = FEs[dim] // pop_size
        for func in functions:
            for repeat in range(repeats):
                tasks.append((func, dim, pop_size, max_gen, algorithms, repeat + 1))

    # Paralelizace pomocí multiprocessing.Pool
    with multiprocessing.Pool() as pool:
        for record in pool.imap_unordered(process_task, tasks):
            records.append(record)

    # Vytvoření DataFrame
    df = pd.DataFrame(records)
    return df

def rank_algorithms(df: pd.DataFrame, algorithms: List[str], dim: int) -> pd.DataFrame:
    """Ranks algorithms for each function based on their fitness."""
    subset = df[df['Dimension'] == dim]
    # Vybrat pouze fitness hodnoty pro algoritmy
    alg_columns = [alg for alg in algorithms]
    # Nižší fitness znamená lepší rank
    ranks = subset.groupby('Function')[alg_columns].mean().rank(method='average', ascending=True)
    return ranks

def calculate_average_ranks(ranks: pd.DataFrame, algorithms: List[str]) -> Dict[str, float]:
    """Calculates average ranks for each algorithm."""
    avg_ranks = ranks.mean()
    return avg_ranks.to_dict()

def select_top_functions(df: pd.DataFrame, algorithms: List[str], dim: int, top_n: int = 5) -> List[str]:
    """Selects top_n functions with the largest differences in algorithm rankings for a given dimension."""
    ranks = rank_algorithms(df, algorithms, dim)
    # Vypočítat rozdíl mezi nejlepším a nejhorším rankem pro každou funkci
    rank_diff = ranks.max(axis=1) - ranks.min(axis=1)
    # Vybrat top_n funkcí s největším rozdílem
    top_functions = rank_diff.nlargest(top_n).index.tolist()
    return top_functions

def perform_friedman_test(df: pd.DataFrame, algorithms: List[str], dim: int) -> Tuple[float, float]:
    """Performs Friedman test for a specific dimension."""
    subset = df[df['Dimension'] == dim]
    # Shromažďujeme fitness výsledky pro algoritmy
    alg_results = [subset[alg].values for alg in algorithms]
    # Friedmanův test
    stat, p = friedmanchisquare(*alg_results)
    return stat, p

def main():
    """Hlavní funkce pro spuštění všech testů a analýzu výsledků."""
    # Definice parametrů
    dimensions = [2]
    pop_sizes = POP_SIZES
    FEs_dict = FEs
    repeats = REPEATS
    algorithms = [de_rand_1_bin, de_best_1_bin, pso]
    algorithm_names = [alg.__name__ for alg in algorithms]

    # Spuštění všech testů
    print("Spouštím všechny testy. Tento proces může chvíli trvat...")
    df = run_all_tests(functions, dimensions, pop_sizes, FEs_dict, repeats, algorithms)
    print("Testy dokončeny.")

    # Uložení výsledků do CSV pro případnou další analýzu
    df.to_csv('optimization_results.csv', index=False)
    print("Výsledky byly uloženy do 'optimization_results.csv'.")

    # Statistická analýza pomocí Friedmanova testu
    print("\nProvádím Friedmanův rank test pro každou dimenzi...")
    for dim in dimensions:
        print(f"\nDimension: {dim}")
        # Shromažďujeme fitness výsledky pro algoritmy
        alg_results = [df[df['Dimension'] == dim][alg].values for alg in algorithm_names]
        # Friedmanův test
        stat, p = friedmanchisquare(*alg_results)
        print(f"Friedman test statistic: {stat:.4f}, p-value: {p:.4f}")
        if p < 0.05:
            print("Existují statisticky významné rozdíly mezi algoritmy.")
        else:
            print("Neexistují statisticky významné rozdíly mezi algoritmy.")

    # Určení pořadí algoritmů a výběr klíčových funkcí
    print("\nUrčuji pořadí algoritmů a vybírám klíčové funkce...")
    for dim in dimensions:
        print(f"\nDimension: {dim}")
        # Vytvoření DataFrame s rankingy
        ranks = rank_algorithms(df, algorithm_names, dim)
        # Výpočet průměrného ranku pro každý algoritmus
        avg_ranks = calculate_average_ranks(ranks, algorithm_names)
        print("Průměrné ranky algoritmů:")
        for alg, rank in avg_ranks.items():
            print(f"{alg}: {rank:.2f}")
        # Výběr top 5 funkcí s největším rozdílem v rankingu
        top_functions = select_top_functions(df, algorithm_names, dim, top_n=5)
        print(f"Top 5 funkcí s největším rozdílem v rankingu algoritmů pro dimenzi {dim}:")
        for func in top_functions:
            print(f"- {func}")
        # Můžeš zde přidat kód pro odůvodnění na základě vlastního chápání

if __name__ == "__main__":
    main()
