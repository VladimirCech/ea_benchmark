""" benchmark_functions.py - Module with benchmark functions for optimization algorithms """

from typing import Callable, List
import numpy as np

def sphere(x: np.ndarray) -> float:
    """
    Sphere function: f(x) = sum(x_i^2)
    """
    return float(np.sum(x ** 2))

def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    """
    A = 10
    n = x.size
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

def ackley(x: np.ndarray) -> float:
    """
    Ackley function: f(x) = -20 * exp(-0.2 * sqrt(sum(x_i^2)/n)) - exp(sum(cos(2*pi*x_i))/n) + 20 + e
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = x.size
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return float(term1 + term2 + a + np.exp(1))

def griewank(x: np.ndarray) -> float:
    """
    Griewank function: f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i)))
    """
    sum_sq = np.sum(x ** 2) / 4000
    i = np.arange(1, x.size + 1)
    prod_cos = np.prod(np.cos(x / np.sqrt(i)))
    return float(1 + sum_sq - prod_cos)

def schwefel(x: np.ndarray) -> float:
    """
    Schwefel function: f(x) = 418.9829 * n - sum(x_i * sin(sqrt(|x_i|)))
    """
    n = x.size
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def zakharov(x: np.ndarray) -> float:
    """
    Zakharov function: f(x) = sum(x_i^2) + (sum(0.5 * i * x_i))^2 + (sum(0.5 * i * x_i))^4
    """
    i = np.arange(1, x.size + 1)
    sum_sq = np.sum(x ** 2)
    sum_linear = np.sum(0.5 * i * x)
    return float(sum_sq + sum_linear ** 2 + sum_linear ** 4)

def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    """
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

def michalewicz(x: np.ndarray) -> float:
    """
    Michalewicz function: f(x) = -sum(sin(x_i) * (sin(i * x_i^2 / pi))^(2m))
    """
    m = 10
    i = np.arange(1, x.size + 1)
    return float(-np.sum(np.sin(x) * (np.sin(i * x ** 2 / np.pi) ** (2 * m))))

def levy(x: np.ndarray) -> float:
    """
    Levy function: f(x) = sin^2(pi*w1) + sum((w_i-1)^2 * (1 + 10*sin^2(pi*w_i+1))) + (w_n-1)^2 * (1 + sin^2(2*pi*w_n))
    where w_i = 1 + (x_i - 1)/4
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:]) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)

def dixon_price(x: np.ndarray) -> float:
    """
    Dixon-Price function: f(x) = (x1 - 1)^2 + sum(i*(2*x_i^2 - x_{i-1})^2) for i=2 to n
    """
    return float((x[0] - 1) ** 2 + np.sum(np.arange(2, x.size + 1) * (2 * x[1:] ** 2 - x[:-1]) ** 2))

def alpine(x: np.ndarray) -> float:
    """
    Alpine function: f(x) = sum(|x_i * sin(x_i) + 0.1 * x_i|)
    """
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))

def bent_cigar(x: np.ndarray) -> float:
    """
    Bent Cigar function: f(x) = x1^2 + 1e6 * sum(x_i^2) for i=2 to n
    """
    return float(x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2))

def schwefel_2_22(x: np.ndarray) -> float:
    """
    Schwefel's 2.22 function: f(x) = sum(|x_i| + sum(|x_i - x_j|)) for all i < j
    """
    sum_abs = np.sum(np.abs(x))
    sum_diff = np.sum([np.abs(x[i] - x[j]) for i in range(x.size) for j in range(i + 1, x.size)])
    return float(sum_abs + sum_diff)

def michalewicz_modified(x: np.ndarray) -> float:
    """
    Modified Michalewicz function for multi-dimensions: same as original but applies for any dimension
    """
    return michalewicz(x)

def schwefel_1_2(x: np.ndarray) -> float:
    """
    Schwefel's 1.2 function: f(x) = (sum(x_i)^2) / 4000 - prod(cos(x_i / sqrt(i))) + 1
    """
    sum_sq = np.sum(x) ** 2 / 4000
    i = np.arange(1, x.size + 1)
    prod_cos = np.prod(np.cos(x / np.sqrt(i)))
    return float(sum_sq - prod_cos + 1)

# Export all functions in a list with optimized type hints
functions: List[Callable[[np.ndarray], float]] = [
    sphere,
    rastrigin,
    ackley,
    griewank,
    schwefel,
    zakharov,
    rosenbrock,
    michalewicz,
    levy,
    dixon_price,
    alpine,
    bent_cigar,
    schwefel_2_22,
    michalewicz_modified,
    schwefel_1_2,
]

# Example usage
if __name__ == "__main__":
    # Definuj dimenzi podle funkce
    dimensions = [2, 10, 20]
    for dim in dimensions:
        print(f"\nTesting with dimension: {dim}")
        x = np.random.uniform(-100, 100, dim)  # Náhodný vektor v daném rozsahu
        for func in functions:
            try:
                result = func(x)
                print(f"{func.__name__}: {result}")
            except Exception as e:
                print(f"{func.__name__}: Error - {e}")
