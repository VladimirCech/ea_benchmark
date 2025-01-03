from typing import Callable
import numpy as np

def sphere(x: np.ndarray) -> float:
    """
    Sphere function: f(x) = sum(x_i^2)
    """
    return float(np.sum(x ** 2))

def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    """
    A = 10
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

def ackley(x: np.ndarray) -> float:
    """
    Ackley function: f(x) = -a * exp(-b * sqrt(mean(x^2))) - exp(mean(cos(c*x))) + a + exp(1)
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    return float(-a * np.exp(-b * np.sqrt(np.sum(x ** 2) / n)) - np.exp(np.sum(np.cos(c * x)) / n) + a + np.exp(1))

def griewank(x: np.ndarray) -> float:
    """
    Griewank function: f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i)))
    """
    i = np.arange(1, len(x) + 1)
    return float(1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))))

def schwefel(x: np.ndarray) -> float:
    """
    Schwefel function: f(x) = 418.9829 * d - sum(x_i * sin(sqrt(abs(x_i))))
    """
    d = len(x)
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

def zakharov(x: np.ndarray) -> float:
    """
    Zakharov function: f(x) = sum(x_i^2) + (sum(0.5 * i * x_i))^2 + (sum(0.5 * i * x_i))^4
    """
    i = np.arange(1, len(x) + 1)
    term2 = 0.5 * i * x
    return float(np.sum(x ** 2) + np.sum(term2) ** 2 + np.sum(term2) ** 4)

def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    """
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

def michalewicz(x: np.ndarray) -> float:
    """
    Michalewicz function: f(x) = -sum(sin(x_i) * (sin(i * x_i^2 / pi))^(2m))
    """
    m = 10
    i = np.arange(1, len(x) + 1)
    return float(-np.sum(np.sin(x) * (np.sin(i * x ** 2 / np.pi) ** (2 * m))))

def levy(x: np.ndarray) -> float:
    """
    Levy function: f(x) = sin^2(pi*w_1) + sum((w_i-1)^2 * (1 + 10*sin^2(pi*w_i+1))) + (w_d-1)^2 * (1 + sin^2(2*pi*w_d))
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:]) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return float(term1 + term2 + term3)

def dixon_price(x: np.ndarray) -> float:
    """
    Dixon-Price function: f(x) = (x_1 - 1)^2 + sum(i * (2*x_i^2 - x_{i-1})^2)
    """
    return float((x[0] - 1) ** 2 + np.sum(np.arange(2, len(x) + 1) * (2 * x[1:] ** 2 - x[:-1]) ** 2))

def eggholder(x: np.ndarray) -> float:
    """
    Eggholder function: f(x) = -(x_2+47)*sin(sqrt(abs(x_2+x_1/2+47))) - x_1*sin(sqrt(abs(x_1-(x_2+47))))
    """
    return float(- (x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47)))))

def bohachevsky(x: np.ndarray) -> float:
    """
    Bohachevsky function: f(x) = x_1^2 + 2*x_2^2 - 0.3*cos(3*pi*x_1) - 0.4*cos(4*pi*x_2) + 0.7
    """
    return float(x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7)

def six_hump_camel(x: np.ndarray) -> float:
    """
    Six-Hump Camel function: f(x) = (4 - 2.1*x_1^2 + (x_1^4)/3)*x_1^2 + x_1*x_2 + (-4 + 4*x_2^2)*x_2^2
    """
    return float((4 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3) * x[0] ** 2 + x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[1] ** 2)

def drop_wave(x: np.ndarray) -> float:
    """
    Drop-Wave function: f(x) = -(1 + cos(12*sqrt(x_1^2 + x_2^2))) / (0.5*(x_1^2 + x_2^2) + 2)
    """
    sq_sum = np.sum(x ** 2)
    return float(-(1 + np.cos(12 * np.sqrt(sq_sum))) / (0.5 * sq_sum + 2))

def bent_cigar(x: np.ndarray) -> float:
    """
    Bent Cigar function: f(x) = x_1^2 + 10^6 * sum(x_2^2)
    """
    return float(x[0] ** 2 + 10 ** 6 * np.sum(x[1:] ** 2))

# Export all functions in a list
functions: list[Callable[[np.ndarray], float]] = [
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
    eggholder,
    bohachevsky,
    six_hump_camel,
    drop_wave,
    bent_cigar,
]

# Example usage
if __name__ == "__main__":
    sample = np.random.uniform(-100, 100, 10)  # Random 10-dimensional vector
    for func in functions:
        print(f"{func.__name__}: {func(sample)}")
