import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum


# 1. Sphere
def sphere(x):
    return sum(i**2 for i in x)


# 2. Rastrigin
def rastrigin(x):
    return 10 * len(x) + sum(i**2 - 10 * np.cos(2 * np.pi * i) for i in x)


# 3. Ackley
def ackley(x):
    n = len(x)
    return (
        -20 * np.exp(-0.2 * np.sqrt(sum(i**2 for i in x) / n))
        - np.exp(sum(np.cos(2 * np.pi * i) for i in x) / n)
        + 20
        + np.e
    )


# 4. Rosenbrock
def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


# 5. Grienwank
def griewank(x):
    part1 = sum(i**2 / 4000 for i in x)
    part2 = 1
    for i, val in enumerate(x):
        part2 *= np.cos(val / np.sqrt(i + 1))
    return 1 + part1 - part2


# 6. Levy
def levy(x):
    w = [(1 + (i - 1) / 4) for i in x]
    part1 = np.sin(np.pi * w[0]) ** 2
    part2 = sum((w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2) for i in range(len(w) - 1))
    part3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return part1 + part2 + part3


# 7. Schwefel
def schwefel(x):
    return 418.9829 * len(x) - sum(i * np.sin(np.sqrt(abs(i))) for i in x)


# 8. Zakharov
def zakharov(x):
    part1 = sum(i**2 for i in x)
    part2 = sum(0.5 * i * x[i] for i in range(len(x)))
    return part1 + part2**2 + part2**4


# 9. Styblinski-Tang
def styblinski_tang(x):
    return 0.5 * sum(i**4 - 16 * i**2 + 5 * i for i in x)


# 10. Expanded Griewank-Rosenbrock
def f8(x):
    return 1/4000 * x**2 - np.cos(x) + 1

def f2(x0, x1):
    return 100 * (x1 - x0**2)**2 + (x0 - 1)**2

def expanded_griewank_rosenbrock(x):
    n = len(x)
    s = 0
    for i in range(n-1):
        s += f8(f2(x[i], x[i+1]))
    return s



# 11. Booth ###
def booth_general(x):
    part1 = sum((x[i] + 2 * x[i + 1] - 7) ** 2 for i in range(len(x) - 1))
    part2 = sum((2 * x[i] + x[i + 1] - 5) ** 2 for i in range(len(x) - 1))
    return part1 + part2


# 12. Three-hump camel
def three_hump_camel_general(x):
    sum_terms = sum(
        2 * x[i] ** 2 - 1.05 * x[i] ** 4 + x[i] ** 6 / 6 + x[i] * x[i + 1] + x[i + 1] ** 2 for i in range(len(x) - 1)
    )
    return sum_terms


# 13. Sum-Squares
def sum2(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1.0, n + 1)
    return sum(j * x**2)


# 14. Powell
def powell(x):
    x = np.asarray_chkfinite(x)
    n = len(x)
    n4 = ((n + 3) // 4) * 4
    if n < n4:
        x = np.append(x, np.zeros(n4 - n))
    x = x.reshape((4, -1))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
    f = np.empty_like(x)
    f[0] = x[0] + 10 * x[1]
    f[1] = sqrt(5) * (x[2] - x[3])
    f[2] = (x[1] - 2 * x[2]) ** 2
    f[3] = sqrt(10) * (x[0] - x[3]) ** 2
    return sum(f**2)


# 15. Trid
def trid(x):
    x = np.asarray_chkfinite(x)
    return sum((x - 1) ** 2) - sum(x[:-1] * x[1:])


# 16. Drop-wave
def drop_wave_general(x):
    sum_terms = sum(
        -(1 + np.cos(12 * np.sqrt(x[i] ** 2 + x[i + 1] ** 2))) / (0.5 * (x[i] ** 2 + x[i + 1] ** 2) + 2)
        for i in range(len(x) - 1)
    )
    return sum_terms


# 17. Perm 
def perm(x, b=0.5):
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1.0, n + 1)
    xbyj = np.fabs(x) / j
    return mean([mean((j**k + b) * (xbyj**k - 1)) ** 2 for k in j / n])


# 18. Michalewicz 
def michalewicz(x):  # mich.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(1.0, n + 1)
    return -sum(sin(x) * sin(j * x**2 / pi) ** (2 * 0.5))


# 19. Saddle
def dixonprice(x):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange(2, n + 1)
    x2 = 2 * x**2
    return sum(j * (x2[1:] - x[:-1]) ** 2) + (x[0] - 1) ** 2


# 20. Saddle
def saddle(x):
    x = np.asarray_chkfinite(x) - 1
    return np.mean(np.diff(x**2)) + 0.5 * np.mean(x**4)


# 21. Ellipse
def ellipse(x):
    x = np.asarray_chkfinite(x)
    return mean((1 - x) ** 2) + 100 * mean(np.diff(x) ** 2)


# 22. Salomon
def salomon(x):
    r = np.sqrt(sum(x**2))
    return 1 - np.cos(2 * np.pi * r) + 0.1 * r


# 23. Sum of Different Powers
def sum_of_different_powers(x):
    return sum(abs(x_i) ** (i + 1) for i, x_i in enumerate(x))


# 24. Alpine
def alpine_function(x):
    return sum(abs(x_i * np.sin(x_i) + 0.1 * x_i) for x_i in x)


# 25. Modified Bent Cigar
def modified_bent_cigar(x):
    return x[0] ** 2 + 10**6 * sum(i**2 for i in x[1:])
