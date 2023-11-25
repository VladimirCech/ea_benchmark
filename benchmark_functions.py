import numpy as np


# 1. Sphere
def sphere(x):
    return sum(i ** 2 for i in x)


# 2. Rastrigin
def rastrigin(x):
    return 10 * len(x) + sum(i ** 2 - 10 * np.cos(2 * np.pi * i) for i in x)


# 3. Ackley
def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(sum(i ** 2 for i in x) / n)) - np.exp(
        sum(np.cos(2 * np.pi * i) for i in x) / n) + 20 + np.e


# 4. Rosenbrock
def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


# 5. Grienwank
def griewank(x):
    part1 = sum(i ** 2 / 4000 for i in x)
    part2 = 1
    for i, val in enumerate(x):
        part2 *= np.cos(val / np.sqrt(i + 1))
    return 1 + part1 - part2


# 6. Levy
def levy(x):
    w = [(1 + (i - 1) / 4) for i in x]
    part1 = np.sin(np.pi * w[0])**2
    part2 = sum((w[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w[i] + 1)**2)
                for i in range(len(w) - 1))
    part3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
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
    return 0.5 * sum(i**4 - 16*i**2 + 5*i for i in x)


# 10. Matyas
def matyas_general(x):
    sum_squares = sum(i**2 for i in x)
    sum_products = sum(x[i] * x[i+1] for i in range(len(x) - 1))
    return 0.26 * sum_squares - 0.48 * sum_products


# 11. Booth
def booth_general(x):
    part1 = sum((x[i] + 2*x[i+1] - 7)**2 for i in range(len(x) - 1))
    part2 = sum((2*x[i] + x[i+1] - 5)**2 for i in range(len(x) - 1))
    return part1 + part2


# 12. Three-hump camel
def three_hump_camel_general(x):
    sum_terms = sum(2*x[i]**2 - 1.05*x[i]**4 + x[i]**6 / 6 +
                    x[i] * x[i+1] + x[i+1]**2 for i in range(len(x) - 1))
    return sum_terms


# 13. Easom
def easom_general(x):
    product_terms = 1
    for i in range(len(x) - 1):
        product_terms *= -np.cos(x[i]) * np.cos(x[i+1]) * \
            np.exp(-((x[i] - np.pi)**2 + (x[i+1] - np.pi)**2))
    return product_terms


# 14. Cross-in-tray
def cross_in_tray_general(x):
    product_terms = 1
    for i in range(len(x) - 1):
        product_terms *= -0.0001 * (abs(np.sin(x[i]) * np.sin(x[i+1]) * np.exp(
            abs(100 - np.sqrt(x[i]**2 + x[i+1]**2) / np.pi)) + 1)**0.1)
    return product_terms


# 15. Six-hump camel
def six_hump_camel_back_general(x):
    sum_terms = sum((4 - 2.1*x[i]**2 + x[i]**4 / 3)*x[i]**2 + x[i] *
                    x[i+1] + (4*x[i+1]**2 - 4)*x[i+1]**2 for i in range(len(x) - 1))
    return sum_terms


# 16. Drop-wave
def drop_wave_general(x):
    sum_terms = sum(- (1 + np.cos(12 * np.sqrt(x[i]**2 + x[i+1]**2))) / (
        0.5 * (x[i]**2 + x[i+1]**2) + 2) for i in range(len(x) - 1))
    return sum_terms


# 17. Bukin_n6
def bukin_n6_general(x):
    sum_terms = sum(100 * np.sqrt(abs(x[i+1] - 0.01*x[i]**2)) +
                    0.01 * abs(x[i] + 10) for i in range(len(x) - 1))
    return sum_terms


# 18. Goldstein-price
def goldstein_price_general(x):
    sum_terms = sum((1 + (x[i] + x[i+1] + 1)**2 * (19 - 14*x[i] + 3*x[i]**2 - 14*x[i+1] + 6*x[i]*x[i+1] + 3*x[i+1]**2)) * (30 + (
        2*x[i] - 3*x[i+1])**2 * (18 - 32*x[i] + 12*x[i]**2 + 48*x[i+1] - 36*x[i]*x[i+1] + 27*x[i+1]**2)) for i in range(len(x) - 1))
    return sum_terms


# 19. De Jong f4
def de_jong_f4(x):
    return sum((i + 1) * x_i**4 for i, x_i in enumerate(x))


# 20. Schaffer n4
def schaffer_n4(x):
    return sum(0.5 + (np.sin(np.sqrt(x[i]**2 + x[i+1]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[i+1]**2))**2 for i in range(len(x) - 1))


# 21. Salomon
def salomon(x):
    r = np.sqrt(sum(i**2 for i in x))
    return 1 - np.cos(2 * np.pi * r) + 0.1 * r


# 22. Schaffer n2
def schaffer_n2(x):
    return sum(0.5 + (np.sin(x[i]**2 - x[i+1]**2)**2 - 0.5) / (1 + 0.001 * (x[i]**2 - x[i+1]**2))**2 for i in range(len(x) - 1))


# 23. Sine-Envelope Sine-Wave
def sine_envelope_sine_wave(x):
    return sum(0.5 + (np.sin(np.sqrt(x[i]**2 + x[i+1]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[i+1]**2))**2 for i in range(len(x) - 1))


# 24. Modified-Exponential
def modified_exponential(x):
    return -np.exp(-0.5 * sum(i**2 for i in x))


# 25. Modofied Bent Cigar
def modified_bent_cigar(x):
    return x[0]**2 + 10**6 * sum(i**2 for i in x[1:])
