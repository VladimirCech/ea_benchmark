import numpy as np

# 1. Rastriginova funkce
def rastrigin(x):
    n = len(x)
    return 10*n + sum([(xi**2 - 10*np.cos(2*np.pi*xi)) for xi in x])

# 2. Ackleyho funkce
def ackley(x):
    n = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([np.cos(2*np.pi*xi) for xi in x])
    return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

# 3. De Jongova funkce 1 (Sphere model)
def dejong(x):
    return sum([xi**2 for xi in x])

# 4. Goldstein-Priceova funkce
def goldstein_price(x):
    x1, x2 = x
    return (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)) * \
           (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))

# 5. Schafferova funkce
def schaffer(x):
    return 0.5 + (np.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001*(x[0]**2 + x[1]**2))**2

# 6. Griewankova funkce
def griewank(x):
    n = len(x)
    sum_term = sum([xi**2 for xi in x]) / 4000.0
    prod_term = np.prod([np.cos(xi/np.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_term - prod_term + 1

# 7. Michalewiczova funkce
def michalewicz(x, m=10):
    return -sum([np.sin(xi) * (np.sin((i+1) * xi**2 / np.pi))**(2*m) for i, xi in enumerate(x)])

# 8. Six-Hump Camelback funkce
def six_hump_camelback(x):
    x1, x2 = x
    return (4 - 2.1*x1**2 + x1**4/3.0) * x1**2 + x1*x2 + (-4 + 4*x2**2) * x2**2

# 9. Easomova funkce
def easom(x):
    x1, x2 = x
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

# 10. Styblinski-Tangova funkce
def styblinski_tang(x):
    return sum([(xi**4 - 16*xi**2 + 5*xi) for xi in x]) / 2.0
