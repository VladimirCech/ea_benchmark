from benchmark_functions import *

x=[1,2,5,10]

print(rastrigin(x), rastrigin.__name__)
print(ackley(x))
print(dejong(x))
# print(goldstein_price(x))
print(schaffer(x))
print(griewank(x))
print(michalewicz(x))
# print(six_hump_camelback(x))
# print(easom(x))
