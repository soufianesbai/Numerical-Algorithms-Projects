from math import *
import numpy as np
import matplotlib.pyplot as plt


## Question 1

def calculate_exponent(x):
    abs_x = abs(x)
    exponent = 0

    if x == 0:
        return 0
    
    if abs_x >= 10:
        while abs_x >= 10:
            abs_x = abs_x / 10
            exponent += 1
    elif abs_x < 1:
        while abs_x < 1:
            abs_x = abs_x * 10
            exponent -= 1
    
    return exponent


def rp(x, p):
    if x == 0:
        return 0
    sign = 1 if x > 0 else -1
    
    x = abs(x)
    exponent = calculate_exponent(x)

    x = x * 10**(-exponent) 
    x = round(x, p - 1)
    result = sign * x * (10**exponent)
    result = round(result, -exponent + p - 1)
    
    return result


## Question 2

def rp_sum(x, y, p):
    result = rp(x, p) + rp(y, p)
    return rp(result, p)

def rp_multiply(x, y, p):
    result = rp(x, p) * rp(y, p)
    return rp(result, p)


## Question 3

def sum_error(x, y, p):
    machine_sum = rp_sum(x, y, p)
    real_sum = x + y
    if real_sum == 0:
        return 0 if machine_sum == 0 else float('inf')
    error = abs((real_sum - machine_sum) / real_sum)
    return error


## Question 4

def multiply_error(x, y, p):
    machine_mul = rp_multiply(x, y, p)
    real_mul = x * y
    if real_mul == 0:
        return 0 if machine_mul == 0 else float('inf')
    error = abs((real_mul - machine_mul)/real_mul)
    return error




## Question 5

def fix_x_p(x, p):
    y_values = np.linspace(-10, 10, 1000)

    sum_errors = [sum_error(x, y, p) for y in y_values]
    mul_errors = [multiply_error(x, y, p) for y in y_values]

    plt.figure(figsize=(12, 6))
    plt.plot(y_values, sum_errors, label='Sum error')
    plt.plot(y_values, mul_errors, label='Multiplication error')
    plt.xlabel('y')
    plt.ylabel('Relative error')
    plt.title(f'Relative errors for x = {x} et p = {p}')
    plt.legend()
    plt.grid(True)
    plt.show()

    max_sum_error = max(sum_errors)
    max_mul_error = max(mul_errors)

    print(f"Maximum relative error for sum: {max_sum_error:.6e}")
    print(f"Maximum relative error for multiplication : {max_mul_error:.6e}")


