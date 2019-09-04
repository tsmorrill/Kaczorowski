from numpy import absolute as abs
from numpy import exp as exp
from numpy import floor as fl
from numpy import pi as pi
from numpy import sin as sin

length = 1000

file = open('zeros1', 'r')
zeros = file.read().split('\n')
zeros = [float(string) for string in zeros[:length]]
tail =  [float(string) for string in zeros[length:2*length]]
file.close()
print('Approximating by using first {} zeros of zeta.'.format(len(zeros)))

def a1(xi, epsilon):
    """Calculate lower bound for a_1(z) on circle of radius epsilon about xi."""
    u, v = xi.real, xi.imag
    gamma = 14.134725142
    rho = 1/2 + gamma
    return exp(-gamma*exp(u + epsilon))/abs(rho)

def a_sum(xi, epsilon):
    """Calculate upper bound for sum_1^length a_i(z) on circle of radius epsilon about xi."""
    u, v = xi.real, xi.imag
    k = fl(u/pi)
    u2 = u - k*pi
    if 0 <= u2 - epsilon <= u2 + epsilon <= pi/2:
        sin_min = sin(u2 - epsilon)
    elif pi/2 <= u2 - epsilon <= u2 + epsilon <= pi:
        sin_min = sin(u2 + epsilon)
    sum = 0
    for i in range(1, length + 1):
        gamma = zeros[i-1]
        rho = 1/2 + gamma*1j
        sum += exp(-gamma*exp(u - epsilon)*sin_min)/abs(rho)

def tail(xi, epsilon):
    """Calculate upper bound for sum_{length+1} a_i(z) on circle of radius epsilon about xi."""
