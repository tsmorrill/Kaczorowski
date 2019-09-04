from numpy import absolute as abs
from numpy import exp as exp
from numpy import floor as fl
from numpy import log as log
from numpy import pi as pi
from numpy import sin as sin

from scipy.special import expn as E1 # -Ei(-x) = E1(x)

length = 1000

file = open('zeros1', 'r')
zeros = file.read().split('\n')
tail = [float(string) for string in zeros[length:2*length]]
zeros = [float(string) for string in zeros[:length]]
file.close()
print('Approximating by using first {} zeros of zeta.'.format(len(zeros)))

def init_sum(xi, epsilon):
    """Calculate lower bound for a_1(z) on circle of radius epsilon about xi."""
    u, v = xi.real, xi.imag
    gamma = 14.134725142
    rho = 1/2 + gamma
    return exp(-gamma*(v + epsilon))/abs(rho)

def middle_sum(xi, epsilon):
    """Calculate upper bound for sum_1^length a_i(z) on circle of radius epsilon about xi."""
    u, v = xi.real, xi.imag
    sum = 0
    for gamma in zeros[1:]:
        rho = 1/2 + gamma*1j
        sum += exp(-gamma*(v - epsilon))/abs(rho)
    return sum

def tail_sum(xi, epsilon):
    """Calculate upper bound for sum_{length+1} a_i(z) on circle of radius epsilon about xi."""
    u, v = xi.real, xi.imag
    y = v - epsilon
    gamma = tail[0]
    Ei = -E1(1, y*gamma)
    return (1/y + 4*log(gamma*(1-1/2/pi)))*exp(-y*gamma) + 2*Ei/log(gamma/2/pi)
