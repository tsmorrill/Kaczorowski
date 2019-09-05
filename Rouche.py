from numpy import absolute as abs
from numpy import exp as exp
from numpy import floor as fl
from numpy import log as log
from numpy import pi as pi
from numpy import sin as sin

from scipy.special import expn as E1 # -Ei(-x) = E1(x)
from scipy.optimize import minimize

length = 1000

file = open('zeros1', 'r')
zeros = file.read().split('\n')
tail = [float(string) for string in zeros[length:2*length]]
zeros = [float(string) for string in zeros[:length]]
file.close()
print('Approximating by using first {} zeros of zeta.'.format(len(zeros)))

def FN(z):
    sum = 0
    for zero in zeros:
        sum += exp(complex(0, zero)*z)/complex(0.5, zero)
    return sum

def FN_min(xi, epsilon):
    """Calculate lower point for FN on circle of radius epsilon about xi."""
    def f(t):
        return abs(FN(xi + epsilon*exp(t*1j)))
    return minimize(f, 0).fun

def GN_max(xi, epsilon):
    """Calculate upper bound for GN on circle of radius epsilon about xi."""
    u, v = xi.real, xi.imag
    y = v - epsilon
    gamma = tail[0]
    Ei = -E1(1, y*gamma)
    return (1/y + 4*log(gamma*(1-1/2/pi)))*exp(-y*gamma) + 2*Ei/log(gamma/2/pi)

def Rouche_test(xi, epsilon):
    return(FN_min(xi, epsilon), GN_max(xi, epsilon))
