from scipy import pi
from scipy import sin
from scipy import exp
import numpy as np
from scipy.optimize import newton
from scipy.optimize import minimize
from scipy.optimize import basinhopping

file = open('zeros1', 'r')
zeros = file.read().split('\n')
zeros = [float(string) for string in zeros[:4000]]
file.close()
print('Approximating by using first {} zeros of zeta.'.format(len(zeros)))

def F(z):
    sum = 0
    for zero in zeros:
        sum += exp(complex(0, zero)*z)/complex(0.5, zero)
    return sum

def f(z):
    sum = 0
    for zero in zeros:
        sum += complex(0, zero)*exp(complex(0, zero)*z)/complex(0.5, zero)
    return sum

def F_N(z, N):
    sum = 0
    for zero in zeros[:N]:
        sum += exp(complex(0, zero)*z)/complex(0.5, zero)
    return sum

def root(guess, silent=False):
    """Find a root of F near guess using Newton's method.
    """
    root = None
    try:
        root = newton(F, guess, fprime=f)
    except:
        if not silent:
            print("Newton's method failed.")
    finally:
        return root

def alpha_a_b(coord, N, silent=True):
    """Calculate alpha, a, b for a rectangle with coordinates coord and
    truncation at N."""
    [x0, x1, y0, y1] = coord

    a = 0
    for zero in zeros[:N]:
        a += exp(-zero*y0)/abs(complex(0.5, zero))
    b = 0
    for zero in zeros[N:]:
        b += exp(-zero*y0)/abs(complex(0.5, zero))

    x_bounds = [(x0, x1)]
    y_bounds = [(y0, y1)]

    def F_north(x):
        return abs(F_N(complex(x, y1), N))
    def F_south(x):
        return abs(F_N(complex(x, y0), N))
    def F_east(y):
        return abs(F_N(complex(x1, y), N))
    def F_west(y):
        return abs(F_N(complex(x0, y), N))

    def x_bounds(f_new, x_new, f_old, x_old):
        return x0 <= x_new <= x1

    def y_bounds(f_new, x_new, f_old, x_old):
        return x0 <= x_new <= x1

    min_north = basinhopping(F_north, 0.5*(x0 + x1), stepsize=0.5*(x1-x0), accept_test=x_bounds).fun
    min_south = basinhopping(F_south, 0.5*(x0 + x1), stepsize=0.5*(x1-x0), accept_test=x_bounds).fun
    min_east = basinhopping(F_east, 0.5*(y0 + y1), stepsize=0.5*(y1-y0), accept_test=y_bounds).fun
    min_west = basinhopping(F_west, 0.5*(y0 + y1), stepsize=0.5*(y1-y0), accept_test=y_bounds).fun

    if not silent:
        tuple = (min_north, min_south, min_east, min_west)
        print("alpha = min{}.".format(tuple))

    alpha = min(min_north, min_south, min_east, min_west)

    return alpha, a, b

def old_box_q(coord, N, root):
    """Calculate alpha, a, b according to Kaczorowski."""
    [alpha, a, b] = alpha_a_b(coord, N)
    if alpha - 3*b < 0:
        return None
    return int(4*pi*a/(alpha - 3*b)) + 1

def box_q(coord, N, root, silent=True, compare=False):
    """Calculate q according to Morrill, Platt and Trudgian, with the option of
    comparing to Kaczorowski.
    """
    [x0, x1, y0, y1] = coord
    if not (x0 <= root.real <= x1 and y0 <= root.imag <= y1) and not silent:
        string = 'Error: Box does not contain root:'
        if not x0 < root.real:
            string += ' x0 too large,'
        if not root.real < x1:
            string += ' x1 too small,'
        if not y0 < root.imag:
            string += ' y0 too large,'
        if not root.imag < y1:
            string += ' y1 too small'
        string = string[:-1] + '.'
        print(string)
        return None
    [alpha, a, b] = alpha_a_b(coord, N, silent=silent)
    if not silent:
        print("alpha = {}.".format(alpha))
        print("a = {}.".format(a))
        print("b = {}.".format(b))
    w = root.imag
    aw = 0
    for zero in zeros[:N]:
        aw += exp(-zero*w)/abs(complex(0.5, zero))
    if not silent:
        print("a_w = {}.".format(aw))
    bw = 0
    for zero in zeros[N:]:
        bw += exp(-zero*w)/abs(complex(0.5, zero))
    if not silent:
        print("b_w = {}.".format(bw))
    q = int(pi*a/(alpha - b - 2*bw))
    i = 0
    if alpha - b - 2*bw <= 0:
        if not silent:
            print('Error: Inadmissable box/N.')
        return None
    while not 2*aw*sin(pi/q) + 2*pi*a/q <= alpha - b - 2*bw:
            q += 1
            i += 1
            if i == 750:
                if not silent:
                    print('Error: Timed out.')
                return None
    if compare:
        Kacz_q = old_box_q(coord, N, root)
        if not Kacz_q:
            Kacz_q = 'no result'
        print('As compared to {},'.format(Kacz_q))
    return q

def opti_box(guess, N, root):
    """Optimize q^(-N) as a function of the coordinates (x0, x1, y0, y1).
    """
    if not box_q(guess, N, root, silent=False):
        return None

    x, y = root.real, root.imag
    class MyBounds(object):
        def __init__(self, xymax=[x, np.inf, y, np.inf], xymin=[0, x, 0, y]):
            self.xymax = np.array(xymax)
            self.xymin = np.array(xymin)
        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            coord_max = bool(np.all(x <= self.xymax))
            coord_min = bool(np.all(x >= self.xymin))
            unit_wide = bool(x[1] - x[0] < 1)
            good_alpha = bool(box_q(x, N, root, silent=True))
            return coord_max and coord_min and unit_wide and good_alpha
    mybounds = MyBounds()

    class RandomDisplacementBounds(object):
        """random displacement with bounds"""
        def __init__(self, x, y, stepsize=0.5):
            self.x = x
            self.y = y
            self.stepsize = stepsize

        def __call__(self, x):
            """take a random step but ensure the new position is within the bounds"""
            while True:
                xnew = x + np.random.uniform(-self.stepsize, self.stepsize,
                                             np.shape(x))
                if x[0] < self.x < x[1] and x[2] < self.y < x[3]:
                    break
            return xnew
    take_step = RandomDisplacementBounds(x, y)

    def opti_q(x):
        coord = x
        q = box_q(coord, N, root, silent=True)
        if not q:
            return 10000
        return q

    result = basinhopping(opti_q, guess, niter=1000, accept_test=mybounds,
                          take_step=take_step)
    if result.fun == 1:
        print("That ain't good.")
    return result

def epsilon_box(x_epsilon, y_epsilon, N, root):
    x, y = root.real, root.imag
    guess = [x - x_epsilon, x + x_epsilon, y - y_epsilon, y + y_epsilon]
    return opti_box(guess, N, root)

def min_N(coord, N, root, silent=False):
    """Find a better for a (coord, N, root) that works."""
    X = box_q(coord, N, root, silent=True)**(-N)
    best_N = N
    for n in range(N+11)[:0:-1]:
        q = box_q(coord, n, root, silent=True)
        if q:
            if not silent:
                print(q**(-n))
            if q**(-n) > X:
                X = q**(-n)
                best_N = n
    return best_N
