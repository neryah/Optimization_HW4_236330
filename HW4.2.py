import numpy as np
import matplotlib.pyplot as plt
from os import path, mkdir


def plot_graph(title, values, x_label, y_label, save_to):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(values)
    plt.savefig(path.join('graphs', save_to) + '.png')
    plt.close()


def modifiedChol(G):
    """
    Modified Cholesky Factorization according to prof' Michael Zibulevsky's Matlab code converted to python

	%  Given a symmetric matrix G, find a vector e of "small" norm and
	%  lower triangular matrix L, and vector d such that  G+diag(e) is Positive Definite, and
	%
	%      G+diag(e) = L*diag(d)*L'
	%
	%  Also, calculate a direction pneg, such that if G is not PSD, then
	%
	%      pneg'*G*pneg < 0
	%
	%  Reference: Gill, Murray, and Wright, "Practical Optimization", p111.
	%  Author: Brian Borchers (borchers@nmt.edu)
	%  Modification (acceleration):  Michael Zibulevsky   (mzib@cs.technion.ac.il)
	%

	:param G: a symmetric Matrix to decompose.
	"""
    assert np.allclose(G, G.T, atol=1e-8)

    #	initialze variables:
    n = G.shape[0]
    eps = np.finfo(float).eps

    diagG = np.diag(G)
    C = np.diag(diagG)

    gamma = np.max(diagG)
    zi = np.max(G - C)
    nu = np.max([1, np.sqrt(n ** 2 - 1)])
    beta2 = np.max([gamma, zi / nu, 1e-15])

    L = np.zeros((n, n))
    d = np.zeros((n, 1))
    e = np.zeros((n, 1))

    theta = np.zeros((n, 1))

    #	Perform for first element seperately (to aviod if statements):
    j = 0
    ee = range(1, n)
    C[ee, j] = G[ee, j]
    theta[j] = np.max(np.abs(C[ee, j]))

    d[j] = np.max(np.array([eps, np.abs(C[j, j]), (theta[j] ** 2) / beta2]).T)
    e[j] = d[j] - C[j, j]

    ind = [(i, i) for i in range(n)]
    for pos, e_i in zip(ind[j + 1:], ee):
        C[pos] = C[pos] - (1.0 / d[j]) * (C[e_i, j] ** 2)

    #	Perform for 2 <= j <= n-1
    for j in range(1, n - 1):
        bb = range(j)
        ee = range(j + 1, n)

        L[j, bb] = np.divide(C[j, bb], d[bb].T)
        C[ee, j] = G[ee, j] - ((C[np.ix_(ee, bb)]) @ (L[j, bb].T))
        theta[j] = np.max(np.abs(C[ee, j]))

        d[j] = np.max(np.array([eps, np.abs(C[j, j]), (theta[j] ** 2) / beta2]).T)
        e[j] = d[j] - C[j, j]

        for pos, e_i in zip(ind[j + 1:], ee):
            C[pos] = C[pos] - (1.0 / d[j]) * (C[e_i, j] ** 2)

    #	Perform for last element seperately (to aviod if statements):
    j = n - 1
    bb = range(j)
    ee = range(j + 1, n)

    L[j, bb] = np.divide(C[j, bb], d[bb].T)
    C[ee, j] = G[ee, j] - ((C[np.ix_(ee, bb)]) @ (L[j, bb].T))
    theta[j] = 0

    d[j] = np.max(np.array([eps, np.abs(C[j, j]), (theta[j] ** 2) / beta2]).T)
    e[j] = d[j] - C[j, j]

    #	Add ones on the diagonal:
    for pos in ind:
        L[pos] = 1

    return L, d, e


def norm(x):
    return np.sqrt(np.sum(x * x))


class solver:
    def __init__(self, A, b):
        self.L, self.D, _ = modifiedChol(A)
        self.b = b

    def direction(self):
        return self.__backward_sub(self.L.T, self.__diag_sub(self.D, self.__forward_sub(self.L, self.b)))

    def __forward_sub(self, A, b):
        x = np.zeros((A.shape[0], 1))

        for i in range(x.shape[0]):
            x[i, 0] = b[i, 0]
            for j in range(i):
                x[i, 0] -= A[i, j] * x[j, 0]

            x[i, 0] /= A[i, i]

        return x

    def __diag_sub(self, A, b):
        return b / A

    def __backward_sub(self, A, b):
        x = np.zeros((A.shape[0], 1))

        for i in range(x.shape[0] - 1, -1, -1):
            x[i, 0] = b[i, 0]
            for j in range(x.shape[0] - 1, i, -1):
                x[i, 0] -= A[i, j] * x[j, 0]

            x[i, 0] /= A[i, i]

        return x


# penalty function from lectures:
class Phi:
    @staticmethod
    def val(t):
        return ((t ** 2) / 2 + t) if t >= -0.5 else (-(1 / 4) * np.log(-2 * t) - 3 / 8)

    @staticmethod
    def grad(t):
        return (t + 1) if t >= -0.5 else (-(1 / (4 * t)))

    @staticmethod
    def hessian(t):
        return 1 if t >= -0.5 else (1 / (4 * (t ** 2)))


class lagrangian:
    def __init__(self, f, constraints, mus, p):
        self.f = f
        self.constraints = constraints
        self.mus = mus
        self.p = p

    def val(self):
        def inner(y):
            return self.f.val(y) + np.sum([(mu / self.p) * Phi.val(self.p * const.val(y))
                                           for mu, const in zip(self.mus, self.constraints)])

        return inner

    def grad(self):
        def inner(y):
            return self.f.grad(y) + sum([(mu * Phi.grad(self.p * const.val(y)) * const.grad(y))
                                         for mu, const in zip(self.mus, self.constraints)])

        return inner

    def hessian(self):
        def inner(y):
            return self.f.hessian(y) + sum(
                [(mu * self.p * Phi.hessian(self.p * const.val(y)) * const.grad(y) @ const.grad(y).T)
                 for mu, const in zip(self.mus, self.constraints)])

        return inner


def armijo_lr(evaluation_function, gradient_function, x, direction, alpha, beta, sigma):
    def phi(step):
        return evaluation_function(x + step * direction) - evaluation_function(x)

    while phi(alpha) >= sigma * gradient_function(x).T @ direction * alpha:
        alpha *= beta
    return alpha


def newton_method(initial_point, evaluation_function,
                  gradient_function, hessian_function, x_star,
                  epsilon=1e-5, alpha=1, sigma=0.25, beta=0.5, return_convergences=False):
    convergence_f = []
    convergence_x = []
    x_series = []
    grad_norm = []
    x = initial_point
    f_star = evaluation_function(x_star)
    g = gradient_function(x)
    while norm(g) >= epsilon:
        convergence_f.append(evaluation_function(x)[0] - f_star[0])
        convergence_x.append(norm(x - x_star))
        x_series.append(x)
        grad_norm.append(norm(g))

        direction = solver(hessian_function(x), -gradient_function(x)).direction()
        lr = armijo_lr(evaluation_function=evaluation_function,
                       gradient_function=gradient_function,
                       x=x,
                       direction=direction,
                       alpha=alpha,
                       beta=beta,
                       sigma=sigma)
        x += lr * direction
        g = gradient_function(x)
    return x, convergence_x, grad_norm, x_series if return_convergences else x


def AugmentedLagrangian(f, x_initial, x_star, optimal_mus, alpha=2, p=2, epsilon=1e-6):
    """
    Augmented Lagrangian Solver
    :param f: Minimization problem, consist of functions g with the structure:
                    g.val = f(x)
                    g.grad = g(x)
                    g.hessian = H_f(x)
        f.minimize: The function to minimize
        f.constraints: list of constraints on the solution
    :param x_initial: The initial point for the solver
    :param x_star: The optimal solution point
    :param optimal_mus: the optimal coefficients of the constraints
    :param alpha: increase rate of p
    :param p: initial value for p
    :param epsilon: tol for aggregated lagrangian gradient
    :return:
    """
    max_p = 1000
    mus = np.ones(len(f.constraints))  # coefficients for the constraints
    x = x_initial
    f_star = f.minimize.val(x_star)
    #grad_lagrangian_aggregate = []
    max_constraint_violation = []
    #residual_objective = []
    #d_optimal_point = []
    d_optimal_mus = []

    newton_grad_lagrangian_aggregate = []
    newton_max_constraint_violation = []
    newton_residual_objective = []
    newton_d_optimal_point = []
    newton_d_optimal_mus = []

    while True:
        lagrangianInstance = lagrangian(f.minimize, f.constraints, mus, p)
        evaluation_function = lagrangianInstance.val()
        gradient_function = lagrangianInstance.grad()
        hessian_function = lagrangianInstance.hessian()

        g_norm = norm(gradient_function(x))

        if g_norm < epsilon or p > max_p:
            break

        x, convergence_x, grad_norm, x_series = newton_method(initial_point=x,
                                                              evaluation_function=evaluation_function,
                                                              gradient_function=gradient_function,
                                                              hessian_function=hessian_function,
                                                              x_star=x_star,
                                                              return_convergences=True)

        mus = np.multiply(mus, [Phi.grad(p * const.val(x))[0] for const in f.constraints])
        # lagrangianInstance.mus = mus
        # gradient_function = lagrangianInstance.grad()

        convergence_f = [abs(evaluation_function(x_) - f_star) for x_ in x_series]

        # grad_lagrangian_aggregate.append(norm(gradient_function(x)))
        max_constraint_violation.append(max([const.val(x) for const in f.constraints] + [0]))
        # residual_objective.append(norm(f.val(x) - f_star))
        # d_optimal_point.append(norm(x - x_star))
        d_optimal_mus.append(norm(np.subtract(mus, optimal_mus)))

        length = len(convergence_f)
        newton_grad_lagrangian_aggregate += grad_norm
        newton_max_constraint_violation += [max_constraint_violation[-1]] * length
        newton_residual_objective += convergence_f
        newton_d_optimal_point += convergence_x
        newton_d_optimal_mus += [d_optimal_mus[-1]] * length

        p *= alpha
    print(f"Solution:\nx:\n{x}\n\n\nmus:{mus}")
    # ALS's iterations graphs
    # plot_graph(title='Gradient of the Augmented Lagrangian aggregate',
    #            values=grad_lagrangian_aggregate,
    #            x_label='ALS iteration',
    #            y_label=r'$||\nabla F_{p,\mu} (x,\lambda)||$',
    #            save_to='grad_aggregate')
    # #
    # plot_graph(title='Maximal constraint violation',
    #            values=max_constraint_violation,
    #            x_label='ALS iteration',
    #            y_label=r'$max_i (g_i (x))$',
    #            save_to='max_violation')
    # #
    # plot_graph(title='Residual in the objective function',
    #            values=residual_objective,
    #            x_label='ALS iteration',
    #            y_label=r'$|f(x) - f(x^*)|$',
    #            save_to='residual_objective')
    # #
    # plot_graph(title='Distance to the optimal point',
    #            values=d_optimal_point,
    #            x_label='ALS iteration',
    #            y_label=r'$||x-x^*||$',
    #            save_to='optimal_point')
    # #
    # plot_graph(title='Distance to the optimal multipliers',
    #            values=d_optimal_mus,
    #            x_label='ALS iteration',
    #            y_label=r'$||\lambda-\lambda ^*||$',
    #            save_to='optimal_mus')
    #
    # Newton's iterations graphs
    plot_graph(title='Gradient of the Augmented Lagrangian aggregate',
               values=newton_grad_lagrangian_aggregate,
               x_label='Newton iteration',
               y_label=r'$||\nabla F_{p,\mu} (x,\lambda)||$',
               save_to='newton_grad_aggregate')

    plot_graph(title='Maximal constraint violation',
               values=newton_max_constraint_violation,
               x_label='Newton iteration',
               y_label=r'$max_i (g_i (x))$',
               save_to='newton_max_violation')

    plot_graph(title='Residual in the objective function',
               values=newton_residual_objective,
               x_label='Newton iteration',
               y_label=r'$|f(x) - f(x^*)|$',
               save_to='newton_residual_objective')

    plot_graph(title='Distance to the optimal point',
               values=newton_d_optimal_point,
               x_label='Newton iteration',
               y_label=r'$||x-x^*||$',
               save_to='newton_optimal_point')

    plot_graph(title='Distance to the optimal multipliers',
               values=newton_d_optimal_mus,
               x_label='Newton iteration',
               y_label=r'$||\lambda-\lambda ^*||$',
               save_to='newton_optimal_mus')

    return x


# part 2: (objective function and constrains from user)
class part2target:
    def __init__(self):
        self.constraints = [self.C1, self.C2, self.C3]

    class minimize:
        @staticmethod
        def val(y):
            return 2 * (y[0] - 5) ** 2 + (y[1] - 1) ** 2

        @staticmethod
        def grad(y):
            dy1 = 4 * (y[0] - 5)
            dy2 = 2 * (y[1] - 1)
            return np.array([[dy1], [dy2]]).reshape((y.shape[0], -1))

        @staticmethod
        def hessian(y):
            return np.array([[4, 0], [0, 2]])


    # The 3 given constrains:
    class C1:
        @staticmethod
        def val(y):
            return y[1] + y[0] / 2 - 1

        @staticmethod
        def grad(y):
            dy1 = 1 / 2
            dy2 = 1
            return np.array([[dy1], [dy2]])

        @staticmethod
        def hessian(y):
            return np.array([[0, 0], [0, 0]])


    class C2:
        @staticmethod
        def val(y):
            return y[0] - y[1]

        @staticmethod
        def grad(y):
            dy1 = 1
            dy2 = -1
            return np.array([[dy1], [dy2]])

        @staticmethod
        def hessian(y):
            return np.array([[0, 0], [0, 0]])


    class C3:
        @staticmethod
        def val(y):
            return -y[0] - y[1]

        @staticmethod
        def grad(y):
            dy1 = -1
            dy2 = -1
            return np.array([[dy1], [dy2]])

        @staticmethod
        def hessian(y):
            return np.array([[0, 0], [0, 0]])


def main():
    if not path.exists('graphs'):
        mkdir('graphs')

    AugmentedLagrangian(f=part2target(),
                        x_initial=np.ones((2, 1)),
                        x_star=np.array([[2 / 3], [2 / 3]]),
                        optimal_mus=[12, 11 + 1 / 3, 0])


if __name__ == '__main__':
    main()
    # Solution:
    # x:
    # [[0.66666667]
    #  [0.66666667]]
    #
    # mus: [1.20000000e+01 1.13333333e+01 6.45937438e-15]
