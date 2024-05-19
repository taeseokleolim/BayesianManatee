cimport cython
import numpy as np
cimport numpy as np


cdef class BaseNUTS:
    cdef double[:, :] X
    cdef double[:] y

    def __init__(self, np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=1] y):
        self.X = X
        self.y = y

    cdef np.ndarray[double, ndim=1] grad_log_likelihood(self, np.ndarray[double, ndim=1] beta):
        # Implement the gradient of the log-likelihood for your specific problem
        return NotImplementedError

    cdef leapfrog(self, np.ndarray[double, ndim=1] beta, np.ndarray[double, ndim=1] p):
        cdef np.ndarray[double, ndim=1] p_half_step = p + self.epsilon / 2 * self.grad_log_likelihood(beta)
        cdef np.ndarray[double, ndim=1] beta_new = beta + self.epsilon * p_half_step
        cdef np.ndarray[double, ndim=1] p_new = p_half_step + self.epsilon / 2 * self.grad_log_likelihood(beta_new)
        return [beta_new, p_new]

    cdef build_tree(self, double u, int v, int j, np.ndarray[double, ndim=1] beta, np.ndarray[double, ndim=1] p, double r, double Emax=1000):
        cdef np.ndarray[double, ndim=1] beta_prime, p_prime, beta_minus, p_minus, beta_plus, p_plus
        cdef int n_prime, s_prime, n_minus, n_plus, s_minus, s_plus

        if j == 0:
            beta_prime, p_prime = self.leapfrog(beta, p)
            if u <= np.exp(self.log_likelihood(beta_prime) - 0.5*np.dot(p_prime, p_prime)):
                n_prime = 1
            else:
                n_prime = 0
            s_prime = int(self.log_likelihood(beta_prime) - 0.5*np.dot(p_prime, p_prime) > u - Emax)
            return beta_prime, p_prime, beta_prime, p_prime, n_prime, s_prime
        else:
            beta_minus, p_minus, beta_plus, p_plus, n_minus, s_minus = self.build_tree(u, v, j-1, beta, p, r, Emax)
            if s_minus == 1:
                if v == -1:
                    beta_minus, p_minus, _, _, n_prime, s_prime = self.build_tree(u, v, j-1, beta_minus, p_minus, r, Emax)
                else:
                    _, _, beta_plus, p_plus, n_prime, s_prime = self.build_tree(u, v, j-1, beta_plus, p_plus, r, Emax)
                if (np.dot(beta_plus-beta_minus, p_minus) >= 0) and (np.dot(beta_plus-beta_minus, p_plus) >= 0):
                    s_prime = 1
                else:
                    s_prime = 0
                n_prime = n_minus + n_prime
            return beta_minus, p_minus, beta_plus, p_plus, n_prime, s_prime

    cpdef np.ndarray[double, ndim=1] NUTS(self, np.ndarray[double, ndim=1] current_beta, int L):
        cdef np.ndarray[double, ndim=1] p = np.random.normal(size=1)
        cdef double u = np.random.uniform(low=0, high=np.exp(self.log_likelihood(current_beta) - 0.5*np.dot(p, p)))
        cdef np.ndarray[double, ndim=1] beta_minus = current_beta
        cdef np.ndarray[double, ndim=1] beta_plus = current_beta
        cdef np.ndarray[double, ndim=1] beta_prime = current_beta
        cdef np.ndarray[double, ndim=1] p_minus = np.empty_like(p)
        p_minus[:] = p
        cdef np.ndarray[double, ndim=1] p_plus = np.empty_like(p)
        p_plus[:] = p
        cdef int j = 0
        cdef int n_prime = 1
        cdef int s_prime = 1
        cdef double r = 1e-10
        cdef int v
        while s_prime == 1:
            v = np.random.choice([-1, 1])
            if v == -1:
                beta_minus, p_minus, _, _, beta_prime, n_prime, s_prime = self.build_tree(u, v, j, beta_minus, p_minus, r)
            else:
                _, _, beta_plus, p_plus, beta_prime, n_prime, s_prime = self.build_tree(u, v, j, beta_plus, p_plus, r)
            r += n_prime
            if s_prime == 1 and np.random.uniform() < min(1, n_prime / r):
                current_beta = beta_prime
            j += 1
        return current_beta

cdef class LogitNUTS(BaseNUTS):
    def __init__(self, np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=1] y, int n_param, double epsilon):
        super().__init__(X, y, n_param, epsilon)

    cdef double log_likelihood(self, np.ndarray[double, ndim=1] beta):
        cdef np.ndarray[double, ndim=1] eta = np.dot(self.X, beta)
        cdef np.ndarray[double, ndim=1] p = 1.0 / (1.0 + np.exp(-eta))
        cdef np.ndarray[double, ndim=1] ones = np.ones_like(p)
        return np.sum(self.y * np.log(p) + (ones - self.y) * np.log(ones - p))

    cdef np.ndarray[double, ndim=1] grad_log_likelihood(self, np.ndarray[double, ndim=1] beta):
        cdef np.ndarray[double, ndim=1] eta = np.dot(self.X, beta)
        cdef np.ndarray[double, ndim=1] p = 1.0 / (1.0 + np.exp(-eta))
        return np.dot(self.X.T, self.y - p)