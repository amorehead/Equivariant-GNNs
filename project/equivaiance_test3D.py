import numpy as np
import math
from sympy.physics.quantum.cg import CG
from scipy.special import sph_harm


def radial_function(x, J, alpha=1, beta=0.5):
    return np.exp(alpha * np.linalg.norm(x) + beta + J * 0.001)


def get_Q_lk_transpose(k, l, J, m):
    Q_lk_transpose = np.zeros((2 * l + 1, 2 * k + 1))
    for i in range(2 * l + 1):
        for j in range(2 * k + 1):
            Q_lk_transpose[i, j] = CG(J, m, k, j - k, l, i - l).doit()
    return Q_lk_transpose


def cart2sph(x, y, z):
    XsqPlusYsq = x ** 2 + y ** 2
    r = math.sqrt(XsqPlusYsq + z ** 2)  # r
    elev = math.atan2(z, math.sqrt(XsqPlusYsq))  # theta
    az = math.atan2(y, x)  # phi
    return r, elev, az


def get_spherical_harmonics(x, l, m):
    _, po, az = cart2sph(x[0], x[1], x[2])
    return sph_harm(l, m, az, po)


def get_kernel_weights(x, k, l, alpha=1, beta=0.5):
    w_re = w_img = np.zeros((2 * l + 1, 2 * k + 1))
    for J in range(np.abs(k - l), k + l + 1):
        QY_re = QY_img = np.zeros((2 * l + 1, 2 * k + 1))
        for m in range(-J, J + 1):
            Q_lk_transpose = get_Q_lk_transpose(k, l, J, m)
            y_Jm = get_spherical_harmonics(x, m, J)
            QY_re = QY_re + Q_lk_transpose * y_Jm.real
            QY_img = QY_img + Q_lk_transpose * y_Jm.imag
        w_re = w_re + radial_function(x, J, alpha, beta) * QY_re
        w_img = w_img + radial_function(x, J, alpha, beta) * QY_img
    return w_re,w_img


def forward_pass(x, f, k, l, alpha=1, beta=0.5):
    w_re, w_img = get_kernel_weights(x, k, l, alpha, beta)
    return np.matmul(w_re, f), np.matmul(w_img, f)


if __name__ == "__main__":
    seed = 2021
    in_degree = 1
    out_degree = 1

    np.random.seed(seed)
    x = np.random.random((3, 1))
    f = np.random.random((3, 1))

    f_out = forward_pass(x, f, in_degree, out_degree, alpha=1, beta=0.5)