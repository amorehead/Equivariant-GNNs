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


def get_real_spherical_harmonics(x, l, m):
    _, po, az = cart2sph(x[0], x[1], x[2])
    Y = sph_harm(abs(m), l, az, po)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    else:
        Y = Y.real
    return Y


def get_kernel_weights(x, k, l, alpha=1, beta=0.5):
    w = np.zeros((2 * l + 1, 2 * k + 1))
    for J in range(np.abs(k - l), k + l + 1):
        QY = np.zeros((2 * l + 1, 2 * k + 1))
        for m in range(-J, J + 1):
            Q_lk_transpose = get_Q_lk_transpose(k, l, J, m)
            y_Jm = get_real_spherical_harmonics(x, J, m)
            QY = QY + Q_lk_transpose * y_Jm
        w = w + radial_function(x, J, alpha, beta) * QY
    return w


def forward_pass(x, f, k, l, alpha=1, beta=0.5):
    w = get_kernel_weights(x, k, l, alpha, beta)
    return np.matmul(w, f)


if __name__ == "__main__":
    seed = 2021
    in_degree = 1
    out_degree = 1

    np.random.seed(seed)
    x = np.random.random((3, 1))
    f = np.random.random((3, 1))
    s = np.random.randn(3, 3)
    r, __ = np.linalg.qr(s)

    J = 1
    y_J_r = np.array([get_real_spherical_harmonics(np.matmul(r,x), J , i) for i in (-1,0,1)])
    Y_J = np.array([get_real_spherical_harmonics(x, J , i) for i in (-1,0,1)])
    DJ_yJ = np.matmul(r,Y_J)

    # y_J_r and DJ_yJ should be equal according to https://arxiv.org/abs/1802.08219 Section 4.1.1
    print(y_J_r)
    print(DJ_yJ)

    # f_out = forward_pass(x, f, in_degree, out_degree, alpha=1, beta=0.5)
    # f_out_rotate = forward_pass(np.matmul(r.T, x), np.matmul(r.T, f), in_degree,
    #                             out_degree, alpha=1, beta=0.5)
    # print(np.matmul(r, f_out))
    # print(f_out_rotate)
