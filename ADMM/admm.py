"""
Solver for problem below using ADMM
min_f ||g-Hf||^2 + tau*||Df||_1 + i(f)
"""

import cupy as cp  # NumPyの代わりにCuPyをインポート
import cupyx.scipy.sparse as csp  # SciPyのスパース行列モジュールをCuPyで置き換え
import cupyx.scipy.sparse.linalg as cspla  # SciPyのスパース線形代数モジュールをCuPyで置き換え


class Admm:
    max_iter = 500
    mu1 = 1e1
    mu2 = 1e-1
    mu3 = 1e-1

    def __init__(self, H, g, D, tau):
        self.H = csp.csr_matrix(cp.asarray(H))
        self.g = cp.asarray(g)
        self.D = csp.csr_matrix(cp.asarray(D))
        self.tau = tau
        self.HTH = self.H.T @ self.H
        self.DTD = self.D.T @ self.D
        self.m, self.n = self.H.shape
        self.r = cp.zeros((self.n, 1))
        self.f = cp.zeros((self.n, 1))
        self.z = cp.zeros((2 * self.n, 1))
        self.y = cp.zeros((self.m, 1))
        self.w = cp.zeros((self.n, 1))
        self.x = cp.zeros((self.n, 1))
        self.eta = self.mu2 * self.D @ self.f
        self.rho = cp.zeros((self.n, 1))
        self.obj = []
        self.obj.append(self.compute_obj())
        self.err = []

    def soft_threshold(self, x, sigma):
        return cp.maximum(0, cp.abs(x) - sigma) * cp.sign(x)

    def compute_obj(self):
        return cp.linalg.norm(self.g - self.H @ self.f) ** 2 + self.eta * cspla.norm(self.D @ self.f, 1)

    def compute_err(self, f):
        return cp.linalg.norm(f - self.f)

    def update_r(self):
        self.r = (
            self.H.T @ (self.mu1 * self.y - self.x)
            + self.D.T @ (self.mu2 * self.z - self.eta)
            + self.mu3 * self.w
            - self.rho
        )

    def update_f(self):
        self.f = cspla.spsolve(self.HTH + self.mu2 * self.DTD + self.mu3 * csp.eye(self.m, self.m), self.r)

    def update_z(self):
        self.z = self.soft_threshold(self.D @ self.f + self.eta / self.mu2, self.tau / self.mu2)

    def update_w(self):
        self.w = cp.clip(self.f + self.rho / self.mu3, 0, 1)

    def update_y(self):
        self.y = self.H @ self.f + self.g

    def update_x(self):
        self.x = self.x + self.mu1 * (self.H @ self.f - self.y)

    def update_eta(self):
        self.eta = self.eta + self.mu2 * (self.D @ self.f - self.z)

    def update_rho(self):
        self.rho = self.rho + self.mu3 * (self.f - self.w)

    def solve(self):
        for i in range(self.max_iter):
            f = self.f
            self.update_r()
            self.update_f()
            self.update_z()
            self.update_w()
            self.update_y()
            self.update_x()
            self.update_eta()
            self.update_rho()
            self.obj.append(self.compute_obj())
            error = self.compute_err(f)
            self.err.append(error)
            if error < 1e-4:
                break
            print("iter =", i, "obj =", self.obj[-1], "err =", self.err[-1])
        return cp.asnumpy(self.f), cp.asnumpy(self.obj), cp.asnumpy(self.err)
