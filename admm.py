"""
Solver for problem below using ADMM
min_f ||g-Hf||^2 + tau*||Df||_1 + i(f)
"""

import cupy as cp


class Admm:
    max_iter = 300
    tau = 1e1
    mu1 = 1e-5
    mu2 = 1e-4
    mu3 = 1e-4
    tol = 1e-2

    def __init__(self, H, g, D):
        self.H = H
        self.g = g.reshape(-1, 1)
        self.D = D
        self.HTH = self.H.T @ self.H
        self.DTD = self.D.T @ self.D
        self.A = self.HTH + self.mu2 * self.DTD + self.mu3 * cp.eye(self.n)
        self.m, self.n = self.H.shape
        self.r = cp.zeros((self.n, 1))
        self.f = cp.ones((self.n, 1))
        self.z = cp.zeros((self.D.shape[0], 1))
        self.y = cp.zeros((self.m, 1))
        self.w = cp.zeros((self.n, 1))
        self.x = cp.zeros((self.m, 1))
        self.eta = self.mu2 * self.D @ self.f
        self.rho = cp.zeros((self.n, 1))
        self.err = []
        print("Initialized")

    def soft_threshold(self, x, sigma):
        return cp.maximum(0, cp.abs(x) - sigma) * cp.sign(x)

    def compute_obj(self):
        return cp.linalg.norm(self.g - self.H @ self.f) ** 2 + self.tau * cp.linalg.norm(self.D @ self.f, 1)

    def compute_err(self, f):
        return cp.abs(cp.linalg.norm(f - self.f) / cp.linalg.norm(f))

    def update_r(self):
        self.r = (
            self.H.T @ (self.mu1 * self.y - self.x)
            + self.D.T @ (self.mu2 * self.z - self.eta)
            + self.mu3 * self.w
            - self.rho
        )

    def update_f(self):
        self.f = cp.linalg.solve(self.A, self.r)

    def update_z(self):
        self.z = self.soft_threshold(
            self.D @ self.f + self.eta / self.mu2, self.tau / self.mu2)

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
            pre_f = self.f.copy()
            self.update_r()
            self.update_f()
            self.update_z()
            self.update_w()
            self.update_y()
            self.update_x()
            self.update_eta()
            self.update_rho()
            error = self.compute_err(pre_f)
            self.err.append(error)
            print("iter =", i, "err =", error)
            if error < self.tol:
                break
        return cp.asnumpy(self.f), self.err
