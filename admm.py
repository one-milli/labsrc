"""
Solver for problem below using ADMM
min_f ||g-Hf||^2 + tau*||Df||_1 + i(f)
"""

import numpy as np
import scipy.sparse as ssp


class Admm:
    max_iter = 300
    tau = 1e0
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
        self.m, self.n = self.H.shape
        self.r = np.zeros((self.n, 1))
        self.f = np.ones((self.n, 1))
        self.z = np.zeros((self.D.shape[0], 1))
        self.y = np.zeros((self.m, 1))
        self.w = np.zeros((self.n, 1))
        self.x = np.zeros((self.m, 1))
        self.eta = self.mu2 * self.D @ self.f
        self.rho = np.zeros((self.n, 1))
        self.err = []
        print("Initialized")

    def soft_threshold(self, x, sigma):
        return np.maximum(0, np.abs(x) - sigma) * np.sign(x)

    def compute_obj(self):
        return np.linalg.norm(self.g - self.H @ self.f) ** 2 + self.tau * np.linalg.norm(self.D @ self.f, 1)

    def compute_err(self, f):
        return np.abs(np.linalg.norm(f - self.f) / np.linalg.norm(f))

    def update_r(self):
        self.r = (
            self.H.T @ (self.mu1 * self.y - self.x)
            + self.D.T @ (self.mu2 * self.z - self.eta)
            + self.mu3 * self.w
            - self.rho
        )

    def update_f(self):
        A = self.HTH + self.mu2 * self.DTD + self.mu3 * np.eye(self.n)
        self.f = np.linalg.solve(A, self.r)

    def update_z(self):
        self.z = self.soft_threshold(self.D @ self.f + self.eta / self.mu2, self.tau / self.mu2)

    def update_w(self):
        self.w = np.clip(self.f + self.rho / self.mu3, 0, 1)

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
        return np.asnumpy(self.f), self.err
