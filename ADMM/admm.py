"""
Solver for problem below using ADMM
min_f ||g-Hf||^2 + tau*||Df||_1 + i(f)
"""


import numpy as np


class Admm:
    max_iter = 500
    mu1 = 1e1
    mu2 = 1e-1
    mu3 = 1e-1

    def __init__(self, H, g, D, tau):
        self.H = H  # System matrix
        self.g = g  # Observation
        self.D = D  # Gradient operator
        self.tau = tau
        self.HTH = self.H.T @ self.H
        self.DTD = self.D.T @ self.D
        self.m, self.n = H.shape
        self.r = np.zeros((self.n, 1))
        self.f = np.zeros((self.n, 1))
        self.z = np.zeros((2 * self.n, 1))
        self.y = np.zeros((self.m, 1))
        self.w = np.zeros((self.n, 1))
        self.x = np.zeros((self.n, 1))
        self.eta = self.mu2 * np.dot(self.D, self.f)
        self.rho = np.zeros((self.n, 1))
        self.obj = []
        self.obj.append(self.compute_obj())
        self.err = []

    def soft_threshold(self, x, sigma):
        return np.maximum(0, abs(x) - sigma) * np.sign(x)

    def compute_obj(self):
        return np.linalg.norm(self.g - np.dot(self.H, self.f))**2 + self.eta * np.linalg.norm(np.dot(self.D, self.f), 1)

    def compute_err(self, f):
        return np.linalg.norm(f - self.f)

    def update_r(self):
        self.r = np.dot(self.H.T, self.mu1 * self.y - self.x) + np.dot(self.D.T, self.mu2 * self.z - self.eta) + self.mu3 * self.w - self.rho

    def update_f(self):
        self.f = np.linalg.inv(np.dot(self.H.T, self.H) + self.mu2 * self.DTD + self.mu3 * np.eye(self.m, self.m)).dot(self.r)

    def update_z(self):
        self.z = self.soft_threshold(np.dot(self.D, self.f) + self.eta / self.mu2, self.tau / self.mu2)

    def update_w(self):
        self.w = np.clip(self.f + self.rho / self.mu3, 0, 1)

    def update_y(self):
        self.y = np.dot(self.H, self.f) + self.g

    def update_x(self):
        self.x = self.x + self.mu1 * (np.dot(self.H, self.f) - self.y)

    def update_eta(self):
        self.eta = self.eta + self.mu2 * (np.dot(self.D, self.f) - self.z)

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
            self.err.append(self.compute_err(f))
            print('iter =', i, 'obj =', self.obj[-1], 'err =', self.err[-1])
        return self.f, self.obj, self.err
