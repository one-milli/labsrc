import numpy as np


# Proximal Gradient Method
def proximal_gradient(grad_f, prox, gamma, objective, init_x, tol=1e-9):
    x = init_x
    result = objective(x)   # 目的関数の値
    while True:
        x_new = prox(x - gamma * grad_f(x), gamma)
        result_new = objective(x_new)
        if (np.abs(result-result_new)/np.abs(result) < tol) == True:
            break
        x = x_new
        result = result_new
    return x_new, result_new


def prox_norm1(v, gamma, lam):
    return soft_thresh(v, gamma*lam)


def soft_thresh(b, lam):
    x_hat = np.zeros(b.shape[0])
    x_hat[b >= lam] = b[b >= lam]-lam
    x_hat[b <= -lam] = b[b <= -lam]+lam
    return x_hat


# Iterative Shrinkage Thresholding Algorithm(LASSO回帰を解く)
def ISTA(W, y, lam):
    def objective(x): return np.sum(np.power(y - np.dot(W, x), 2)
                                    )/2.0 + lam * np.sum(np.abs(x))

    def grad_f(x): return np.dot(W.T, np.dot(W, x) - y)
    # 最大特異値の逆数より小さい値をgammaとすると収束する
    (u, l, v) = np.linalg.svd(W)
    gamma = 1/max(l.real*l.real)
    # proximal operatorを再定義
    def prox(v, gamma): return prox_norm1(v, gamma, lam)

    x_init = np.random.randn(W.shape[1])
    (x_hat, result) = proximal_gradient(
        grad_f, prox, gamma, objective, x_init, 1e-5)
    return x_hat, result
