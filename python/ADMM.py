import os
import sys
import copy
import numpy as np
import scipy as sci
from scipy import sparse
from numpy.random import *
from PIL import Image
import cv2


def arr2image(arr):
    arr = copy.deepcopy(arr)
    return Image.fromarray(np.uint8(arr))


def float_arr2image(arr):
    return arr2image(float2im_scale(arr))


def im_scale2float(vec):
    return vec.astype(np.float)


def float2im_scale(vec):
    vec[vec > 255.0] = 255.0
    vec[vec < 0.0] = 0.0
    return vec


# 画像データからパッチを切り出す関数
# shift_numで切り出す際のずらし量を指定
def gray_im2patch_vec(im, patch_hight, patch_width, shift_num):
    patchs = np.zeros((patch_hight*patch_width,
                       int(1 + (im.size[0] - patch_hight)/shift_num + (1 if (im.size[0] - patch_hight) % shift_num != 0 else 0)) *
                       int(1 + (im.size[1] - patch_width)/shift_num + (1 if (im.size[1] - patch_width) % shift_num != 0 else 0))))
    im_arr = im_scale2float(np.array(im))
    k = 0
    for i in range(1 + int((im.size[0] - patch_hight)/shift_num + (1 if (im.size[0] - patch_hight) % shift_num != 0 else 0))):
        # ずらしたことによって画像からはみ出る量(縦方向)
        h_over_shift = min(0, im.size[0] - ((i * shift_num) + patch_hight))
        for j in range(1 + int((im.size[1] - patch_width)/shift_num + (1 if (im.size[1] - patch_width) % shift_num != 0 else 0))):
            # ずらしたことによって画像からはみ出る量(横方向)
            w_over_shift = min(0, im.size[1] - ((j * shift_num) + patch_width))
            # パッチの切り出し　画像からはみ出たら、そのはみ出た分を戻して切り出し
            patchs[:, k] = im_arr[(i * shift_num) + h_over_shift:((i * shift_num) + patch_hight) + h_over_shift,
                                  (j * shift_num) + w_over_shift:((j * shift_num) + patch_width) + w_over_shift].reshape(-1)
            k += 1
    return patchs


# 画像パッチをつなぎ合わせて画像を再構成する関数
# shift_numがずらし量、パッチがオーバーラップしている個所は足し合わせて平均化
def patch_vecs2gray_im(patchs, im_hight, im_width, patch_hight, patch_width, shift_num):
    im_arr = np.zeros((im_hight, im_width))
    sum_count_arr = np.zeros((im_hight, im_width))  # 各領域何回パッチを足したか表す行列
    i = 0
    j = 0
    w_over_shift = 0
    h_over_shift = 0
    for k in range(patchs.shape[1]):
        P = patchs[:, k].reshape((patch_hight, patch_width))
        im_arr[(i * shift_num) + h_over_shift:((i * shift_num) + patch_hight) + h_over_shift,
               (j * shift_num) + w_over_shift:((j * shift_num) + patch_width) + w_over_shift] += P  # 指定の領域にパッチの足しこみ
        sum_count_arr[(i * shift_num) + h_over_shift:(i * shift_num) + patch_hight + h_over_shift,
                      (j * shift_num) + w_over_shift:(j * shift_num) + patch_width + w_over_shift] += 1  # 当該領域のパッチを足しこんだ回数をカウントアップ
        if j < ((im_width - patch_width)/shift_num + (1 if (im_width - patch_width) % shift_num != 0 else 0)):
            j += 1
            # パッチを足しこむ領域が画像からはみ出た時、そのはみ出た分を戻すための変数（横方向）
            w_over_shift = min(0, im_width - ((j * shift_num) + patch_width))
        else:
            j = 0
            i += 1
            # パッチを足しこむ領域が画像からはみ出た時、そのはみ出た分を戻すための変数（縦方向）
            h_over_shift = min(0, im_hight - ((i * shift_num) + patch_hight))
            w_over_shift = 0
    im_arr /= sum_count_arr
    return float_arr2image(im_arr), im_arr


# ブロックソフト閾値計算
def block_soft_thresh(b, lam):
    return max(0, 1 - lam/np.linalg.norm(b, 2)) * b


# グループ正則化のproximal operator
def prox_group_norm(v, groups, gamma, lam):
    u = np.zeros(v.shape[0])
    for i in range(1, np.max(groups) + 1):
        u[groups == i] = block_soft_thresh(v[groups == i], gamma * lam)
    return u


# グループ正則化計算
def gloup_l1_norm(x, groups, p):
    s = 0
    for i in range(1, np.max(groups) + 1):
        s += np.linalg.norm(x[groups == i], p)
    return s


# 関数をメモ化する関数
def memoize(f):
    table = {}

    def func(*args):
        if not args in table:
            table[args] = f(*args)
        return table[args]
    return func


# ADMMを行う関数
# argmin_x, argmin_y, update_lam, objectiveはそれぞれ変数名に準ずる計算を行う関数
def ADMM(argmin_x, argmin_y, update_lam, p, objective, x_init, y_init, lam_init, tol=1e-8):
    x = x_init
    y = y_init
    lam = lam_init
    result = objective(x)
    iter = 0
    while 1:
        x_new = argmin_x(y, lam, p)
        y_new = argmin_y(x_new, lam, p)
        lam_new = update_lam(x_new, y_new, lam, p)
        result_new = objective(x_new)
        if result_new < tol or (np.abs(result - result_new)/np.abs(result) < tol) == True:
            break
        x = x_new
        y = y_new
        lam = lam_new
        result = result_new
        print(iter, ':', result_new)
        iter += 1
    return x_new, result_new


# TV正則化付きのスムージング
# N,M:画像のピクセル行数、列数
# v:入力画像のベクトル
# B:正則化の変換行列（スパースなので、scipyのsparse行列を渡す）
# groups:変換後ベクトルの各要素の所属グループ
# C:正則化の係数
# p：拡張ラグランジュの係数
def TV_reg_smoothing(N, M, v, B, groups, C, p):
    # inv(2I + pB^T B)を計算する関数。アルゴリズムによってはpが可変なので、pを引数として受け取る。
    # 関数をメモ化して同一のpが入力された場合、再計算不要としている。
    inv_H = memoize(lambda p: np.array(np.linalg.inv(
        2.0 * np.eye(B.shape[1], B.shape[1]) + p * (B.T * B))))

    def argmin_x(y, lam, p): return np.dot(
        inv_H(p), 2.0 * v - np.array(B.T * (-p * y + lam)))
    def argmin_y(x, lam, p): return prox_group_norm(
        (B * x) + lam/p, groups, 1.0/p, C)

    def update_lam(x, y, lam, p): return lam + p*((B * x) - y)
    def objective(x): return np.linalg.norm(v - x, 2)**2 + \
        C * gloup_l1_norm((B * x), groups, 2)

    x_init = np.random.randn(B.shape[1])
    y_init = (B * x_init)
    lam_init = np.zeros(B.shape[0])

    (x, result) = ADMM(argmin_x, argmin_y, update_lam,
                       p, objective, x_init, y_init, lam_init, 1e-8)

    return x, result


# グループ変換行列を計算する関数
# N,M:画像のピクセル行数、列数
def calc_group_trans_mat(N, M):
    B = sci.sparse.lil_matrix(
        (2 * (M - 1) * (N - 1) + (M - 1) + (N - 1), M * N))
    groups = np.zeros(B.shape[0], 'int')
    k = 0
    for i in range(N):
        for j in range(M):
            base = i * M + j
            if i < N - 1 and j < M - 1:
                B[k, base] = 1
                B[k, base + 1] = -1
                B[k + 1, base] = 1
                B[k + 1, base + M] = -1
                groups[k] = int(k/2) + int(k % 2) + 1
                groups[k + 1] = int(k/2) + int(k % 2) + 1
                k += 2
            # 一番下の行のピクセルは右隣のピクセルとの差分のみ計算
            elif i >= N - 1 and j < M - 1:
                B[k, base] = 1
                B[k, base + 1] = -1
                groups[k] = int(k/2) + int(k % 2) + 1
                k += 1
            # 一番右の劣のピクセルは下のピクセルとの差分のみ計算
            elif i < N - 1 and j >= M - 1:
                B[k, base] = 1
                B[k, base + M] = -1
                groups[k] = int(k/2) + int(k % 2) + 1
                k += 1
    return B, groups


# グレースケール画像に対するノイズ除去
# img:画像データ（PILのimageクラス）
# patch_hight:切り出す画像パッチの高さ
# patch_width:切り出す画像パッチの幅
# shift_num:画像を切り出す際のずらし量（重複して切り出してもOK）
# C:正則化の係数
# p:拡張ラグランジュの係数
def denoise_gray_img(img, patch_hight, patch_width, shift_num, C, p):
    # グループ変換行列の計算
    [B, groups] = calc_group_trans_mat(patch_hight, patch_width)

    # 画像パッチの切り出し
    patchs = gray_im2patch_vec(img, patch_hight, patch_width, shift_num)

    new_patchs = np.zeros((patch_hight * patch_width, patchs.shape[1]))
    for i in range(patchs.shape[1]):
        # 各パッチに対してノイズ除去を施す
        print('i=', i)
        new_patchs[:, i] = TV_reg_smoothing(
            patch_hight, patch_width, patchs[:, i], B, groups, C, p)[0]

    # パッチをつなぎ合わせて画像の再構成
    [new_img, img_arr] = patch_vecs2gray_im(
        new_patchs, img.size[0], img.size[1], patch_hight, patch_width, shift_num)
    return new_img


patch_hight = 32
patch_width = 32
shift_num = 16
C = 30
p = 1

print('-----RED-----')
img_r = Image.open('reconst/CGM_230107/emoji_R.png')
grayimg_r = (np.array(img_r)/65535)*255
img_r = Image.fromarray(np.uint8(grayimg_r))
img_r2 = denoise_gray_img(img_r, patch_hight, patch_width, shift_num, C, p)

print('-----GREEN-----')
img_g = Image.open('reconst/CGM_230107/emoji_G.png')
grayimg_g = (np.array(img_g)/65535)*255
img_g = Image.fromarray(np.uint8(grayimg_g))
img_g2 = denoise_gray_img(img_g, patch_hight, patch_width, shift_num, C, p)

print('-----BLUE-----')
img_b = Image.open('reconst/CGM_230107/emoji_B.png')
grayimg_b = (np.array(img_b)/65535)*255
img_b = Image.fromarray(np.uint8(grayimg_b))
img_b2 = denoise_gray_img(img_b, patch_hight, patch_width, shift_num, C, p)

zeros = np.zeros((64, 64), 'uint8')
img_z = Image.fromarray(zeros)

denoise_img = Image.merge("RGB", (img_r2, img_g2, img_b2))
denoise_img.save('reconst/CGM_230107/emoji_denoise.png')
img_r2.save('reconst/CGM_230107/emoji_R_denoise.png')
img_g2.save('reconst/CGM_230107/emoji_G_denoise.png')
img_b2.save('reconst/CGM_230107/emoji_B_denoise.png')
