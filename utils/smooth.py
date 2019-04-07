# import torch


def winsmooth(mat, kernelsize=1):
    mat = mat.data
    n = mat.shape[0]
    out = mat.clone()
    for m in range(n):
        a = max(0, m - kernelsize)
        b = min(n - 1, m + kernelsize)
        out[m, :] = mat[a:b + 1, :].mean(0)
    return out
