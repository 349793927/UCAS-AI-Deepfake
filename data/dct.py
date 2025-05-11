import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def shuffle_patch(image, n=8):
    """将图像切分为 n*n 个 patch，随机打乱后重新拼接"""
    _, h, w = image.shape
    patch_h, patch_w = h // n, w // n

    patches = []
    for i in range(n):
        for j in range(n):
            patch = image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            patches.append(patch)

    random.shuffle(patches)

    # 重新组合
    new_image = torch.zeros_like(image)
    for idx, patch in enumerate(patches):
        i, j = divmod(idx, n)
        new_image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = patch

    return new_image





        


