import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# # simple pytorch dilation from https://www.zhihu.com/question/466370919/answer/1952767030
# def tensor_dilation(bin_img, ksize=5):
#     # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
#     B, C, H, W = bin_img.shape
#     pad = (ksize - 1) // 2
#     bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

#     # 将原图 unfold 成 patch
#     patches = bin_img.unfold(dimension=2, size=ksize, step=1)
#     patches = patches.unfold(dimension=3, size=ksize, step=1)
#     # B x C x H x W x k x k

#     # 取每个 patch 中最大的值，i.e., 1
#     dilated, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
#     return dilated


def loss(ncc, similar, radius=5, eps=0.001, device='cpu'):
    
    # get ncc_max
    v1, _ = torch.max(ncc, dim=2, keepdim=True)
    p_max, _ = torch.max(v1, dim=3, keepdim=True)
    mask_p = ncc.ge(p_max-eps).float()

    # create a mask which is used to ignore ncc_max
    # mask_p = tensor_dilation(mask_p, radius*2+1)

    # the tensor_dilation need too much memory, use conv2d
    kernel = torch.full((1, 1, radius*2+1, radius*2+1), fill_value=1.).float()
    kernel = kernel.to(device)
    mask_p = F.conv2d(mask_p, kernel, padding=radius)
    mask_p = mask_p.lt(1.).float()
    
    # focus on the second max val
    ncc_masked = torch.mul(ncc, mask_p)
    v2, _ = torch.max(ncc_masked, dim=2, keepdim=True)
    p_max_2, _ = torch.max(v2, dim=3, keepdim=True)

    # the loss for similar pair
    p_max = torch.squeeze(p_max)
    p_max_2 = torch.squeeze(p_max_2)
    l = -(p_max - p_max_2)

    # if a pair is not similar, try to make p_max smaller
    l = torch.where(similar.gt(0), l, torch.abs(p_max))

    # get mean loss
    l = torch.mean(l)
    l.requires_grad_(True)

    return l