import cv2
import random
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.functional as TF
from PIL import Image


def _random_warp(im, ptn=20, power=15, sz=1024):
    '''
    :param im: 图像
    :param ptn: 采点对数量 每个点对贡献一个形变
    :param power: 非线性形变的程度 越小越低
    :param sz: 形变场大小
    :return:
    '''
    kp1 = []
    kp2 = []
    dmatch = []
    # 薄板样条插值算法 此处只是用来之后变形的
    tps = cv2.createThinPlateSplineShapeTransformer()
    for i in range(ptn):
        # pt1随机采点 范围[power, sz-power]
        pt1 = [random.randint(0 + power, sz - power), random.randint(0 + power, sz - power)]
        # pt2随机在pt1的结果附近随机采点
        pt2 = [pt1[0] + random.randint(-power, power), pt1[1] + random.randint(-power, power)]
        # 加入点集
        kp1.append(pt1)
        kp2.append(pt2)
        # 建立映射关系
        dmatch.append(cv2.DMatch(i, i, 0))
    # 变为列向量
    kp1 = np.array(kp1).reshape(1, -1, 2)
    kp2 = np.array(kp2).reshape(1, -1, 2)
    # print(kp1, '\n', kp2)
    # 根据映射关系得到形变场
    tps.estimateTransformation(kp1, kp2, dmatch)
    # 列表：成组变形 否则返回变形图像
    if isinstance(im, list):
        warped = []
        for img in im:
            warped.append(tps.warpImage(img))
        return warped
    return tps.warpImage(im)

def random_affine(im):
    # 随机仿射变换 各个给定值在一定范围内 具体如下
    degree = random.random() * 20 - 10
    transforms_x = random.random() * 0.06 - 0.03
    transforms_y = random.random() * 0.06 - 0.03
    scale = random.random() * 0.06 + 0.97
    sheer = random.random() * 4 - 2 # 错切度数
    warp = []
    for img in im:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        warp.append(TF.affine(img, degree, [transforms_x, transforms_y], scale, sheer))
    return warp



def random_change(im, pairs=4, source_size=512, template_size=224):
    # input must be a pair of images with same shape

    im = random_affine(im)
    for i in range(len(im)):
        im[i] = np.array(im[i])
    imsz = im[0].shape[0]
    img_s = im[0]
    img_t = im[1]
    im = _random_warp(im, ptn=20, power=40, sz=imsz)
    ss = []
    ts = []
    for i in range(pairs):
        source_u = random.randint(int(0.1*imsz), int(0.9*imsz-source_size))
        source_l = random.randint(int(0.1*imsz), int(0.9*imsz-source_size))
        template_us = random.randint(0, source_size-template_size)
        template_ls = random.randint(0, source_size-template_size)
        template_u = source_u + template_us
        template_l = source_l + template_ls
        source = img_s[source_u:source_u+source_size, source_l:source_l+source_size]
        template = img_t[template_u:template_u+template_size, template_l:template_l+template_size]
        ss.append(source)
        ts.append(template)
        
    return ss, ts

def whole_change(pimg, nimg):

        ss, ts = random_change([pimg, nimg])

        pimg2 = TF.hflip(pimg)
        nimg2 = TF.hflip(nimg)
        tss, tts = random_change([pimg2, nimg2])
        ss = ss + tss
        ts = ts + tts

        pimg = TF.vflip(pimg)
        nimg = TF.vflip(nimg)
        tss, tts = random_change([pimg, nimg])
        ss = ss + tss
        ts = ts + tts

        pimg = TF.hflip(pimg)
        nimg = TF.hflip(nimg)
        tss, tts = random_change([pimg, nimg])
        ss = ss + tss
        ts = ts + tts

        return ss, ts