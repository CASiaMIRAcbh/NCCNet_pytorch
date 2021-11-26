import Deform
import random
import numpy
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import os

def gene_data_in_folders(srcpath, dstpath, pairnum=1000, begin=1):
    # subpath = 'E:\\zbf\\zbf-small\\4-8'
    imgsInSub = os.listdir(srcpath)
    startName = imgsInSub[0]
    sstr1 = startName.split('layer')[-1]
    sstr2 = sstr1.split('_')[0]
    suffix = sstr1.split(sstr2)[-1]
    startId = int(sstr2)
    endName = imgsInSub[-1]
    estr1 = endName.split('layer')[-1]
    estr2 = estr1.split('_')[0]
    endId = int(estr2)
    imgId = begin
    if endId-startId < pairnum:
        pairnum = endId-startId
    for imgIndex in range(pairnum):
        pimgId = imgIndex + startId
        nimgId = pimgId + 1
        pimg = Image.open(os.path.join(srcpath, 'layer{imgid}'.format(imgid=pimgId) + suffix))
        nimg = Image.open(os.path.join(srcpath, 'layer{imgid}'.format(imgid=nimgId) + suffix))
        # 4600->1150   downsample 4
        pimg = pimg.resize((1150, 1150))
        nimg = nimg.resize((1150, 1150))
        ss, ts = Deform.whole_change(pimg, nimg)
        for source, template in zip(ss, ts):
            cv2.imwrite(os.path.join(dstpath + '\\' + '{:0>8d}_s.tif'.format(imgId)), source)
            cv2.imwrite(os.path.join(dstpath + '\\' + '{:0>8d}_t.tif'.format(imgId)), template)
            imgId = imgId + 1

    return imgId