import torch.utils.data as data
import os
import numpy as np
import cv2
import glob
import random
import torchvision
import torch

def default_loader(image_path, normalize=True):
    a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    
    tran = torchvision.transforms.ToTensor()
    ta = tran(a)
    # mean, std = ta.mean(keepdim=True), ta.std(keepdim=True)
    # tran_norm = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(mean, std)
    # ])
    # tn = tran_norm(a)
    
    if normalize:
        return ta
    else:
        return ta*255.0

class ListDataset(data.Dataset):
    def __init__(self, path_list, loader=default_loader):
        self.path_list = path_list
        self.loader = loader

    def __getitem__(self, index):
        source, template, bsimilar = self.path_list[index]
        source = self.loader(source)
        template = self.loader(template)

        return source, template, bsimilar

    def __len__(self):
        return len(self.path_list)

def random_similar(train_sets, prop=0.5):
    dispairs = random.sample(train_sets, int(prop * len(train_sets)))
    sources = []
    templates = []
    out = []
    for pair in dispairs:
        sources.append(pair[0])
        templates.append(pair[1])
    random.shuffle(sources)
    for s, t in zip(sources, templates):
        out.append([s, t, -1.])
    return out

def make_dataset(dir, split_val=0.9, pairs=-1):
    # for torch-NCCNet train data and eval data
    images = []
    for img in glob.iglob(os.path.join(dir, '*_s.tif')):
        img = os.path.basename(img)
        index = img.split('_')[0]
        source = os.path.join(dir, index + '_s.tif')
        template = os.path.join(dir, index + '_t.tif')
        if not (os.path.isfile(os.path.join(dir, source)) or os.path.isfile(os.path.join(dir, template))):
            continue
        images.append([source, template, 1.])
        
    split_judge = np.random.uniform(0, 1, len(images)) < split_val
    train_set = [sample for sample, split in zip(images, split_judge) if split]
    eval_set = [sample for sample, split in zip(images, split_judge) if not split]
    
    dis_set = random_similar(train_set)
    train_set = train_set + dis_set
    
    random.shuffle(train_set)
    random.shuffle(eval_set)

    if pairs > 0:
        train_set = train_set[:pairs]
        eval_set = eval_set[:round(0.2*pairs)]

    return ListDataset(train_set), ListDataset(eval_set)