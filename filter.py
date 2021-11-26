import torch
import nccnet
from NCC import batch_ncc
from dataset import default_loader
import os
import matplotlib.pyplot as plt
import helper

def load_model(filepath='D:\\cbh_workspace\\OtherCode\\python\\NCCNet-torch', filename='NCCNet.pth.tar'):
    model = nccnet.NCCNet()
    model.load_state_dict(torch.load(os.path.join(filepath, filename)))
    model.eval()
    return model

dir = 'E:\\zbf\\zbf-test'

if __name__ == '__main__':
    model = load_model()
    for i in range(5):
        source_name = '{:0>8d}_s.tif'.format(i)
        template_name = '{:0>8d}_t.tif'.format(i)

        source_img = default_loader(os.path.join(dir, source_name))
        template_img = default_loader(os.path.join(dir, template_name))
        source_img = torch.unsqueeze(source_img, 0)
        template_img = torch.unsqueeze(template_img, 0)
        
        ncc_pre = batch_ncc(source_img, template_img)
        
        source_f, template_f = model(source_img, template_img)
        
        source_f = source_f.cpu().detach()
        template_f = template_f.cpu().detach()
        
        ncc = batch_ncc(source_f, template_f)

        ncc_pre = ncc_pre.squeeze().numpy()
        ncc = ncc.squeeze().numpy()
        source_f = source_f.squeeze().numpy()
        template_f = template_f.squeeze().numpy()
        source_img_n = source_img.squeeze().numpy()
        template_img_n = template_img.squeeze().numpy()
        
        helper.plot_img_array([source_img_n, source_f], ncol=2)
        helper.plot_img_array([template_img_n, template_f], ncol=2)
