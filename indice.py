# NCC  and  SSIM

import torch
from torch.nn import functional as F
from torch.autograd import Variable, Function
import numpy as np
from math import exp, ceil

class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        output = -(W_col.unsqueeze(2)-X_col.unsqueeze(0)).abs().sum(1)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col

def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    out = adder.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out


"""
Normalized Cross-Correlation for pattern matching.
pytorch implementation

roger.bermudez@epfl.ch
CVLab EPFL 2019
"""

def patch_mean(images, patch_shape):
    """
    Computes the local mean of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local means computed independently for each channel.

    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> means = patch_mean(images, patch_shape)
        >>> expected_mean = images[3, 2, :5, :5].mean()  # mean of the third image, channel 2, top left 5x5 patch
        >>> computed_mean = means[3, 2, 5//2, 5//2]      # computed mean whose 5x5 neighborhood covers same patch
        >>> computed_mean.isclose(expected_mean).item()
        1
    """
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    # padding = tuple(side // 2 for side in patch_size)
    padding = 0

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    # channel_selector = torch.eye(channels).byte()
    channel_selector = torch.eye(channels).bool()
    # weights[1 - channel_selector] = 0
    weights[torch.logical_not(channel_selector)] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):
    """
    Computes the local standard deviations of an image or set of images.

    Args:
        images (Tensor): Expected size is (n_images, n_channels, *image_size). 1d, 2d, and 3d images are accepted.
        patch_shape (tuple): shape of the patch tensor (n_channels, *patch_size)

    Returns:
        Tensor same size as the image, with local standard deviations computed independently for each channel.

    Example::
        >>> images = torch.randn(4, 3, 15, 15)           # 4 images, 3 channels, 15x15 pixels each
        >>> patch_shape = 3, 5, 5                        # 3 channels, 5x5 pixels neighborhood
        >>> stds = patch_std(images, patch_shape)
        >>> patch = images[3, 2, :5, :5]
        >>> expected_std = patch.std(unbiased=False)     # standard deviation of the third image, channel 2, top left 5x5 patch
        >>> computed_std = stds[3, 2, 5//2, 5//2]        # computed standard deviation whose 5x5 neighborhood covers same patch
        >>> computed_std.isclose(expected_std).item()
        1
    """
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)

def batch_conv(source_batch, template_batch):
    sN, sC, sH, sW = source_batch.shape
    tN, tC, tH, tW = template_batch.shape

    source_batch_trans = source_batch.view_as(torch.Tensor(sC, sN, sH, sW))
    res_t = F.conv2d(source_batch_trans, template_batch, groups=sN)
    res = res_t.view_as(torch.Tensor(sN, sC, (sH-tH+1), (sW-tW+1)))

    return res


def batch_ncc(source_batch, template_batch, eps=1e-7):
    sN, sC, sH, sW = source_batch.shape
    tN, tC, tH, tW = template_batch.shape

    tbmean = torch.mean(template_batch, dim=[2, 3], keepdim=True)
    tbstd = torch.std(template_batch, dim=[2, 3], keepdim=True)
    normalized_tb = (template_batch - tbmean) / tbstd
    
    sbstd = patch_std(source_batch, (tC, tH, tW))

    patch_elements = torch.Tensor((tC, tH, tW)).prod().item()

    res = batch_conv(source_batch, normalized_tb)

    # Try to avoid loss is NaN
    res = res.div_(patch_elements * sbstd + eps)
    return res


# patch ssim

def gaussian(window_size, sigma, device='cpu'):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    gauss.to(device)
    return gauss/gauss.sum()

def create_window(channel, device='cpu'):
    sigma = 1.5
    window_size = 2*ceil(3*sigma)+1
    _1D_window = gaussian(window_size, sigma, device=device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window.to(device)
    return window, window_size

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, size_average = True):
    (_, channel, _, _) = img1.size()
    window, window_size = create_window(channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def part_ssim(img1, img2, mu1, mu2, mu1_sq, mu2_sq, sigma1_sq, sigma2_sq, window, window_size, channel, C1, C2):

    mu1_mu2 = mu1*mu2

    sigma12 = F.conv2d(img1*img2, window, padding = window_size, groups = channel) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    res = ssim_map.mean(dim=[2, 3])

    del mu1_mu2, sigma12, ssim_map, img1, img2, mu1, mu2, mu1_sq, mu2_sq, sigma1_sq, sigma2_sq, window

    return res

    





def batch_ssim_(source_batch, template_batch):

    C1 = 0.01**2
    C2 = 0.03**2
    
    device = source_batch.device
    (_, channel, sH, sW) = source_batch.size()
    (_, tC, tH, tW) = template_batch.size()
    window, window_size = create_window(channel, device=device)
    window = window.to(device)

    sb_mu = F.conv2d(source_batch, window, padding=window_size//2, groups=channel).to(device)
    sb_mu_sq = sb_mu.pow(2)
    sb_sigma = F.conv2d(source_batch*source_batch, window, padding=window_size//2, groups=channel) - sb_mu_sq
    

    tb_mu = F.conv2d(template_batch, window, padding=window_size//2, groups=channel).to(device)
    tb_mu_sq = tb_mu.pow(2)
    tb_sigma = F.conv2d(template_batch*template_batch, window, padding=window_size//2, groups=channel) - tb_mu_sq
    

    stb_mu = batch_conv(sb_mu, tb_mu)
    stb_sigma = batch_conv(source_batch, template_batch)
    stb_sigma = F.conv2d(stb_sigma, window, padding=window_size//2, groups=channel).to(device) - stb_mu

    tb_mu_sq = tb_mu_sq.mean(dim=[2,3], keepdim=True)
    tb_sigma = tb_sigma.mean(dim=[2,3], keepdim=True)

    sb_mu_sq = patch_mean(sb_mu_sq, (tC, tH, tW))
    sb_sigma = patch_mean(sb_sigma, (tC, tH, tW))

    num = (2*stb_mu + C1) * (2*stb_sigma + C2)
    den = (sb_mu_sq + tb_mu_sq + C1) * (sb_sigma + tb_sigma+ C2)

    res = num/den

    return res

def batch_ssim_silly(source_batch, template_batch):

    device = source_batch.device
    (sN, sC, sH, sW) = source_batch.size()
    (_, _, tH, tW) = template_batch.size()

    window, window_size = create_window(sC)
    window_size = window_size // 2
    if source_batch.is_cuda:
        window = window.to(device)

    C1 = 0.01**2
    C2 = 0.03**2
    
    source_batch_mu = F.conv2d(source_batch, window, padding=window_size, groups=sC)
    template_batch_mu = F.conv2d(template_batch, window, padding=window_size, groups=sC)
    source_batch_mu_sq = source_batch_mu.pow(2)
    template_batch_mu_sq = template_batch_mu.pow(2)

    source_batch_sigma_sq = F.conv2d(source_batch*source_batch, window, padding=window_size, groups=sC) - source_batch_mu_sq
    template_batch_sigma_sq = F.conv2d(template_batch*template_batch, window, padding=window_size, groups=sC) - template_batch_mu_sq

    res = torch.zeros(sN, sC, sH-tH+1, sW-tW+1).to(device)
    for i in range(sH-tH+1):
        for j in range(sW-tW+1):
            res[:, :, i, j] = part_ssim(source_batch[:,:,i:i+tH,j:j+tW], 
            template_batch,
            source_batch_mu[:,:,i:i+tH,j:j+tW],
            template_batch_mu,
            source_batch_mu_sq[:,:,i:i+tH,j:j+tW],
            template_batch_mu_sq,
            source_batch_sigma_sq[:,:,i:i+tH,j:j+tW],
            template_batch_sigma_sq,
            window,
            window_size,
            sC,
            C1, C2)

    return res
    

# def batch_ssim(source_batch, template_batch):
#     sN, sC, sH, sW = source_batch.shape
#     tN, tC, tH, tW = template_batch.shape

#     # get mu_s mu_t sigma_s sigma_t
#     mu_t = torch.mean(template_batch, dim=[2, 3], keepdim=True)
#     sigma_t = torch.std(template_batch, dim=[2, 3], keepdim=True)
    
#     mu_s = patch_mean(source_batch, (tC, tH, tW))
#     sigma_s = patch_std(source_batch, (tC, tH, tW))

#     # get c1 c2 in SSIM
#     C1 = 0.01**2
#     C2 = 0.03**2

#     # calc sigma_st
#     sub_tb = template_batch - mu_t
#     source_batch_trans = source_batch.view_as(torch.Tensor(sC, sN, sH, sW))
#     conv_res = F.conv2d(source_batch_trans, sub_tb, groups=sN)
#     sigma_st = conv_res.view_as(torch.Tensor(sN, sC, (sH-tH+1), (sW-tW+1)))
#     patch_elements = torch.Tensor((tC, tH, tW)).prod().item()
#     sigma_st = sigma_st.div_(patch_elements)

#     # calc ssim
#     numerator_p1 = 2 * mu_t * mu_s + C1
#     numerator_p2 = 2 * sigma_st + C2
#     numerator = torch.mul(numerator_p1, numerator_p2)

#     denominator_p1 = mu_t.pow(2) + mu_s.pow(2) + C1
#     denominator_p2 = sigma_t.pow(2) + sigma_s.pow(2) + C2
#     denominator = torch.mul(denominator_p1, denominator_p2)

#     res = torch.div(numerator, denominator)
#     return res