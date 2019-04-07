import numpy as np
import cv2
from PIL import Image
import torch


class IMG_resize(object):
    def __init__(self, w, h):
        self.W = w
        self.H = h

    def __call__(self, IMG_clip):
        # list input
        IMG = np.array(IMG_clip)
        resized_Img = cv2.resize(IMG, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        out = Image.fromarray(resized_Img)
        return out


class CenterCrop(object):
    def __init__(self, size):
        self.size = (int(size), int(size))
        self.th = self.size[0]
        self.tw = self.size[1]

    def __call__(self, imgs):
        Images = np.array(imgs)
        h, w, c = Images.shape
        i = int(np.round((h - self.th) / 2.))
        j = int(np.round((w - self.tw) / 2.))
        out = Images[i:i + self.th, j:j + self.tw, :]
        out = Image.fromarray(out)
        return out

class To_Tensor_My(object):
    # def __init__(self, w, h):
    def __call__(self, IMG_clip):
        # list input
        tensor = torch.from_numpy(np.array(IMG_clip)).float()
        # tensor = torch.from_numpy(np.array(IMG_clip)).float()[10:200,:,:]
        out = tensor.permute(2, 0, 1).contiguous()
        return out


class Norm_my(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
