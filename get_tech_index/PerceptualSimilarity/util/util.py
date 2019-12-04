from __future__ import print_function

import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

def load_source_image(path1):
    import cv2
    img1=cv2.imread(path1)
    img1_size=img1.shape
    if(img1_size[0]<img1_size[1]):
        img1=img1[0:img1_size[0],int((img1_size[1]-img1_size[0])/2):int((img1_size[1]+img1_size[0])/2)]
    else:
        img1=img1[int((img1_size[0]-img1_size[1])/2):int((img1_size[0]+img1_size[1])/2),0:img1_size[1]]
    img1=cv2.resize(img1, (256,256), interpolation=cv2.INTER_AREA)
    return img1[:,:,::-1]

def load_image(path):
    if(path[-3:] == 'dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png'):#返回一个bgr的图片array
        import cv2
        return cv2.imread(path)[:,:,::-1]
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')

    return img

def save_image(image_numpy, image_path, ):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
