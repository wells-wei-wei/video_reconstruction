import cv2
import numpy as np
import time
import os
from skimage.measure import compare_ssim 
#from numba import jit,njit

def load_source_image(path1):
    img1=cv2.imread(path1)
    img1_size=img1.shape
    if(img1_size[0]<img1_size[1]):
        img1=img1[0:img1_size[0],int((img1_size[1]-img1_size[0])/2):int((img1_size[1]+img1_size[0])/2)]
    else:
        img1=img1[int((img1_size[0]-img1_size[1])/2):int((img1_size[0]+img1_size[1])/2),0:img1_size[1]]
    img1=cv2.resize(img1, (256,256), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)

#相关操作
#由于使用的高斯函数圆对称，因此相关操作和卷积操作结果相同
#@njit
def correlation(img,kernal):
    kernal_heigh = kernal.shape[0]
    kernal_width = kernal.shape[1]
    cor_heigh = img.shape[0] - kernal_heigh + 1
    cor_width = img.shape[1] - kernal_width + 1
    result = np.zeros((cor_heigh, cor_width), dtype=np.float64)
    for i in range(cor_heigh):
        for j in range(cor_width):
            result[i][j] = (img[i:i + kernal_heigh, j:j + kernal_width] * kernal).sum()
    return result

#产生二维高斯核函数
#这个函数参考自：https://blog.csdn.net/qq_16013649/article/details/78784791
#@jit
def gaussian_2d_kernel(kernel_size=11, sigma=1.5):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    return kernel * sum_val


#ssim模型
#@jit
def ssim(distorted_image,original_image,window_size=11,gaussian_sigma=1.5,K1=0.01,K2=0.03,alfa=1,beta=1,gama=1):
    distorted_image=np.array(distorted_image,dtype=np.float64)
    original_image=np.array(original_image,dtype=np.float64)
    if not distorted_image.shape == original_image.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(distorted_image.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    kernal=gaussian_2d_kernel(window_size,gaussian_sigma)

    #求ux uy ux*uy ux^2 uy^2 sigma_x^2 sigma_y^2 sigma_xy等中间变量
    ux=correlation(distorted_image,kernal)
    uy=correlation(original_image,kernal)
    distorted_image_sqr=distorted_image**2
    original_image_sqr=original_image**2
    dis_mult_ori=distorted_image*original_image
    uxx=correlation(distorted_image_sqr,kernal)
    uyy=correlation(original_image_sqr,kernal)
    uxy=correlation(dis_mult_ori,kernal)
    ux_sqr=ux**2
    uy_sqr=uy**2
    uxuy=ux*uy
    sx_sqr=uxx-ux_sqr
    sy_sqr=uyy-uy_sqr
    sxy=uxy-uxuy
    C1=(K1*255)**2
    C2=(K2*255)**2
    #常用情况的SSIM
    if(alfa==1 and beta==1 and gama==1):
        ssim=(2*uxuy+C1)*(2*sxy+C2)/(ux_sqr+uy_sqr+C1)/(sx_sqr+sy_sqr+C2)
        return np.mean(ssim)
    #计算亮度相似性
    l=(2*uxuy+C1)/(ux_sqr+uy_sqr+C1)
    l=l**alfa
    #计算对比度相似性
    sxsy=np.sqrt(sx_sqr)*np.sqrt(sy_sqr)
    c=(2*sxsy+C2)/(sx_sqr+sy_sqr+C2)
    c=c**beta
    #计算结构相似性
    C3=0.5*C2
    s=(sxy+C3)/(sxsy+C3)
    s=s**gama
    ssim=l*c*s
    return np.mean(ssim)

if __name__ == '__main__':
    files = os.listdir("../demo/results/frames")
    files.sort()
    all_dist=0
    for file in files:
        file_num=file[5:13]
        file_num_start=0
        for i in range(len(file_num)):
            if(file_num[i]!="0"):
                file_num_start=i
                break
            if(i==len(file_num)-1):
                file_num_start=i
                break
        d1_file_name=file_num[file_num_start:]+".jpg"
        if(os.path.exists(os.path.join("../get_keyframe/frames",d1_file_name))):
            img1=cv2.imread(os.path.join("../demo/results/frames",file),cv2.IMREAD_GRAYSCALE)
            img2=load_source_image(os.path.join("../get_keyframe/frames",d1_file_name))
            #ssmi_dist=ssim(img1,img2)
            (ssmi_dist, diff) = compare_ssim(img1, img2, full=True)
            print('%s: %.3f'%(file,ssmi_dist))
            all_dist+=ssmi_dist
    print("总距离：%.3f"%all_dist)
    avg_dist=all_dist/len(files)
    print("平均距离：%.3f"%avg_dist)
