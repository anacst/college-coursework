# Ana Luisa Teixeira Costa
# SCC0251 - Processamento de imagens 
# 2020/1
# Assigment 1: Intesity Transformations

import numpy as np
import imageio


# Apply inversion function
# T(i) = 255 - i
def inversion(img):
    return 255-(img).astype(np.float32)


# Apply contrast modulation function
# T(i) = (i-a) * \frac{d-c}{b-a} + c
# [a, b] begin the interval of pixels intensities (min,max) from image
# [c, d] begin the new interval desired  
def contrastModulation(c, d, img):
    a = img.min() 
    b = img.max()
    return ((img-a)*((d-c)/(int(b)-int(a))))+c


# Apply logarithmic function
# T(i) = 255 * \frac{(log_2{1 + i})}{log_2{1 + R}} 
# R begin the maximum pixel intensity from image
def logarithmicFunction(img):
    R = img.max()
    img_log2 = (255 * np.log2(1+img.astype(np.float32)) / np.log2(1+R))
    return img_log2


# Apply gamma adjustment
# T(i) = W * i^lambd
def gammaAdjustment(W, lambd, img):
    return W * img**lambd
    

filename = str(input()).rstrip()        # Name of file(image) to be manipulated
input_img = imageio.imread(filename)    # Reference to image file 
method = int(input())                   # Determines which function to apply
save = int(input())                     # Variable that determines whether the processed image should be saved or not

if method == 1:
    output_img = inversion(input_img)

if method == 2:
    c = int(input())
    d = int(input())
    output_img = contrastModulation(c, d, input_img)

if method == 3:
    output_img = logarithmicFunction(input_img)

if method == 4:
    W = int(input())
    lambd = float(input())
    output_img = gammaAdjustment(W, lambd, input_img)


# Saves image in memory if variable save is set to 1
if save == 1:
    imageio.imwrite('output_img.png', output_img)


# Saves image in memory if variable save is set to 1
rse = (np.sqrt(np.sum(np.square((output_img - input_img).astype(float)))))

print('%.4f' % rse)
