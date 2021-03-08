# SCC251 - Image Processing - 2020/2
# Asignment 2: Image Enhancement and Filtering
# Authors: Ana Laura Chioca, 9866531 
#          Ana Luisa Costa, 11218963
# Repo: https://gitlab.com/anachioca/dip-2

import numpy as np
import imageio

def error(f, r):
    """Computing error from image reference to processed image"""
    return np.sqrt(np.sum(np.square(np.subtract(f.astype(np.float64), r.astype(np.float64)))))


def G(x, sigma):
    """Gaussian kernel equation"""
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x) ** 2 / (2 * sigma ** 2)))

def E(x, y):
    """Calculates euclidean distance between two points"""
    return np.sqrt(x**2 + y**2)

def bilateral_filter(img, n, s, r):
    """Function to apply bilateral filter, smoothing the image while preserving its edges
    Parameters:
        n - size of filter
        s - spatial Gaussian
        r - range Gaussian
    """
    
    # Calculating padding necessary to process all pixels from image
    a = int((n-1)/2)
    b = int((n-1)/2)
    
    # Padding image with zeros
    img = np.pad(img, ((a,a),(b,b)), 'constant')
    
    # Converting image to type float64 in order to process it
    img = img.astype(np.float64)
    
    # New array to store filtered image
    img_mod = np.zeros(img.shape, dtype=np.float32)
    
    N, M = img.shape
    gs = np.zeros((n,n))
    gr = np.zeros((n,n))
    
    # Computing the spatial Gaussian component
    for x in range(n):
        for y in range(n):
            gs[x,y] = G(E(x-a,y-b), s)
            
    # Convolving the original image r with the chosen kernel
    for x in range(a, N-a):
        for y in range(b, M-b):
            subimg = img[x-a : x+a+1 , y-b : y+b+1]
            I = img[x,y]
            W = 0
            for u in range(n): 
                for v in range(n):
                    gr[u,v] = G(subimg[u,v]-I, r)   # Computing the range Gaussian component
            w = np.multiply(gr,gs)
            W = np.sum(w)
            I = np.sum(np.multiply(w,subimg))
            img_mod[x,y] = I / W

    
    img_mod = img_mod[a:N-a, b:M-b]

    return img_mod.astype(np.uint8)

def unsharp_mask(img, c, k):
    n, m = k.shape
    a = int((n - 1) / 2)
    b = int((m - 1) / 2)

    # Converting image to type float64 in order to process it and padding it with zeros
    img = img.astype(np.float32)
    img = np.pad(img, ((a, a), (b, b)), 'constant')

    # New array to store filtered image
    filtered_img = np.array(img, copy=True)

    N, M = img.shape

    # Convolving the original image r with the chosen kernel
    for x in range(a, N - a):
        for y in range(b, M - b):
            subimg = img[x - a: x + a + 1, y - b: y + b + 1]
            filtered_img[x, y] = np.sum(np.multiply(subimg, k))

    # Image Scaling:
    filtered_img = (((filtered_img - filtered_img.min()) * 255) / (filtered_img.max() - filtered_img.min()))

    # Adding the filtered image, multiplied by the parameter c, back to the original image:
    filtered_img = (c * (filtered_img)) + img

    # Image Scaling:
    filtered_img = (((filtered_img - filtered_img.min()) * 255) / (filtered_img.max() - filtered_img.min()))

    filtered_img = filtered_img[a:N - a, b:M - b]

    return filtered_img.astype(np.uint8)


def vignette_filter(img, row, col):
    """Function that applies Vignette filter"""

    # Converting image to type float64 in order to process it
    img = img.astype(np.float64)

    # Padding image with zeros
    img = np.pad(img, ((1, 1), (1, 1)), 'constant')

    # New array to stored filtered image
    img_mod = np.zeros(img.shape, dtype=np.float64)

    a, b = img.shape
    k1 = np.zeros(a)
    k2 = np.zeros(b)

    # Computing the Gaussian Kernel
    i = 0
    for x in range(-int((a - 1) / 2), int(((a - 1) / 2) + 1)):
        k1[i] = G(x, row)
        i += 1

    j = 0
    for x in range(-int((b - 1) / 2), int(((b - 1) / 2) + 1)):
        k2[j] = G(x, col)
        j += 1

    # Transposition
    k1_transp = k1.reshape(k1.shape + (1,))

    img_mod = np.zeros((a, b))
    prod = np.multiply(k1_transp, k2)
    img_mod = np.multiply(img, prod)

    # Image Scaling
    img_mod = ((img_mod - img_mod.min()) * 255) / img_mod.max()

    return img_mod.astype(np.uint8)


# Parameters input
filename = str(input()).strip()
img = imageio.imread(filename)
method = int(input())
save = int(input())

if method == 1:
    n = int(input())
    s = float(input())
    r = float(input())
    img_mod = bilateral_filter(img, n, s, r)

if method == 2:

    k1 = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

    k2 = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

    c = float(input())  # parâmetro c
    kernel = int(input())  # escolha do kernel (1 ou 2)

    if kernel == 1:
        k = k1
    if kernel == 2:
        k = k2

    img_mod = unsharp_mask(img, c, k)

if method == 3:
    row = float(input())  # parâmetro row
    col = float(input())  # parâmetro col
    img_mod = vignette_filter(img)

if save == 1:
    imageio.imwrite("output_img.png", img_mod)

print('%.4f' % error(img_mod, img))
