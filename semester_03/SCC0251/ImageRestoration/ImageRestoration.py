from scipy.fftpack import fftn, ifftn, fftshift
import numpy as np
import imageio

def gaussian_filter(k=3, sigma=1.0):
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma) )
    return filt/np.sum(filt)

def normalize(f, max):
    return (max * (f - f.min()))/(f.max() - f.min())

def denoising(g, w):
    """Function that denoises image
    Params:
    g - image input
    w - guassian filter
    """
    # padding filter
    pad = int(g.shape[0]//2 - w.shape[0]//2) 
    wp = np.pad(w, (pad,pad-1), 'constant', constant_values=(0))

    #applying the fourier transform to image and filter
    G = fftn(g)
    W = fftn(wp)

    denoised_img = np.multiply(G, W)
    denoised_img = np.real(fftshift(ifftn(denoised_img)))

    return denoised_img

def deblurring(g, h, gamma):
    """Function that restores the blur using the Constrained Least Squares method
    Params:
    g - degraded image
    h - degradation function (gaussian filter)
    gamma - regularization factor
    """

    # defining laplace operator
    p = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    pad = int(g.shape[0]//2 - h.shape[0]//2) 
    hp = np.pad(h, (pad,pad-1), 'constant', constant_values=(0))
    pad = int(g.shape[0]//2 - p.shape[0]//2) 
    pp = np.pad(p, (pad,pad-1), 'constant', constant_values=(0))

    # applying the fourier transform
    G = fftn(g)
    H = fftn(hp)
    P = fftn(pp)

    F = (np.conjugate(H) / (np.square(np.abs(H)) + gamma*np.square(np.abs(P)))) * G
    f = np.real(fftshift(ifftn(F)))

    return f

def main():
    # parameters input
    filename = str(input()).strip()
    k = int(input())
    sigma = float(input())
    gamma = float(input())

    img = imageio.imread(filename)
    w = gaussian_filter(k, sigma)
    maxg = img.max()

    denoised_img = denoising(img, w)
    maxd = denoised_img.max()
    denoised_img = normalize(denoised_img, maxg)

    restored_img = deblurring(denoised_img, w, gamma)
    restored_img = normalize(restored_img, maxd)

    print(np.round(np.std(restored_img),1))

if __name__ == '__main__':
    main()
