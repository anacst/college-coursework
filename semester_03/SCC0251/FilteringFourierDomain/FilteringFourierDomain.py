import numpy as np
import imageio

def DFT2D(f):
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape[0:2]
    
    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1,m)
    for u in np.arange(n):
        for v in np.arange(m):
            F[u,v] += np.sum(f * np.exp( (-1j*2*np.pi) * (((u*x)/n)+((v*y)/m)) ))
                
    return F/np.sqrt(n*m)

def IDFT2D(f):
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape[0:2]
    
    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1,m)
    for u in np.arange(n):
        for v in np.arange(m):
            F[u,v] += np.sum(f * np.exp( (1j*2*np.pi) * (((u*x)/n)+((v*y)/m)) ))
                
    return F/np.sqrt(n*m)


# parameters input
filename = str(input()).strip()
img = imageio.imread(filename)
T = float(input())

# compute fourier transform of the image
F = DFT2D(img)

temp = F[0,0]
F[0,0] = 0
p2 = np.abs(F).max()
F[0,0] = temp

F[np.where(np.abs(F) < (p2*T))] = 0
nCoefficients = np.count_nonzero(F == 0)

# computing inverse fourier transform
IF = IDFT2D(F)

mean = np.mean(img)
new_mean = np.mean(np.abs(IF))

print("Threshold=%.4f" % (p2*T)) 
print("Filtered Coefficients=%d" % nCoefficients)
print("Original Mean=%.2f" % mean)
print("New Mean=%.2f" %  new_mean)