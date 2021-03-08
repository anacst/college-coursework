import numpy as np
import imageio

#filename = str(input()).strip()
img = imageio.imread("images/gradient_noise_small.png")
#T = float(input())

F = np.zeros(img.shape, dtype=np.complex64)
n,m = img.shape

print("Mean:")
print(np.sum(img)/(n*m))

#x = np.arange(n)
#for u in np.arange(n):
#    for v in np.arange(m):
#        for y in np.arange(m):
#           F[u,v] = np.sum(img[:,y]*np.exp( (-1j*2*np.pi) * ((u*x)/n + (v*y)/m) )) / np.sqrt(n*m)

#F = np.fft.fft2(img)

def DFT2D(f):
    # create empty array of complex coefficients
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape[0:2]

    # creating indices for x, to compute multiplication using numpy (f*exp)
    x = np.arange(n)
    # for each frequency 'u,v'
    for u in np.arange(n):
        for v in np.arange(m):
            for y in np.arange(m):
                F[u,v] += np.sum(f[:,y] * np.exp( (-1j*2*np.pi) * (((u*x)/n)+((v*y)/m)) ))

    print("Image size: " + str(n*m))

    return F/np.sqrt(n*m)


#F = DFT2D(img)
F = np.fft.fft2(img)/np.sqrt(n*m)

print("Image shape:")
print(img.shape)

print("Transformed shape")
print(F.shape)
print(F)
print("Coeficient [0,0]:")
print(np.abs(F.max()))
print(np.abs(F[0,0]))

p1 = np.abs(F[0,0])
p2 = np.abs(F[0,1])
for i in np.arange(n):
    for j in np.arange(m):
        if (np.abs(F[i,j]) != p1 and np.abs(F[i,j]) > p2):
            p2 = np.abs(F[i,j])

t = p2*0.05
nCoefficients = 0
for i in np.arange(n):
    for j in np.arange(m):
        if (np.abs(F[i,j]) < t):
            F[i,j] = 0
            nCoefficients = nCoefficients + 1
   
IF = np.fft.ifft2(F)/np.sqrt(n*m)

print("Threshold=" + str(t))
print("Filtered Coefficients=" + str(nCoefficients))
print("Original Mean=" + str(np.sum(img)/(n*m)))
print("New Mean=" + str(np.sum(np.abs(IF))/(n*m)))
