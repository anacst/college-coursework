**Course:** SCC0251 - Digital Image Processing
**Activity 2:** Denoising and Debluring

In this assignment you are going to perform denoising followed by deblurring.

The denoising is going to be performed by using a Gaussian filter h with size k and standard deviation sigma, (k and sigma are given as input), in the Fourier domain. Perform the transformation of the image g and the filter to the Fourier domain, and multiply the transforms.

* Use FFT functions available at numpy or scipy libraries. The fftpack is specially useful for computing the Fast Fourier Transforms (and the inverse) for the images and degradation functions.

Create the filter using the python code available below. After denosing normalise the image so that its values are in the range [0, maxg], in which maxg is the maximum value found at the input image (before denoising).

Afterwards, restore the blur using the Constrained Least Squares method, which approximates a solution via a minimization of the mean squared error, using a priori assumptions, in this case a Laplacian regularization.

The Constrained Least Squares filter is applied in the Fourier domain taking as input G(u) to be the Fourier transform of the degraded image, and H(u) the Fourier transform of the degradation function (the point spread function), as well as P(u), the Fourier transform of a Laplacian operator and finally a gamma float point variable which controls the influence of the regularization.

We are going to assume that the function H is a Gaussian filter with size k and standard deviation sigma, given as input as in the denoising step.

After blur removal normalise again the image so that its values are in the range [0, maxd], in which maxd is the maximum value found at the denoised image (after denoising, before deblurring).

You should copy this source code and use in your program, but we recommend you take a time to understand it:

```python
def gaussian_filter(k=3, sigma=1.0):
	arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
	x, y = np.meshgrid(arx, arx)
	filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma) )
	return filt/np.sum(filt)
```

The input must be in this order:

* image filename
* k (filter size, a positive and odd integer)
* sigma (standard deviation, a float point number > 0 and <=1)
* gamma (regularization factor, a float point number >= 0 and <=1)

The output is the standard deviation value of the image pixels after restoration, rounded to 1 decimal place.
