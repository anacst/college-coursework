**Disciplina:** SCC0251 - Processamento de Imagens
**Activity 1:** Discrete Fourier Transform

In this short assignment you'll implement DFT 2D and Inverse DFT 2D using only numpy. \*Use vector/matrix processing whenever possible to decrease running time

The program takes as input an image filename (that must be loaded into memory) and a float number.

Implement a function that takes as input an image and a threshold value 0 < T <= 1 This function have to:

1. Compute the Fourier Transform, F, of the image;
2. Find the second peak of the Fourier Spectrum, p2, i.e. the maximum not counting the coefficient 0;
3. Set to zero (0) all coefficients for which the Fourier Spectrum is below (less than) T% of the second peak, that is, |F|< p2\*T
4. Compute the Inverse Fourier Transform of the filtered version of F (after step 3 above)
5. Print on the screen the following information: p2\*T, the number of coefficients below the threshold, the original mean of the image and the mean of the image after processing, using the following format:

```python
Threshold=%.4f
Filtered Coefficients=%d
Original Mean=%.2f
New Mean=%.2f
```

**Input example:**
```
images/gradient_noise_small.png
0.05
```

**Output example:**
```
Threshold=28.3364
Filtered Coefficients=961
Original Mean=116.06
New Mean=116.06
```
