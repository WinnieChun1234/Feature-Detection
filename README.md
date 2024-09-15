# Feature-Detection

This is the second assignmnet of my computer vision course in the university.  The assignment comes with partially completed Python programs and the tasks are to perform RGB to grayscale conversion, 1D smoothing using a 1D horizontal Gaussian filter, 2D smoothing using 1D convolutions, and Harris corner detection on image.


### Implementation Workflow (My Tasks)

1. Use the formula for the Y-channel of the YIQ model in performing the color-to-grayscale image conversion.
This step is done by using np.dot() on the RGB-channel vector and the vector of [0.299,0.587,0.114].

2. Compute Ix and Iy correctly by finite differences.
I calculate Ix and Iy by np.gradient() for finding the gradient horizontally and vertically.


3. Construct images of Ix2, Iy2 , and IxIy correctly.
I simply calculate the Ix2 = Ix * Ix, Iy2 = Iy * Iy, IxIy = Ix * Iy.


4. Compute a proper filter size for a Gaussian filter based on its sigma value.
I calculated the filter size based on the instruction in tutorial 3, and so the filter size is generated.


5. Construct a proper 1D Gaussian filter.
The img_filer is formed by convolve1d().


6. Smooth a 2D image by convolving it with two 1D Gaussian filters.
I apply the smooth1D() once on the img and this smooth the image along the horizontal direction.
Then I apply the smooth1D() again on the transpose of smooth_Horizontal (the image being smoothed horizontally) and this smooth the image along the vertical direction.

7. Handle the image border using partial filters in smoothing.
I created a matrix with the same size as the img, form the weight by applying convolve1d() on it.
Finally, let the img_filter to be divided by the weight as indicated in tutorial 3 for purpose of dealing with borders.


8. Construct an image of the cornerness function R correctly.
I have done this by the formula R = det_A - K * trace_A **2. This step make the smoothed Ix2, Iy2 and IxIy to form an element-wise R.


9. harris()
I set up nested for-loop to loop over the R horizontally and vertically from the range 1 to the R.shape[]-1 in order to ignore the border.
Then I formed an 8-neighbours matrix for iterating on R. If the center pixel is the maxima, then we look into sub-pixel accuracy by considering the 4-neighbors.


10. Identify potential corners at local maxima in the image of the cornerness function R. Compute the cornerness value and coordinates of the potential corners up to sub-pixel accuracy by quadratic approximation.
By finding all the variables a, b, c , d, e so that xx = -c/2a and yy = -d/2b can be calculated.
We can then use 2D Quadratic approximation to find out maxima r2.


11. Use the threshold value to identify strong corners for output.
If r2 is greater than the threshold value, then append (x+xx, y+yy, r2) as in the corner array.
 
