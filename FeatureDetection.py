################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Conrner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion

    img_gray = np.dot(img_color[...,:3], [0.299, 0.587, 0.114])
    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result


    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    n = int(sigma * (2*np.log(1000))**0.5)
    x = np.arange(-n, n + 1)
    filter = np.exp((x ** 2) / -2 / (sigma ** 2))
    matrix_one = np.ones_like(img)
    weight = convolve1d(matrix_one, filter, 1, np.float64, 'constant', 0,0)

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    img_filter = convolve1d(img, filter, 1, np.float64, 'constant', 0, 0)
    img_smoothed = img_filter/weight
    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the horizontal direction
    smooth_Horizontal = smooth1D(img,sigma)
    # TODO: smooth the image along the vertical direction
    img_smoothed = smooth1D(smooth_Horizontal.T,sigma).T
    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy

    Ix = np.gradient(img,axis=1)
    Iy = np.gradient(img,axis=0)

    
    # TODO: compute Ix2, Iy2 and IxIy

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # TODO: smooth the squared derivatives
    smooth_Ix2 = smooth2D(Ix2, sigma)
    smooth_Iy2 = smooth2D(Iy2, sigma)
    smooth_IxIy = smooth2D(IxIy, sigma)

    # TODO: compute cornesness functoin R
    K = 0.04
    det_A = smooth_Ix2 * smooth_Iy2 - smooth_IxIy **2
    trace_A = smooth_Ix2 + smooth_Iy2
    R = det_A - K * trace_A **2

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy

    # TODO: perform thresholding and discard weak corners
    corners = []

    for y in range(1,R.shape[0]-1): #ignore border (i.e. ignore first an last index)
        for x in range(1,R.shape[1]-1): #ignore border (i.e. ignore first an last index)

            f=R[y-1:y+2,x-1:x+2]; # 3x3 matrix (8-neighbors and itself)
            if R[y,x] == np.amax(f):

                north = f[0,1]
                south = f[2,1]
                west = f[1,0]
                east = f[1,2]

                e = f[1,1]
                a = (west+east-2*e)/2
                b = (south+north-2*e)/2
                c = (east-west)/2
                d = (south-north)/2

                xx=-c/2.0/a if a!=0 else 0
                yy=-d/2.0/b if b!=0 else 0
                r2 =a*xx*xx+b*yy*yy+c*xx+d*yy+e

                if r2 > threshold:
                    corners.append((x+xx, y+yy, r2))

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap = 'gray')
    # plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)

if __name__ == '__main__':
    main()
