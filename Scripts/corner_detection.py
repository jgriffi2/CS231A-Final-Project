import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from skimage.feature import corner_harris

"""
detect_corners: Detects the corners of an image using harris corner detection.

filename: name of the image we want to detect corners on

thresh: threshold for the algorithm to detect the corner, should be a value
        between 0 and 1
"""
def detect_corners(filename, thresh):
    #img = io.imread(filename, as_grey=True)
    img = filename
    img = img / np.max(img)

    dx, dy = np.gradient(img)
    edge = np.hypot(dx, dy)
    edge = edge / np.max(edge)

    #plt.imshow(edge, cmap='gray')
    #plt.show()

    corners = edge

    #corners = np.where(edge > thresh, edge, 0)
    #plt.imshow(corners)
    #plt.show()

    corners = corner_harris(img)
    corners = corners / np.max(corners)

    corners_inds = np.where(corners > thresh)
    #corners_inds = np.nonzero(corners)
    corners_indsX = corners_inds[0].reshape((len(corners_inds[0]), 1))
    corners_indsY = corners_inds[1].reshape((len(corners_inds[1]), 1))
    inds = np.concatenate((corners_indsX, corners_indsY), axis=1)

    plt.imshow(corners, cmap='gray')
    plt.plot(inds[:, 1], inds[:, 0], 'ro')
    plt.show()

    return corners, inds
