import numpy as np
import cv2
from scipy.spatial.distance import cdist
from collections import defaultdict
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import math

import mesh
import quad

def find_surfaces(img, projection_img, K=2, point_thresh=100, line_thresh=200, use_aspect_ratio=False, project_mesh=False):
    projected_img = cv2.cvtColor(projection_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.float32) / 25
    gray = cv2.filter2D(gray, -1, kernel)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if (len(lines[0]) == 0):
        return gray

    suppressed_lines = non_max_suppression_lines(lines, line_thresh)
    if (len(suppressed_lines[0]) == 0):
        return gray

    segmented = segment_by_angle_kmeans(suppressed_lines, k=K)

    intersections = segmented_intersections(segmented)
    if (intersections.shape[0] < 4):
        return gray

    legit_quads = quad.get_planes(intersections)
    if (len(legit_quads) == 0):
        return gray

    final_img = gray / np.max(gray)
    h, w = projected_img.shape
    for i in range(legit_quads.shape[0]):
        q = legit_quads[i]
        if use_aspect_ratio:
            ratio = quad.get_aspect_ratio(q, final_img)
            h = int(w * ratio)
            pimg = np.copy(projected_img)
            pimg = np.resize(pimg, (h, w))
        else:
            pimg = projected_img

        H = find_transformation_matrix(q, h, w)

        if project_mesh:
            if h < w:
                cube = mesh.get_cube_mesh()
            else:
                cube = mesh.get_cube_mesh()
            K = mesh.camera_calibration(h, w)
            proj = mesh.get_camera_projection(K, H)
            newpts = mesh.apply_projection(proj, cube)
            plt.imshow(gray, cmap='gray')
            mesh.display_mesh(newpts)
            plt.show()

        warped_img = cv2.warpPerspective(pimg, H, (img.shape[1], img.shape[0]), None, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        final_img += warped_img /np.max(warped_img)
        break # TODO: breaking after first quad for testing purposes

    return final_img

def resize_projected_img(img, h, w):
    return img.resize((h, w))

#checks if the given points are a closed shape
def is_closed_shape(points):
    n = points.shape[0]
    counter = Counter()

    for i in range(n):
        counter.update((points[i, 2], points[i, 3]))
        counter.update((points[i, 4], points[i, 5]))
        if (counter[(points[i, 2], points[i, 3])] > 2):
            return False
        if (counter[(points[i, 4], points[i, 5])] > 2):
            return False

    for element in counter:
        if (counter[element] != 2):
            return False

    return True

def get_rho_theta(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    theta = math.atan2(x, y)
    rho = np.abs(x) * np.cos(theta) + np.abs(y) * np.sin(theta)
    return np.array([rho, theta])

def non_max_suppression_lines(lines, thresh):
    l = np.empty((0, 1, 2))

    for i in range(lines.shape[0]):
        if (len(l) > 0):
            closest_lines = np.where(np.abs(l[:, 0, 0] - lines[i, 0, 0]) < thresh)
            if (len(closest_lines[0]) == 0):
                l = np.append(l, [lines[i]], axis=0)
        else:
            l = np.append(l, [lines[i]], axis=0)

    return l

def non_max_suppression_points(points, thresh):
    p = np.empty((0, points.shape[1]))

    for i in range(points.shape[0]):
        if (len(p) > 0):
            dist = np.sqrt(np.sum(np.square(p[:, 0:2] - points[i, 0:2]), axis=1))
            inds = np.where(dist < thresh)
            indY = inds[0]
            if (len(indY) == 0):
                p = np.append(p, points[i][None, :], axis=0)
        else:
            p = np.append(p, points[i][None, :], axis=0)

    return p

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.lstsq(A, b, rcond=None)[0]
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return np.array([x0, y0, rho1, theta1, rho2, theta2])

def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return np.asarray(intersections)

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def find_transformation_matrix(matrix, h, w):
    src = np.array([
    [0, 0],
    [w, 0],
    [0, h],
    [w, h]
    ])
    dst = matrix[:, 0:2]
    return cv2.getPerspectiveTransform( np.float32(src), np.float32(dst) )
