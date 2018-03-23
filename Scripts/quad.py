import numpy as np
import itertools
# from orientation.py import *
import find_rectangular_surfaces
from collections import Counter

#determines if a closed quad is concave
#get an array of line params and intersection point coordinates
# each row -> (x, y, r1, t1, r2, t2)
#check that the diagonals do not intersect the lines at any point
#besides the side points
def is_concave(quad):
	#get one set of diagonal points (assume other pair is diagonal)
	#know that these exist because the shape is closed
	p1, p3 = get_diagonal_points(quad)
	points = np.arange(4)
	points = np.where(np.logical_and(points != p1, points != p3))
	p2 = points[0][0]
	p4 = points[0][1]

	#TODO: Check that diagonals don't intersect with any of the outside lines

	return True

# get the pair of diagonal lines across a quad
def get_diagonal_lines(quad):
	#get one set of diagonal points (assume other pair is diagonal)
	#know that these exist because the shape is closed
	p1, p3 = get_diagonal_points(quad)
	points = np.arange(4)
	points = np.where(np.logical_and(points != p1, points != p3))
	p2 = points[0][0]
	p4 = points[0][1]

	# d1 = find_rectangular_surfaces.get_rho_theta(quad[p1], quad[p3])
	# d2 = find_rectangular_surfaces.get_rho_theta(quad[p2], quad[p4])

	ordered_points = get_ordered_points(quad, p1, p3, p2, p4)
	# print(ordered_points)

	# return d1, d2, ordered_points
	return None, None, ordered_points

# order the points starting with top left, top right, bottom left, bottom right
def get_ordered_points(quad, p1, p3, p2, p4):
	#switch each pair if one is smaller by x
	# print(quad.shape, quad[p1, 0].shape, quad[p3, 0].shape, p1, quad[p1, 0])
	if (p1 == None or p2 == None or p3 == None or p4 == None):
		return None

	if quad[p1, 0] > quad[p3, 0]:
		temp = p1
		p1 = p3
		p3 = temp
	if quad[p2, 0] > quad[p4, 0]:
		temp = p2
		p2 = p4
		p4 = temp

	#sort based on smallest y value
	if quad[p1, 1] > quad[p2, 1]:
		return quad[[p2, p1, p3, p4]]
	return quad[[p1, p2, p4, p3]]


#gets the midpoint of the quad
def get_midpoint(d1, d2):
	#get intersection function
	mp = find_rectangular_surfaces.intersection([d1.tolist()], [d2.tolist()])

	return mp


#get the full quad information for one quad
def get_full_quad_matrix(quad):
	#get the diagonal lines
	d1, d2, ordered_points = get_diagonal_lines(quad)
	# mp = get_midpoint(d1, d2)

	# data = np.zeros((5, 6))
	# data[0:4] = ordered_points
	# data[4] = mp

	# return data
	return ordered_points

#Gets a pair of diagonal points given a quad
def get_diagonal_points(quad):
	for i in range(4):
		for j in range(i+1, 4):
			if are_diagonal_points(quad[i, 2:6], quad[j, 2:6]):
				return i, j
	return None, None

# checks to see if two points are diagonal
def are_diagonal_points(intersect1, intersect2):
	lines1 = np.repeat(intersect1.reshape((2,2)), 2, axis=0)
	lines2 = np.tile(intersect2.reshape((2,2)), (2, 1))
	return np.all(lines1 != lines2)

#runs tests on the found quad (defined by array points)
#points 1 and 3 are opposite, 2 and 4 are opposite
#tests if the disparity is about the same (test a window for robustness?)
#tests if the images are rotations of one another? (TODO)
#tests if there isn't that much salience on the table? (TODO)
def is_plane(im, disparity_map, points):
	if not test_disparity(disparity_map, points, thresh=.5):
		return False
	if not test_similarity(im, points, thresh=.5):
		return False
	if not test_salience(im, points, thresh=.5):
		return False
	return True

#tests if four points have similar disparities
# based upon if their average diff b/w pts is < thresh
def test_disparity(disparity_map, points, thresh):
	disps = disparity_map[points[:, 0], points[:, 1]]
	dists =  np.abs(np.tile(disps[:, np.newaxis], (1, 4)) - disps[np.newaxis, :])
	d = np.sum(dists) / 12
	print(d)

	return (d <= thresh)

#tests if the corners are similar enough
#HoG and reorient based on the largest theta?
def test_similarity(im, points, thresh):
	#TODO
	return True


#tests if the body of the space isn't very salient
#aka it's a plain plane
def test_salience(im, points, thresh):
	#TODO
	return True

#tests every possible quad to see if it's a real quad
#passes back an array of quads that's N x 4 x 6
def get_quads(intersections):
	n, m = intersections.shape
	possible_shapes = np.array(list(itertools.combinations(np.arange(n), 4)))
	test_quads = intersections[possible_shapes]
	good_quads = []

	for quad in test_quads:
		# print(quad)
		#test if closed shape
		if not find_rectangular_surfaces.is_closed_shape(quad):
			continue

		good_quads.append(quad)

	return np.asarray(good_quads)

#gets planes - quad with a midpoint for now
def get_planes(intersections):
	quads = get_quads(intersections)
	planes = []

	for quad in quads:
		# if not is_plane(im, disparity_map, quad):
		# 	continue

		#get appropriate plane information and add to planes
		plane = get_full_quad_matrix(quad)
		if (np.all(plane != None)):
			planes.append(plane)

	return np.asarray(planes)

#gets the aspect ratio of the rectangle (w/h)
#assumes that the focal length is about 38mm (typical cellphone)
#method used here:
#https://www.microsoft.com/en-us/research/uploads/prod/2016/11/Digital-Signal-Processing.pdf
def get_aspect_ratio(quad, im):
	w, h = im.shape
	u = h/2
	v = w/2

	#get properties
	points = quad[0:4, 0:2]
	print(points)
	points = np.hstack((points, np.ones((4, 1))))

	m1 = points[2]
	m2 = points[3]
	m3 = points[0]
	m4 = points[1]

	k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
	k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

	n2 = (k2 * m2 - m1)
	n3 = (k3 * m3 - m1)

	if k2 == 1 or k3 == 1:
		ratio = np.sqrt((n2[0] ** 2 + n2[1] ** 2) / (n3[0] ** 2 + n3[1] ** 2))
		return ratio

	#get focal length
	fp = n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u + n2[2] * n3[2] * u ** 2
	print(fp)
	fp = fp + n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v + n2[2] * n3[2] * v ** 2
	print(fp)
	fp = fp *  -1 / n2[2] / n3[2]
	print(fp)
	f = np.sqrt(np.abs(fp))

	Ainv = np.linalg.inv(np.array([[f, 0, u], [0, f, v], [0, 0, 1]]))

	numerator = np.dot(np.dot(n2.T, Ainv.T), np.dot(Ainv, n2))
	denom = np.dot(np.dot(n3.T, Ainv.T), np.dot(Ainv, n3))
	ratio = np.sqrt(denom / numerator)

	return ratio


if __name__ == '__main__':
	#set of test rhos and thetas and intersections
	quad = np.array([[0, 0, 1, 1, 2, 2],
		[0, 1, 2, 2, 3, 3],
		[1, 1, 3, 3, 4, 4],
		[1, 0, 4, 4, 1, 1]])

	im = np.zeros((20, 20))

	test = np.arange(30).reshape((5, 6))

	planes = get_planes(quad)

	ratio = get_aspect_ratio(planes[0], im)

	#height should be 1/ratio (if width is 1)
	height = ratio

	print(height)
