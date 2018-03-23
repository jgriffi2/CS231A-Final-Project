import numpy as np
import cv2
# import pip
from stl import mesh
import matplotlib.pyplot as plt


#loads a ply or obj file with a mesh in it
def get_mesh(filename):
	new_mesh = mesh.Mesh.from_file(filename)
	return new_mesh

#creates a cube mesh to work with
def get_cube_mesh(center=[0, 0, 0.1], size=0.2, height_scale=0.05):
	COUNT = 12
	data = np.zeros(12, dtype=mesh.Mesh.dtype)
	cx, cy, cz = center
	# Top of the cube
	data['vectors'][0] = np.array([[0, size, size],
	                                  [size, 0, size],
	                                  [0, 0, size]])

	data['vectors'][1] = np.array([[size, 0, size],
	                                  [0, size, size],
	                                  [size, size, size]])
	# Right face
	data['vectors'][2] = np.array([[size, 0, 0],
	                                  [size, 0, size],
	                                  [size, size, 0]])

	data['vectors'][3] = np.array([[size, size, size],
	                                  [size, 0, size],
	                                  [size, size, 0]])
	# Front face
	data['vectors'][4] = np.array([[0, 0, 0],
	                                  [size, 0, 0],
	                                  [size, 0, size]])

	data['vectors'][5] = np.array([[0, 0, 0],
	                                  [0, 0, size],
	                                  [size, 0, size]])

	# Bottom of the cube
	data['vectors'][6] = np.array([[0, size, 0],
	                                  [size, 0, 0],
	                                  [0, 0, 0]])

	data['vectors'][7] = np.array([[size, 0, 0],
	                                  [0, size, 0],
	                                  [size, size, 0]])

	# Left face
	data['vectors'][8] = np.array([[0, 0, 0],
	                                  [0, 0, size],
	                                  [0, size, 0]])

	data['vectors'][9] = np.array([[0, size, size],
	                                  [0, 0, size],
	                                  [0, size, 0]])
	# Back face
	data['vectors'][10] = np.array([[0, size, 0],
	                                  [size, size, 0],
	                                  [size, size, size]])

	data['vectors'][11] = np.array([[0, size, 0],
	                                  [0, size, size],
	                                  [size, size, size]])

	data['vectors'] -= .5 * size
	data['vectors'][:, :, 0] += cx
	data['vectors'][:, :, 1] += cy
	data['vectors'][:, :, 2] += cz
	data['vectors'][:, :, 2] *= height_scale

	new_mesh = mesh.Mesh(data)
	return new_mesh

#returns the coordinates of the points to plot
def get_unique_points(mesh_obj):
	a = np.ascontiguousarray(get_points_list(mesh_obj))
	unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
	return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

#gets list of all points in triangles in rows
def get_points_list(mesh_obj):
	flattened = mesh_obj.points.flatten()
	flattened = flattened.reshape((flattened.shape[0] // 3, 3))
	return flattened


#applies a 3x3 projective transform to the mesh_obj
#returns nx3x2 where n = num of triangles, and three 2d points listed in each triangle
def apply_projection(K, mesh_obj):
	#stl version wants a 4x4 rotation + translation matrix - write own?
	# transformed = mesh_obj.transform(pmatrix)

	# columns are points, get columns as resulting points
	pts = get_points_list(mesh_obj)
	pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
	n, d = pts.shape
	# print(pts)
	result = np.dot(K, pts.T)

	# print(result)

	result = result[0:2] / result[2]
	# result = result[0:2]
	result = result.T

	#reshape to get triangles
	result = result.reshape((n // 3, 3, 2))

	# for i in range(result.shape[0]):
	# 	print(result[i])
		# print(pts[i*3:i*3+3])

	return result

#plots the wireframe of the mesh from nx3x2 points
def display_mesh(triangles):
	for triangle in triangles:
		poly = plt.Polygon(triangle, edgecolor='y', facecolor='blue', alpha=.3, capstyle='round')
		plt.gca().add_patch(poly)

#get the camera matrix
def get_camera_projection(K, H):
	p1 = np.hstack((K, np.dot(K, np.array([[0], [0], [-1]]))))

	print(p1)
	test = np.array([[.1, .1, 0, 1], [-.1, -.1, 0, 1], [.1, .1, 0, 1], [.1, .1, .2, 1]])
	rtest = np.dot(p1, test.T)
	rtest = rtest[0:2] / rtest[2]
	print(rtest.T)

	print(H)

	p2 = np.dot(H, p1)
	A = np.dot(np.linalg.inv(K), p2[:,:3])
	A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 1], A[:, 0])]).T
	p2[:, :3] = np.dot(K, A)

	print(p2)

	return p2

#calibrate camera
def camera_calibration(row, col, size=.2):
	# print(shape)
	K = np.diag([128 / size * col / 128,  128 / size * row / 128, 1])
	K[0,2] = 0.5*col
	K[1,2] = 0.5*row
	return K

if __name__ == '__main__':
	# installed_packages = pip.get_installed_distributions()
	# installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
	# for i in installed_packages])
	# print(installed_packages_list)

	cube = get_cube_mesh(center=[0, 0, .1], height_scale=.1)
	test = np.eye(3)
	img = np.zeros((128, 128))

	#mesh.points has all the triangles
	vertices = get_unique_points(cube)
	print(vertices)

	# print(cube.points.shape)

	#try applying the transform
	# K = camera_calibration(img, 1, 1)
	# proj = get_camera_projection(K, test)
	# print(proj)
	# print(apply_projection(proj, cube))

	# test_triangles = np.array([[[0, 1], [1, 0], [1, 1]]])
	# plt.figure()
	# display_mesh(test_triangles)
	# plt.show()

