import cv2
import numpy as np
import sys
import os

def image_to_grayscale(img_name, file_loc, dest_loc):
	img = cv2.imread(file_loc + '/' + img_name)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	w, h = gray.shape
	max_height = 1024
	if h > max_height:
		resizeFactor = 1/ (h // max_height + 1)
		gray = cv2.resize(gray, None, fx=resizeFactor, fy=resizeFactor, interpolation = cv2.INTER_CUBIC)

	#cv2.imshow('img',gray)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	# print(gray.shape)

	cv2.imwrite(dest_loc + img_name, gray)

if __name__ == '__main__':
	print("Converting images to grayscale")
	file_loc = "Samples/PosSamples"
	dest_loc = "Samples/PosSamplesGray/"
	for filename in os.listdir(file_loc):
		#print(filename)
		image_to_grayscale(filename.strip(), file_loc, dest_loc)
