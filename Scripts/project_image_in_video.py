import find_rectangular_surfaces as frs
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

def project_img(video, img_to_project, frame_to_read=100, line_thresh=100):
    start_time = time.time()
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    index = 0
    success = True
    p_img = cv2.imread(img_to_project)
    while success:
        success, image = vidcap.read()
        if (success and index % frame_to_read == 0):
            projected_img = frs.find_surfaces(image, p_img, line_thresh=line_thresh)
            plt.imshow(projected_img)
            plt.show()
            # cv2.imwrite("projected_image%d.jpg" % count, projected_img)
            count += 1
        index += 1
    vidcap.release()
    print(time.time() - start_time)
