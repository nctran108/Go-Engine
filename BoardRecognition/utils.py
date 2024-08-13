import math
import numpy as np
import cv2

def make_line_image(shape, lines, img=None, is_HoughLines=False):
    if img is None:
        img = np.full(shape, 255 ,dtype=np.uint8)
    if not is_HoughLines:
        if lines is not None:
            for line in lines:
                pt1 = (line[0][0], line[0][1])
                pt2 = (line[0][2],line[0][3])
                cv2.line(img, pt1, pt2, (0,0,255), 2)
    else:
        # draw lines
        if lines is not None:
            for line in lines:
                rho = line[0][0]
                theta = line[0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(img, pt1, pt2, (0,0,255), 2)
    return img

