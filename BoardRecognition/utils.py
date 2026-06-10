import os

import cv2
from matplotlib import pyplot as plt
import numpy as np


def visulization(images = None, title=None):
    if images is None:
        raise ValueError('images cannot be None')
    if isinstance(images, list):
        figure, axis = plt.subplots(1,len(images))
        for i,image in enumerate(images):
            if len(image.shape) > 2:
                axis[i].imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            else:
                axis[i].imshow(image.astype(np.uint8))
            if title:
                axis[i].set_title(title[i])
        plt.tight_layout()
        plt.show()
    else:
        if len(images.shape) > 2:
            plt.imshow(cv2.cvtColor(images.astype(np.uint8), cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images.astype(np.uint8))
        if title:
            plt.title(title)
        plt.show()

def writeImage(image, name):
    path = "./data/result"
    if not os.path.exists(path):
        os.makedirs(path)

    cv2.imwrite(os.path.join(path,name), image.astype(np.uint8))

def writeImageWithLines(image, lines, name):
    path = "./data/result"
    output = image.copy()
    output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if not os.path.exists(path):
        os.makedirs(path)
    if lines is not None:
        # Sort safety: slice to ONLY take the top 8 strongest lines
        strongest_lines = lines[:8]
        for r_theta in strongest_lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            rho , theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(output, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(os.path.join(path, name), img= output)


def writeImageWithLineSegments(image, lines, name):
    path = "./data/result"
    output = image.copy()
    if len(output.shape) == 2:  # Check if it's grayscale
        output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if not os.path.exists(path):
        os.makedirs(path)

    if lines is not None:
        for line in lines:
            # HoughLinesP returns direct x,y coordinates for start and end points
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imwrite(os.path.join(path, name), img=output)