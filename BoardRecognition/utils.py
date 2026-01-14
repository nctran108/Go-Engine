import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


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
    if not os.path.exists(path):
        os.makedirs(path)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(os.path.join(path, name), image.astype(np.uint8))