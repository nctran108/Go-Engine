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