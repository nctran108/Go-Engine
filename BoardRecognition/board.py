import cv2
import numpy as np
import image_processing
import matplotlib.pyplot as plt

class BoardRecognition:
    def __init__(self):
        pass

    def process_image(self, image):
        # pre processing
        preProcessImage = image_processing.PreProcessImage(7)
        hdr = preProcessImage.process(image)
        plt.imshow(hdr)
        plt.show()

        # board regconition

        # stone recognition

        return

def main():
    input = cv2.imread("./data/11.png", cv2.IMREAD_COLOR)

    recognition = BoardRecognition()
    recognition.process_image(input)

if __name__ == "__main__":
    main()