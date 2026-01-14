import cv2
import numpy as np
import image_processing
import matplotlib.pyplot as plt
import utils

class SGTM:
    def __init__(self):
        self.VERBOSE = True

    def process_image(self, image):
        # pre processing
        preProcessImage = image_processing.PreProcessImage(7)
        hdr = preProcessImage.process(image)
        # utils.visulization(hdr, "hpf image")

        # board regconition
        board = image_processing.BoardRecognition()
        _, lines = board.process(hdr)
        # stone recognition

        if self.VERBOSE:
            utils.writeImage(hdr, "hpf.png")
            utils.writeImageWithLines(image, lines, "lines.png")
        return

def main():
    input = cv2.imread("./data/Zrzut ekranu 2020-04-11 o 15.42.14.png", cv2.IMREAD_COLOR)

    recognition = SGTM()
    recognition.process_image(input)

if __name__ == "__main__":
    main()