import cv2
import numpy as np
import image_processing
import matplotlib.pyplot as plt
import utils

def main():
    input = cv2.imread("./data/Zrzut ekranu 2020-04-11 o 15.42.14.png", cv2.IMREAD_COLOR)

    preProcess_img = image_processing.PreProcessImage().process(input)

    board = image_processing.BoardRecognition()
    board.findFourCorners(preProcess_img)

if __name__ == "__main__":
    main()