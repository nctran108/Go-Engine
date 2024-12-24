import cv2
import numpy as np
import image_processing

class BoardRecognition:
    def __init__(self, board_size):
        self.board_size = board_size

    def process_image(self, image):
        # get 4 conners of the board, either manually or using opencv
        # assume the board is static and not move because this will give many more problem,
        # this can be something in the future for more robust version.
        coners = image_processing.conner_detector(image)
        # assume the board in board size that input by the user rather than detect from the real board for easier programming
        # can add detect lines to guess the board size in future when find a good robust method.

        # transform the board into up-side down perspective to have better view on the board

        # draw lines base on board size and gett all coordinate of crossing points

        # detect the stones colors and positions base on crossing points

        # return the computer version of board game
        return