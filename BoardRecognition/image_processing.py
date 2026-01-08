import numpy as np
import cv2

def PreProcess(image):
    # transform into gray scale

    # Laplace operator for edge detection to reduce noices

    # high-pass filter

    #return image after preprocessing
    return

class BoardRecognition:
    # the process of SGTM (Simple Geometric Transform Model)

    # (1) Locate coordinate of four corners and return Corner[0], Corner[1], Corner[2], Corner[3]

    # (2) base on 4 corners get the central position of the image and return the middle of the cental position

    # find the interections of the midline with the upper edge line and the lower edge line of the grid
    # return Mid_line[0], Mid_line[1]

    # (3) Corner[0], Mid_line[0], Mid_line[1], Corner[3] as a retangular corner points
    # then repeat (2) to find Mid_line_1[0], Mid_line_1[1] for new retangular
    # line between Mid_line_1[0] and Mid_line[1]
    # line between Corner[3] and Mid_line[0]
    # the interection1 between two lines above
    # interection2(L[0]), interection3(L[1]) between the upper edge line and the lower edge line through the intersection1

    # (4) using L[0], COrner[0], Corner[3], L[1] as a rectangular corner points
    # the midline L2, and the intersections L2[0], L2[1] are obtain between L2 with the upper edge line and the lower edge line.Mid_line.Mid_line

    # (6) same way as step (5), find 8 lines (excluding edge lines) on the right side of the midline
    # plus 2 edge lines and middle line, find 19 vertical lines

    # (7) for the four corner coordinates from (1), repeat (2)~(6)
    # identify the 19 horizontal lines

    # (8) find 361 interections of go-board base on 19x19 lines


class PieceRecognition:
    pass