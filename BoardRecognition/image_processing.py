import numpy as np
import cv2

class PreProcessImage:
    def __init__(self, kernelSize):
        self.ddepth = cv2.CV_64F
        self.kernelSize = kernelSize

    def gradients(self, image_bw):
        Ix = cv2.filter2D(image_bw.astype(np.float32), -1, self.SOBEL_X, borderType=cv2.BORDER_CONSTANT)
        Iy = cv2.filter2D(image_bw.astype(np.float32), -1, self.SOBEL_Y, borderType=cv2.BORDER_CONSTANT)
        return Ix, Iy

    def process(self, image):
        # transform into gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Laplace operator for edge detection to reduce noices
        laplacian = cv2.Laplacian(gray, self.ddepth, self.kernelSize)
        laplacian = cv2.convertScaleAbs(laplacian)
        # high-pass filter
        hpf = gray - cv2.GaussianBlur(gray, (self.kernelSize, self.kernelSize), 3) + 127

        #return image after preprocessing
        return hpf

class BoardRecognition:
    def __init__(self):
        pass

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