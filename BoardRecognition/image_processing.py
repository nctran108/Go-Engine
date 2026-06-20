import numpy as np
import cv2
import utils

class PreProcessImage:
    def __init__(self):
        self.ddepth = cv2.CV_64F
        self.kernal = np.array([[-1,-1,-1],
                                [-1,8,-1],
                                [-1,-1,-1]])
        self.laplace_kernal = np.array([[0,1,0],
                                        [1,-4,1],
                                        [0,1,0]])

    def process(self, image):
        # transform into gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # The Laplacian of Gaussian (LoG) Method
        blurred = cv2.GaussianBlur(gray, (5, 5), 3)
        utils.writeImage(blurred, "blurred.png")

        # Laplace operator for edge detection to reduce noices
        laplacian = cv2.filter2D(blurred, self.ddepth, self.laplace_kernal, borderType=cv2.BORDER_REFLECT)
        utils.writeImage(laplacian, "laplace.png")

        #return image after preprocessing
        return laplacian.astype(np.uint8)
        
class BoardRecognition:
    def __init__(self):
        pass

    def findFourCorners(self, img):
        ### tried HoughLines but not work well but use HoughlineP is work better, do not know why
        # lines = cv2.HoughLines(img,1, np.pi/180.0,200)
        # utils.writeImageWithLines(img,lines,"find four corners lines.png")

        # find lines
        lines = cv2.HoughLinesP(img,1, np.pi/180.0,200,minLineLength=100,maxLineGap=10)
        utils.writeImageWithLineSegments(img,lines,"find four corners line using points.png")

        corners = []
        # find horizontal lines

        # find verticle lines

        # find unique lines for both verticle and horizontal

        # remove lines not in the board

        # find 4 coners

        return corners

    def findCentral(self, corners):
        pass

    def midlineInterections(self, centralPosition):
        pass

    def findInterection(self, line1, line2):
        pass

    def SGTM(self, img):
        pass
        # the process of SGTM (Simple Geometric Transform Model)

        # (1) use hough transsform Locate coordinate of four corners and return Corner[0], Corner[1], Corner[2], Corner[3]
        corners = self.findFourCorners(img)

        # (2) base on 4 corners get the central position of the image and return the middle of the cental position
        # find the interections of the midline with the upper edge line and the lower edge line of the grid
        # return Mid_line[0], Mid_line[1]
        #centralPosition = self.findCentral(corners)
        #Mid_line = self.midlineInterection(centralPosition)

        # (3) Corner[0], Mid_line[0], Mid_line[1], Corner[3] as a retangular corner points
        # then repeat (2) to find Mid_line_1[0], Mid_line_1[1] for new retangular
        # line between Mid_line_1[0] and Mid_line[1]
        # line between Corner[3] and Mid_line[0]
        # the interection1 between two lines above
        # interection2(L[0]), interection3(L[1]) between the upper edge line and the lower edge line through the intersection1
        #corners_1 = np.array([corners[0], Mid_line[0], Mid_line[1], corners[3]])
        #centralPosition_1 = self.findCentral(corners_1)
        #Mid_line_1 = self.midlineInterections(centralPosition_1)
        #line_0 = self.line(Mid_line_1[0], Mid_line[1])
        #line_1 = self.line(corners[3], Mid_line[0])
        #interection1 = self.findInterection(line_0, line_1)

        # (4) using L[0], COrner[0], Corner[3], L[1] as a rectangular corner points
        # the midline L2, and the intersections L2[0], L2[1] are obtain between L2 with the upper edge line and the lower edge line.Mid_line.Mid_line

        # (6) same way as step (5), find 8 lines (excluding edge lines) on the right side of the midline
        # plus 2 edge lines and middle line, find 19 vertical lines

        # (7) for the four corner coordinates from (1), repeat (2)~(6)
        # identify the 19 horizontal lines

        # (8) find 361 interections of go-board base on 19x19 lines

    def process(self, image):
        self.SGTM(image)
        pass


class PieceRecognition:
    pass