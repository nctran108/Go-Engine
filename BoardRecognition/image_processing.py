import cv2
import numpy as np
import math

class ImageProcessing:
    def __init__(self) -> None:
        pass

    def _resize_image(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def edge_detection(self, image, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # edge detection
        edges = cv2.Canny(blur, 50, 150, apertureSize=3) 

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)   
        
        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assume the largest contour corresponds to the chessboard
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to a polygon
        epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Extract the corners if we have a quadrilateral
        if len(approx) == 4:
            corners = approx.reshape((4, 2))
            return corners, lines
        else:
            return None

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self._order_points(pts)

        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped

    def _order_points(self,pts):
        # initialize a list of coordinates
        # in the top-left, top-right, bottom-right, and bottom-left
        rect = np.zeros((4, 2), dtype= "float32")

        x_values = pts[:,0]

        y_values = pts[:,1]


        rect[0] = pts[np.argmin(x_values)]
        rect[2] = pts[np.argmax(x_values)]

        rect[1] = pts[np.argmin(y_values)]
        rect[3] = pts[np.argmax(y_values)]

        return rect
    """
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        old_set = set(map(tuple, pts))
        new_set = set(map(tuple,np.array([rect[0],rect[2]])))

        # Find the difference
        difference_set = np.array(list(old_set - new_set))
        print(difference_set)
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect
    """



def main():
    image = cv2.imread('BoardRecognition/data/board_13x13.jpg')

    #image = cv2.resize(image, (0,0), fx = 0.1, fy = 0.1)
    edges, lines = ImageProcessing().edge_detection(image)

    print(len(lines))

    for point in edges:
        cv2.circle(image, tuple(point), radius=5, color=(0,255,0), thickness=-1)

    if lines is not None:
        for line in lines:
            pt1 = (line[0][0], line[0][1])
            pt2 = (line[0][2],line[0][3])
            cv2.line(image, pt1, pt2, (0,0,255), 2)

    image = ImageProcessing().four_point_transform(image, edges)

    cv2.imshow('Go Board Corners', image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
