import cv2
import numpy as np

class ImageProcessing:
    def __init__(self) -> None:
        pass

    def edge_detection(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)    
        
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
            return corners
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
        rect = np.zeros((4, 2), dtype= "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

def main():
    image = cv2.imread('BoardRecognition/data/go_board_55.jpg')
    edges = ImageProcessing().edge_detection(image)
    image = ImageProcessing().four_point_transform(image, edges)
    cv2.imshow('Go Board Corners', image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
