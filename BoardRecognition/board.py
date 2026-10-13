import sys
import cv2
import image_processing


def main(path="./data/11.png"):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)

    mask = image_processing.PreProcessImage().process(image)

    board = image_processing.BoardRecognition()
    corners, (cols, rows), points = board.process(image, mask)

    print("corners (tl, tr, br, bl):", corners.round(1).tolist())
    print(f"board size: {cols} x {rows}")
    print("intersections:", points.shape, "-> see data/result/detected grid.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
