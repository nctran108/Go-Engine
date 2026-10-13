"""
SGTM — Simple Geometric Transform Model
From: "A Robust Algorithm for Go Image Recognition in Go Game" (the paper in data/).

A LEARNING implementation. It is separate from image_processing.py and changes nothing there.
It follows the paper's steps (1)-(8) and can save one picture per step (data/result/sgtm/).

Run:  python sgtm.py            (uses ./data/11.png)
      python sgtm.py my.png

------------------------------------------------------------------------------------------
THE BIG IDEA
In a tilted photo, the board's squares get smaller further away, so you can NOT find the
middle of an edge by measuring halfway in pixels. But some things survive perspective:
  * straight lines stay straight
  * where lines cross stays where they cross
So SGTM finds every grid line using ONLY "draw a line" and "find where two lines cross".

Three tools do all the work:
  midline(quad)    -> the line 1/2 of the way across   (step 2: the diagonals cross at the true centre)
  third_line(quad) -> the line 1/3 of the way across   (step 3: a clever pair of crossing lines)
  then cut the pieces again until all 19 lines exist   (steps 4-6)

A quad is always given as 4 points: (top-left, top-right, bottom-right, bottom-left).
"Across" means from its left edge to its right edge.
------------------------------------------------------------------------------------------
"""
import os
import sys
import cv2
import numpy as np


# ======================================================================================
# Geometry helpers
# ======================================================================================
# The paper writes a line as  a*x + b*y = c  and gives formula (3) for where two lines cross.
# Here we use "homogeneous coordinates", which is the same maths in a neater form:
#   a point (x, y)        is stored as [x, y, 1]
#   the line through p, q is  cross(p, q)        -> gives [a, b, -c]
#   where lines l1, l2 cross is cross(l1, l2)    -> gives [x*w, y*w, w], divide by w
# Bonus: if two lines are parallel, w = 0, which means "they meet infinitely far away".
# That case happens when the board is photographed straight-on, and this form handles it
# without any special code.

def P(x, y):
    """A point as [x, y, 1]."""
    return np.array([x, y, 1.0])


def line_through(p, q):
    """The line through points p and q."""
    return np.cross(p, q)


def intersect(l1, l2):
    """Where two lines cross. Same result as the paper's formula (3).
    The point may be 'at infinity' (last value 0) when the lines are parallel."""
    p = np.cross(l1, l2)
    if abs(p[2]) > 1e-12:
        p = p / p[2]
    return p


def xy(p):
    """Back to normal (x, y) pixel coordinates."""
    return np.array([p[0] / p[2], p[1] / p[2]])


# ======================================================================================
# SGTM
# ======================================================================================
class SGTM:
    def __init__(self, corners, debug_image=None, out_dir="./data/result/sgtm"):
        """
        corners: 4 points (x, y), clockwise from top-left:
                 Corner[0] = top-left, Corner[1] = top-right, Corner[2] = bottom-right, Corner[3] = bottom-left
        debug_image: the photo, if you want a picture saved for each step.
        """
        self.corners = [P(*c) for c in np.asarray(corners, dtype=float)]
        self.debug_image = debug_image
        self.out_dir = out_dir
        self.picture = 0

    # ----------------------------------------------------------------------------------
    # Step (2): the midline of a quad
    # ----------------------------------------------------------------------------------
    def findCentral(self, quad):
        """The true centre of a quad = where its two diagonals cross.
        This works in a tilted photo too, because diagonals are straight lines."""
        tl, tr, br, bl = quad
        return intersect(line_through(tl, br), line_through(tr, bl))

    def line_parallel_to_sides(self, quad, point):
        """The line through `point` that runs the same way as the quad's left/right sides,
        e.g. a vertical grid line.
        In the photo, the left and right edges are not parallel: extended, they meet at a far
        'vanishing point'. Every vertical grid line also passes through that point.
        So: the vertical line through `point` = the line through `point` and the vanishing point."""
        tl, tr, br, bl = quad
        vanishing = intersect(line_through(tl, bl), line_through(tr, br))
        return line_through(point, vanishing)

    def line_to_edges(self, quad, line):
        """Where a vertical line hits the top edge and the bottom edge -> (top point, bottom point)."""
        tl, tr, br, bl = quad
        top = intersect(line, line_through(tl, tr))
        bottom = intersect(line, line_through(bl, br))
        return top, bottom

    def midlineIntersections(self, quad):
        """Step (2): centre -> midline -> Mid_line[0] (on the top edge) and Mid_line[1] (on the bottom edge).
        This line is exactly 1/2 of the way across the quad."""
        center = self.findCentral(quad)
        mid = self.line_parallel_to_sides(quad, center)
        return self.line_to_edges(quad, mid)

    # ----------------------------------------------------------------------------------
    # Step (3): the 1/3 line of a quad (the clever part)
    # ----------------------------------------------------------------------------------
    def thirdLine(self, quad, from_left=True):
        """Step (3): find the line 1/3 of the way across the quad, using only lines and crossings.

          1. Mid_line   = midline of the whole quad                     (1/2 across)
          2. Mid_line1  = midline of the LEFT half                      (1/4 across)
          3. line A: from Mid_line1[0] (top, 1/4) to Mid_line[1] (bottom, 1/2)
             line B: from Corner[3]   (bottom-left) to Mid_line[0] (top, 1/2)
          4. A and B cross at intersection1, which is exactly 1/3 across (and 1/3 down).
             Proof in flat coordinates: A is u = 1/4 + v/4, B is u = 1/2 - v/2.
             Equal when v = 1/3, so u = 1/4 + 1/12 = 1/3.
          5. The vertical line through intersection1 gives L[0] (top) and L[1] (bottom).

        from_left=False mirrors everything and gives the line 1/3 from the RIGHT (= 2/3 across).
        """
        tl, tr, br, bl = quad
        if not from_left:  # mirror left <-> right, solve, done
            return self.thirdLine([tr, tl, bl, br], from_left=True)

        mid0, mid1 = self.midlineIntersections(quad)                 # 1/2 across
        q0, q1 = self.midlineIntersections([tl, mid0, mid1, bl])    # 1/4 across (midline of left half)
        line_a = line_through(q0, mid1)
        line_b = line_through(bl, mid0)
        intersection1 = intersect(line_a, line_b)
        L = self.line_to_edges(quad, self.line_parallel_to_sides(quad, intersection1))

        self._draw_step3(quad, (mid0, mid1), (q0, q1), (q0, mid1), (bl, mid0), intersection1, L)
        return L

    # ----------------------------------------------------------------------------------
    # Steps (4)-(6): cut the board into 18 cells -> 19 lines
    # ----------------------------------------------------------------------------------
    def trisect(self, quad):
        """Cut a quad into 3 equal strips -> the 2 lines inside it (1/3 and 2/3 across)."""
        return [self.thirdLine(quad, from_left=True), self.thirdLine(quad, from_left=False)]

    def verticalLines(self, quad):
        """Steps (2)-(6): all 19 lines between the top and bottom edges, left to right.
        19 lines = 18 cells. Each line is returned as (top point, bottom point).

          step (2): Mid  = midline                       -> line 9
          step (3): L    = 1/3 line from the left          -> line 6
          step (4): L2   = midline of [left edge .. L]     -> line 3
          step (5): trisect the strips 0-3, 3-6, 6-9       -> lines 1,2  4,5  7,8
          step (6): same on the right side (mirrored)      -> lines 12, 15, then 10,11 13,14 16,17
        """
        tl, tr, br, bl = quad
        left_edge, right_edge = (tl, bl), (tr, br)

        mid = self.midlineIntersections(quad)                        # (2) line 9
        self._draw_step2(quad, mid)
        L = self.thirdLine(quad, from_left=True)                     # (3) line 6
        L2 = self.midlineIntersections([tl, L[0], L[1], bl])        # (4) line 3
        R = self.thirdLine(quad, from_left=False)                    # (6) line 12
        R2 = self.midlineIntersections([R[0], tr, br, R[1]])        # (6) line 15

        main = [left_edge, L2, L, mid, R, R2, right_edge]            # lines 0, 3, 6, 9, 12, 15, 18
        lines = [left_edge]
        for left, right in zip(main[:-1], main[1:]):                 # (5)+(6) each 3-cell strip
            strip = [left[0], right[0], right[1], left[1]]
            lines += self.trisect(strip) + [right]
        self._draw_main(quad, main)
        return lines                                                 # 19 lines

    # ----------------------------------------------------------------------------------
    # Steps (7)-(8)
    # ----------------------------------------------------------------------------------
    def process(self):
        c0, c1, c2, c3 = self.corners
        vertical = self.verticalLines([c0, c1, c2, c3])
        self._draw_lines(vertical, "step6_19_vertical_lines.png", (255, 120, 40))

        # step (7): same thing, but start the corners from Corner[1] (rotate the board a quarter turn).
        # "Left to right" in the rotated quad = "top to bottom" in the photo.
        horizontal = self.verticalLines([c1, c2, c3, c0])
        self._draw_lines(vertical + horizontal, "step7_19x19_lines.png", (255, 120, 40))

        # step (8): every vertical line crosses every horizontal line -> 19 x 19 = 361 points
        points = np.zeros((19, 19, 2))
        for r, (h0, h1) in enumerate(horizontal):
            for c, (v0, v1) in enumerate(vertical):
                points[r, c] = xy(intersect(line_through(h0, h1), line_through(v0, v1)))
        self._draw_points(vertical + horizontal, points)
        return vertical, horizontal, points

    # ==================================================================================
    # Pictures (only used to learn; skip if debug_image is None)
    # ==================================================================================
    def _save(self, img, name):
        if self.debug_image is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.out_dir, name), img)

    def _canvas(self, quad=None):
        img = self.debug_image.copy()
        board = [tuple(xy(p).astype(int)) for p in self.corners]
        cv2.polylines(img, [np.array(board)], True, (60, 60, 230), 4)
        if quad is not None:
            cv2.polylines(img, [np.array([xy(p).astype(int) for p in quad])], True, (0, 220, 255), 5)
        return img

    @staticmethod
    def _seg(img, p, q, color, w=4):
        cv2.line(img, tuple(xy(p).astype(int)), tuple(xy(q).astype(int)), color, w, cv2.LINE_AA)

    @staticmethod
    def _dot(img, p, color, r=14):
        cv2.circle(img, tuple(xy(p).astype(int)), r, color, -1, cv2.LINE_AA)

    def _draw_step2(self, quad, mid):
        if self.debug_image is None or self.picture > 0:
            return
        img = self._canvas()
        tl, tr, br, bl = quad
        self._seg(img, tl, br, (0, 200, 255), 4)
        self._seg(img, tr, bl, (0, 200, 255), 4)
        self._dot(img, self.findCentral(quad), (200, 40, 255), 18)
        self._seg(img, mid[0], mid[1], (200, 40, 255), 7)
        self._dot(img, mid[0], (60, 200, 60)); self._dot(img, mid[1], (60, 200, 60))
        self._save(img, "step2_midline.png")

    def _draw_step3(self, quad, mid, q, a, b, i1, L):
        if self.debug_image is None or self.picture > 0:
            return
        self.picture += 1  # only draw the FIRST time step 3 runs (on the left half of the board)
        img = self._canvas(quad)
        self._seg(img, mid[0], mid[1], (200, 40, 255), 4)        # 1/2
        self._seg(img, q[0], q[1], (255, 160, 60), 4)             # 1/4
        self._seg(img, a[0], a[1], (0, 140, 255), 4)              # line A
        self._seg(img, b[0], b[1], (0, 140, 255), 4)              # line B
        self._dot(img, i1, (40, 40, 255), 18)                     # intersection1
        self._seg(img, L[0], L[1], (60, 200, 60), 7)              # result: 1/3 line
        self._save(img, "step3_one_third_line.png")

    def _draw_main(self, quad, main):
        if self.debug_image is None or self.picture > 1:
            return
        self.picture += 1
        img = self._canvas()
        for (p, q), col in zip(main, [(60, 60, 230), (255, 160, 60), (60, 200, 60), (200, 40, 255),
                                       (60, 200, 60), (255, 160, 60), (60, 60, 230)]):
            self._seg(img, p, q, col, 6)
        self._save(img, "step4_lines_0_3_6_9_12_15_18.png")

    def _draw_lines(self, lines, name, color):
        if self.debug_image is None:
            return
        img = self._canvas()
        for p, q in lines:
            self._seg(img, p, q, color, 3)
        self._save(img, name)

    def _draw_points(self, lines, points):
        if self.debug_image is None:
            return
        img = self._canvas()
        for p, q in lines:
            self._seg(img, p, q, (255, 120, 40), 2)
        for pt in points.reshape(-1, 2):
            cv2.circle(img, tuple(pt.astype(int)), 7, (60, 200, 60), -1)
        self._save(img, "step8_361_intersections.png")


# ======================================================================================
def main(path="./data/11.png"):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)

    # Step (1): four corners. The paper uses a Hough transform on the EMPTY board at the start
    # of the game. Here we borrow the corners from the current pipeline (image_processing.py)
    # so this file can focus on SGTM itself.
    import image_processing
    mask = image_processing.PreProcessImage().process(image)
    board = image_processing.BoardRecognition()
    corners = board.findFourCorners(mask)

    sgtm = SGTM(corners, debug_image=image)
    vertical, horizontal, points = sgtm.process()
    print("19 vertical + 19 horizontal lines, intersections:", points.shape)

    # Check: SGTM should agree with the one-step perspective transform
    board.boardSize(mask)          # builds the one-step perspective transform (board.M)
    board.size = (19, 19)
    ours = board.intersections()
    diff = np.linalg.norm(points - ours, axis=2)
    print(f"difference to cv2.getPerspectiveTransform grid: max {diff.max():.6f} px, mean {diff.mean():.6f} px")
    print("pictures saved in ./data/result/sgtm/")


if __name__ == "__main__":
    main(*sys.argv[1:])
