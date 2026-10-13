import numpy as np
import cv2
import utils


class PreProcessImage:
    """Turn a camera photo into a black & white "line mask":
    white = dark thin things (grid lines, stone outlines), black = everything else.
    Pictures: data/explain/1_histogram_equalization.png ... 4_why_it_matters.png
    """

    def __init__(self, clahe_clip=2.0, threshold_c=7):
        # CLAHE = contrast boost done separately in each of 8 x 8 tiles of the photo.
        # clipLimit stops it from over-boosting flat areas (cloth, plain wood) into noise.
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        # How much darker than its neighbourhood a pixel must be to count as "line".
        self.threshold_c = threshold_c

    def process(self, image):
        # (1) Colour is not needed to find lines -> grayscale (one brightness value per pixel).
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # (2) Even out the lighting: lines in shadow get as much contrast as lines in bright areas.
        gray = self.clahe.apply(gray)

        # (3) A very small blur removes single-pixel camera noise before thresholding.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        utils.writeImage(gray, "blurred.png")

        # (4) Adaptive threshold: for EVERY pixel, compare it with the average of the
        #     block x block square around it. If it is darker than that average by more than
        #     threshold_c, it becomes white (a line); otherwise black.
        #     "Local" average = works even when one side of the board is darker than the other.
        #     THRESH_BINARY_INV = dark lines become WHITE in the mask (OpenCV likes white objects).
        h, w = gray.shape
        block = max(15, (min(h, w) // 40) | 1)  # block size grows with the photo; "| 1" makes it odd (required)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, block, self.threshold_c)
        utils.writeImage(mask, "line_mask.png")
        return mask


class BoardRecognition:
    """Find the board in the line mask: 4 corners, board size (9/13/19...) and all intersections.
    Pictures: data/explain/B0_pipeline_overview.png ... B5_sgtm_vs_ours.png
    """
    STANDARD_SIZES = (9, 13, 19)

    def __init__(self, warp_size=900, candidate_sizes=range(5, 26)):
        self.warp_size = warp_size              # size in pixels of the flattened, top-down board
        self.candidate_sizes = candidate_sizes  # line counts to try (5, 6, ..., 25)
        self.corners = None      # tl, tr, br, bl of the outer grid lines, in photo pixels
        self.size = None         # (columns, rows), e.g. (19, 19)
        self.M = None            # perspective transform: photo -> top-down square

    # ------------------------------------------------------------------ step 1: find the grid
    def findGridComponent(self, mask):
        """Find which white piece of the mask is the Go grid.

        Idea: all grid lines touch each other, so the whole grid is ONE connected white piece.
        It is also a "net": it encloses many small squares (cells) of about the same size.
        Board edges, cloth wrinkles and noise enclose few or no squares, so they lose.

        Pictures for every step: data/explain/E0_code_map.png ... E4_steps_10_11.png
        """
        # (1) Give every separate white piece its own number.
        #     n      = number of pieces + 1 (label 0 is the black background)
        #     labels = image where each pixel holds the number of the piece it belongs to
        #     stats  = one row per piece: [x, y, width, height, area]  (x, y = top-left of its box)
        #     connectivity=8: pixels touching only at a corner still count as the same piece,
        #     which keeps thin tilted grid lines joined together.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h, w = mask.shape

        best, best_score = None, 0   # the best piece found so far, and its score

        for i in range(1, n):        # look at every piece (start at 1 to skip the background)
            x, y, bw, bh, area = stats[i]

            # (2) Quick filter: the grid must be big. If the piece's box is narrower or shorter
            #     than 15% of the photo, skip it. (On 11.png this skips 1,108 of 1,112 pieces.)
            if bw < w * 0.15 or bh < h * 0.15:
                continue

            # (3) Cut out ONLY this piece. The box can also contain bits of other pieces,
            #     so keep just the pixels whose label is i. Result: white = this piece, black = rest.
            sub = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255

            # (4) Find the outlines of this piece.
            #     RETR_CCOMP gives 2 levels: the outside outline(s) AND the outline of every hole.
            #     hier[0][k] = [next, previous, first_child, parent] for contour k.
            cnts, hier = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hier is None:          # no outlines at all -> nothing to check
                continue

            # (5) Keep only the HOLES. A hole sits inside the outside outline,
            #     so it has a parent (hh[3] >= 0). The outside outline has no parent (hh[3] == -1).
            #     For each hole, store its area in pixels.
            holes = [cv2.contourArea(c) for c, hh in zip(cnts, hier[0]) if hh[3] >= 0]

            # (6) Drop tiny holes (20 pixels or less): they are specks of noise inside a line,
            #     not grid cells.
            holes = [a for a in holes if a > 20]

            # (7) Even a tiny grid has many cells. Fewer than 9 holes -> this piece is not a grid.
            if len(holes) < 9:
                continue

            # (8) The typical cell size = the median hole area.
            #     Median (the middle value), not the average, so a few odd holes cannot pull it.
            med = np.median(holes)

            # (9) Score = how many holes have a cell-like size: between 0.3x and 3x the typical size.
            #     Holes outside this range are cells merged together (a stone broke a line)
            #     or other odd shapes, so they do not count.
            score = sum(1 for a in holes if 0.3 * med < a < 3 * med)

            # (10) Remember the piece with the highest score so far.
            if score > best_score:
                best, best_score = i, score

        # No piece passed all the checks -> there is no grid in this photo.
        if best is None:
            raise RuntimeError("Could not find a grid in the image")

        # (11) Turn the winning piece into its own mask (white = grid, black = everything else).
        #      This mask is what findFourCorners() uses next.
        comp = (labels == best).astype(np.uint8) * 255
        utils.writeImage(comp, "grid_component.png")
        return comp

    # ------------------------------------------------------------------ step 2: four corners
    @staticmethod
    def orderCorners(pts):
        """Put 4 points in a fixed order: top-left, top-right, bottom-right, bottom-left.
        Trick: x + y is smallest at top-left and largest at bottom-right;
               y - x is smallest at top-right and largest at bottom-left."""
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(1)               # x + y for each point
        d = pts[:, 1] - pts[:, 0]    # y - x for each point
        return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                         pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # tl, tr, br, bl

    def roughQuad(self, contour):
        """A first guess of the 4 corners from the grid's outline (may be pulled off by edge stones).
        Picture: data/explain/B2_corners.png (panel A)"""
        # (1) Convex hull = a rubber band stretched around the outline (removes all dents).
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)   # length of the rubber band

        # (2) approxPolyDP simplifies the hull into fewer, straighter sides.
        #     eps = how much simplification is allowed (as a fraction of the length).
        #     Increase it bit by bit until exactly 4 corners are left.
        for eps in np.linspace(0.005, 0.1, 40):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return self.orderCorners(approx)

        # (3) Backup plan if it never gives exactly 4: take the most extreme points
        #     (smallest/largest x+y and y-x, same trick as orderCorners).
        s = hull.reshape(-1, 2)
        return self.orderCorners([s[np.argmin(s.sum(1))], s[np.argmin(s[:, 1] - s[:, 0])],
                                  s[np.argmax(s.sum(1))], s[np.argmax(s[:, 1] - s[:, 0])]])

    @staticmethod
    def ransacLine(pts, tol, iters=300, seed=0):
        """Fit a straight line to pts while IGNORING outliers (bumps where stones sit on the edge).
        RANSAC = "RANdom SAmple Consensus": try many random guesses, keep the one most points agree with.
        Picture: data/explain/B2_corners.png (panel B: green = kept, red = ignored)"""
        rng = np.random.default_rng(seed)   # fixed seed -> same result every run
        best = None
        for _ in range(iters):
            # (1) Pick 2 random points; they define a candidate line.
            p, q = pts[rng.choice(len(pts), 2, replace=False)]
            d = q - p
            nrm = np.hypot(*d)
            if nrm < 1e-6:                  # the 2 points are the same -> no line, try again
                continue
            # (2) normal = unit arrow at 90 degrees to the line.
            #     (pts - p) @ normal = each point's distance from the line.
            normal = np.array([-d[1], d[0]]) / nrm
            # (3) Inliers = points closer to the line than tol pixels.
            inliers = np.abs((pts - p) @ normal) < tol
            # (4) Keep the guess with the most inliers.
            if best is None or inliers.sum() > best.sum():
                best = inliers
        # (5) Final, precise line through ONLY the inliers.
        vx, vy, x0, y0 = cv2.fitLine(pts[best].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        return np.array([vx, vy]), np.array([x0, y0])  # (direction, a point on the line)

    @staticmethod
    def findInterection(line1, line2):
        """Where two lines cross. Each line = (direction d, point p): all points p + t*d.
        Solve p1 + t*d1 = p2 + s*d2 for t and s, then return p1 + t*d1."""
        (d1, p1), (d2, p2) = line1, line2
        t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
        return p1 + t[0] * d1

    def findFourCorners(self, mask, image=None):
        """The 4 corners of the outer grid lines. Picture: data/explain/B2_corners.png"""
        # (1) Keep only the grid piece of the mask (see findGridComponent).
        comp = self.findGridComponent(mask)

        # (2) The grid's outside outline (RETR_EXTERNAL = outside only, no holes;
        #     CHAIN_APPROX_NONE = keep every point, we need them all for line fitting).
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(cnts, key=cv2.contourArea)

        # (3) Rough corners (good enough to know roughly where each side is).
        quad = self.roughQuad(contour)
        pts = contour.reshape(-1, 2).astype(np.float64)
        # How close (in pixels) a point must be to count as "on the line": 0.4% of the board's diagonal.
        tol = max(2.0, 0.004 * np.hypot(*(quad[2] - quad[0])))

        # (4) For each of the 4 sides: collect the outline points along it and fit a straight line.
        edge_lines = []
        for k in range(4):
            a, b = quad[k], quad[(k + 1) % 4]   # rough start and end corner of this side
            L = np.linalg.norm(b - a)           # side length
            d = (b - a) / L                     # unit arrow ALONG the side
            normal = np.array([-d[1], d[0]])    # unit arrow ACROSS the side
            t = (pts - a) @ d                   # how far along the side each point is
            dist = np.abs((pts - a) @ normal)   # how far away from the side each point is
            # keep points in the middle 90% of the side (corners are messy) and near the side
            sel = (t > 0.05 * L) & (t < 0.95 * L) & (dist < 0.08 * L)
            side = pts[sel] if sel.sum() > 20 else np.array([a, b], dtype=np.float64)
            edge_lines.append(self.ransacLine(side, tol))

        # (5) Each corner = where two neighbouring side lines cross
        #     (k-1 and k: for k = 0 that is side 3 (left) and side 0 (top) -> top-left).
        corners = [self.findInterection(edge_lines[k - 1], edge_lines[k]) for k in range(4)]
        self.corners = self.orderCorners(corners)

        if image is not None:
            utils.writeImageWithPoints(image, self.corners, "four corners.png")
        return self.corners

    # ------------------------------------------------------------------ step 3: count lines
    def countLines(self, profile):
        """Find how many evenly spaced lines fit the profile best.
        profile[i] = amount of line pixels in row (or column) i of the top-down board,
        so every grid line makes a bump. Picture: data/explain/B4_count_lines.png

        For every candidate n: put n evenly spaced "ticks" from the first to the last row.
          good fit = a strong bump at every tick AND nothing between ticks.
          (10 or 7 on a 19x19 board also hit bumps, but real lines hide between their ticks.)
        """
        # (1) Rescale so the numbers do not depend on photo size or brightness:
        #     average becomes 0, typical spread becomes 1.
        profile = profile.astype(np.float64)
        profile = (profile - profile.mean()) / (profile.std() + 1e-9)
        L = len(profile) - 1          # position of the last row (first row is 0)

        scores = {}
        for n in self.candidate_sizes:
            # (2) Spacing between n evenly spaced lines.
            step = L / (n - 1)
            # (3) r = search window around each tick: about a line's thickness
            #     (a line may be a few pixels off if the corners are slightly off).
            r = max(2, int(min(0.2 * step, L / 120)))
            # (4) Where the n lines should be.
            pos = np.round(np.arange(n) * step).astype(int)
            # (5) Highest value near each tick -> should be high (a line is there).
            peaks = [profile[max(0, p - r):p + r + 1].max() for p in pos]
            # (6) Highest value in each gap between ticks -> should be low (no line in between).
            gaps = [profile[a + r + 1:b - r].max() for a, b in zip(pos[:-1], pos[1:]) if b - r > a + r + 1]
            # (7) Score: strong lines minus anything found in the gaps.
            scores[n] = np.mean(peaks) - (np.mean(gaps) if gaps else 0)

        # (8) The best-scoring n is the number of lines.
        best = max(scores, key=scores.get)
        return best, scores

    def boardSize(self, mask):
        """Flatten the board and count its columns and rows.
        Pictures: data/explain/B3_flatten.png and B4_count_lines.png"""
        # (1) Perspective transform: move the 4 corners to the corners of an S x S square.
        #     After this, every grid line is straight and evenly spaced.
        S = self.warp_size
        dst = np.float32([[0, 0], [S, 0], [S, S], [0, S]])   # tl, tr, br, bl of the square
        self.M = cv2.getPerspectiveTransform(self.corners, dst)
        top_down = cv2.warpPerspective(mask, self.M, (S + 1, S + 1))
        utils.writeImage(top_down, "top_down_mask.png")

        # (2) Keep only long straight strokes, so stone outlines do not create fake bumps.
        #     MORPH_OPEN with a k x 1 bar keeps only white parts at least k pixels wide
        #     -> horizontal lines survive, everything shorter disappears. Same with a 1 x k bar
        #     for vertical lines. k = 1/30 of the board, shorter than one cell even on a 25x25 board.
        k = max(5, S // 30)
        horiz = cv2.morphologyEx(top_down, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        vert = cv2.morphologyEx(top_down, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))

        # (3) Add up each row (for horizontal lines) and each column (for vertical lines),
        #     then count the bumps.
        rows, self.row_scores = self.countLines(horiz.sum(axis=1))
        cols, self.col_scores = self.countLines(vert.sum(axis=0))
        self.size = (cols, rows)
        return self.size

    # ------------------------------------------------------------------ step 4: intersections
    def intersections(self):
        """All cols x rows intersection points in ORIGINAL photo pixels, shape (rows, cols, 2).
        These are where stones can be, ready for stone recognition."""
        cols, rows = self.size
        S = self.warp_size
        # (1) In the flat square, intersections are simply evenly spaced.
        xs = np.arange(cols) * S / (cols - 1)
        ys = np.arange(rows) * S / (rows - 1)
        grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
        # (2) Send them back into the tilted photo with the INVERSE transform (square -> photo).
        pts = cv2.perspectiveTransform(grid, np.linalg.inv(self.M))
        return pts.reshape(rows, cols, 2)

    def process(self, image, mask):
        """Run everything: corners -> size -> intersections, and save a picture of the result."""
        self.findFourCorners(mask, image)   # step 2
        self.boardSize(mask)                # step 3
        points = self.intersections()       # step 4
        utils.writeImageWithGrid(image, self.corners, points, self.size, "detected grid.png")
        return self.corners, self.size, points


class PieceRecognition:
    pass
