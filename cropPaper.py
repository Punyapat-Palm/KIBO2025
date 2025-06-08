import cv2
import numpy as np
import sys

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit("[ERROR] Could not load image. Check the file path.")
    print("[DEBUG] Input image shape:", img.shape)
    print("[DEBUG] Input image min/max pixel values:", img.min(), img.max())
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect_aruco(gray, img_draw):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        print(f"[SUCCESS] ArUco marker detected: {len(ids)}")
        cv2.aruco.drawDetectedMarkers(img_draw, corners, ids)
        pts = corners[0].reshape(4, 2)
        dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
        print(f"[INFO] ArUco rotation angle: {angle:.2f}°")
        return True
    print("[WARNING] No ArUco markers detected")
    return False

def preprocess_edges(gray):
    blur = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(blur, 10, 20)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    return edged

def biggest_contour(contours):
    max_area, biggest = 0, np.array([])
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            approx = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and area > max_area:
                biggest, max_area = approx, area
    print("[DEBUG] Biggest contour points (before reshape):\n", biggest)
    return biggest

def order_points(pts):
    # Ensure points are unique and valid
    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) < 4:
        print("[ERROR] Less than 4 unique points detected:", unique_pts)
        sys.exit()
    
    # Sort points by y-coordinate first, then x-coordinate for top/bottom pairs
    pts = pts.reshape(-1, 2)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top_pts = sorted_by_y[:2]
    bottom_pts = sorted_by_y[2:]
    
    # Sort top and bottom points by x-coordinate
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
    
    ordered = np.array([
        top_pts[0],      # Top-left
        top_pts[1],      # Top-right
        bottom_pts[1],   # Bottom-right
        bottom_pts[0]    # Bottom-left
    ], dtype="float32")
    
    print("[DEBUG] Ordered points:\n", ordered)
    
    # Check for points too close together
    min_dist = 10.0  # Minimum distance between points
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(ordered[i] - ordered[j])
            if dist < min_dist:
                print(f"[ERROR] Points {i} and {j} are too close: distance={dist}")
                sys.exit()
    
    return ordered

def calculate_dimensions(pts):
    w1 = np.linalg.norm(pts[0] - pts[1])
    w2 = np.linalg.norm(pts[2] - pts[3])
    h1 = np.linalg.norm(pts[0] - pts[3])
    h2 = np.linalg.norm(pts[1] - pts[2])
    w = int(max(w1, w2))
    h = int(max(h1, h2))
    w = max(w, 200)  # Minimum size
    h = max(h, 200)
    print("[DEBUG] Calculated dimensions: width=", w, "height=", h)
    return w, h

def validate_corners(points):
    if len(points) != 4:
        return False, "Not exactly 4 corners"
    unique_points = np.unique(points, axis=0)
    if len(unique_points) < 4:
        return False, f"Only {len(unique_points)} unique points detected"
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    signs = [np.sign(cross(points[i], points[(i+1)%4], points[(i+2)%4])) for i in range(4)]
    if len(set(signs)) != 1:
        return False, "Points don't form a convex quadrilateral"
    ordered = order_points(points)
    w, h = calculate_dimensions(ordered)
    if min(w, h) < 100:
        return False, f"Too small: {w}x{h}"
    if max(w, h) / min(w, h) > 10:
        return False, f"Extreme aspect ratio: {w}/{h}"
    return True, "Valid corners"

def draw_points(img, pts, filename="debug_points.png"):
    img_copy = img.copy()
    for i, p in enumerate(pts):
        cv2.circle(img_copy, tuple(p.astype(int)), 10, (0,0,255), -1)
        cv2.putText(img_copy, f"{i}", tuple(p.astype(int) + np.array([10,-10])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    if not cv2.imwrite(filename, img_copy):
        print(f"[ERROR] Failed to save {filename}")
    print(f"[INFO] Saved {filename}")
    return img_copy

def perspective_transform(img, pts):
    ordered = order_points(pts)
    w, h = calculate_dimensions(ordered)
    dst = np.array([[0,0],[w,0],[w,h],[0,h]], dtype="float32")

    print("[DEBUG] Destination points:\n", dst)

    # Additional validation for source points
    if np.any(np.isnan(ordered)) or np.any(np.isinf(ordered)):
        print("[ERROR] Invalid source points (NaN or Inf)")
        sys.exit()

    try:
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        print("[DEBUG] Perspective matrix:\n", matrix)
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            raise ValueError("Invalid perspective matrix (NaN or Inf values)")
    except cv2.error as e:
        print(f"[ERROR] Failed to compute perspective transform: {e}")
        sys.exit()

    warped = cv2.warpPerspective(img, matrix, (w, h))
    print("[DEBUG] Warped image shape:", warped.shape)
    print("[DEBUG] Warped image min/max pixel values:", warped.min(), warped.max())
    if warped.sum() == 0:
        print("[WARNING] Warped image is completely black/empty")

    # Save raw warped image
    if not cv2.imwrite("debug_warped_raw.png", warped):
        print("[ERROR] Failed to save debug_warped_raw.png")

    # Fallback transform with fixed points
    print("[INFO] Performing fallback transform with fixed points")
    test_w, test_h = 500, 500
    test_dst = np.array([[0,0],[test_w,0],[test_w,test_h],[0,test_h]], dtype="float32")
    img_h, img_w = img.shape[:2]
    margin = min(img_w, img_h) // 4
    test_ordered = np.array([
        [img_w//2 - margin, img_h//2 - margin],
        [img_w//2 + margin, img_h//2 - margin],
        [img_w//2 + margin, img_h//2 + margin],
        [img_w//2 - margin, img_h//2 + margin]
    ], dtype="float32")
    try:
        test_matrix = cv2.getPerspectiveTransform(test_ordered, test_dst)
        test_warped = cv2.warpPerspective(img, test_matrix, (test_w, test_h))
        print("[DEBUG] Test warped image min/max pixel values:", test_warped.min(), test_warped.max())
        if not cv2.imwrite("debug_test_warped.png", test_warped):
            print("[ERROR] Failed to save debug_test_warped.png")
        if test_warped.sum() == 0:
            print("[WARNING] Test warped image is completely black/empty")
    except cv2.error as e:
        print(f"[ERROR] Fallback transform failed: {e}")

    return warped, w, h

def correct_image_orientation(image_intput):
    img = image_intput
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if ids is not None:
        print(f"Detected ArUco markers with IDs: {ids}")
        marker_corners = corners[0][0]

        p1 = marker_corners[0]  # Top-left
        p2 = marker_corners[1]  # Top-right

        angle_rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        angle_deg = np.degrees(angle_rad)
        print(f"Calculated marker angle: {angle_deg} degrees")

        rotation_angle = angle_deg
        rotation_angle = round(angle_deg / 90) * 90

        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)

        # Compute rotation matrix
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Compute new bounding dimensions of the rotated image
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust rotation matrix to take into account translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Rotate with the new bounds
        rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)

        return rotated

    else:
        print("No ArUco markers detected in the image.")
        return None

def main(image_path):
    img, gray = load_image(image_path)
    img_copy = img.copy()

    aruco_detected = detect_aruco(gray, img)

    edged = preprocess_edges(gray)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.imwrite("debug_edges.png", edged)
        print("[ERROR] No contours found. Check 'debug_edges.png'")
        sys.exit()

    biggest = biggest_contour(contours)
    if biggest.size == 0:
        cv2.imwrite("debug_edges.png", edged)
        print("[ERROR] No valid rectangle found. Check 'debug_edges.png'")
        sys.exit()

    # Save raw contour points for debugging
    points = biggest.reshape(-1, 2)
    print("[DEBUG] Raw contour points:\n", points)
    draw_points(img_copy, points, "debug_raw_points.png")

    cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)
    valid, msg = validate_corners(points)
    if not valid:
        print(f"[ERROR] Corner validation failed: {msg}")
        sys.exit()
    print(f"[SUCCESS] Corners validated: {msg}")

    ordered_points = order_points(points)
    draw_points(img_copy, ordered_points, "debug_ordered_points.png")

    try:
        warped, w, h = perspective_transform(img_copy, points)
        rotated = correct_image_orientation(warped)
        if not cv2.imwrite("Warped_perspective_rotated.png", rotated, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            raise IOError("Failed to save Warped_perspective_improved.png")
        print(f"[SUCCESS] Perspective transform completed: {w}x{h}")
    except Exception as e:
        print(f"[ERROR] Perspective transform failed: {e}")
        print("[DEBUG] Exception type:", type(e).__name__)
        import traceback
        traceback.print_exc()

    if not cv2.imwrite("Contour_detection.png", img):
        print("[ERROR] Failed to save Contour_detection.png")
    print("[INFO] Saved 'Contour_detection.png'")

    print("\n=== DETECTION SUMMARY ===")
    print(f"ArUco Detection: {'YES' if aruco_detected else 'NO'}")
    print(f"Corner Detection: {'YES' if biggest.size != 0 else 'NO'}")

if __name__ == "__main__":
    main("/home/palm/kibo/test/undistort.png")